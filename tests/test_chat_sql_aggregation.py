from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.chat.orchestrator import _build_hybrid_prompt
from cultural_mood_tracker.chat.prompt_compression import (
    MAX_HYBRID_TOKENS,
    MAX_RAG_CHUNKS,
    compress_rag_chunks,
    estimate_tokens,
    format_sql_block,
)
from cultural_mood_tracker.chat.retrieval_rerank import chunk_type_distribution, filter_and_rerank_chunks
from cultural_mood_tracker.chat.sql_client import MotherDuckClient, extract_title
from cultural_mood_tracker.chat.sql_schemas import validate_sql_payload
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.retrieval import RetrievedChunk


class ChatSqlAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_project_environment(Path.cwd())
        if not os.getenv("MOTHERDUCK_TOKEN"):
            raise unittest.SkipTest("MOTHERDUCK_TOKEN is not configured")
        cls.client = MotherDuckClient()

    @classmethod
    def tearDownClass(cls) -> None:
        client = getattr(cls, "client", None)
        if client is not None:
            client.close()

    def test_get_title_ratings_returns_single_aggregate_object(self) -> None:
        result = self.client.get_title_ratings("Obsession")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["query_type"], "rating")
        self.assertEqual(result["title"], "Obsession")
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("rating_aggregate", result["results"][0])
        self.assertNotIn("rating_id", result)
        self.assertNotIn("author", result)

    def test_get_title_profile_is_compact_and_not_raw_rows(self) -> None:
        result = self.client.get_title_profile("Obsession")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["query_type"], "aggregate")
        self.assertEqual(result["title"], "Obsession")
        self.assertNotIsInstance(result, list)
        self.assertNotIn("review", str(result).lower())
        self.assertNotIn("rating_id", str(result))
        self.assertLessEqual(len(result["results"][0].get("cast", [])), 15)

    def test_hybrid_prompt_stays_under_3000_approx_tokens(self) -> None:
        structured_data = self.client.get_title_profile("Obsession")
        chunks = [
            RetrievedChunk(
                chunk_id=f"chunk_{index}",
                chunk_text="Obsession is described with concise retrieved context. " * 80,
                metadata={"title_name": "Obsession", "source_name": "tmdb", "document_type": "overview"},
                distance=0.2,
            )
            for index in range(5)
        ]

        prompt = _build_hybrid_prompt(
            query="Why is Obsession trending?",
            structured_data=structured_data,
            chunks=chunks,
        )

        self.assertLess(estimate_tokens(prompt.system_prompt + prompt.user_prompt), MAX_HYBRID_TOKENS)

    def test_raw_sql_rows_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_sql_payload([{"rating_id": "raw"}])  # type: ignore[arg-type]

    def test_sql_block_is_flat_and_compact(self) -> None:
        structured_data = self.client.get_title_profile("Obsession")
        block = format_sql_block(structured_data)

        self.assertIn('"query_type":"aggregate"', block)
        self.assertIn('"title":"Obsession"', block)
        self.assertNotIn("rating_id", block)
        self.assertLess(len(block), 1200)

    def test_rag_compression_removes_internal_metadata(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id=f"chunk_{index}",
                chunk_text="This is a retrieved interpretation snippet with extra text. " * 20,
                metadata={"title_name": "Obsession", "source_name": "guardian", "document_type": "review"},
                distance=0.2,
            )
            for index in range(8)
        ]

        block = compress_rag_chunks(chunks)

        self.assertEqual(block.count("["), MAX_RAG_CHUNKS)
        self.assertNotIn("chunk_id", block)
        self.assertNotIn("distance", block)

    def test_global_analytics_methods_are_aggregated(self) -> None:
        top_rated = self.client.get_top_rated_titles(limit=5)
        top_attention = self.client.get_top_attention_titles(limit=5)
        stats = self.client.get_rating_stats("Obsession")

        self.assertEqual(top_rated["query_type"], "aggregate")
        self.assertLessEqual(len(top_rated["results"]), 5)
        self.assertIn("avg_rating", top_rated["results"][0])
        self.assertNotIn("rating_id", str(top_rated))

        self.assertEqual(top_attention["query_type"], "aggregate")
        self.assertLessEqual(len(top_attention["results"]), 5)
        self.assertIn("attention_score", top_attention["results"][0])

        self.assertEqual(stats["query_type"], "rating")
        self.assertIn("avg_rating", stats["results"][0])
        self.assertNotIn("author", str(stats))

    def test_popularity_question_extracts_title(self) -> None:
        self.assertEqual(extract_title("Why is Obsession popular?"), "Obsession")

    def test_reranker_balances_chunk_types(self) -> None:
        chunks = [
            RetrievedChunk("overview", "overview text", {"document_type": "tmdb_overview", "source_name": "tmdb"}, 0.3),
            RetrievedChunk("editorial", "editorial text", {"document_type": "critic_article", "source_name": "guardian"}, 0.31),
            RetrievedChunk("review1", "review text", {"document_type": "user_review", "source_name": "tmdb"}, 0.32),
            RetrievedChunk("review2", "review text", {"document_type": "critic_review", "source_name": "vulture"}, 0.33),
            RetrievedChunk("theme", "theme text", {"document_type": "theme_summary", "source_name": "analytics"}, 0.34),
            RetrievedChunk("other", "other text", {"document_type": "other", "source_name": "other"}, 0.1),
        ]

        selected = filter_and_rerank_chunks(chunks)
        distribution = chunk_type_distribution(selected)

        self.assertLessEqual(len(selected), 4)
        self.assertGreaterEqual(distribution.get("tmdb_review", 0), 1)
        self.assertGreaterEqual(distribution.get("overview", 0), 1)
        self.assertGreaterEqual(distribution.get("editorial", 0), 1)

    def test_hybrid_prompt_uses_strict_sections(self) -> None:
        structured_data = self.client.get_title_profile("Obsession")
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_1",
                chunk_text="Obsession is discussed as culturally resonant.",
                metadata={"title_name": "Obsession", "source_name": "guardian", "document_type": "critic_article"},
                distance=0.2,
            )
        ]
        prompt = _build_hybrid_prompt(
            query="Why is Obsession popular?",
            structured_data=structured_data,
            chunks=chunks,
        )

        self.assertIn("STRUCTURED DATA (JSON):", prompt.user_prompt)
        self.assertIn("EVIDENCE (TOP 3 CHUNKS ONLY):", prompt.user_prompt)
        self.assertIn("If data is missing, explicitly say so", prompt.user_prompt)
        self.assertNotIn("chunk_id=", prompt.user_prompt)


if __name__ == "__main__":
    unittest.main()
