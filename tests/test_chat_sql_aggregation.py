from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.chat.orchestrator import (
    _build_comparison_hybrid_prompt,
    _build_comparison_hybrid_analytics_prompt,
    _build_hybrid_analytics_prompt,
    _build_comparison_rag_prompt,
    _build_hybrid_prompt,
    _build_recommendation_prompt,
    _classify_hybrid_query,
    _format_fast_sql_answer,
    rank_recommendation_candidates,
)
from cultural_mood_tracker.chat.prompt_compression import (
    MAX_HYBRID_TOKENS,
    MAX_RAG_CHUNKS,
    compress_rag_chunks,
    estimate_tokens,
    format_sql_block,
)
from cultural_mood_tracker.chat.retrieval_rerank import chunk_type_distribution, filter_and_rerank_chunks
from cultural_mood_tracker.chat.router import route_query, route_query_debug
from cultural_mood_tracker.chat.sql_client import MotherDuckClient, extract_comparison_titles, extract_title
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

    def test_recommendation_queries_route_to_recommendation_mode(self) -> None:
        self.assertEqual(route_query("What should I watch if I feel sad?"), "recommendation")
        self.assertEqual(route_query("Recommend something comforting."), "recommendation")
        self.assertEqual(route_query("Suggest a nostalgic sci-fi movie."), "recommendation")

    def test_structured_queries_route_to_fast_sql(self) -> None:
        self.assertEqual(route_query("What is the rating for Obsession?"), "fast_sql")
        self.assertEqual(route_query("List the top rated titles."), "fast_sql")
        self.assertEqual(route_query("What is the cast of Obsession?"), "fast_sql")

    def test_weighted_router_debug_scores_and_boundaries(self) -> None:
        debug = route_query_debug("What should I watch if I feel sad?")

        self.assertEqual(debug["mode"], "recommendation")
        self.assertGreater(debug["scores"]["recommendation"], debug["scores"]["rag"])
        self.assertTrue(debug["matched_signals"])

        self.assertEqual(route_query("Compare Obsession vs Leviticus."), "hybrid")
        self.assertEqual(route_query("Explain the darkness of Obsession."), "rag")
        self.assertEqual(route_query("Is Obsession more popular than Leviticus?"), "hybrid")
        self.assertEqual(route_query("Compare Obsession and Leviticus in terms of audience reception."), "hybrid")

    def test_emotional_comparisons_route_to_hybrid_not_recommendation(self) -> None:
        self.assertEqual(route_query("Which is more uplifting: Obsession or Leviticus?"), "hybrid")
        self.assertEqual(route_query("Which is more emotional: Obsession or Disclosure Day?"), "hybrid")
        self.assertEqual(route_query("Compare dark tone in Obsession vs Leviticus."), "hybrid")

    def test_emotion_words_need_selection_intent_for_recommendation(self) -> None:
        self.assertEqual(route_query("uplifting"), "rag")
        self.assertEqual(route_query("Recommend something uplifting."), "recommendation")
        self.assertEqual(route_query("Suggest a nostalgic sci-fi movie."), "recommendation")
        self.assertEqual(route_query("What should I watch if I feel sad or nostalgic?"), "recommendation")

    def test_recommendation_summary_methods_are_compact(self) -> None:
        title_rows = self.client.get_title_theme_summary("Recommend something comforting.")
        genre_rows = self.client.get_genre_theme_summary("Recommend something comforting.")
        audience_rows = self.client.get_audience_editorial_summary("Recommend something comforting.")

        self.assertLessEqual(len(title_rows), 8)
        self.assertLessEqual(len(genre_rows), 6)
        self.assertLessEqual(len(audience_rows), 8)
        self.assertNotIn("chunk_text", str(title_rows))
        self.assertNotIn("review_text", str(audience_rows))
        if title_rows:
            self.assertIn("dominant_themes", title_rows[0])

    def test_recommendation_prompt_is_summary_first(self) -> None:
        prompt = _build_recommendation_prompt(
            query="Recommend something comforting.",
            candidates=[
                {
                    "title": "Scary Movie",
                    "dominant_themes": ["comfort", "nostalgia", "identity", "anxiety"],
                    "emotional_tone": "mixed",
                    "avg_rating": 8.1,
                    "attention_score": 34000,
                    "evidence_count": 10,
                }
            ],
            genre_theme_rows=[{"genre": "Comedy", "dominant_themes": ["comfort"], "title_count": 3}],
            fallback_chunks=[],
        )

        self.assertIn("CANDIDATES", prompt.user_prompt)
        self.assertIn("Scary Movie: themes=[comfort, nostalgia, identity]", prompt.user_prompt)
        self.assertIn("GENRE SIGNALS", prompt.user_prompt)
        self.assertIn("Recommend exactly 3 titles", prompt.user_prompt)
        self.assertIn("Why it fits: max 2 sentences", prompt.user_prompt)
        self.assertNotIn("AUDIENCE VS EDITORIAL", prompt.user_prompt)

    def test_recommendation_candidates_are_ranked_and_prompt_is_compact(self) -> None:
        ranked = rank_recommendation_candidates(
            [
                {"title": "Low", "avg_rating": 5.0, "attention_score": 100, "dominant_themes": ["comfort"]},
                {"title": "High", "avg_rating": 8.0, "attention_score": 30000, "dominant_themes": ["comfort", "nostalgia"]},
            ]
        )
        self.assertEqual(ranked[0]["title"], "High")

        prompt = _build_recommendation_prompt(
            query="Recommend something comforting.",
            candidates=ranked[:3],
            genre_theme_rows=[],
            fallback_chunks=[],
        )
        self.assertLess(estimate_tokens(prompt.system_prompt + prompt.user_prompt), 1200)
        self.assertNotIn("full", prompt.user_prompt.lower())

    def test_comparison_title_extraction_preserves_both_titles(self) -> None:
        self.assertEqual(
            self.client.extract_titles("Compare Obsession and Leviticus.")[:2],
            ["Obsession", "Leviticus"],
        )
        self.assertEqual(
            self.client.extract_titles("Which is more positively received: Leviticus or Obsession?")[:2],
            ["Leviticus", "Obsession"],
        )
        self.assertEqual(
            self.client.extract_titles("Compare audience reception of Disclosure Day and Obsession.")[:2],
            ["Disclosure Day", "Obsession"],
        )
        self.assertEqual(
            extract_comparison_titles("Compare Obsession and Leviticus — which is more positively received?"),
            ("Obsession", "Leviticus"),
        )

    def test_comparison_rag_prompt_groups_evidence_by_title(self) -> None:
        grouped_chunks = {
            "Obsession": [
                RetrievedChunk(
                    "obsession_1",
                    "Audience responses describe Obsession as tense and ambiguous.",
                    {"title_name": "Obsession", "source_name": "tmdb", "document_type": "review"},
                    0.2,
                )
            ],
            "Leviticus": [
                RetrievedChunk(
                    "leviticus_1",
                    "Reviewers describe Leviticus as darker and more severe.",
                    {"title_name": "Leviticus", "source_name": "guardian", "document_type": "critic_article"},
                    0.25,
                )
            ],
        }

        prompt = _build_comparison_rag_prompt(
            query="Compare Obsession and Leviticus.",
            grouped_chunks=grouped_chunks,
            max_context_chars=2000,
        )

        self.assertIn("TITLE A\nObsession\nEvidence", prompt.user_prompt)
        self.assertIn("TITLE B\nLeviticus\nEvidence", prompt.user_prompt)
        self.assertLess(prompt.user_prompt.index("TITLE A"), prompt.user_prompt.index("TITLE B"))
        self.assertEqual(prompt.included_chunk_ids, ["obsession_1", "leviticus_1"])

    def test_comparison_hybrid_prompt_separates_sql_and_text_by_title(self) -> None:
        structured_data = {
            "query_type": "comparison",
            "title": None,
            "results": [
                {"title": "Obsession", "rating_aggregate": 7.5, "themes": {"dominant_themes": ["anxiety"]}},
                {"title": "Leviticus", "rating_aggregate": 6.8, "themes": {"dominant_themes": ["loneliness"]}},
            ],
            "summary_metrics": {"titles": ["Obsession", "Leviticus"]},
        }
        grouped_chunks = {
            "Obsession": [
                RetrievedChunk("obsession_1", "Obsession evidence.", {"source_name": "tmdb"}, 0.2),
            ],
            "Leviticus": [
                RetrievedChunk("leviticus_1", "Leviticus evidence.", {"source_name": "guardian"}, 0.25),
            ],
        }

        prompt = _build_comparison_hybrid_prompt(
            query="Compare Obsession and Leviticus.",
            structured_data=structured_data,
            grouped_chunks=grouped_chunks,
        )

        self.assertIn("STRUCTURED DATA BY TITLE", prompt.user_prompt)
        self.assertIn("TEXTUAL EVIDENCE BY TITLE", prompt.user_prompt)
        self.assertIn("TITLE A\nObsession\nEvidence", prompt.user_prompt)
        self.assertIn("TITLE B\nLeviticus\nEvidence", prompt.user_prompt)

    def test_hybrid_query_type_classification(self) -> None:
        self.assertEqual(_classify_hybrid_query("Why is Obsession popular?"), "POPULARITY_EXPLANATION")
        self.assertEqual(
            _classify_hybrid_query("Compare audience reception vs ratings for Obsession"),
            "ATTENTION_VS_RECEPTION",
        )
        self.assertEqual(
            _classify_hybrid_query("How does attention compare to rating for Obsession?"),
            "ATTENTION_VS_RATING",
        )
        self.assertEqual(_classify_hybrid_query("Compare Obsession and Leviticus."), "TITLE_COMPARISON")

    def test_fast_sql_answer_formats_structured_data_without_llm(self) -> None:
        structured = self.client.get_title_ratings("Obsession")
        answer = _format_fast_sql_answer(structured)

        self.assertIn("Obsession", answer)
        self.assertIn("rating", answer)

    def test_hybrid_analytics_sql_helpers_return_metrics_and_summaries(self) -> None:
        metrics = self.client.get_title_metrics("Obsession")
        analytics = self.client.get_title_analytical_summaries("Obsession")
        avr = self.client.get_attention_vs_reception("Obsession")

        self.assertEqual(metrics["title"], "Obsession")
        self.assertIn("avg_rating", metrics)
        self.assertIn("attention_score", metrics)
        self.assertEqual(analytics["title"], "Obsession")
        self.assertIn("title_theme_summary", analytics)
        self.assertIn("audience_vs_editorial_summary", analytics)
        self.assertIn("attention_vs_reception", analytics)
        self.assertIn("attention_percentile", avr)

    def test_hybrid_analytics_prompt_uses_required_sections_and_caps_reviews(self) -> None:
        chunks = [
            RetrievedChunk(
                f"chunk_{index}",
                "Review evidence about audience response and interpretation. " * 10,
                {"title_name": "Obsession", "source_name": "tmdb", "document_type": "review"},
                0.2,
            )
            for index in range(5)
        ]
        prompt = _build_hybrid_analytics_prompt(
            query="Why is Obsession popular?",
            metrics={"title": "Obsession", "avg_rating": 8.1, "attention_score": 100.0},
            analytics={"attention_vs_reception": {"attention_percentile": 1.0}},
            chunks=chunks,
        )

        self.assertIn("STRUCTURED METRICS", prompt.user_prompt)
        self.assertIn("ANALYTICAL SUMMARIES", prompt.user_prompt)
        self.assertIn("REVIEW EVIDENCE", prompt.user_prompt)
        self.assertIn("1.\nKey facts", prompt.user_prompt)
        self.assertEqual(len(prompt.included_chunk_ids), 3)

    def test_comparison_hybrid_analytics_prompt_groups_metrics_summaries_and_evidence(self) -> None:
        structured_data = self.client.compare_titles("Obsession", "Leviticus")
        grouped_chunks = {
            "Obsession": [RetrievedChunk("obsession_1", "Obsession review evidence.", {"source_name": "tmdb"}, 0.2)],
            "Leviticus": [RetrievedChunk("leviticus_1", "Leviticus review evidence.", {"source_name": "guardian"}, 0.25)],
        }
        prompt = _build_comparison_hybrid_analytics_prompt(
            query="Compare Obsession and Leviticus.",
            structured_data=structured_data,
            grouped_chunks=grouped_chunks,
        )

        self.assertIn("STRUCTURED METRICS", prompt.user_prompt)
        self.assertIn("ANALYTICAL SUMMARIES", prompt.user_prompt)
        self.assertIn("TITLE A\nObsession\nEvidence", prompt.user_prompt)
        self.assertIn("TITLE B\nLeviticus\nEvidence", prompt.user_prompt)


if __name__ == "__main__":
    unittest.main()
