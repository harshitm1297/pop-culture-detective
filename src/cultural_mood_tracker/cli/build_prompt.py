from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.prompting import (
    DEFAULT_MAX_CONTEXT_CHARS,
    build_prompt,
    load_retrieved_chunks_json,
)
from cultural_mood_tracker.rag.retrieval import DEFAULT_QUERY_INSTRUCTION, query_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a grounded RAG prompt from retrieved chunks. Either point at a saved "
            "retrieve.py --output-path JSON file (offline, no ChromaDB/model calls), or pass "
            "--query to retrieve live."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-path",
        type=Path,
        help="Path to a JSON file written by `retrieve.py --output-path` (offline mode).",
    )
    source.add_argument("--query", help="Natural-language query to retrieve live before building the prompt.")

    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve in live mode.")
    parser.add_argument(
        "--persist-dir",
        default=Path(DEFAULT_CHROMA_DB_DIR),
        type=Path,
        help=f"ChromaDB persist directory for live mode. Defaults to ./{DEFAULT_CHROMA_DB_DIR}.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_CHROMA_COLLECTION,
        help=f"ChromaDB collection name for live mode. Defaults to {DEFAULT_CHROMA_COLLECTION}.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model for live mode. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Character budget for assembled context. Defaults to {DEFAULT_MAX_CONTEXT_CHARS}.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Drop chunks below this cosine similarity before building the prompt (e.g. 0.3).",
    )
    parser.add_argument("--output-path", type=Path, default=None, help="Optional path to write the full PromptResult as JSON.")
    return parser.parse_args()


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.input_path:
        input_path = args.input_path if args.input_path.is_absolute() else project_root / args.input_path
        query, chunks = load_retrieved_chunks_json(input_path)
        print(f"Loaded {len(chunks)} chunk(s) from {input_path} (offline mode, no retrieval call made)")
    else:
        persist_dir = args.persist_dir if args.persist_dir.is_absolute() else project_root / args.persist_dir
        query = args.query
        chunks = query_collection(
            query,
            persist_dir=persist_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            top_k=args.top_k,
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )
        print(f"Retrieved {len(chunks)} chunk(s) live from {persist_dir}/{args.collection_name}")

    result = build_prompt(
        query,
        chunks,
        max_context_chars=args.max_context_chars,
        min_similarity=args.min_similarity,
    )

    print(f"\nIncluded chunks ({len(result.included_chunk_ids)}): {result.included_chunk_ids}")
    print(f"Excluded chunks ({len(result.excluded_chunk_ids)}): {result.excluded_chunk_ids}")
    print(f"Context size: {result.context_char_count} chars\n")
    print("=" * 80)
    print(result.flattened)
    print("=" * 80)

    if args.output_path:
        output_path = args.output_path if args.output_path.is_absolute() else project_root / args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": query,
            "system_prompt": result.system_prompt,
            "user_prompt": result.user_prompt,
            "messages": result.messages,
            "included_chunk_ids": result.included_chunk_ids,
            "excluded_chunk_ids": result.excluded_chunk_ids,
            "context_char_count": result.context_char_count,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull prompt result written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
