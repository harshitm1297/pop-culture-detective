from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.retrieval import (
    DEFAULT_QUERY_INSTRUCTION,
    RetrievedChunk,
    query_collection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve the top-k most relevant document chunks for a natural-language query."
    )
    parser.add_argument("--query", required=True, help="Natural-language search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--persist-dir",
        default=Path(DEFAULT_CHROMA_DB_DIR),
        type=Path,
        help=f"Directory of the persistent ChromaDB database. Defaults to ./{DEFAULT_CHROMA_DB_DIR}.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_CHROMA_COLLECTION,
        help=f"ChromaDB collection name. Defaults to {DEFAULT_CHROMA_COLLECTION}.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "SentenceTransformer model name. Must match the model used at ingest time, "
            f"otherwise vector dimensions or semantics will not line up. Defaults to {DEFAULT_EMBEDDING_MODEL}."
        ),
    )
    parser.add_argument(
        "--content-type",
        choices=["movie", "tv"],
        default=None,
        help="Optional metadata filter on content_type.",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Optional metadata filter on source_name (e.g. tmdb, guardian, tvmaze, wikidata).",
    )
    parser.add_argument(
        "--no-query-instruction",
        action="store_true",
        help="Disable the BGE query instruction prefix (for debugging/comparison only).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to also write results as JSON.",
    )
    return parser.parse_args()


def _build_where(content_type: str | None, source_name: str | None) -> dict | None:
    clauses = []
    if content_type:
        clauses.append({"content_type": content_type})
    if source_name:
        clauses.append({"source_name": source_name})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _print_results(query: str, chunks: list[RetrievedChunk]) -> None:
    print(f"Query: {query}")
    print(f"Retrieved {len(chunks)} chunk(s)\n")
    for rank, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        preview = chunk.chunk_text[:220] + ("..." if len(chunk.chunk_text) > 220 else "")
        print(f"[{rank}] similarity={chunk.similarity:.4f} distance={chunk.distance:.4f}")
        print(f"    chunk_id: {chunk.chunk_id}")
        print(f"    title:    {meta.get('title_name')} ({meta.get('release_year')}) [{meta.get('content_type')}]")
        print(f"    source:   {meta.get('source_name')} / {meta.get('document_type')}")
        print(f"    text:     {preview}")
        print()


def _write_output(output_path: Path, project_root: Path, query: str, chunks: list[RetrievedChunk]) -> Path:
    resolved = output_path if output_path.is_absolute() else project_root / output_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "results": [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.chunk_text,
                "metadata": chunk.metadata,
                "distance": chunk.distance,
                "similarity": chunk.similarity,
            }
            for chunk in chunks
        ],
    }
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    persist_dir = args.persist_dir
    if not persist_dir.is_absolute():
        persist_dir = project_root / persist_dir

    chunks = query_collection(
        args.query,
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
        top_k=args.top_k,
        where=_build_where(args.content_type, args.source_name),
        query_instruction="" if args.no_query_instruction else DEFAULT_QUERY_INSTRUCTION,
    )

    _print_results(args.query, chunks)

    if args.output_path:
        written = _write_output(args.output_path, project_root, args.query, chunks)
        print(f"Results written to: {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
