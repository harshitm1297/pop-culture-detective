from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chatbot import Chatbot, RetrievalRoute
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS
from cultural_mood_tracker.rag.retrieval import DEFAULT_QUERY_INSTRUCTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive local RAG chatbot.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per question.")
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
        "--embedding-model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model used for retrieval. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--llm-model-name",
        default=DEFAULT_MODEL,
        help=f"Local HuggingFace instruction model. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Character budget for retrieved context. Defaults to {DEFAULT_MAX_CONTEXT_CHARS}.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Drop retrieved chunks below this cosine similarity before prompt construction.",
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
        help="Optional metadata filter on source_name.",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print retrieved title/source/similarity evidence after each answer.",
    )
    parser.add_argument(
        "--no-query-instruction",
        action="store_true",
        help="Disable the BGE query instruction prefix.",
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


def _print_evidence(response) -> None:
    print("\nEvidence:")
    for index, item in enumerate(response.evidence, start=1):
        title = item.title or "Unknown title"
        source = item.source or "unknown_source"
        doc_type = item.document_type or "text"
        print(f"  [{index}] similarity={item.similarity:.4f} title={title} source={source}/{doc_type}")


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    persist_dir = args.persist_dir if args.persist_dir.is_absolute() else project_root / args.persist_dir
    chatbot = Chatbot(
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model_name,
        llm_model_name=args.llm_model_name,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        min_similarity=args.min_similarity,
        query_instruction="" if args.no_query_instruction else DEFAULT_QUERY_INSTRUCTION,
    )
    route = RetrievalRoute(where=_build_where(args.content_type, args.source_name))

    print("RAG chatbot ready. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return 0

        try:
            response = chatbot.answer_question(question, route=route)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"\nAnswer:\n{response.answer}")
        if args.show_evidence:
            _print_evidence(response)


if __name__ == "__main__":
    raise SystemExit(main())
