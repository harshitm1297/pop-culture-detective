from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.chat.orchestrator import ChatOrchestrator
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Cultural Mood Tracker chat orchestrator.")
    parser.add_argument(
        "--query",
        default=None,
        help="Natural-language question to answer. If omitted, starts an interactive chat loop.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of ChromaDB chunks for RAG/hybrid mode.")
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
        help=f"Groq chat model. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Context character budget for RAG mode. Defaults to {DEFAULT_MAX_CONTEXT_CHARS}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the required structured response as JSON.",
    )
    return parser.parse_args()


def _print_response(response: dict) -> None:
    print(f"Mode: {response['mode']}")
    print(f"Used SQL: {response['used_sql']}")
    if response["retrieved_chunk_ids"]:
        print(f"Retrieved chunks: {', '.join(response['retrieved_chunk_ids'])}")
    print()
    print(response["answer"])


def _answer_once(orchestrator: ChatOrchestrator, query: str, *, as_json: bool) -> None:
    result = orchestrator.answer(query)
    response = result.to_response()
    if as_json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        _print_response(response)


def _interactive_loop(orchestrator: ChatOrchestrator, *, as_json: bool) -> int:
    print("Unified Cultural Mood Tracker chat ready. Type a question, or 'exit' to quit.")
    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            return 0

        try:
            _answer_once(orchestrator, query, as_json=as_json)
        except Exception as exc:
            print(f"Error: {exc}")


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    persist_dir = args.persist_dir if args.persist_dir.is_absolute() else project_root / args.persist_dir
    orchestrator = ChatOrchestrator(
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model_name,
        llm_model_name=args.llm_model_name,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )

    if args.query:
        _answer_once(orchestrator, args.query, as_json=args.json)
        return 0

    _interactive_loop(orchestrator, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
