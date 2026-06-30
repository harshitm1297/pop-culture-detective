from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_DIR,
    ingest_embeddings_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load precomputed document chunk embeddings into a persistent ChromaDB collection."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        type=Path,
        help="Path to document_chunk_embeddings.jsonl.",
    )
    parser.add_argument(
        "--persist-dir",
        default=Path(DEFAULT_CHROMA_DB_DIR),
        type=Path,
        help=f"Directory for the persistent ChromaDB database. Defaults to ./{DEFAULT_CHROMA_DB_DIR}.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_CHROMA_COLLECTION,
        help=f"ChromaDB collection name. Defaults to {DEFAULT_CHROMA_COLLECTION}.",
    )
    parser.add_argument(
        "--batch-size",
        default=500,
        type=int,
        help="Number of vectors to upsert per ChromaDB call.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    input_path = args.input_path
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not input_path.exists():
        raise RuntimeError(f"Missing embedding JSONL file: {input_path}")

    persist_dir = args.persist_dir
    if not persist_dir.is_absolute():
        persist_dir = project_root / persist_dir

    inserted = ingest_embeddings_file(
        input_path,
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
    )
    print(f"ChromaDB persisted at: {persist_dir}")
    print(f"collection={args.collection_name} inserted={inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
