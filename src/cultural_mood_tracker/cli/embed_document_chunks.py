from __future__ import annotations

import argparse
from pathlib import Path

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag import DEFAULT_EMBEDDING_MODEL, embed_document_chunks, load_document_chunks
from cultural_mood_tracker.rag.document_chunks import DEFAULT_LOCAL_DOCUMENT_CHUNKS_PATH


def parse_args(default_process_run_id: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed processed document chunks into ChromaDB-ready JSONL records."
    )
    parser.add_argument(
        "--process-run-id",
        default=default_process_run_id,
        help="Processed run ID used for the output embedding file.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model name. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--output-name",
        default="document_chunk_embeddings.jsonl",
        help="Output filename written under the processed run directory.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding normalization. Normalized vectors are recommended for cosine retrieval.",
    )
    return parser.parse_args()


def run_embed_document_chunks(
    project_root: Path,
    process_run_id: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    output_name: str = "document_chunk_embeddings.jsonl",
    normalize_embeddings: bool = True,
) -> Path:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    processed_dir = paths.processed_root / process_run_id
    output_path = processed_dir / output_name
    chunks = load_document_chunks(settings.rag_data_source)
    count = embed_document_chunks(
        chunks,
        output_path,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    print(f"Embedded chunks written to: {output_path}")
    print(f"model={model_name} chunks={count} normalized={normalize_embeddings}")
    return output_path


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    default_process_run_id = DEFAULT_LOCAL_DOCUMENT_CHUNKS_PATH.parts[2]

    args = parse_args(default_process_run_id)
    if not args.process_run_id:
        raise RuntimeError("No processed run available. Run transform first or pass --process-run-id.")

    run_embed_document_chunks(
        project_root,
        args.process_run_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_name=args.output_name,
        normalize_embeddings=not args.no_normalize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
