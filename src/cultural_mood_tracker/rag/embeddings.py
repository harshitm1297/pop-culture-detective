from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_RESERVED_FIELDS = {"chunk_id", "chunk_text"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _metadata_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _chroma_metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    source_metadata = row.get("metadata")
    if isinstance(source_metadata, dict):
        items = source_metadata.items()
    else:
        items = ((key, value) for key, value in row.items() if key not in CHROMA_RESERVED_FIELDS)

    for key, value in items:
        if not isinstance(key, str):
            continue
        chroma_value = _metadata_value(value)
        if chroma_value is not None:
            metadata[key] = chroma_value
    return metadata


def _validate_chunk(row: dict[str, Any], index: int) -> tuple[str, str]:
    chunk_id = row.get("chunk_id")
    chunk_text = row.get("chunk_text")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"Chunk row {index} is missing a non-empty string chunk_id")
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        raise ValueError(f"Chunk row {index} is missing non-empty string chunk_text")
    return chunk_id, chunk_text


def _readable_row_number(batch_start: int, offset: int) -> int:
    return batch_start + offset + 1


def build_chroma_embedding_records(
    chunks: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> list[dict[str, Any]]:
    """Embed document chunks into records that can be passed directly to ChromaDB add/upsert."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install sentence-transformers before embedding chunks."
        ) from exc

    model = SentenceTransformer(model_name)
    records: list[dict[str, Any]] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        validated = [
            _validate_chunk(row, _readable_row_number(batch_start, offset))
            for offset, row in enumerate(batch)
        ]
        texts = [chunk_text for _, chunk_text in validated]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )

        for row, (chunk_id, chunk_text), embedding in zip(batch, validated, embeddings, strict=True):
            records.append(
                {
                    "id": chunk_id,
                    "document": chunk_text,
                    "metadata": _chroma_metadata(row),
                    "embedding": embedding.tolist(),
                }
            )

    return records


def embed_document_chunks(
    chunks: list[dict[str, Any]],
    output_path: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> int:
    records = build_chroma_embedding_records(
        chunks,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    return _write_jsonl(output_path, records)


def embed_document_chunks_file(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> int:
    chunks = _load_jsonl(input_path)
    return embed_document_chunks(
        chunks,
        output_path,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
