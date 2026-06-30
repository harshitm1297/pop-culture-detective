from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHROMA_DB_DIR = "chroma_db"
DEFAULT_CHROMA_COLLECTION = "movie_chunks"
LOGGER = logging.getLogger(__name__)


PrimitiveMetadata = str | int | float | bool


@dataclass(frozen=True)
class ChromaEmbeddingRecord:
    id: str
    document: str
    embedding: list[float]
    metadata: dict[str, PrimitiveMetadata]


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
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
            yield row


def _metadata_value(value: Any) -> PrimitiveMetadata | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _sanitize_metadata(metadata: Any, line_number: int) -> dict[str, PrimitiveMetadata]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Record {line_number} has non-object metadata")

    sanitized: dict[str, PrimitiveMetadata] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"Record {line_number} has a non-string metadata key")
        chroma_value = _metadata_value(value)
        if chroma_value is not None:
            sanitized[key] = chroma_value
    return sanitized


def _coerce_embedding(value: Any, line_number: int) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Record {line_number} has missing or invalid embedding")

    embedding: list[float] = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)):
            raise ValueError(f"Record {line_number} embedding[{index}] is not numeric")
        embedding.append(float(item))
    return embedding


def _parse_record(row: dict[str, Any], line_number: int) -> ChromaEmbeddingRecord:
    record_id = row.get("id")
    document = row.get("document")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"Record {line_number} has missing or invalid id")
    if not isinstance(document, str) or not document.strip():
        raise ValueError(f"Record {line_number} has missing or invalid document")

    return ChromaEmbeddingRecord(
        id=record_id,
        document=document,
        embedding=_coerce_embedding(row.get("embedding"), line_number),
        metadata=_sanitize_metadata(row.get("metadata"), line_number),
    )


def load_embedding_records(path: Path) -> list[ChromaEmbeddingRecord]:
    return [_parse_record(row, line_number) for line_number, row in enumerate(_load_jsonl(path), start=1)]


def _batches(records: list[ChromaEmbeddingRecord], batch_size: int) -> Iterator[list[ChromaEmbeddingRecord]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _ids(records: Iterable[ChromaEmbeddingRecord]) -> list[str]:
    return [record.id for record in records]


def _documents(records: Iterable[ChromaEmbeddingRecord]) -> list[str]:
    return [record.document for record in records]


def _embeddings(records: Iterable[ChromaEmbeddingRecord]) -> list[list[float]]:
    return [record.embedding for record in records]


def _metadatas(records: Iterable[ChromaEmbeddingRecord]) -> list[dict[str, PrimitiveMetadata]]:
    return [record.metadata for record in records]


def _disable_chroma_telemetry_noise() -> None:
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    try:
        import posthog
    except ImportError:
        return
    posthog.disabled = True


def ingest_embedding_records(
    records: list[ChromaEmbeddingRecord],
    *,
    persist_dir: Path,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    batch_size: int = 500,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install chromadb before ingesting vectors.") from exc

    persist_dir.mkdir(parents=True, exist_ok=True)
    _disable_chroma_telemetry_noise()
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    inserted = 0
    for batch in _batches(records, batch_size):
        collection.upsert(
            ids=_ids(batch),
            embeddings=_embeddings(batch),
            documents=_documents(batch),
            metadatas=_metadatas(batch),
        )
        inserted += len(batch)
        LOGGER.info("Inserted %s/%s records into ChromaDB", inserted, len(records))

    return inserted


def ingest_embeddings_file(
    input_path: Path,
    *,
    persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    batch_size: int = 500,
) -> int:
    records = load_embedding_records(input_path)
    LOGGER.info("Loaded %s embedding records from %s", len(records), input_path)
    inserted = ingest_embedding_records(
        records,
        persist_dir=persist_dir,
        collection_name=collection_name,
        batch_size=batch_size,
    )
    LOGGER.info("Inserted %s embedding records into %s/%s", inserted, persist_dir, collection_name)
    return inserted
