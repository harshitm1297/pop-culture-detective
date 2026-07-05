from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment


DEFAULT_LOCAL_DOCUMENT_CHUNKS_PATH = Path("data/processed/20260702T131134Z/document_chunks.jsonl")
SUPPORTED_DOCUMENT_CHUNK_SOURCES = {"local", "motherduck"}


def _project_root() -> Path:
    return load_project_environment(Path.cwd())


def _parse_metadata(value: Any, *, row_number: int) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Row {row_number} has invalid JSON metadata") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Row {row_number} has non-object metadata")


def _normalize_chunk_row(row: dict[str, Any], *, row_number: int) -> dict[str, Any]:
    chunk_id = row.get("chunk_id")
    chunk_text = row.get("chunk_text")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"Row {row_number} is missing a non-empty chunk_id")
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        raise ValueError(f"Row {row_number} is missing non-empty chunk_text")

    if "metadata" in row:
        metadata = _parse_metadata(row.get("metadata"), row_number=row_number)
    else:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"chunk_id", "chunk_text"} and value is not None
        }

    return {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "metadata": metadata,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
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
            chunks.append(_normalize_chunk_row(row, row_number=line_number))
    return chunks


def _load_csv(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            chunks.append(_normalize_chunk_row(dict(row), row_number=row_number))
    return chunks


def _resolve_local_path(project_root: Path) -> Path:
    jsonl_path = project_root / DEFAULT_LOCAL_DOCUMENT_CHUNKS_PATH
    if jsonl_path.exists():
        return jsonl_path

    csv_path = jsonl_path.with_suffix(".csv")
    if csv_path.exists():
        return csv_path

    raise RuntimeError(f"Missing local document chunks file: {jsonl_path} or {csv_path}")


def _load_local_document_chunks() -> list[dict[str, Any]]:
    path = _resolve_local_path(_project_root())
    if path.suffix.lower() == ".csv":
        chunks = _load_csv(path)
    else:
        chunks = _load_jsonl(path)
    print(f"RAG document chunk source: local ({path})")
    print(f"Loaded document chunks: {len(chunks)}")
    return chunks


def _load_motherduck_document_chunks() -> list[dict[str, Any]]:
    _project_root()
    settings = load_settings()
    if not settings.motherduck_token:
        raise RuntimeError("Missing required environment variable: MOTHERDUCK_TOKEN")
    if not settings.motherduck_database:
        raise RuntimeError("Missing required environment variable: MOTHERDUCK_DATABASE")

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install duckdb before loading chunks from MotherDuck.") from exc

    con = duckdb.connect(
        f"md:{settings.motherduck_database}",
        config={"motherduck_token": settings.motherduck_token},
    )
    try:
        relation = con.execute("SELECT * FROM document_chunks")
        column_names = [column[0] for column in relation.description]
        rows = relation.fetchall()
    finally:
        con.close()

    chunks = [
        _normalize_chunk_row(dict(zip(column_names, row, strict=True)), row_number=index)
        for index, row in enumerate(rows, start=1)
    ]
    print(f"RAG document chunk source: motherduck (md:{settings.motherduck_database}.document_chunks)")
    print(f"Loaded document chunks: {len(chunks)}")
    return chunks


def load_document_chunks(source: str) -> list[dict[str, Any]]:
    normalized_source = source.strip().lower()
    if normalized_source not in SUPPORTED_DOCUMENT_CHUNK_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_CHUNK_SOURCES))
        raise ValueError(f"Unsupported RAG document chunk source: {source!r}. Expected one of: {supported}")
    if normalized_source == "local":
        return _load_local_document_chunks()
    return _load_motherduck_document_chunks()
