from __future__ import annotations

from pathlib import Path
from typing import Any

from cultural_mood_tracker.transform.common import load_json


BASE_TABLE_NAMES = (
    "titles",
    "documents",
    "document_chunks",
    "ratings",
    "attention_signals",
)

OPTIONAL_TABLE_NAMES = (
    "people",
    "title_cast",
    "title_crew",
)

TABLE_CASTS: dict[str, dict[str, str]] = {
    "titles": {
        "release_date": "DATE",
    },
    "documents": {
        "published_at": "TIMESTAMPTZ",
    },
    "document_chunks": {
        "published_at": "TIMESTAMPTZ",
    },
    "ratings": {
        "published_at": "TIMESTAMPTZ",
    },
    "attention_signals": {
        "timestamp_utc": "TIMESTAMPTZ",
    },
}


def _import_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError(
            "Missing dependency duckdb. Install requirements before running MotherDuck upload."
        ) from exc
    return duckdb


def create_motherduck_connection(*, token: str):
    duckdb = _import_duckdb()
    if not token.strip():
        raise RuntimeError("Missing required environment variable: MOTHERDUCK_TOKEN")
    return duckdb.connect(f"md:?motherduck_token={token.strip()}")


def ensure_database(*, connection, database_name: str) -> None:
    if not database_name.strip():
        raise RuntimeError("Missing required environment variable: MOTHERDUCK_DATABASE")
    safe_name = database_name.replace('"', '""')
    connection.execute(f'CREATE DATABASE IF NOT EXISTS "{safe_name}"')
    connection.execute(f'USE "{safe_name}"')


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _build_create_table_sql(*, table_name: str, source_path: Path) -> str:
    cast_config = TABLE_CASTS.get(table_name, {})
    source_sql = _sql_path(source_path)
    if not cast_config:
        return f'''
            CREATE OR REPLACE TABLE "{table_name}" AS
            SELECT *
            FROM read_json_auto('{source_sql}')
        '''

    replacements = ",\n                ".join(
        (
            f"TRY_CAST({column_name} AS {target_type}) AS {column_name}"
            for column_name, target_type in cast_config.items()
        )
    )
    return f'''
        CREATE OR REPLACE TABLE "{table_name}" AS
        WITH source AS (
            SELECT *
            FROM read_json_auto('{source_sql}')
        )
        SELECT * REPLACE (
                {replacements}
        )
        FROM source
    '''


def _load_process_manifest(processed_dir: Path) -> dict[str, Any]:
    manifest_path = processed_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing processed manifest: {manifest_path}")
    return load_json(manifest_path)


def load_processed_tables(
    *,
    connection,
    processed_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    loaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for table_name in (*BASE_TABLE_NAMES, *OPTIONAL_TABLE_NAMES):
        source_path = processed_dir / f"{table_name}.jsonl"
        if not source_path.exists():
            skipped.append({"table_name": table_name, "reason": "missing_file"})
            continue

        safe_table = table_name.replace('"', '""')
        connection.execute(
            _build_create_table_sql(table_name=safe_table, source_path=source_path)
        )
        row_count = connection.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone()[0]
        loaded.append(
            {
                "table_name": table_name,
                "row_count": row_count,
                "source_path": str(source_path),
            }
        )

    return {"loaded_tables": loaded, "skipped_tables": skipped}


def build_manifest(
    *,
    pipeline_run_id: str | None,
    source_run_id: str,
    process_run_id: str,
    database_name: str,
    process_manifest: dict[str, Any],
    load_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "pipeline_run_id": pipeline_run_id,
        "source_run_id": source_run_id,
        "process_run_id": process_run_id,
        "motherduck_database": database_name,
        "processed_outputs": process_manifest.get("outputs", []),
        "loaded_table_count": len(load_results["loaded_tables"]),
        "skipped_table_count": len(load_results["skipped_tables"]),
        "loaded_tables": load_results["loaded_tables"],
        "skipped_tables": load_results["skipped_tables"],
    }
