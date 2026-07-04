from __future__ import annotations

from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.transform.common import write_json

from .motherduck import (
    build_manifest,
    create_motherduck_connection,
    ensure_database,
    load_processed_tables,
    _load_process_manifest,
)


def run_motherduck_load(
    *,
    project_root: Path,
    source_run_id: str,
    process_run_id: str,
    pipeline_run_id: str | None = None,
    enable_motherduck_load: bool | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    md_enabled = (
        settings.enable_motherduck_load
        if enable_motherduck_load is None
        else enable_motherduck_load
    )
    if not md_enabled:
        return {
            "enabled": False,
            "source_run_id": source_run_id,
            "process_run_id": process_run_id,
            "message": "MotherDuck load skipped because ENABLE_MOTHERDUCK_LOAD is disabled.",
        }

    processed_dir = paths.processed_root / process_run_id
    reports_dir = paths.reports_root / process_run_id
    if not processed_dir.exists():
        raise RuntimeError(f"Missing processed directory: {processed_dir}")
    if not reports_dir.exists():
        raise RuntimeError(f"Missing reports directory: {reports_dir}")

    process_manifest = _load_process_manifest(processed_dir)
    connection = create_motherduck_connection(token=settings.motherduck_token)
    try:
        ensure_database(connection=connection, database_name=settings.motherduck_database)
        load_results = load_processed_tables(connection=connection, processed_dir=processed_dir)
    finally:
        connection.close()

    manifest = build_manifest(
        pipeline_run_id=pipeline_run_id,
        source_run_id=source_run_id,
        process_run_id=process_run_id,
        database_name=settings.motherduck_database,
        process_manifest=process_manifest,
        load_results=load_results,
    )
    manifest_path = reports_dir / "motherduck_load_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        **manifest,
    }
