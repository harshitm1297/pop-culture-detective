from __future__ import annotations

import argparse
from pathlib import Path

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.load import run_cloud_load
from cultural_mood_tracker.transform.common import find_latest_run_id, load_json


def parse_args(default_process_run_id: str | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload ETL outputs to GCS and load canonical tables into BigQuery."
    )
    parser.add_argument(
        "--process-run-id",
        default=default_process_run_id,
        help="Processed run ID to publish. Defaults to the latest processed run.",
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Source run ID. If omitted, it is read from the processed run manifest.",
    )
    parser.add_argument(
        "--enable-gcs-upload",
        action="store_true",
        help="Upload raw, processed, and report outputs to GCS for this run.",
    )
    parser.add_argument(
        "--enable-bigquery-load",
        action="store_true",
        help="Load processed JSONL tables from GCS into BigQuery.",
    )
    return parser.parse_args()


def _resolve_source_run_id(processed_dir: Path, explicit_source_run_id: str | None) -> str:
    if explicit_source_run_id:
        return explicit_source_run_id
    manifest_path = processed_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing processed manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    source_run_id = manifest.get("source_run_id")
    if not source_run_id:
        raise RuntimeError(f"Processed manifest does not contain source_run_id: {manifest_path}")
    return source_run_id


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    default_process_run_id = None
    if paths.processed_root.exists():
        try:
            default_process_run_id = find_latest_run_id(paths.processed_root)
        except RuntimeError:
            default_process_run_id = None

    args = parse_args(default_process_run_id)
    if not args.process_run_id:
        raise RuntimeError("No processed run found. Run transform first or pass --process-run-id.")

    processed_dir = paths.processed_root / args.process_run_id
    if not processed_dir.exists():
        raise RuntimeError(f"Missing processed directory: {processed_dir}")

    source_run_id = _resolve_source_run_id(processed_dir, args.source_run_id)
    enable_gcs_upload = args.enable_gcs_upload or settings.enable_gcs_upload
    enable_bigquery_load = args.enable_bigquery_load or settings.enable_bigquery_load

    manifest = run_cloud_load(
        project_root=project_root,
        source_run_id=source_run_id,
        process_run_id=args.process_run_id,
        enable_gcs_upload=enable_gcs_upload,
        enable_bigquery_load=enable_bigquery_load,
    )
    print(
        f"Cloud load completed | source_run_id={source_run_id} "
        f"| process_run_id={args.process_run_id} "
        f"| manifest={manifest.get('manifest_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
