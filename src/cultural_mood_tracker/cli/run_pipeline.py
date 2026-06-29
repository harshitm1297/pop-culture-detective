from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from cultural_mood_tracker.cli.extract_multisource import run_extraction
from cultural_mood_tracker.cli.transform_canonical import run_transform
from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.load import run_cloud_load
from cultural_mood_tracker.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Cultural Mood Tracker ETL pipeline with explicit orchestration."
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Skip extraction and transform this existing aligned raw run ID.",
    )
    parser.add_argument("--movie-count", type=int, default=None)
    parser.add_argument("--tv-count", type=int, default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--guardian-api-key", default=None)
    parser.add_argument("--guardian-page-size", type=int, default=None)
    parser.add_argument("--gdelt-max-records", type=int, default=None)
    parser.add_argument(
        "--enable-gdelt",
        action="store_true",
        help="Enable GDELT during extraction.",
    )
    parser.add_argument(
        "--cleanup-old-raw",
        action="store_true",
        help="Remove previous generated raw source folders before extraction.",
    )
    parser.add_argument(
        "--enable-gcs-upload",
        action="store_true",
        help="Upload raw, processed, and report outputs to GCS after transform.",
    )
    parser.add_argument(
        "--enable-bigquery-load",
        action="store_true",
        help="Load processed JSONL tables into BigQuery after transform.",
    )
    return parser.parse_args()


def build_extract_namespace(settings, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        movie_count=args.movie_count if args.movie_count is not None else 100,
        tv_count=args.tv_count if args.tv_count is not None else 100,
        language=args.language or settings.tmdb_language,
        start_date=args.start_date or settings.tmdb_start_date,
        end_date=args.end_date or settings.tmdb_end_date,
        output_root=args.output_root or str(settings.local_data_root),
        guardian_api_key=args.guardian_api_key or settings.guardian_api_key,
        guardian_page_size=(
            args.guardian_page_size
            if args.guardian_page_size is not None
            else settings.guardian_page_size
        ),
        gdelt_max_records=(
            args.gdelt_max_records
            if args.gdelt_max_records is not None
            else settings.gdelt_max_records
        ),
        enable_gdelt=args.enable_gdelt,
        cleanup_old_raw=args.cleanup_old_raw,
    )


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    args = parse_args()
    extract_args = build_extract_namespace(settings, args)
    load_args = SimpleNamespace(
        enable_gcs_upload=(args.enable_gcs_upload or settings.enable_gcs_upload),
        enable_bigquery_load=(args.enable_bigquery_load or settings.enable_bigquery_load),
    )

    manifest = run_pipeline(
        project_root=project_root,
        extract_fn=run_extraction,
        extract_args=extract_args,
        transform_fn=run_transform,
        load_fn=run_cloud_load,
        load_args=load_args,
        enable_load=bool(load_args.enable_gcs_upload or load_args.enable_bigquery_load),
        source_run_id=args.source_run_id,
    )

    print(
        f"Pipeline completed | pipeline_run_id={manifest['pipeline_run_id']} "
        f"| source_run_id={manifest.get('source_run_id')} "
        f"| process_run_id={manifest.get('process_run_id')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
