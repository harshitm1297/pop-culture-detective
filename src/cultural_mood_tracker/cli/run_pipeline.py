from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from cultural_mood_tracker.cli.extract_multisource import run_extraction
from cultural_mood_tracker.cli.transform_canonical import run_transform
from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.load import run_motherduck_load
from cultural_mood_tracker.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Cultural Mood Tracker ETL pipeline end to end. "
            "By default, this extracts source data, transforms it, and uploads the "
            "processed tables to MotherDuck."
        )
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Skip extraction and transform this existing aligned raw run ID.",
    )
    parser.add_argument("--movie-count", type=int, default=None, help="Defaults to TMDB_MOVIE_SAMPLE_SIZE from .env.")
    parser.add_argument("--tv-count", type=int, default=None, help="Defaults to TMDB_TV_SAMPLE_SIZE from .env.")
    parser.add_argument("--language", default=None, help="Defaults to TMDB_LANGUAGE from .env.")
    parser.add_argument("--start-date", default=None, help="Defaults to TMDB_START_DATE from .env.")
    parser.add_argument("--end-date", default=None, help="Defaults to TMDB_END_DATE from .env.")
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
        "--disable-critic-blogs",
        action="store_true",
        help="Skip curated critic blog extraction during the extraction stage.",
    )
    parser.add_argument(
        "--skip-motherduck-load",
        action="store_true",
        help="Skip the MotherDuck upload step and stop after local transform.",
    )
    parser.add_argument(
        "--keep-full-local",
        action="store_true",
        help="Do not downsample local raw and processed outputs after MotherDuck upload.",
    )
    parser.add_argument(
        "--local-retain-movie-count",
        type=int,
        default=None,
        help="Local movie titles to keep after MotherDuck upload. Defaults to LOCAL_RETAIN_MOVIE_COUNT or 30.",
    )
    parser.add_argument(
        "--local-retain-tv-count",
        type=int,
        default=None,
        help="Local TV titles to keep after MotherDuck upload. Defaults to LOCAL_RETAIN_TV_COUNT or 30.",
    )
    return parser.parse_args()


def build_extract_namespace(settings, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        movie_count=(
            args.movie_count
            if args.movie_count is not None
            else settings.tmdb_movie_sample_size
        ),
        tv_count=(
            args.tv_count
            if args.tv_count is not None
            else settings.tmdb_tv_sample_size
        ),
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
        disable_critic_blogs=getattr(args, "disable_critic_blogs", False),
    )


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    args = parse_args()
    extract_args = build_extract_namespace(settings, args)
    load_args = SimpleNamespace(
        enable_motherduck_load=(
            False if args.skip_motherduck_load else settings.enable_motherduck_load
        ),
    )

    manifest = run_pipeline(
        project_root=project_root,
        extract_fn=run_extraction,
        extract_args=extract_args,
        transform_fn=run_transform,
        load_fn=run_motherduck_load,
        load_args=load_args,
        enable_load=bool(load_args.enable_motherduck_load),
        enable_local_retention=(
            False
            if args.keep_full_local or args.skip_motherduck_load
            else settings.enable_local_sample_retention
        ),
        local_retain_movie_count=args.local_retain_movie_count,
        local_retain_tv_count=args.local_retain_tv_count,
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
