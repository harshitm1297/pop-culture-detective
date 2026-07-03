from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.extract.aligned import (
    build_anchor_entries,
    call_source,
    clean_previous_outputs,
    download_imdb_and_filter,
    fetch_critic_blog_sources,
    fetch_gdelt,
    fetch_guardian,
    fetch_tvmaze,
    fetch_wikidata_entity,
    fetch_wikipedia_pageviews,
)
from cultural_mood_tracker.sources import ensure_dir, save_json


def parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch aligned raw movie/TV data across multiple sources."
    )
    parser.add_argument("--movie-count", type=int, default=100)
    parser.add_argument("--tv-count", type=int, default=100)
    parser.add_argument("--language", default=settings.tmdb_language)
    parser.add_argument("--start-date", default=settings.tmdb_start_date)
    parser.add_argument("--end-date", default=settings.tmdb_end_date)
    parser.add_argument(
        "--output-root",
        default=str(settings.local_data_root),
        help="Base local data directory.",
    )
    parser.add_argument(
        "--guardian-api-key",
        default=settings.guardian_api_key,
        help="Guardian API key. Defaults to 'test'.",
    )
    parser.add_argument(
        "--guardian-page-size",
        type=int,
        default=settings.guardian_page_size,
    )
    parser.add_argument(
        "--gdelt-max-records",
        type=int,
        default=settings.gdelt_max_records,
    )
    parser.add_argument(
        "--enable-gdelt",
        action="store_true",
        help="Enable GDELT extraction. Disabled by default because the public API is heavily rate-limited.",
    )
    parser.add_argument(
        "--cleanup-old-raw",
        action="store_true",
        help="Remove previous generated raw source folders before writing the new run.",
    )
    parser.add_argument(
        "--disable-critic-blogs",
        action="store_true",
        help="Skip curated critic blog extraction.",
    )
    return parser.parse_args()


def write_run_manifest(run_root: Path, payload: dict) -> None:
    save_json(run_root / "run_manifest.json", payload)


def run_extraction(project_root: Path, args: argparse.Namespace) -> str:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()
    api_key = settings.tmdb_api_key.strip()
    if not api_key:
        raise RuntimeError("Missing TMDB_API_KEY in .env")

    data_root = Path(args.output_root).expanduser()
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    raw_root = data_root / "raw"
    ensure_dir(raw_root)

    if args.cleanup_old_raw:
        clean_previous_outputs(raw_root)

    run_id = make_run_id()
    anchor_dir = raw_root / "anchors" / run_id
    tmdb_dir = raw_root / "tmdb" / run_id
    imdb_dir = raw_root / "imdb" / run_id
    tvmaze_dir = raw_root / "tvmaze" / run_id
    wikidata_dir = raw_root / "wikidata" / run_id
    wikipedia_dir = raw_root / "wikipedia" / run_id
    guardian_dir = raw_root / "guardian" / run_id
    gdelt_dir = raw_root / "gdelt" / run_id
    critic_dirs = [raw_root / name / run_id for name in ("rogerebert", "indiewire", "vulture", "slant", "slashfilm")]
    for path in [anchor_dir, tmdb_dir, imdb_dir, tvmaze_dir, wikidata_dir, wikipedia_dir, guardian_dir, gdelt_dir, *critic_dirs]:
        ensure_dir(path)

    print("[anchors] building TMDB anchor list")
    anchors = build_anchor_entries(
        api_key=api_key,
        language=args.language,
        start_date=args.start_date,
        end_date=args.end_date,
        movie_count=args.movie_count,
        tv_count=args.tv_count,
        anchor_dir=anchor_dir,
        tmdb_dir=tmdb_dir,
    )
    print(f"[anchors] built {len(anchors)} total anchors")

    print("[imdb] downloading and filtering datasets")
    download_imdb_and_filter(anchors, imdb_dir)

    for index, anchor in enumerate(anchors, start=1):
        print(
            f"[aligned] {index}/{len(anchors)} {anchor['content_type']} "
            f"{anchor['title_name']} | IMDb={anchor.get('imdb_id')} | Wikidata={anchor.get('wikidata_id')}"
        )
        call_source("wikidata", lambda: fetch_wikidata_entity(anchor, wikidata_dir), anchor)
        call_source(
            "wikipedia",
            lambda: fetch_wikipedia_pageviews(
                anchor,
                wikipedia_dir,
                wikidata_dir,
                args.start_date,
                args.end_date,
            ),
            anchor,
        )
        call_source(
            "guardian",
            lambda: fetch_guardian(
                anchor,
                guardian_dir,
                args.guardian_api_key,
                args.start_date,
                args.end_date,
                args.guardian_page_size,
            ),
            anchor,
        )
        if args.enable_gdelt and args.gdelt_max_records > 0:
            call_source("gdelt", lambda: fetch_gdelt(anchor, gdelt_dir, args.gdelt_max_records), anchor)
        call_source("tvmaze", lambda: fetch_tvmaze(anchor, tvmaze_dir), anchor)
        time.sleep(0.05)

    if settings.enable_critic_blog_sources and not args.disable_critic_blogs:
        print("[critic] fetching curated critic/blog sources")
        fetch_critic_blog_sources(
            anchors,
            raw_root=raw_root,
            run_id=run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            entry_limit=settings.critic_feed_entry_limit,
        )

    manifest = {
        "run_at_utc": run_id,
        "movie_count": args.movie_count,
        "tv_count": args.tv_count,
        "language": args.language,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "sources": [
            "tmdb",
            "imdb",
            "tvmaze",
            "wikidata",
            "wikipedia",
            "guardian",
            "rogerebert",
            "indiewire",
            "vulture",
            "slant",
            "slashfilm",
            *(["gdelt"] if args.enable_gdelt and args.gdelt_max_records > 0 else []),
        ],
        "same_anchor_titles": True,
        "gdelt_enabled": bool(args.enable_gdelt and args.gdelt_max_records > 0),
        "note": (
            "All non-TMDB sources are fetched against the same TMDB-derived anchor list. "
            "This ensures aligned target titles, but does not guarantee every source returns a match for every title."
        ),
    }
    write_run_manifest(anchor_dir, manifest)
    print(f"[done] raw aligned outputs written under {raw_root}")
    return run_id


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    args = parse_args(settings)
    try:
        run_extraction(project_root, args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
