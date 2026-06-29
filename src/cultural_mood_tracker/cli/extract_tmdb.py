from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.sources import ensure_dir, save_json, slugify
from cultural_mood_tracker.sources.tmdb import discover_titles, fetch_details, fetch_reviews
from cultural_mood_tracker.sources.wikipedia import fetch_pageviews


def parse_args(settings, default_output_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test TMDB extraction for the Cultural Mood Tracker project."
    )
    parser.add_argument(
        "--api-key",
        default=settings.tmdb_api_key,
        help="TMDB API key. Defaults to TMDB_API_KEY env var.",
    )
    parser.add_argument(
        "--content-type",
        choices=["movie", "tv", "both"],
        default="both",
        help="Which TMDB content type to extract.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "How many titles per content type to fetch in detail. If omitted, uses "
            "TMDB_MOVIE_SAMPLE_SIZE and TMDB_TV_SAMPLE_SIZE from .env."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=settings.tmdb_start_date,
        help="Lower bound for release/air date filtering.",
    )
    parser.add_argument(
        "--end-date",
        default=settings.tmdb_end_date,
        help="Upper bound for release/air date filtering.",
    )
    parser.add_argument(
        "--language",
        default=settings.tmdb_language,
        help="TMDB response language.",
    )
    parser.add_argument(
        "--with-pageviews",
        action="store_true",
        help="Also fetch Wikipedia pageviews for the extracted titles.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Base output directory for raw responses.",
    )
    return parser.parse_args()


def write_discovery_snapshot(
    output_root: Path,
    content_type: str,
    payload: list[dict[str, Any]],
) -> None:
    save_json(output_root / f"{content_type}_discover.json", {"results": payload})


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()
    default_output_dir = paths.raw_root / "tmdb_smoke"
    args = parse_args(settings, default_output_dir)
    if not args.api_key:
        print(
            "TMDB API key missing. Set TMDB_API_KEY in .env, pass --api-key, "
            "or export TMDB_API_KEY in the shell.",
            file=sys.stderr,
        )
        return 1

    output_base = Path(args.output_dir).expanduser()
    if not output_base.is_absolute():
        output_base = (project_root / output_base).resolve()
    output_root = output_base / make_run_id()
    ensure_dir(output_root)

    content_types = ["movie", "tv"] if args.content_type == "both" else [args.content_type]
    default_sample_sizes = {
        "movie": settings.tmdb_movie_sample_size,
        "tv": settings.tmdb_tv_sample_size,
    }
    summary: dict[str, Any] = {
        "run_at_utc": output_root.name,
        "content_types": content_types,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "language": args.language,
        "with_pageviews": args.with_pageviews,
        "results": {},
    }

    for content_type in content_types:
        started = time.perf_counter()
        sample_size = args.sample_size if args.sample_size is not None else default_sample_sizes[content_type]
        records = discover_titles(
            api_key=args.api_key,
            content_type=content_type,
            language=args.language,
            start_date=args.start_date,
            end_date=args.end_date,
            sample_size=sample_size,
        )
        type_dir = output_root / content_type
        ensure_dir(type_dir)
        write_discovery_snapshot(type_dir, content_type, records)

        titles_summary: list[dict[str, Any]] = []
        for index, item in enumerate(records, start=1):
            tmdb_id = item["id"]
            title = item.get("title") or item.get("name") or f"{content_type}-{tmdb_id}"
            safe_title = slugify(title)
            print(f"[{content_type}] fetching title {index}/{len(records)}: {title} (TMDB {tmdb_id})")

            details = fetch_details(args.api_key, content_type, tmdb_id, args.language)
            reviews = fetch_reviews(args.api_key, content_type, tmdb_id, args.language)
            save_json(type_dir / f"{tmdb_id}_{safe_title}_details.json", details)
            save_json(type_dir / f"{tmdb_id}_{safe_title}_reviews.json", reviews)

            entry = {
                "tmdb_id": tmdb_id,
                "title": title,
                "imdb_id": details.get("imdb_id") or details.get("external_ids", {}).get("imdb_id"),
                "popularity": item.get("popularity"),
                "vote_average": item.get("vote_average"),
                "review_count": len(reviews.get("results", [])),
            }

            if args.with_pageviews:
                try:
                    pageviews = fetch_pageviews(title, args.start_date, args.end_date)
                    save_json(type_dir / f"{tmdb_id}_{safe_title}_pageviews.json", pageviews)
                    entry["pageview_points"] = len(pageviews.get("items", []))
                    print(
                        f"[{content_type}] pageviews fetched for {title}: "
                        f"{entry['pageview_points']} daily points"
                    )
                except RuntimeError as exc:
                    entry["pageviews_error"] = str(exc)
                    print(f"[{content_type}] pageviews failed for {title}: {exc}")

            print(
                f"[{content_type}] saved {title} | IMDb={entry.get('imdb_id')} | "
                f"reviews={entry['review_count']}"
            )
            titles_summary.append(entry)

        duration = round(time.perf_counter() - started, 2)
        summary["results"][content_type] = {
            "requested_sample_size": sample_size,
            "count": len(titles_summary),
            "duration_seconds": duration,
            "titles": titles_summary,
        }

    save_json(output_root / "run_summary.json", summary)

    print(f"Saved raw responses to: {output_root.resolve()}")
    for content_type, payload in summary["results"].items():
        print(f"\n[{content_type}] extracted {payload['count']} titles in {payload['duration_seconds']}s")
        for title in payload["titles"]:
            print(
                f"  - {title['title']} | TMDB={title['tmdb_id']} | "
                f"IMDb={title.get('imdb_id')} | reviews={title['review_count']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
