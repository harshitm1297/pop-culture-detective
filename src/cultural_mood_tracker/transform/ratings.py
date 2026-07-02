from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .common import find_matching_file, maybe_load_json, normalize_datetime


def _load_imdb_ratings(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["tconst"]: row for row in reader if row.get("tconst")}


def build_ratings(
    anchors: list[dict[str, Any]],
    tmdb_run_dir: Path,
    imdb_run_dir: Path,
    *,
    source_run_id: str,
) -> list[dict[str, Any]]:
    imdb_ratings = _load_imdb_ratings(imdb_run_dir / "matched_title_ratings.tsv")
    rows: list[dict[str, Any]] = []

    for anchor in anchors:
        title_id = f"{anchor['content_type']}_{anchor['tmdb_id']}"
        rows.append(
            {
                "rating_id": f"{title_id}:tmdb_aggregate",
                "title_id": title_id,
                "source_run_id": source_run_id,
                "source_name": "tmdb",
                "content_type": anchor["content_type"],
                "rating_scope": "title_aggregate",
                "source_record_id": anchor["tmdb_id"],
                "rating_value": anchor.get("tmdb_vote_average"),
                "scale_min": 0,
                "scale_max": 10,
                "rating_count": anchor.get("tmdb_vote_count"),
                "author": None,
                "published_at": normalize_datetime(anchor.get("release_date")),
            }
        )

        imdb_row = imdb_ratings.get(anchor.get("imdb_id") or "")
        if imdb_row:
            rows.append(
                {
                    "rating_id": f"{title_id}:imdb_aggregate",
                    "title_id": title_id,
                    "source_run_id": source_run_id,
                    "source_name": "imdb",
                    "content_type": anchor["content_type"],
                    "rating_scope": "title_aggregate",
                    "source_record_id": anchor.get("imdb_id"),
                    "rating_value": imdb_row.get("averageRating"),
                    "scale_min": 0,
                    "scale_max": 10,
                    "rating_count": imdb_row.get("numVotes"),
                    "author": None,
                    "published_at": normalize_datetime(anchor.get("release_date")),
                }
            )

        reviews_path = find_matching_file(tmdb_run_dir / anchor["content_type"], f"{anchor['tmdb_id']}_", "_reviews.json")
        reviews = maybe_load_json(reviews_path) if reviews_path else {}
        for review in (reviews or {}).get("results", []):
            author_rating = (review.get("author_details") or {}).get("rating")
            if author_rating is None:
                continue
            rows.append(
                {
                    "rating_id": f"{title_id}:tmdb_review:{review.get('id')}",
                    "title_id": title_id,
                    "source_run_id": source_run_id,
                    "source_name": "tmdb",
                    "content_type": anchor["content_type"],
                    "rating_scope": "review_author",
                    "source_record_id": review.get("id"),
                    "rating_value": author_rating,
                    "scale_min": 0,
                    "scale_max": 10,
                    "rating_count": 1,
                    "author": review.get("author"),
                    "published_at": normalize_datetime(review.get("updated_at") or review.get("created_at")),
                }
            )

    return rows
