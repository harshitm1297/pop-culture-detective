from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .common import clean_text, find_matching_file, load_json, maybe_load_json, normalize_name
from cultural_mood_tracker.sources.wikidata import extract_enwiki_title


def _load_imdb_table(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["tconst"]: row for row in reader if row.get("tconst")}


def _wikidata_fields(
    payload: dict[str, Any] | None,
    wikidata_id: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not payload or not wikidata_id:
        return None, None, None
    entity = payload.get("entities", {}).get(wikidata_id, {})
    label = entity.get("labels", {}).get("en", {}).get("value")
    description = entity.get("descriptions", {}).get("en", {}).get("value")
    enwiki_title = extract_enwiki_title(payload, wikidata_id)
    return clean_text(label or ""), clean_text(description or ""), clean_text(enwiki_title or "")


def build_titles(
    anchors: list[dict[str, Any]],
    tmdb_run_dir: Path,
    imdb_run_dir: Path,
    tvmaze_run_dir: Path,
    wikidata_run_dir: Path,
    *,
    source_run_id: str,
) -> list[dict[str, Any]]:
    imdb_basics = _load_imdb_table(imdb_run_dir / "matched_title_basics.tsv")
    imdb_ratings = _load_imdb_table(imdb_run_dir / "matched_title_ratings.tsv")
    rows: list[dict[str, Any]] = []

    for anchor in anchors:
        title_id = f"{anchor['content_type']}_{anchor['tmdb_id']}"
        type_dir = tmdb_run_dir / anchor["content_type"]
        details_path = find_matching_file(type_dir, f"{anchor['tmdb_id']}_", "_details.json")
        details = maybe_load_json(details_path) if details_path else {}

        imdb_basic = imdb_basics.get(anchor.get("imdb_id") or "", {})
        imdb_rating = imdb_ratings.get(anchor.get("imdb_id") or "", {})

        tvmaze_path = find_matching_file(tvmaze_run_dir / anchor["content_type"], f"{anchor['tmdb_id']}_", ".json")
        tvmaze = maybe_load_json(tvmaze_path) if tvmaze_path else {}
        if isinstance(tvmaze, dict) and "error" in tvmaze:
            tvmaze = {}
        tvmaze_embedded = tvmaze.get("_embedded", {}) if isinstance(tvmaze, dict) else {}

        wikidata_path = find_matching_file(wikidata_run_dir / anchor["content_type"], f"{anchor['tmdb_id']}_", ".json")
        wikidata = maybe_load_json(wikidata_path) if wikidata_path else {}
        label_en, description_en, enwiki_title = _wikidata_fields(
            wikidata if isinstance(wikidata, dict) else {},
            anchor.get("wikidata_id"),
        )

        genres = [genre.get("name") for genre in details.get("genres", []) if genre.get("name")]
        spoken_languages = [item.get("english_name") or item.get("name") for item in anchor.get("spoken_languages", []) if item.get("english_name") or item.get("name")]
        production_countries = [item.get("iso_3166_1") for item in details.get("production_countries", []) if item.get("iso_3166_1")]
        tmdb_credits = details.get("credits", {}) if isinstance(details, dict) else {}
        tmdb_videos = ((details.get("videos") or {}).get("results")) if isinstance(details, dict) else []

        rows.append(
            {
                "title_id": title_id,
                "source_run_id": source_run_id,
                "content_type": anchor["content_type"],
                "tmdb_id": anchor["tmdb_id"],
                "imdb_id": anchor.get("imdb_id"),
                "wikidata_id": anchor.get("wikidata_id"),
                "title_name": clean_text(anchor.get("title_name") or ""),
                "original_title_name": clean_text(anchor.get("original_title_name") or ""),
                "normalized_title": normalize_name(anchor.get("title_name") or ""),
                "release_date": anchor.get("release_date"),
                "release_year": anchor.get("release_year"),
                "original_language": anchor.get("original_language"),
                "spoken_languages": spoken_languages,
                "origin_country": anchor.get("origin_country", []),
                "production_countries": production_countries,
                "genres": genres,
                "tmdb_overview": clean_text(details.get("overview") or ""),
                "tmdb_popularity": anchor.get("tmdb_popularity"),
                "tmdb_vote_average": anchor.get("tmdb_vote_average"),
                "tmdb_vote_count": anchor.get("tmdb_vote_count"),
                "tmdb_review_count": anchor.get("tmdb_review_count"),
                "tmdb_cast_count": len(tmdb_credits.get("cast", [])),
                "tmdb_crew_count": len(tmdb_credits.get("crew", [])),
                "tmdb_video_count": len(tmdb_videos or []),
                "imdb_title_type": imdb_basic.get("titleType"),
                "imdb_primary_title": clean_text(imdb_basic.get("primaryTitle") or ""),
                "imdb_original_title": clean_text(imdb_basic.get("originalTitle") or ""),
                "imdb_start_year": imdb_basic.get("startYear"),
                "imdb_end_year": imdb_basic.get("endYear"),
                "imdb_runtime_minutes": imdb_basic.get("runtimeMinutes"),
                "imdb_genres": [part for part in (imdb_basic.get("genres") or "").split(",") if part and part != "\\N"],
                "imdb_average_rating": imdb_rating.get("averageRating"),
                "imdb_num_votes": imdb_rating.get("numVotes"),
                "tvmaze_id": tvmaze.get("id"),
                "tvmaze_language": tvmaze.get("language"),
                "tvmaze_status": tvmaze.get("status"),
                "tvmaze_genres": tvmaze.get("genres", []) if isinstance(tvmaze.get("genres"), list) else [],
                "tvmaze_network": (tvmaze.get("network") or {}).get("name") if isinstance(tvmaze, dict) else None,
                "tvmaze_summary": clean_text(tvmaze.get("summary") or ""),
                "tvmaze_episode_count": len(tvmaze_embedded.get("episodes", []) or []),
                "wikidata_label_en": label_en or None,
                "wikidata_description_en": description_en or None,
                "wikipedia_article_title": enwiki_title or None,
            }
        )

    return rows
