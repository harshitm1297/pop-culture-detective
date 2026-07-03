from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import clean_text, find_matching_file, maybe_load_json, normalize_datetime


def build_people_and_credits(
    anchors: list[dict[str, Any]],
    tmdb_run_dir: Path,
    *,
    source_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    people_by_id: dict[str, dict[str, Any]] = {}
    title_cast: list[dict[str, Any]] = []
    title_crew: list[dict[str, Any]] = []

    for anchor in anchors:
        title_id = f"{anchor['content_type']}_{anchor['tmdb_id']}"
        details_path = find_matching_file(
            tmdb_run_dir / anchor["content_type"],
            f"{anchor['tmdb_id']}_",
            "_details.json",
        )
        details = maybe_load_json(details_path) if details_path else {}
        credits = details.get("credits", {}) if isinstance(details, dict) else {}

        for member in credits.get("cast", []):
            person_id = f"tmdb_person_{member.get('id')}"
            if member.get("id") is None:
                continue
            people_by_id.setdefault(
                person_id,
                {
                    "person_id": person_id,
                    "source_run_id": source_run_id,
                    "source_name": "tmdb",
                    "source_person_id": member.get("id"),
                    "name": clean_text(member.get("name") or member.get("original_name") or ""),
                    "original_name": clean_text(member.get("original_name") or member.get("name") or ""),
                    "gender": member.get("gender"),
                    "known_for_department": clean_text(member.get("known_for_department") or ""),
                    "popularity": member.get("popularity"),
                    "profile_path": member.get("profile_path"),
                },
            )
            title_cast.append(
                {
                    "title_id": title_id,
                    "person_id": person_id,
                    "source_run_id": source_run_id,
                    "source_name": "tmdb",
                    "source_credit_id": member.get("credit_id"),
                    "character_name": clean_text(member.get("character") or ""),
                    "billing_order": member.get("order"),
                    "cast_position": member.get("cast_id"),
                }
            )

        for member in credits.get("crew", []):
            person_id = f"tmdb_person_{member.get('id')}"
            if member.get("id") is None:
                continue
            people_by_id.setdefault(
                person_id,
                {
                    "person_id": person_id,
                    "source_run_id": source_run_id,
                    "source_name": "tmdb",
                    "source_person_id": member.get("id"),
                    "name": clean_text(member.get("name") or member.get("original_name") or ""),
                    "original_name": clean_text(member.get("original_name") or member.get("name") or ""),
                    "gender": member.get("gender"),
                    "known_for_department": clean_text(member.get("known_for_department") or ""),
                    "popularity": member.get("popularity"),
                    "profile_path": member.get("profile_path"),
                },
            )
            title_crew.append(
                {
                    "title_id": title_id,
                    "person_id": person_id,
                    "source_run_id": source_run_id,
                    "source_name": "tmdb",
                    "source_credit_id": member.get("credit_id"),
                    "department": clean_text(member.get("department") or ""),
                    "job": clean_text(member.get("job") or ""),
                }
            )

    people = list(people_by_id.values())
    return people, title_cast, title_crew


def build_title_videos(
    anchors: list[dict[str, Any]],
    tmdb_run_dir: Path,
    *,
    source_run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for anchor in anchors:
        title_id = f"{anchor['content_type']}_{anchor['tmdb_id']}"
        details_path = find_matching_file(
            tmdb_run_dir / anchor["content_type"],
            f"{anchor['tmdb_id']}_",
            "_details.json",
        )
        details = maybe_load_json(details_path) if details_path else {}
        videos = (((details or {}).get("videos") or {}).get("results")) or []
        for video in videos:
            key = clean_text(video.get("key") or "")
            rows.append(
                {
                    "video_id": f"{title_id}:tmdb_video:{video.get('id') or key}",
                    "title_id": title_id,
                    "source_run_id": source_run_id,
                    "source_name": "tmdb",
                    "source_video_id": video.get("id"),
                    "site": clean_text(video.get("site") or ""),
                    "video_type": clean_text(video.get("type") or ""),
                    "official": bool(video.get("official")),
                    "published_at": normalize_datetime(video.get("published_at")),
                    "language": clean_text(video.get("iso_639_1") or ""),
                    "region": clean_text(video.get("iso_3166_1") or ""),
                    "name": clean_text(video.get("name") or ""),
                    "key": key,
                    "watch_url": (
                        f"https://www.youtube.com/watch?v={key}"
                        if clean_text(video.get("site") or "").lower() == "youtube" and key
                        else None
                    ),
                }
            )

    return rows


def build_episodes(
    anchors: list[dict[str, Any]],
    tvmaze_run_dir: Path,
    *,
    source_run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for anchor in anchors:
        if anchor.get("content_type") != "tv":
            continue

        title_id = f"tv_{anchor['tmdb_id']}"
        tvmaze_path = find_matching_file(tvmaze_run_dir / "tv", f"{anchor['tmdb_id']}_", ".json")
        tvmaze = maybe_load_json(tvmaze_path) if tvmaze_path else {}
        if not isinstance(tvmaze, dict) or "error" in tvmaze:
            continue

        embedded = tvmaze.get("_embedded") or {}
        episodes = embedded.get("episodes") or []
        for episode in episodes:
            episode_id = episode.get("id")
            if episode_id is None:
                continue
            rows.append(
                {
                    "episode_id": f"tvmaze_episode_{episode_id}",
                    "title_id": title_id,
                    "source_run_id": source_run_id,
                    "source_name": "tvmaze",
                    "source_episode_id": episode_id,
                    "season_number": episode.get("season"),
                    "episode_number": episode.get("number"),
                    "episode_type": clean_text(episode.get("type") or ""),
                    "name": clean_text(episode.get("name") or ""),
                    "airdate": episode.get("airdate"),
                    "airstamp": normalize_datetime(episode.get("airstamp")),
                    "runtime_minutes": episode.get("runtime"),
                    "rating_value": (episode.get("rating") or {}).get("average"),
                    "summary": clean_text(episode.get("summary") or ""),
                    "url": episode.get("url"),
                }
            )

    return rows
