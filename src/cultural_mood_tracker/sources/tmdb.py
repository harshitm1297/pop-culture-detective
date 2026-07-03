from __future__ import annotations

from typing import Any

from .base import http_get_json


BASE_TMDB_URL = "https://api.themoviedb.org/3"
TMDB_USER_AGENT = "cultural-mood-tracker-tmdb/0.1"


def discover_titles(
    api_key: str,
    content_type: str,
    language: str,
    start_date: str,
    end_date: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    endpoint = f"{BASE_TMDB_URL}/discover/{content_type}"
    date_key = "primary_release_date" if content_type == "movie" else "first_air_date"
    collected: list[dict[str, Any]] = []
    page = 1

    while len(collected) < sample_size:
        params = {
            "api_key": api_key,
            "language": language,
            "sort_by": "popularity.desc",
            "vote_count.gte": 20,
            f"{date_key}.gte": start_date,
            f"{date_key}.lte": end_date,
            "with_original_language": "en",
            "page": page,
        }
        payload = http_get_json(endpoint, params, user_agent=TMDB_USER_AGENT, timeout=30)
        results = payload.get("results", [])
        if not results:
            break

        collected.extend(results)
        total_pages = int(payload.get("total_pages", page))
        print(
            f"[tmdb:{content_type}] discover page {page}/{total_pages} "
            f"({len(results)} records, collected {min(len(collected), sample_size)}/{sample_size})"
        )

        if page >= total_pages:
            break
        page += 1

    return collected[:sample_size]


def fetch_details(api_key: str, content_type: str, tmdb_id: int, language: str) -> dict[str, Any]:
    endpoint = f"{BASE_TMDB_URL}/{content_type}/{tmdb_id}"
    params = {
        "api_key": api_key,
        "language": language,
        "append_to_response": "external_ids,credits,videos",
    }
    return http_get_json(endpoint, params, user_agent=TMDB_USER_AGENT)


def fetch_reviews(
    api_key: str,
    content_type: str,
    tmdb_id: int,
    language: str,
    *,
    max_pages: int = 3,
) -> dict[str, Any]:
    endpoint = f"{BASE_TMDB_URL}/{content_type}/{tmdb_id}/reviews"
    first_page = http_get_json(
        endpoint,
        {"api_key": api_key, "language": language, "page": 1},
        user_agent=TMDB_USER_AGENT,
    )
    results = list(first_page.get("results", []))
    total_pages = int(first_page.get("total_pages", 1) or 1)
    collected_pages = 1

    for page in range(2, min(total_pages, max_pages) + 1):
        payload = http_get_json(
            endpoint,
            {"api_key": api_key, "language": language, "page": page},
            user_agent=TMDB_USER_AGENT,
        )
        results.extend(payload.get("results", []))
        collected_pages += 1

    first_page["results"] = results
    first_page["collected_pages"] = collected_pages
    first_page["collected_result_count"] = len(results)
    return first_page
