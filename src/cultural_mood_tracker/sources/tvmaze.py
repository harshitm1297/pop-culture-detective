from __future__ import annotations

from .base import http_get_json


BASE_TVMAZE_URL = "https://api.tvmaze.com"
TVMAZE_USER_AGENT = "cultural-mood-tracker-tvmaze/0.1"


def fetch_show(imdb_id: str) -> dict:
    return http_get_json(
        (
            f"{BASE_TVMAZE_URL}/lookup/shows"
            f"?imdb={imdb_id}&embed[]=cast&embed[]=crew&embed[]=episodes"
        ),
        user_agent=TVMAZE_USER_AGENT,
    )
