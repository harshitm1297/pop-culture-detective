from __future__ import annotations

import time

from .base import http_get_json


BASE_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_USER_AGENT = "cultural-mood-tracker-gdelt/0.1"


def fetch_articles(title_name: str, max_records: int, *, max_attempts: int = 3) -> dict:
    query = f"\"{title_name}\" sourcecountry:US OR sourcecountry:GB"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    for attempt in range(1, max_attempts + 1):
        try:
            return http_get_json(BASE_GDELT_URL, params, user_agent=GDELT_USER_AGENT)
        except RuntimeError as exc:
            if "HTTP 429" not in str(exc) or attempt == max_attempts:
                raise
            time.sleep(5.25)
    raise RuntimeError("GDELT fetch failed without a terminal exception.")
