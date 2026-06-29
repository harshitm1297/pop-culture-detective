from __future__ import annotations

from urllib.parse import quote

from .base import http_get_json


BASE_WIKIPEDIA_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
WIKIPEDIA_USER_AGENT = "cultural-mood-tracker-wikipedia/0.1"


def fetch_pageviews(
    title: str,
    start_date: str,
    end_date: str,
    *,
    article_title: str | None = None,
) -> dict:
    article_name = article_title or title
    article = quote(article_name.replace(" ", "_"), safe="")
    start = start_date.replace("-", "") + "00"
    end = end_date.replace("-", "") + "00"
    url = (
        f"{BASE_WIKIPEDIA_URL}/en.wikipedia.org/all-access/user/{article}/daily/"
        f"{start}/{end}"
    )
    return http_get_json(url, user_agent=WIKIPEDIA_USER_AGENT)
