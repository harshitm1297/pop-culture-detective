from __future__ import annotations

from .base import http_get_json


BASE_GUARDIAN_URL = "https://content.guardianapis.com/search"
GUARDIAN_USER_AGENT = "cultural-mood-tracker-guardian/0.1"


def fetch_articles(
    title_name: str,
    content_type: str,
    api_key: str,
    start_date: str,
    end_date: str,
    page_size: int,
) -> dict:
    section = "film" if content_type == "movie" else "tv-and-radio"
    params = {
        "api-key": api_key,
        "q": f"\"{title_name}\"",
        "section": section,
        "from-date": start_date,
        "to-date": end_date,
        "page-size": page_size,
        "show-fields": "headline,trailText,bodyText,byline,publication",
        "order-by": "newest",
    }
    return http_get_json(BASE_GUARDIAN_URL, params, user_agent=GUARDIAN_USER_AGENT)
