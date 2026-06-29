from __future__ import annotations

from .base import http_get_json


BASE_WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData"
WIKIDATA_USER_AGENT = "cultural-mood-tracker-wikidata/0.1"


def fetch_entity(wikidata_id: str) -> dict:
    url = f"{BASE_WIKIDATA_ENTITY_URL}/{wikidata_id}.json"
    return http_get_json(url, user_agent=WIKIDATA_USER_AGENT)


def extract_enwiki_title(payload: dict, wikidata_id: str | None) -> str | None:
    if not wikidata_id:
        return None
    entity = payload.get("entities", {}).get(wikidata_id, {})
    sitelink = entity.get("sitelinks", {}).get("enwiki", {})
    title = sitelink.get("title")
    if not title:
        return None
    return str(title).replace("_", " ")
