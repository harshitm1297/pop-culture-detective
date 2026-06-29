from __future__ import annotations

from pathlib import Path
from typing import Any

from cultural_mood_tracker.sources.wikidata import extract_enwiki_title

from .common import find_matching_file, maybe_load_json, parse_wikipedia_timestamp


def build_attention_signals(
    anchors: list[dict[str, Any]],
    wikipedia_run_dir: Path,
    wikidata_run_dir: Path,
    *,
    source_run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for anchor in anchors:
        title_id = f"{anchor['content_type']}_{anchor['tmdb_id']}"

        wikidata_path = find_matching_file(
            wikidata_run_dir / anchor["content_type"],
            f"{anchor['tmdb_id']}_",
            ".json",
        )
        wikidata_payload = maybe_load_json(wikidata_path) if wikidata_path else {}
        expected_article = None
        if isinstance(wikidata_payload, dict):
            expected_article = extract_enwiki_title(wikidata_payload, anchor.get("wikidata_id"))

        source_path = find_matching_file(
            wikipedia_run_dir / anchor["content_type"],
            f"{anchor['tmdb_id']}_",
            ".json",
        )
        payload = maybe_load_json(source_path) if source_path else {}
        if not isinstance(payload, dict) or "error" in payload:
            continue

        items = payload.get("items", [])
        if not items:
            continue

        actual_article = items[0].get("article")
        if expected_article and actual_article and actual_article != expected_article.replace(" ", "_"):
            continue

        for item in items:
            rows.append(
                {
                    "attention_id": f"{title_id}:wikipedia:{item.get('timestamp')}",
                    "title_id": title_id,
                    "source_run_id": source_run_id,
                    "source_name": "wikipedia",
                    "content_type": anchor["content_type"],
                    "signal_name": "pageviews",
                    "signal_value": item.get("views"),
                    "timestamp_utc": parse_wikipedia_timestamp(item.get("timestamp") or ""),
                    "article": item.get("article"),
                    "expected_article": expected_article,
                    "granularity": item.get("granularity"),
                }
            )

    return rows
