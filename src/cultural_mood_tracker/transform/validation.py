from __future__ import annotations

from collections import Counter
from typing import Any


def build_validation_report(
    *,
    source_run_id: str,
    process_run_id: str,
    titles: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    ratings: list[dict[str, Any]],
    attention_signals: list[dict[str, Any]],
    document_dedup_stats: dict[str, Any],
    chunks: list[dict[str, Any]],
    people: list[dict[str, Any]] | None = None,
    title_cast: list[dict[str, Any]] | None = None,
    title_crew: list[dict[str, Any]] | None = None,
    episodes: list[dict[str, Any]] | None = None,
    title_videos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    people = people or []
    title_cast = title_cast or []
    title_crew = title_crew or []
    episodes = episodes or []
    title_videos = title_videos or []
    document_flag_counts = Counter()
    for row in documents:
        for flag in row.get("quality_flags", []):
            document_flag_counts[flag] += 1

    titles_missing_imdb = sum(1 for row in titles if not row.get("imdb_id"))
    titles_missing_wikidata = sum(1 for row in titles if not row.get("wikidata_id"))
    docs_without_url = sum(1 for row in documents if not row.get("source_url"))
    ratings_without_value = sum(1 for row in ratings if row.get("rating_value") in ("", None))
    usable_documents = [row for row in documents if row.get("is_usable_for_rag")]

    return {
        "source_run_id": source_run_id,
        "process_run_id": process_run_id,
        "title_count": len(titles),
        "document_count": len(documents),
        "usable_document_count": len(usable_documents),
        "chunk_count": len(chunks),
        "attention_signal_count": len(attention_signals),
        "people_count": len(people),
        "title_cast_count": len(title_cast),
        "title_crew_count": len(title_crew),
        "episode_count": len(episodes),
        "title_video_count": len(title_videos),
        "titles_missing_imdb_id": titles_missing_imdb,
        "titles_missing_wikidata_id": titles_missing_wikidata,
        "documents_without_source_url": docs_without_url,
        "ratings_without_value": ratings_without_value,
        "document_quality_flag_counts": dict(document_flag_counts),
        "document_deduplication": document_dedup_stats,
        "checks": {
            "has_titles": len(titles) > 0,
            "has_documents": len(documents) > 0,
            "has_usable_documents": len(usable_documents) > 0,
            "has_chunks": len(chunks) > 0,
        },
    }
