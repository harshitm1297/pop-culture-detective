from __future__ import annotations

from typing import Any

from .common import stable_text_hash, stable_value_hash


def deduplicate_documents(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: dict[str, str] = {}
    seen_texts: dict[tuple[str, str], str] = {}
    duplicate_rows: list[dict[str, Any]] = []

    for row in documents:
        text = row.get("text") or ""
        source_name = row.get("source_name") or ""
        url = (row.get("source_url") or "").strip()
        text_hash = stable_text_hash(text) if text else ""
        duplicate_of = None
        duplicate_reason = None

        if url:
            url_hash = stable_value_hash(url.lower())
            duplicate_of = seen_urls.get(url_hash)
            if duplicate_of:
                duplicate_reason = "duplicate_source_url"
            else:
                seen_urls[url_hash] = row["document_id"]

        if not duplicate_of and text_hash:
            key = (source_name, text_hash)
            duplicate_of = seen_texts.get(key)
            if duplicate_of:
                duplicate_reason = "duplicate_source_text"
            else:
                seen_texts[key] = row["document_id"]

        if duplicate_of:
            duplicate_rows.append(
                {
                    "document_id": row["document_id"],
                    "duplicate_of": duplicate_of,
                    "reason": duplicate_reason,
                }
            )
            continue

        row["text_hash"] = text_hash
        deduped.append(row)

    stats = {
        "input_document_count": len(documents),
        "output_document_count": len(deduped),
        "duplicate_document_count": len(duplicate_rows),
        "duplicates": duplicate_rows,
    }
    return deduped, stats
