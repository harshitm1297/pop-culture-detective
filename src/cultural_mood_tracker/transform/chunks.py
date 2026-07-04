from __future__ import annotations

import re
from typing import Any


def _split_words(text: str) -> list[str]:
    return text.split()


def _chunk_policy(document_type: str) -> tuple[int, int, int]:
    policies = {
        "user_review": (200, 40, 50),
        "critic_article": (300, 60, 70),
        "news_article": (260, 50, 60),
        "show_summary": (140, 30, 40),
        "overview": (140, 30, 40),
        "entity_description": (80, 10, 20),
    }
    return policies.get(document_type, (180, 40, 40))


def _chunk_priority(document: dict[str, Any]) -> str:
    if document.get("document_type") == "user_review":
        return "high"
    if document.get("document_type") == "critic_article":
        return "high"
    if document.get("document_type") == "show_summary":
        return "medium"
    if document.get("document_type") == "overview":
        return "medium"
    return "low"


def build_document_chunks(
    documents: list[dict[str, Any]],
    titles_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for document in documents:
        if not document.get("is_usable_for_rag"):
            continue

        target_words, overlap_words, min_chunk_words = _chunk_policy(document.get("document_type") or "")
        step = max(target_words - overlap_words, 1)
        words = _split_words(document.get("text") or "")
        if not words:
            continue

        title = titles_by_id.get(document["title_id"], {})
        if len(words) <= target_words:
            spans = [(0, len(words))]
        else:
            spans = []
            start = 0
            while start < len(words):
                end = min(start + target_words, len(words))
                if end - start < min_chunk_words and spans:
                    break
                spans.append((start, end))
                if end == len(words):
                    break
                start += step

        for index, (start, end) in enumerate(spans, start=1):
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            chunk_text = re.sub(r"https?://\S+", "", chunk_text).strip()
            if len(chunk_words) < min_chunk_words and spans and len(spans) > 1:
                continue
            rows.append(
                {
                    "chunk_id": f"{document['document_id']}:chunk_{index:03d}",
                    "document_id": document["document_id"],
                    "title_id": document["title_id"],
                    "source_run_id": document["source_run_id"],
                    "title_name": document["title_name"],
                    "content_type": document["content_type"],
                    "source_name": document["source_name"],
                    "document_type": document["document_type"],
                    "published_at": document.get("published_at"),
                    "language": document.get("language"),
                    "genre": title.get("genres", []),
                    "release_year": title.get("release_year"),
                    "source_match_method": document.get("source_match_method"),
                    "source_match_confidence": document.get("source_match_confidence"),
                    "chunk_source_type": document.get("document_type"),
                    "chunk_priority": _chunk_priority(document),
                    "chunk_index": index,
                    "chunk_word_count": len(chunk_words),
                    "chunk_text": chunk_text,
                    "is_usable_for_rag": True,
                }
            )

    return rows
