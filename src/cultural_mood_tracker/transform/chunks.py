from __future__ import annotations

import re
from typing import Any


def _split_words(text: str) -> list[str]:
    return text.split()


def build_document_chunks(
    documents: list[dict[str, Any]],
    titles_by_id: dict[str, dict[str, Any]],
    *,
    target_words: int = 180,
    overlap_words: int = 40,
    min_chunk_words: int = 40,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    step = max(target_words - overlap_words, 1)

    for document in documents:
        if not document.get("is_usable_for_rag"):
            continue

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
                    "chunk_index": index,
                    "chunk_word_count": len(chunk_words),
                    "chunk_text": chunk_text,
                    "is_usable_for_rag": True,
                }
            )

    return rows
