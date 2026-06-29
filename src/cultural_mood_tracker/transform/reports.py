from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_coverage_report(
    *,
    source_run_id: str,
    process_run_id: str,
    titles: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    ratings: list[dict[str, Any]],
    attention_signals: list[dict[str, Any]],
    document_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    docs_by_source = Counter(row["source_name"] for row in documents)
    usable_docs_by_source = Counter(
        row["source_name"] for row in documents if row.get("is_usable_for_rag")
    )
    titles_with_docs: defaultdict[str, set[str]] = defaultdict(set)
    for row in documents:
        titles_with_docs[row["source_name"]].add(row["title_id"])

    return {
        "source_run_id": source_run_id,
        "process_run_id": process_run_id,
        "title_count": len(titles),
        "document_count": len(documents),
        "usable_document_count": sum(1 for row in documents if row.get("is_usable_for_rag")),
        "chunk_count": len(document_chunks),
        "rating_count": len(ratings),
        "attention_signal_count": len(attention_signals),
        "documents_by_source": dict(docs_by_source),
        "usable_documents_by_source": dict(usable_docs_by_source),
        "titles_with_documents_by_source": {
            source: len(title_ids) for source, title_ids in titles_with_docs.items()
        },
        "titles_with_any_documents": len({row["title_id"] for row in documents}),
        "titles_with_attention_signals": len({row["title_id"] for row in attention_signals}),
        "titles_with_ratings": len({row["title_id"] for row in ratings}),
    }
