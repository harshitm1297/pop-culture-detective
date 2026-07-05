from __future__ import annotations

from collections import Counter

from cultural_mood_tracker.rag.retrieval import RetrievedChunk


MAX_RERANKED_CHUNKS = 4
SOURCE_PRIORITY = {
    "tmdb_review": 0,
    "editorial": 1,
    "overview": 2,
    "other": 3,
}


def classify_chunk(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata
    values = " ".join(
        str(metadata.get(key) or "").casefold()
        for key in ("source_name", "document_type", "chunk_source_type")
    )
    if "tmdb" in values and "review" in values:
        return "tmdb_review"
    if "guardian" in values or "indiewire" in values or "vulture" in values or "editorial" in values or "critic" in values:
        return "editorial"
    if "overview" in values or "tmdb_overview" in values:
        return "overview"
    return "other"


def _title_id(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("title_id") or chunk.metadata.get("title_name") or "")


def _source_key(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("source_name") or chunk.metadata.get("document_type") or "")


def _semantic_key(chunk: RetrievedChunk) -> str:
    words = " ".join(chunk.chunk_text.casefold().split()[:35])
    return f"{_title_id(chunk)}::{classify_chunk(chunk)}::{words}"


def _score(chunk: RetrievedChunk, seen_sources: Counter[str], seen_titles: Counter[str]) -> float:
    priority_bonus = (4 - SOURCE_PRIORITY.get(classify_chunk(chunk), 3)) * 0.08
    source_penalty = 0.10 * seen_sources[_source_key(chunk)]
    title_penalty = 0.04 * max(0, seen_titles[_title_id(chunk)] - 1)
    return chunk.similarity + priority_bonus - source_penalty - title_penalty


def filter_and_rerank_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    seen_semantics: set[str] = set()
    seen_sources: Counter[str] = Counter()
    seen_titles: Counter[str] = Counter()

    for chunk_type in ("tmdb_review", "editorial", "overview", "other"):
        candidates = [chunk for chunk in chunks if classify_chunk(chunk) == chunk_type]
        ranked = sorted(candidates, key=lambda chunk: _score(chunk, seen_sources, seen_titles), reverse=True)
        for chunk in ranked:
            if len(selected) >= MAX_RERANKED_CHUNKS:
                return selected
            semantic_key = _semantic_key(chunk)
            if chunk.chunk_id in selected_ids or semantic_key in seen_semantics:
                continue
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            seen_semantics.add(semantic_key)
            seen_sources[_source_key(chunk)] += 1
            seen_titles[_title_id(chunk)] += 1
            break

    remaining = sorted(chunks, key=lambda chunk: _score(chunk, seen_sources, seen_titles), reverse=True)
    for chunk in remaining:
        if len(selected) >= MAX_RERANKED_CHUNKS:
            break
        semantic_key = _semantic_key(chunk)
        if chunk.chunk_id in selected_ids or semantic_key in seen_semantics:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        seen_semantics.add(semantic_key)
        seen_sources[_source_key(chunk)] += 1
        seen_titles[_title_id(chunk)] += 1

    return selected[:MAX_RERANKED_CHUNKS]


def rerank_chunks(chunks: list[RetrievedChunk], *, limit: int = MAX_RERANKED_CHUNKS) -> list[RetrievedChunk]:
    return filter_and_rerank_chunks(chunks)[: min(limit, MAX_RERANKED_CHUNKS)]


def chunk_type_distribution(chunks: list[RetrievedChunk]) -> dict[str, int]:
    counts = Counter(classify_chunk(chunk) for chunk in chunks)
    return dict(sorted(counts.items()))
