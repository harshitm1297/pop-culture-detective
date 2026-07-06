from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .retrieval import RetrievedChunk


LOGGER = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant answering questions about movies and TV shows using ONLY the "
    "context provided below. Ground every claim in the given context. If the context does not "
    "contain enough information to answer, say so explicitly instead of guessing or using outside "
    "knowledge."
)

DEFAULT_NO_CONTEXT_MESSAGE = (
    "No relevant context was found in the knowledge base for this question. Say so explicitly "
    "and do not attempt to answer from outside knowledge."
)

DEFAULT_GROUNDING_REMINDER = (
    "Answer using only the context above. If it is insufficient, say so explicitly."
)

# Character-based budget, not a real tokenizer. ~4 characters/token is a common rough heuristic
# for English text. This remains an approximate budget because hosted chat APIs do tokenization
# server-side.
DEFAULT_MAX_CONTEXT_CHARS = 6000


@dataclass(frozen=True)
class PromptResult:
    system_prompt: str
    user_prompt: str
    included_chunk_ids: list[str]
    excluded_chunk_ids: list[str]
    context_char_count: int

    @property
    def messages(self) -> list[dict[str, str]]:
        """Chat-style message list, for LLM runtimes that expect role/content turns."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

    @property
    def flattened(self) -> str:
        """Single concatenated string, for plain-completion style LLM runtimes."""
        return f"{self.system_prompt}\n\n{self.user_prompt}"


def _label_chunk(index: int, chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    title = meta.get("title_name", "Unknown title")
    source = meta.get("source_name", "unknown_source")
    doc_type = meta.get("document_type", "text")
    published_at = meta.get("published_at")
    suffix = f", published {published_at}" if published_at else ""
    return f"[{index}] {title} — {source}/{doc_type}{suffix}"


def build_context_blocks(
    chunks: list[RetrievedChunk],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    min_similarity: float | None = None,
) -> tuple[str, list[str], list[str]]:
    """Assemble a labeled context block from ranked chunks within a character budget.

    Returns (context_text, included_chunk_ids, excluded_chunk_ids). Chunks are considered in
    rank order; once the budget is exhausted, every remaining chunk is excluded, even if it
    would individually fit -- this preserves the retrieval ranking instead of packing greedily.
    """
    included_ids: list[str] = []
    excluded_ids: list[str] = []
    blocks: list[str] = []
    used_chars = 0
    budget_exhausted = False

    for chunk in chunks:
        if min_similarity is not None and chunk.similarity < min_similarity:
            excluded_ids.append(chunk.chunk_id)
            continue

        if budget_exhausted:
            excluded_ids.append(chunk.chunk_id)
            continue

        label = _label_chunk(len(included_ids) + 1, chunk)
        block = f"{label}\n{chunk.chunk_text}"
        block_chars = len(block) + 2  # + blank-line separator

        if used_chars + block_chars > max_context_chars and included_ids:
            # Only refuse a chunk once we already have at least one; always allow the first
            # chunk through even if it alone exceeds the budget, so we never emit an empty
            # context purely because of a single oversized chunk.
            budget_exhausted = True
            excluded_ids.append(chunk.chunk_id)
            continue

        blocks.append(block)
        included_ids.append(chunk.chunk_id)
        used_chars += block_chars

    return "\n\n".join(blocks), included_ids, excluded_ids


def build_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    min_similarity: float | None = None,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    no_context_message: str = DEFAULT_NO_CONTEXT_MESSAGE,
) -> PromptResult:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    context_text, included_ids, excluded_ids = build_context_blocks(
        chunks,
        max_context_chars=max_context_chars,
        min_similarity=min_similarity,
    )

    if not included_ids:
        user_prompt = f"Question: {query.strip()}\n\n{no_context_message}"
    else:
        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query.strip()}\n\n"
            f"{DEFAULT_GROUNDING_REMINDER}"
        )

    LOGGER.info(
        "Built prompt: %s chunk(s) included, %s excluded, %s context chars",
        len(included_ids),
        len(excluded_ids),
        len(context_text),
    )

    return PromptResult(
        system_prompt=system_instruction,
        user_prompt=user_prompt,
        included_chunk_ids=included_ids,
        excluded_chunk_ids=excluded_ids,
        context_char_count=len(context_text),
    )


def load_retrieved_chunks_json(path: Path) -> tuple[str, list[RetrievedChunk]]:
    """Load a query + ranked chunks previously saved by `cli/retrieve.py --output-path`.

    This lets prompt construction be developed and tested completely decoupled from
    ChromaDB/sentence-transformers: run retrieve.py once, then iterate on prompting.py
    purely against the saved JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    query = payload.get("query")
    results = payload.get("results")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{path} is missing a non-empty 'query' field")
    if not isinstance(results, list):
        raise ValueError(f"{path} is missing a 'results' list")

    chunks: list[RetrievedChunk] = []
    for index, row in enumerate(results, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} results[{index}] is not an object")
        chunk_id = row.get("chunk_id")
        chunk_text = row.get("chunk_text")
        distance = row.get("distance")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"{path} results[{index}] missing chunk_id")
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError(f"{path} results[{index}] missing chunk_text")
        if not isinstance(distance, (int, float)):
            raise ValueError(f"{path} results[{index}] missing numeric distance")
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                metadata=row.get("metadata") or {},
                distance=float(distance),
            )
        )

    return query, chunks
