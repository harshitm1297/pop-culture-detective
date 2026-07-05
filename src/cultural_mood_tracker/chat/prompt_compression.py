from __future__ import annotations

import json
from typing import Any

from cultural_mood_tracker.rag.retrieval import RetrievedChunk

from .sql_schemas import MAX_SQL_RESULTS, validate_sql_payload


MAX_HYBRID_TOKENS = 3000
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_CONTEXT_CHARS = 1200
MAX_RAG_CHUNKS = 4
HYBRID_EVIDENCE_CHUNKS = 3
DEFAULT_SNIPPET_CHARS = 280


def compact_sql_json(sql_data: dict[str, Any]) -> str:
    validate_sql_payload(sql_data)
    safe_payload = {
        "query_type": sql_data["query_type"],
        "title": sql_data.get("title"),
        "results": sql_data.get("results", [])[:MAX_SQL_RESULTS],
        "summary_metrics": sql_data.get("summary_metrics", {}),
    }
    return json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"), default=str)


def format_sql_block(sql_data: dict[str, Any]) -> str:
    return compact_sql_json(sql_data)


def _clean_snippet(text: str, *, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{truncated}..."


def _source_label(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata
    source = metadata.get("source_name") or "unknown source"
    document_type = metadata.get("document_type") or metadata.get("chunk_source_type")
    return f"{source} {document_type}".strip() if document_type else str(source)


def compress_rag_chunks(
    chunks: list[RetrievedChunk],
    *,
    max_chunks: int = MAX_RAG_CHUNKS,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> str:
    selected = chunks[: max(0, min(max_chunks, MAX_RAG_CHUNKS))]
    blocks: list[str] = []
    for index, chunk in enumerate(selected, start=1):
        metadata = chunk.metadata
        title = metadata.get("title_name") or "Unknown title"
        snippet = _clean_snippet(chunk.chunk_text, max_chars=snippet_chars)
        blocks.append(f"[{index}] {title} ({_source_label(chunk)})\n- {snippet}")
    return "\n\n".join(blocks)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def build_hybrid_user_prompt(
    *,
    sql_data: dict[str, Any],
    chunks: list[RetrievedChunk],
    question: str,
    max_tokens: int = MAX_HYBRID_TOKENS,
) -> tuple[str, int, int]:
    sql_json = compact_sql_json(sql_data)
    chunks = chunks[:HYBRID_EVIDENCE_CHUNKS]

    for snippet_chars in (280, 220, 160, 120):
        evidence = compress_rag_chunks(chunks, max_chunks=HYBRID_EVIDENCE_CHUNKS, snippet_chars=snippet_chars)
        prompt = _hybrid_template(sql_json=sql_json, evidence=evidence, question=question)
        token_estimate = estimate_tokens(prompt)
        if token_estimate <= max_tokens and len(evidence) <= MAX_CONTEXT_CHARS:
            return prompt, len(chunks), token_estimate

    evidence = compress_rag_chunks(chunks[:1], max_chunks=1, snippet_chars=120)
    prompt = _hybrid_template(sql_json=sql_json, evidence=evidence, question=question)
    return prompt, min(len(chunks), 1), estimate_tokens(prompt)


def _hybrid_template(*, sql_json: str, evidence: str, question: str) -> str:
    return (
        f"STRUCTURED DATA (JSON):\n{sql_json}\n\n"
        f"EVIDENCE (TOP 3 CHUNKS ONLY):\n{evidence}\n\n"
        f"QUESTION:\n{question}\n\n"
        "RULES:\n"
        "- Use structured data for all numerical claims\n"
        "- Use evidence only for interpretation\n"
        "- If data is missing, explicitly say so"
    )
