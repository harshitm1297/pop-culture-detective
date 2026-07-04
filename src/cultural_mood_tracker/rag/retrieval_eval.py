from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from .embeddings import DEFAULT_EMBEDDING_MODEL
from .retrieval import DEFAULT_QUERY_INSTRUCTION, open_collection, query_collection


LOGGER = logging.getLogger(__name__)

DEFAULT_GOLDEN_SET_PATH = "data/eval/retrieval_golden_set.jsonl"
DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    difficulty: str | None = None
    notes: str | None = None


@dataclass
class QueryEvalResult:
    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    missing_relevant_ids: list[str]
    reciprocal_rank: float
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_golden_set(path: Path) -> list[GoldenQuery]:
    """Load a hand-labeled (query -> relevant chunk_ids) evaluation set.

    This is a curated ground-truth file, not pipeline output: it records what a human
    (or someone reading the corpus closely) judged to be the relevant chunk(s) for a
    given natural-language query. See data/eval/retrieval_golden_set.jsonl for the format.
    """
    if not path.exists():
        raise RuntimeError(f"Golden set file not found: {path}")

    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(_load_jsonl(path), start=1):
        query_id = row.get("query_id")
        query = row.get("query")
        relevant_chunk_ids = row.get("relevant_chunk_ids")

        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"Golden set row {index} is missing a non-empty string query_id")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id in golden set: {query_id}")
        seen_ids.add(query_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Golden set row {index} ({query_id}) is missing non-empty query text")
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise ValueError(f"Golden set row {index} ({query_id}) needs a non-empty relevant_chunk_ids list")

        queries.append(
            GoldenQuery(
                query_id=query_id,
                query=query,
                relevant_chunk_ids=[str(cid) for cid in relevant_chunk_ids],
                difficulty=row.get("difficulty"),
                notes=row.get("notes"),
            )
        )

    return queries


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def _precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant_ids) / len(top)


def _reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _existing_chunk_ids(collection: Any, chunk_ids: list[str]) -> set[str]:
    if not chunk_ids:
        return set()
    result = collection.get(ids=chunk_ids, include=[])
    return set(result.get("ids") or [])


def evaluate_golden_set(
    golden_set: list[GoldenQuery],
    *,
    persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
) -> dict[str, Any]:
    """Run every golden query through retrieval and score it against known-relevant chunk_ids."""
    if not golden_set:
        raise ValueError("golden_set must not be empty")

    max_k = max(k_values)
    per_query: list[QueryEvalResult] = []

    # Open the collection once and reuse it across every golden query, instead of
    # reopening a PersistentClient per query (query_collection() would do that by default).
    collection = open_collection(persist_dir, collection_name)

    for item in golden_set:
        retrieved = query_collection(
            item.query,
            model_name=model_name,
            top_k=max_k,
            query_instruction=query_instruction,
            collection=collection,
        )
        retrieved_ids = [chunk.chunk_id for chunk in retrieved]
        relevant_ids = set(item.relevant_chunk_ids)

        present_ids = _existing_chunk_ids(collection, item.relevant_chunk_ids)
        missing_relevant_ids = sorted(relevant_ids - present_ids)

        per_query.append(
            QueryEvalResult(
                query_id=item.query_id,
                query=item.query,
                relevant_chunk_ids=item.relevant_chunk_ids,
                retrieved_chunk_ids=retrieved_ids,
                missing_relevant_ids=missing_relevant_ids,
                reciprocal_rank=_reciprocal_rank(retrieved_ids, relevant_ids),
                recall_at_k={k: _recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
                precision_at_k={k: _precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
            )
        )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    aggregate = {
        "mrr": _mean([r.reciprocal_rank for r in per_query]),
        "recall_at_k": {
            k: _mean([r.recall_at_k[k] for r in per_query]) for k in k_values
        },
        "precision_at_k": {
            k: _mean([r.precision_at_k[k] for r in per_query]) for k in k_values
        },
    }

    all_missing = sorted({cid for r in per_query for cid in r.missing_relevant_ids})

    return {
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "model_name": model_name,
        "k_values": list(k_values),
        "num_queries": len(per_query),
        "aggregate": aggregate,
        "missing_relevant_chunk_ids": all_missing,
        "per_query": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "relevant_chunk_ids": r.relevant_chunk_ids,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "missing_relevant_ids": r.missing_relevant_ids,
                "reciprocal_rank": r.reciprocal_rank,
                "recall_at_k": r.recall_at_k,
                "precision_at_k": r.precision_at_k,
            }
            for r in per_query
        ],
    }
