from __future__ import annotations

from typing import Any, Literal


QueryType = Literal["rating", "cast", "attention", "aggregate", "comparison"]
VALID_QUERY_TYPES = {"rating", "cast", "attention", "aggregate", "comparison"}
MAX_SQL_RESULTS = 10


def normalize_sql_output(
    rows: list[dict[str, Any]] | dict[str, Any],
    query_type: QueryType,
    *,
    title: str | None = None,
    summary_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if query_type not in VALID_QUERY_TYPES:
        raise RuntimeError(f"Unsupported SQL query_type: {query_type!r}")

    if isinstance(rows, dict):
        normalized_rows = [rows]
    elif isinstance(rows, list):
        normalized_rows = rows[:MAX_SQL_RESULTS]
    else:
        raise RuntimeError("SQL rows must be a dict or list of dicts before normalization.")

    for row in normalized_rows:
        if not isinstance(row, dict):
            raise RuntimeError("SQL results must contain only dict rows.")

    return {
        "query_type": query_type,
        "title": title,
        "results": [_remove_missing(row) for row in normalized_rows],
        "summary_metrics": _remove_missing(summary_metrics or {}),
    }


def _remove_missing(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value is not None}


def validate_sql_payload(payload: dict[str, Any]) -> None:
    # SQL must always return aggregated or single-object results.
    # Never return raw table rows to orchestrator.
    if not isinstance(payload, dict):
        raise RuntimeError("SQL payload must be a normalized object, not raw rows.")
    if payload.get("query_type") not in VALID_QUERY_TYPES:
        raise RuntimeError(f"Unsupported SQL query_type: {payload.get('query_type')!r}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("SQL payload results must be a list.")
    if len(results) > MAX_SQL_RESULTS:
        raise RuntimeError(f"SQL payload exceeds max_sql_results={MAX_SQL_RESULTS}.")
    for row in results:
        if not isinstance(row, dict):
            raise RuntimeError("SQL payload results must contain only dict objects.")
        forbidden_raw_keys = {"rating_id", "source_record_id", "author", "review_text", "chunk_text"}
        if forbidden_raw_keys.intersection(row):
            raise RuntimeError(f"Raw SQL fields are not allowed in LLM payloads: {forbidden_raw_keys.intersection(row)}")
