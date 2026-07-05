from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from cultural_mood_tracker.core import load_project_environment

from .sql_schemas import normalize_sql_output


READ_ONLY_PREFIXES = {"select", "with", "show", "describe"}


def _import_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install duckdb before querying MotherDuck.") from exc
    return duckdb


def _ensure_read_only_sql(sql: str) -> str:
    candidate = sql.strip().rstrip(";")
    if not candidate:
        raise RuntimeError("SQL query cannot be empty.")
    first_token = candidate.split(maxsplit=1)[0].lower()
    if first_token not in READ_ONLY_PREFIXES:
        raise RuntimeError("Only read-only SELECT/WITH/SHOW/DESCRIBE queries are allowed.")
    return candidate


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")


def _title_filter(title: str, column: str = "title_name") -> str:
    safe_title = _escape_like(title)
    return f"{column} ILIKE '%{safe_title}%' ESCAPE '\\'"


def _rows_from_cursor(cursor) -> list[dict[str, Any]]:
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _time_window(start: Any, end: Any) -> str:
    if start is None and end is None:
        return "unknown"
    if start == end:
        return str(start)
    return f"{start} to {end}"


class MotherDuckClient:
    def __init__(self, *, database: str | None = None, token: str | None = None) -> None:
        load_project_environment(Path.cwd())
        self.database = database or os.getenv("MOTHERDUCK_DATABASE", "cultural_mood_tracker").strip()
        self.token = token or os.getenv("MOTHERDUCK_TOKEN", "").strip()
        self._connection = None

    def connect(self):
        if self._connection is not None:
            return self._connection
        if not self.database:
            raise RuntimeError("Missing required environment variable: MOTHERDUCK_DATABASE")
        if not self.token:
            raise RuntimeError("Missing required environment variable: MOTHERDUCK_TOKEN")

        duckdb = _import_duckdb()
        self._connection = duckdb.connect(
            f"md:{self.database}",
            config={"motherduck_token": self.token},
        )
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _query_rows(self, sql: str) -> list[dict[str, Any]]:
        read_only_sql = _ensure_read_only_sql(sql)
        cursor = self.connect().execute(read_only_sql)
        return _rows_from_cursor(cursor)

    def table_exists(self, table_name: str) -> bool:
        safe_name = table_name.replace("'", "''")
        rows = self._query_rows(
            f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE lower(table_name) = lower('{safe_name}')
            LIMIT 1
            """
        )
        return bool(rows)

    def get_title_ratings(self, title: str) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            WITH matched_titles AS (
                SELECT title_id, title_name
                FROM titles
                WHERE {_title_filter(title)}
                ORDER BY CASE WHEN lower(title_name) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title_name
                LIMIT 1
            ),
            title_ratings AS (
                SELECT
                    mt.title_name AS title,
                    lower(r.source_name) AS source_name,
                    TRY_CAST(r.rating_value AS DOUBLE) AS rating_value,
                    TRY_CAST(r.rating_count AS BIGINT) AS rating_count
                FROM matched_titles mt
                LEFT JOIN ratings r USING (title_id)
                WHERE r.rating_scope = 'title_aggregate'
            )
            SELECT
                COALESCE(MAX(title), '{_escape_like(title)}') AS title,
                AVG(CASE WHEN source_name = 'tmdb' THEN rating_value END) AS rating_tmdb,
                AVG(CASE WHEN source_name = 'imdb' THEN rating_value END) AS rating_imdb,
                AVG(rating_value) AS rating_aggregate,
                SUM(rating_count) AS rating_count
            FROM title_ratings
            """
        )
        row = rows[0] if rows else {}
        resolved_title = str(row.get("title") or title)
        result = {
            "title": resolved_title,
            "rating_tmdb": _float_or_none(row.get("rating_tmdb")),
            "rating_imdb": _float_or_none(row.get("rating_imdb")),
            "rating_aggregate": _float_or_none(row.get("rating_aggregate")),
            "rating_count": _int_or_none(row.get("rating_count")),
        }
        return normalize_sql_output(
            result,
            "rating",
            title=resolved_title,
            summary_metrics={
                "rating_aggregate": result["rating_aggregate"],
                "rating_count": result["rating_count"],
            },
        )

    def get_attention(self, title: str) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            WITH matched_titles AS (
                SELECT title_id, title_name
                FROM titles
                WHERE {_title_filter(title)}
                ORDER BY CASE WHEN lower(title_name) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title_name
                LIMIT 1
            )
            SELECT
                COALESCE(MAX(mt.title_name), '{_escape_like(title)}') AS title,
                COALESCE(AVG(TRY_CAST(a.signal_value AS DOUBLE)), 0.0) AS attention_score,
                MIN(a.timestamp_utc) AS window_start,
                MAX(a.timestamp_utc) AS window_end
            FROM matched_titles mt
            LEFT JOIN attention_signals a USING (title_id)
            """
        )
        row = rows[0] if rows else {}
        resolved_title = str(row.get("title") or title)
        attention_score = _float_or_none(row.get("attention_score"))
        result = {
            "title": resolved_title,
            "attention_score": attention_score,
            "time_window": _time_window(row.get("window_start"), row.get("window_end")),
        }
        return normalize_sql_output(
            result,
            "attention",
            title=resolved_title,
            summary_metrics={"attention_score": attention_score},
        )

    def get_cast(self, title: str) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            WITH matched_titles AS (
                SELECT title_id, title_name
                FROM titles
                WHERE {_title_filter(title)}
                ORDER BY CASE WHEN lower(title_name) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title_name
                LIMIT 1
            ),
            cast_rows AS (
                SELECT DISTINCT
                    mt.title_name AS title,
                    p.name AS person_name,
                    MIN(c.billing_order) AS billing_order
                FROM matched_titles mt
                LEFT JOIN title_cast c USING (title_id)
                LEFT JOIN people p USING (person_id)
                WHERE p.name IS NOT NULL AND p.name <> ''
                GROUP BY mt.title_name, p.name
                ORDER BY billing_order NULLS LAST, person_name
                LIMIT 15
            )
            SELECT
                COALESCE(MAX(title), '{_escape_like(title)}') AS title,
                list(person_name ORDER BY billing_order NULLS LAST, person_name) AS cast_names
            FROM cast_rows
            """
        )
        row = rows[0] if rows else {}
        cast_names = row.get("cast_names") or []
        resolved_title = str(row.get("title") or title)
        cast = [str(name) for name in cast_names if name]
        return normalize_sql_output(
            {"title": resolved_title, "cast": cast},
            "cast",
            title=resolved_title,
            summary_metrics={"cast_count": len(cast)},
        )

    def get_comparison(self, title_a: str, title_b: str) -> dict[str, Any]:
        rating_a = self.get_title_ratings(title_a)
        rating_b = self.get_title_ratings(title_b)
        attention_a = self.get_attention(title_a)
        attention_b = self.get_attention(title_b)
        row_a = rating_a["results"][0] if rating_a["results"] else {}
        row_b = rating_b["results"][0] if rating_b["results"] else {}
        att_a = attention_a["results"][0] if attention_a["results"] else {}
        att_b = attention_b["results"][0] if attention_b["results"] else {}
        result = {
            "title_a": row_a.get("title") or title_a,
            "title_b": row_b.get("title") or title_b,
            "rating_a": row_a.get("rating_aggregate"),
            "rating_b": row_b.get("rating_aggregate"),
            "attention_a": att_a.get("attention_score"),
            "attention_b": att_b.get("attention_score"),
        }
        return normalize_sql_output(
            result,
            "comparison",
            title=None,
            summary_metrics=result,
        )

    def get_top_rated_titles(self, limit: int = 10) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            SELECT
                t.title_name AS title,
                AVG(TRY_CAST(r.rating_value AS DOUBLE)) AS avg_rating,
                SUM(TRY_CAST(r.rating_count AS BIGINT)) AS rating_count
            FROM ratings r
            JOIN titles t USING (title_id)
            WHERE r.rating_scope = 'title_aggregate'
            GROUP BY t.title_id, t.title_name
            HAVING avg_rating IS NOT NULL
            ORDER BY avg_rating DESC, rating_count DESC NULLS LAST
            LIMIT {int(limit)}
            """
        )
        results = [
            {
                "title": row.get("title"),
                "avg_rating": _float_or_none(row.get("avg_rating")),
                "rating_count": _int_or_none(row.get("rating_count")),
            }
            for row in rows[:10]
        ]
        return normalize_sql_output(
            results,
            "aggregate",
            title=None,
            summary_metrics={"metric": "top_rated_titles", "result_count": len(results)},
        )

    def get_top_attention_titles(self, limit: int = 10) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            SELECT
                t.title_name AS title,
                AVG(TRY_CAST(a.signal_value AS DOUBLE)) AS attention_score,
                MIN(a.timestamp_utc) AS window_start,
                MAX(a.timestamp_utc) AS window_end
            FROM attention_signals a
            JOIN titles t USING (title_id)
            GROUP BY t.title_id, t.title_name
            HAVING attention_score IS NOT NULL
            ORDER BY attention_score DESC
            LIMIT {int(limit)}
            """
        )
        results = [
            {
                "title": row.get("title"),
                "attention_score": _float_or_none(row.get("attention_score")),
                "time_window": _time_window(row.get("window_start"), row.get("window_end")),
            }
            for row in rows[:10]
        ]
        return normalize_sql_output(
            results,
            "aggregate",
            title=None,
            summary_metrics={"metric": "top_attention_titles", "result_count": len(results)},
        )

    def get_rating_stats(self, title: str) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            WITH matched_titles AS (
                SELECT title_id, title_name
                FROM titles
                WHERE {_title_filter(title)}
                ORDER BY CASE WHEN lower(title_name) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title_name
                LIMIT 1
            )
            SELECT
                COALESCE(MAX(mt.title_name), '{_escape_like(title)}') AS title,
                AVG(TRY_CAST(r.rating_value AS DOUBLE)) AS avg_rating,
                MIN(TRY_CAST(r.rating_value AS DOUBLE)) AS min_rating,
                MAX(TRY_CAST(r.rating_value AS DOUBLE)) AS max_rating,
                SUM(TRY_CAST(r.rating_count AS BIGINT)) AS rating_count
            FROM matched_titles mt
            LEFT JOIN ratings r USING (title_id)
            WHERE r.rating_scope = 'title_aggregate'
            """
        )
        row = rows[0] if rows else {}
        resolved_title = str(row.get("title") or title)
        result = {
            "title": resolved_title,
            "avg_rating": _float_or_none(row.get("avg_rating")),
            "min_rating": _float_or_none(row.get("min_rating")),
            "max_rating": _float_or_none(row.get("max_rating")),
            "rating_count": _int_or_none(row.get("rating_count")),
        }
        return normalize_sql_output(
            result,
            "rating",
            title=resolved_title,
            summary_metrics={
                "avg_rating": result["avg_rating"],
                "min_rating": result["min_rating"],
                "max_rating": result["max_rating"],
                "rating_count": result["rating_count"],
            },
        )

    def get_title_profile(self, title: str) -> dict[str, Any]:
        ratings = self.get_title_ratings(title)
        attention = self.get_attention(title)
        cast = self.get_cast(title)
        rating_row = ratings["results"][0] if ratings["results"] else {}
        attention_row = attention["results"][0] if attention["results"] else {}
        cast_row = cast["results"][0] if cast["results"] else {}
        resolved_title = str(rating_row.get("title") or attention_row.get("title") or cast_row.get("title") or title)
        result = {
            "title": resolved_title,
            "rating_tmdb": rating_row.get("rating_tmdb"),
            "rating_imdb": rating_row.get("rating_imdb"),
            "rating_aggregate": rating_row.get("rating_aggregate"),
            "rating_count": rating_row.get("rating_count"),
            "attention_score": attention_row.get("attention_score"),
            "time_window": attention_row.get("time_window"),
            "cast": cast_row.get("cast", []),
        }
        return normalize_sql_output(
            result,
            "aggregate",
            title=resolved_title,
            summary_metrics={
                "rating_aggregate": result["rating_aggregate"],
                "rating_count": result["rating_count"],
                "attention_score": result["attention_score"],
            },
        )

    def run_structured_query(self, user_query: str) -> dict[str, Any]:
        title_a, title_b = extract_comparison_titles(user_query)
        if title_a and title_b:
            return self.get_comparison(title_a, title_b)

        title = extract_title(user_query) or user_query
        normalized = user_query.casefold()

        if any(word in normalized for word in ("cast", "acted", "actor", "actors", "starring")):
            return self.get_cast(title)
        if any(word in normalized for word in ("top", "best")) and any(word in normalized for word in ("attention", "trend", "popular", "popularity")):
            return self.get_top_attention_titles()
        if any(word in normalized for word in ("top", "best")):
            return self.get_top_rated_titles()
        if "stats" in normalized or "statistics" in normalized:
            return self.get_rating_stats(title)
        if any(word in normalized for word in ("rating", "ratings", "score")):
            return self.get_title_ratings(title)
        if any(word in normalized for word in ("attention", "trend", "trending", "popular", "popularity")):
            return self.get_title_profile(title)
        return self.get_title_profile(title)


def extract_comparison_titles(query: str) -> tuple[str | None, str | None]:
    quoted = [double or single for double, single in re.findall(r'"([^"]+)"|\'([^\']+)\'', query)]
    quoted = [value.strip() for value in quoted if value.strip()]
    if len(quoted) >= 2:
        return quoted[0], quoted[1]

    match = re.search(r"\bcompare\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .?!"), match.group(2).strip(" .?!")
    match = re.search(r"\b(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .?!"), match.group(2).strip(" .?!")
    return None, None


def extract_title(query: str) -> str | None:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    for double_quoted, single_quoted in quoted:
        value = (double_quoted or single_quoted).strip()
        if value:
            return value

    patterns = (
        r"\bwhy\s+is\s+(.+?)\s+(?:popular|trending|so\s+popular)(?:\?|$)",
        r"\b(?:about|for|of|in|from)\s+(.+?)(?:\?|$)",
        r"\b(?:movie|show|title)\s+(.+?)(?:\?|$)",
    )
    stop_words = {
        "rating",
        "ratings",
        "score",
        "cast",
        "actors",
        "actor",
        "acted",
        "director",
        "directors",
        "attention",
        "trend",
        "trending",
        "popular",
        "popularity",
    }
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip(" .?!")
        words = [word for word in candidate.split() if word.casefold() not in stop_words]
        cleaned = " ".join(words).strip()
        if cleaned:
            return cleaned
    return None
