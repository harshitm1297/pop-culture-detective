from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from cultural_mood_tracker.core import load_project_environment

from .sql_schemas import normalize_sql_output


READ_ONLY_PREFIXES = {"select", "with", "show", "describe"}
SQL_CACHE_TTL_SECONDS = 300


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


RECOMMENDATION_THEME_ALIASES = {
    "comfort": {"comfort", "comforting", "cozy", "heartwarming", "feel sad", "feeling sad", "sad", "hopeful"},
    "nostalgia": {"nostalgia", "nostalgic"},
    "anxiety": {"dark", "horror", "tense", "unsettling", "anxious"},
    "escapism": {"escapism", "escape", "fun", "funny", "lighthearted", "uplifting", "feel happy", "feeling happy"},
    "loneliness": {"loneliness", "lonely", "isolation", "isolated"},
    "identity": {"identity", "emotional", "self", "coming of age"},
}

RECOMMENDATION_GENRE_ALIASES = {
    "Comedy": {"comedy", "funny", "lighthearted", "feel good", "feel-good"},
    "Drama": {"drama", "emotional", "heartwarming", "hopeful"},
    "Horror": {"horror", "scary", "dark", "unsettling"},
    "Science Fiction": {"science fiction", "sci-fi", "scifi", "nostalgic sci-fi"},
    "Thriller": {"thriller", "tense", "dark"},
    "Romance": {"romance", "romantic", "heartwarming"},
}


def _sql_literal(value: str) -> str:
    return f"'{_escape_like(value)}'"


def _sql_list_contains_any(column: str, values: list[str]) -> str:
    if not values:
        return "FALSE"
    checks = [f"COALESCE(list_contains({column}, {_sql_literal(value)}), FALSE)" for value in values]
    return "(" + " OR ".join(checks) + ")"


def _recommendation_terms(query: str) -> tuple[list[str], list[str]]:
    normalized = query.casefold()
    themes = [
        theme
        for theme, aliases in RECOMMENDATION_THEME_ALIASES.items()
        if any(alias in normalized for alias in aliases)
    ]
    genres = [
        genre
        for genre, aliases in RECOMMENDATION_GENRE_ALIASES.items()
        if any(alias in normalized for alias in aliases)
    ]
    return themes, genres


def _sentiment_tone(value: Any) -> str | None:
    score = _float_or_none(value)
    if score is None:
        return None
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "mixed"


def _compact_recommendation_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for row in rows[:limit]:
        compact = {
            "title": row.get("title"),
            "genres": row.get("genres") or [],
            "source_group": row.get("source_group"),
            "dominant_themes": row.get("dominant_themes") or [],
            "audience_themes": row.get("audience_themes") or [],
            "editorial_themes": row.get("editorial_themes") or [],
            "emotional_tone": row.get("emotional_tone") or _sentiment_tone(row.get("avg_sentiment_score")),
            "evidence_count": _int_or_none(row.get("evidence_count")),
            "title_count": _int_or_none(row.get("title_count")),
            "avg_sentiment_score": _float_or_none(row.get("avg_sentiment_score")),
        }
        compact_rows.append({key: value for key, value in compact.items() if value not in (None, [], "")})
    return compact_rows


def _unique_titles(titles: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for title in titles:
        cleaned = _cleanup_comparison_title_candidate(str(title))
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def _cleanup_comparison_title_candidate(value: str) -> str:
    cleaned = " ".join(str(value).split()).strip(" .?!:;")
    cleaned = re.sub(
        r"\s*(?:—|-|,|\?)\s*which\s+is\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*(?:—|-|,|\?)\s*(?:which\s+title|which\s+movie)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\bwhich\s+is\s+more\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" .?!:;")


def _comparison_title_candidates(query: str) -> list[str]:
    quoted = [double or single for double, single in re.findall(r'"([^"]+)"|\'([^\']+)\'', query)]
    candidates = [value.strip() for value in quoted if value.strip()]

    patterns = (
        r"\bcompare\s+(.+?)\s+(?:and|vs|versus|or)\s+(.+?)(?:\?|$)",
        r"\b(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)",
        r":\s*(.+?)\s+or\s+(.+?)(?:\?|$)",
        r"\bdo\s+(.+?)\s+and\s+(.+?)\s+have\b",
        r"\bdoes\s+(.+?)\s+feel\s+.+?\s+than\s+(.+?)(?:\?|$)",
        r"\b(?:between|of)\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\b(.+?)\s+or\s+(.+?)(?:\?|$)",
    )
    leading_noise = re.compile(
        r"^(?:which\s+(?:is|movie|title)\s+.*?|do|does|is|are|movie|show|title)\s+",
        flags=re.IGNORECASE,
    )
    topic_noise = re.compile(
        r"^(?:audience\s+(?:mood\s+in|reception\s+of)|reception\s+of|mood\s+in)\s+",
        flags=re.IGNORECASE,
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        for group in match.groups():
            candidate = leading_noise.sub("", group).strip(" .?!:;")
            if ":" in candidate:
                candidate = candidate.rsplit(":", 1)[-1].strip(" .?!:;")
            candidate = topic_noise.sub("", candidate).strip(" .?!:;")
            candidate = _cleanup_comparison_title_candidate(candidate)
            if candidate:
                candidates.append(candidate)
    return _unique_titles(candidates)


class MotherDuckClient:
    def __init__(self, *, database: str | None = None, token: str | None = None) -> None:
        load_project_environment(Path.cwd())
        self.database = database or os.getenv("MOTHERDUCK_DATABASE", "cultural_mood_tracker").strip()
        self.token = token or os.getenv("MOTHERDUCK_TOKEN", "").strip()
        self._connection = None
        self._query_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

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
        cached = self._query_cache.get(read_only_sql)
        now = time.monotonic()
        if cached and now - cached[0] <= SQL_CACHE_TTL_SECONDS:
            return [dict(row) for row in cached[1]]
        cursor = self.connect().execute(read_only_sql)
        rows = _rows_from_cursor(cursor)
        self._query_cache[read_only_sql] = (now, rows)
        return [dict(row) for row in rows]

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

    def get_title_theme_profile(self, title: str) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            WITH matched_titles AS (
                SELECT title_id, title_name, COALESCE(genres, imdb_genres, tvmaze_genres) AS genres
                FROM titles
                WHERE {_title_filter(title)}
                ORDER BY CASE WHEN lower(title_name) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title_name
                LIMIT 1
            )
            SELECT
                mt.title_name AS title,
                mt.genres,
                s.dominant_themes,
                s.chunk_count AS evidence_count,
                s.avg_sentiment_score,
                CASE
                    WHEN s.avg_sentiment_score >= 0.05 THEN 'positive'
                    WHEN s.avg_sentiment_score <= -0.05 THEN 'negative'
                    ELSE 'mixed'
                END AS emotional_tone
            FROM matched_titles mt
            LEFT JOIN title_theme_summary s USING (title_id)
            LIMIT 1
            """
        )
        compact = _compact_recommendation_rows(rows, limit=1)
        return compact[0] if compact else {"title": title}

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

    def get_title_metrics(self, title: str) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            WITH matched_titles AS (
                SELECT title_id, title_name
                FROM titles
                WHERE {_title_filter(title)}
                ORDER BY CASE WHEN lower(title_name) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title_name
                LIMIT 1
            ),
            rating_metrics AS (
                SELECT
                    mt.title_id,
                    mt.title_name AS title,
                    AVG(TRY_CAST(r.rating_value AS DOUBLE)) AS avg_rating,
                    AVG(CASE WHEN lower(r.source_name) = 'imdb' THEN TRY_CAST(r.rating_value AS DOUBLE) END) AS imdb_rating,
                    AVG(CASE WHEN lower(r.source_name) = 'tmdb' THEN TRY_CAST(r.rating_value AS DOUBLE) END) AS tmdb_rating,
                    SUM(TRY_CAST(r.rating_count AS BIGINT)) AS rating_count
                FROM matched_titles mt
                LEFT JOIN ratings r USING (title_id)
                WHERE r.rating_scope = 'title_aggregate'
                GROUP BY mt.title_id, mt.title_name
            ),
            attention_ranked AS (
                SELECT
                    a.title_id,
                    AVG(TRY_CAST(a.signal_value AS DOUBLE)) AS attention_score,
                    RANK() OVER (ORDER BY AVG(TRY_CAST(a.signal_value AS DOUBLE)) DESC) AS attention_rank,
                    COUNT(*) OVER () AS ranked_title_count
                FROM attention_signals a
                GROUP BY a.title_id
            )
            SELECT
                COALESCE(MAX(rm.title), '{_escape_like(title)}') AS title,
                AVG(rm.avg_rating) AS avg_rating,
                AVG(rm.imdb_rating) AS imdb_rating,
                AVG(rm.tmdb_rating) AS tmdb_rating,
                SUM(rm.rating_count) AS rating_count,
                AVG(ar.attention_score) AS attention_score,
                MIN(ar.attention_rank) AS attention_rank,
                MAX(ar.ranked_title_count) AS ranked_title_count
            FROM rating_metrics rm
            LEFT JOIN attention_ranked ar USING (title_id)
            """
        )
        row = rows[0] if rows else {}
        return {
            "title": str(row.get("title") or title),
            "avg_rating": _float_or_none(row.get("avg_rating")),
            "rating_count": _int_or_none(row.get("rating_count")),
            "imdb_rating": _float_or_none(row.get("imdb_rating")),
            "tmdb_rating": _float_or_none(row.get("tmdb_rating")),
            "attention_score": _float_or_none(row.get("attention_score")),
            "attention_rank": _int_or_none(row.get("attention_rank")),
            "ranked_title_count": _int_or_none(row.get("ranked_title_count")),
        }

    def get_attention_vs_reception(self, title: str) -> dict[str, Any]:
        rows = self._query_rows(
            f"""
            WITH ranked AS (
                SELECT
                    t.title_id,
                    t.title_name AS title,
                    avr.avg_rating_value,
                    avr.attention_total,
                    avr.attention_peak,
                    avr.avg_sentiment_score,
                    avr.dominant_themes,
                    avr.chunk_count,
                    CUME_DIST() OVER (ORDER BY avr.attention_total) AS attention_percentile
                FROM attention_vs_reception avr
                JOIN titles t USING (title_id)
                WHERE avr.attention_total IS NOT NULL
            )
            SELECT *
            FROM ranked
            WHERE {_title_filter(title, "title")}
            ORDER BY CASE WHEN lower(title) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title
            LIMIT 1
            """
        )
        row = rows[0] if rows else {}
        title_name = str(row.get("title") or title)
        sentiment = _float_or_none(row.get("avg_sentiment_score"))
        reception_summary = None
        if row:
            reception_summary = (
                f"{title_name} has attention_total={_float_or_none(row.get('attention_total'))}, "
                f"avg_rating={_float_or_none(row.get('avg_rating_value'))}, "
                f"avg_sentiment={sentiment}, and themes={row.get('dominant_themes') or []}."
            )
        return {
            "title": title_name,
            "attention_score": _float_or_none(row.get("attention_total")),
            "attention_peak": _float_or_none(row.get("attention_peak")),
            "attention_percentile": _float_or_none(row.get("attention_percentile")),
            "avg_rating": _float_or_none(row.get("avg_rating_value")),
            "rating_count": _int_or_none(row.get("chunk_count")),
            "avg_sentiment_score": sentiment,
            "dominant_themes": row.get("dominant_themes") or [],
            "reception_summary": reception_summary,
        }

    def get_title_analytical_summaries(self, title: str) -> dict[str, Any]:
        theme = self.get_title_theme_profile(title)
        audience_rows = self._query_rows(
            f"""
            WITH matched_titles AS (
                SELECT title_id, title_name
                FROM titles
                WHERE {_title_filter(title)}
                ORDER BY CASE WHEN lower(title_name) = lower('{_escape_like(title)}') THEN 0 ELSE 1 END, title_name
                LIMIT 1
            )
            SELECT
                mt.title_name AS title,
                s.source_group,
                s.chunk_count AS evidence_count,
                s.dominant_themes,
                s.avg_sentiment_score,
                CASE
                    WHEN s.avg_sentiment_score >= 0.05 THEN 'positive'
                    WHEN s.avg_sentiment_score <= -0.05 THEN 'negative'
                    ELSE 'mixed'
                END AS emotional_tone
            FROM matched_titles mt
            JOIN audience_vs_editorial_summary s USING (title_id)
            WHERE s.source_group IN ('audience', 'editorial')
            ORDER BY s.source_group
            LIMIT 4
            """
        )
        attention_reception = self.get_attention_vs_reception(title)
        return {
            "title": theme.get("title") or attention_reception.get("title") or title,
            "title_theme_summary": theme,
            "audience_vs_editorial_summary": _compact_recommendation_rows(audience_rows, limit=4),
            "attention_vs_reception": attention_reception,
        }

    def compare_titles(self, title_a: str, title_b: str) -> dict[str, Any]:
        results = []
        for title in (title_a, title_b):
            metrics = self.get_title_metrics(title)
            analytics = self.get_title_analytical_summaries(title)
            results.append(
                {
                    "title": metrics.get("title") or title,
                    "metrics": metrics,
                    "analytics": analytics,
                }
            )
        return {
            "query_type": "comparison",
            "title": None,
            "results": results,
            "summary_metrics": {"titles": [title_a, title_b], "result_count": len(results)},
        }

    def get_title_theme_summary(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        themes, genres = _recommendation_terms(query)
        theme_match = _sql_list_contains_any("s.dominant_themes", themes)
        genre_match = _sql_list_contains_any("COALESCE(t.genres, t.imdb_genres, t.tvmaze_genres)", genres)
        where_clause = f"WHERE {theme_match} OR {genre_match}" if themes or genres else ""
        rows = self._query_rows(
            f"""
            SELECT
                t.title_name AS title,
                COALESCE(t.genres, t.imdb_genres, t.tvmaze_genres) AS genres,
                s.dominant_themes,
                s.chunk_count AS evidence_count,
                s.avg_sentiment_score,
                CASE
                    WHEN s.avg_sentiment_score >= 0.05 THEN 'positive'
                    WHEN s.avg_sentiment_score <= -0.05 THEN 'negative'
                    ELSE 'mixed'
                END AS emotional_tone,
                (
                    CASE WHEN {theme_match} THEN 2 ELSE 0 END
                    + CASE WHEN {genre_match} THEN 1 ELSE 0 END
                    + COALESCE(s.chunk_count, 0) / 1000.0
                ) AS recommendation_score
            FROM title_theme_summary s
            JOIN titles t USING (title_id)
            {where_clause}
            ORDER BY recommendation_score DESC, s.chunk_count DESC NULLS LAST, t.title_name
            LIMIT {int(limit)}
            """
        )
        return _compact_recommendation_rows(rows, limit=limit)

    def get_genre_theme_summary(self, query: str, *, limit: int = 6) -> list[dict[str, Any]]:
        themes, genres = _recommendation_terms(query)
        theme_match = _sql_list_contains_any("dominant_themes", themes)
        genre_checks = [f"genre ILIKE '%{_escape_like(genre)}%'" for genre in genres]
        genre_match = "(" + " OR ".join(genre_checks) + ")" if genre_checks else "FALSE"
        where_clause = f"WHERE {theme_match} OR {genre_match}" if themes or genres else ""
        rows = self._query_rows(
            f"""
            SELECT
                genre,
                title_count,
                dominant_themes,
                avg_sentiment_score,
                CASE
                    WHEN avg_sentiment_score >= 0.05 THEN 'positive'
                    WHEN avg_sentiment_score <= -0.05 THEN 'negative'
                    ELSE 'mixed'
                END AS emotional_tone,
                (
                    CASE WHEN {theme_match} THEN 2 ELSE 0 END
                    + CASE WHEN {genre_match} THEN 1 ELSE 0 END
                    + COALESCE(title_count, 0) / 1000.0
                ) AS recommendation_score
            FROM genre_theme_summary
            {where_clause}
            ORDER BY recommendation_score DESC, title_count DESC NULLS LAST, genre
            LIMIT {int(limit)}
            """
        )
        compact_rows: list[dict[str, Any]] = []
        for row in rows[:limit]:
            compact_rows.append(
                {
                    key: value
                    for key, value in {
                        "genre": row.get("genre"),
                        "title_count": _int_or_none(row.get("title_count")),
                        "dominant_themes": row.get("dominant_themes") or [],
                        "emotional_tone": row.get("emotional_tone") or _sentiment_tone(row.get("avg_sentiment_score")),
                        "avg_sentiment_score": _float_or_none(row.get("avg_sentiment_score")),
                    }.items()
                    if value not in (None, [], "")
                }
            )
        return compact_rows

    def get_audience_editorial_summary(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        themes, genres = _recommendation_terms(query)
        theme_match = _sql_list_contains_any("s.dominant_themes", themes)
        genre_match = _sql_list_contains_any("COALESCE(t.genres, t.imdb_genres, t.tvmaze_genres)", genres)
        where_clause = f"WHERE s.source_group IN ('audience', 'editorial') AND ({theme_match} OR {genre_match})"
        if not themes and not genres:
            where_clause = "WHERE s.source_group IN ('audience', 'editorial')"
        rows = self._query_rows(
            f"""
            SELECT
                t.title_name AS title,
                COALESCE(t.genres, t.imdb_genres, t.tvmaze_genres) AS genres,
                any_value(CASE WHEN s.source_group = 'audience' THEN s.dominant_themes END) AS audience_themes,
                any_value(CASE WHEN s.source_group = 'editorial' THEN s.dominant_themes END) AS editorial_themes,
                SUM(COALESCE(s.chunk_count, 0)) AS evidence_count,
                AVG(s.avg_sentiment_score) AS avg_sentiment_score,
                CASE
                    WHEN AVG(s.avg_sentiment_score) >= 0.05 THEN 'positive'
                    WHEN AVG(s.avg_sentiment_score) <= -0.05 THEN 'negative'
                    ELSE 'mixed'
                END AS emotional_tone,
                (
                    MAX(CASE WHEN {theme_match} THEN 2 ELSE 0 END)
                    + MAX(CASE WHEN {genre_match} THEN 1 ELSE 0 END)
                    + SUM(COALESCE(s.chunk_count, 0)) / 1000.0
                ) AS recommendation_score
            FROM audience_vs_editorial_summary s
            JOIN titles t USING (title_id)
            {where_clause}
            GROUP BY t.title_id, t.title_name, COALESCE(t.genres, t.imdb_genres, t.tvmaze_genres)
            ORDER BY recommendation_score DESC, evidence_count DESC NULLS LAST, t.title_name
            LIMIT {int(limit)}
            """
        )
        return _compact_recommendation_rows(rows, limit=limit)

    def extract_titles(self, query: str, *, limit: int = 4) -> list[str]:
        if not isinstance(query, str) or not query.strip():
            return []

        titles: list[str] = []
        for candidate in _comparison_title_candidates(query):
            resolved = self._resolve_title_candidate(candidate)
            titles.append(resolved or candidate)
            if len(_unique_titles(titles)) >= limit:
                break
        titles = _unique_titles(titles)
        if len(titles) >= 2:
            return titles[:limit]

        rows = self._query_rows(
            f"""
            SELECT DISTINCT title_name AS title
            FROM titles
            WHERE title_name IS NOT NULL
              AND strpos(lower({_sql_literal(query)}), lower(title_name)) > 0
            ORDER BY length(title_name) DESC, title_name
            LIMIT {int(limit)}
            """
        )
        exact_titles = _unique_titles([str(row.get("title")) for row in rows if row.get("title")])
        return _unique_titles(titles + exact_titles)[:limit]

    def _resolve_title_candidate(self, candidate: str) -> str | None:
        cleaned = " ".join(candidate.split()).strip(" .?!:;")
        if not cleaned:
            return None
        rows = self._query_rows(
            f"""
            SELECT title_name AS title
            FROM titles
            WHERE lower(title_name) = lower('{_escape_like(cleaned)}')
               OR lower(normalized_title) = lower('{_escape_like(cleaned)}')
               OR title_name ILIKE '%{_escape_like(cleaned)}%' ESCAPE '\\'
               OR normalized_title ILIKE '%{_escape_like(cleaned)}%' ESCAPE '\\'
            ORDER BY
                CASE
                    WHEN lower(title_name) = lower('{_escape_like(cleaned)}') THEN 0
                    WHEN lower(normalized_title) = lower('{_escape_like(cleaned)}') THEN 1
                    WHEN title_name ILIKE '{_escape_like(cleaned)}%' ESCAPE '\\' THEN 2
                    ELSE 3
                END,
                length(title_name),
                title_name
            LIMIT 1
            """
        )
        if not rows:
            return None
        title = rows[0].get("title")
        return str(title) if title else None

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
        return _cleanup_comparison_title_candidate(quoted[0]), _cleanup_comparison_title_candidate(quoted[1])

    match = re.search(r"\bcompare\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if match:
        return _cleanup_comparison_title_candidate(match.group(1)), _cleanup_comparison_title_candidate(match.group(2))
    match = re.search(r"\b(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if match:
        return _cleanup_comparison_title_candidate(match.group(1)), _cleanup_comparison_title_candidate(match.group(2))
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


def extract_titles(query: str) -> list[str]:
    client = MotherDuckClient()
    try:
        return client.extract_titles(query)
    finally:
        client.close()
