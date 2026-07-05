from __future__ import annotations

from .schemas import ChatMode


SQL_KEYWORDS = {
    "rating",
    "ratings",
    "score",
    "top",
    "best",
    "cast",
    "acted",
    "actor",
    "actors",
    "starring",
    "director",
    "directors",
}

TREND_KEYWORDS = {
    "trend",
    "trending",
    "attention",
    "popular",
    "popularity",
    "over time",
    "increase",
    "decrease",
}

RAG_KEYWORDS = {
    "about",
    "what is",
    "explain",
    "summary",
    "review",
    "reviews",
}

HYBRID_KEYWORDS = {
    "compare",
    "comparison",
    "difference",
    "vs",
    "versus",
    "why",
}


def _contains_any(query: str, keywords: set[str]) -> bool:
    return any(keyword in query for keyword in keywords)


def route_query(query: str) -> ChatMode:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    normalized = query.casefold()

    if _contains_any(normalized, HYBRID_KEYWORDS):
        return "hybrid"
    if _contains_any(normalized, TREND_KEYWORDS):
        return "hybrid"
    if _contains_any(normalized, SQL_KEYWORDS):
        return "sql"
    if _contains_any(normalized, RAG_KEYWORDS):
        return "rag"
    return "rag"
