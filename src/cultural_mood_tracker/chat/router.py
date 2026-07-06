from __future__ import annotations

import re
import logging
from typing import TypedDict

from .schemas import ChatMode


LOGGER = logging.getLogger(__name__)


class RouterDebug(TypedDict):
    mode: ChatMode
    scores: dict[str, int]
    matched_signals: list[str]


RECOMMENDATION_SIGNALS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (5, ("what should i watch", "what movie should i watch", "what show should i watch")),
    (3, ("similar to", "if i liked", "looking for", "recommend", "recommendation", "suggest")),
)

RECOMMENDATION_SELECTION_SIGNALS = {
    "what should i watch",
    "what movie should i watch",
    "what show should i watch",
    "similar to",
    "if i liked",
    "looking for",
    "recommend",
    "recommendation",
    "suggest",
}

EMOTION_MODIFIER_SIGNALS = {
    "feel sad",
    "feeling sad",
    "feeling happy",
    "comforting",
    "uplifting",
    "heartwarming",
    "nostalgic",
    "emotional",
    "dark",
    "funny",
    "lighthearted",
}

STRUCTURAL_COMPARISON_SIGNALS = {
    "which is more",
    "which is better",
    "compare",
    "compared to",
    "comparison",
    "difference between",
    "vs",
    "versus",
    "more than",
    "better than",
    "outperform",
    "popular than",
    "received better",
    "audience reception",
    "critics vs audience",
}

HYBRID_SIGNALS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (5, ("vs", "versus", "compare", "compared to", "comparison", "difference between")),
    (4, ("which is more", "which is better", "stronger criticism", "more positively received", "audience reception", "critics vs audience")),
    (3, ("why is", "overperforming", "underperforming", "outperform", "popular than", "received better", "better than", "more than")),
)

FAST_SQL_SIGNALS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (5, ("rating", "ratings", "score", "how many", "cast", "crew", "director", "directors", "actor", "actors")),
    (4, ("top", "best", "highest", "count", "list", "attention")),
)

RAG_SIGNALS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (4, ("what is", "explain", "summary")),
    (3, ("review", "reviews")),
)


def _normalize_query(query: str) -> str:
    normalized = query.casefold()
    normalized = re.sub(r"[\u2018\u2019]", "'", normalized)
    normalized = re.sub(r"[\u201c\u201d]", '"', normalized)
    normalized = re.sub(r"[^a-z0-9'\"&]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _signal_pattern(signal: str) -> re.Pattern[str]:
    escaped = re.escape(signal.casefold()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def _matches_signal(query: str, signal: str) -> bool:
    return bool(_signal_pattern(signal).search(query))


def _contains_any(query: str, keywords: set[str]) -> bool:
    normalized = _normalize_query(query)
    return any(_matches_signal(normalized, keyword) for keyword in keywords)


def _apply_signals(
    *,
    query: str,
    mode: str,
    signals: tuple[tuple[int, tuple[str, ...]], ...],
    scores: dict[str, int],
    matched_signals: list[str],
) -> None:
    for weight, phrases in signals:
        for phrase in phrases:
            if _matches_signal(query, phrase):
                scores[mode] += weight
                matched_signals.append(f"{mode}:{phrase}+{weight}")


def _has_any_signal(query: str, signals: set[str]) -> bool:
    return any(_matches_signal(query, signal) for signal in signals)


def _matches_comparison_pattern(query: str) -> bool:
    patterns = (
        r"(?<![a-z0-9])\w[\w'\s]{1,80}\s+vs\s+\w[\w'\s]{1,80}(?![a-z0-9])",
        r"(?<![a-z0-9])\w[\w'\s]{1,80}\s+versus\s+\w[\w'\s]{1,80}(?![a-z0-9])",
        r"(?<![a-z0-9])compare\s+.+?\s+and\s+.+",
        r"(?<![a-z0-9])is\s+.+?\s+or\s+.+",
        r"(?<![a-z0-9])how\s+does\s+.+?\s+compare\s+.+",
        r"(?<![a-z0-9]).+?\s+more\s+\w+\s+than\s+.+",
        r"(?<![a-z0-9]).+?\s+better\s+than\s+.+",
    )
    return any(re.search(pattern, query) for pattern in patterns)


def _looks_like_two_entity_more_comparison(query: str) -> bool:
    if not (_matches_signal(query, "which is") and _matches_signal(query, "more")):
        return False
    if _matches_signal(query, "or") or _matches_signal(query, "vs") or _matches_signal(query, "versus"):
        return True
    return bool(re.search(r"which\s+is\s+more\s+\w+\s+\w[\w'\s]{1,80}\s+\w[\w'\s]{1,80}", query))


def route_query_debug(query: str) -> RouterDebug:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    normalized = _normalize_query(query)
    scores = {
        "recommendation": 0,
        "hybrid": 0,
        "fast_sql": 0,
        "rag": 0,
    }
    matched_signals: list[str] = []

    if (
        _has_any_signal(normalized, STRUCTURAL_COMPARISON_SIGNALS)
        or _matches_comparison_pattern(normalized)
        or _looks_like_two_entity_more_comparison(normalized)
    ):
        scores["hybrid"] += 5
        matched_signals.append("hybrid:structural_comparison+5")
        decision: RouterDebug = {
            "mode": "hybrid",
            "scores": scores,
            "matched_signals": matched_signals,
        }
        LOGGER.info("routing_decision=%s, matched_keywords=%s", decision["mode"], decision["matched_signals"])
        return decision

    _apply_signals(
        query=normalized,
        mode="recommendation",
        signals=RECOMMENDATION_SIGNALS,
        scores=scores,
        matched_signals=matched_signals,
    )
    _apply_signals(
        query=normalized,
        mode="hybrid",
        signals=HYBRID_SIGNALS,
        scores=scores,
        matched_signals=matched_signals,
    )
    _apply_signals(
        query=normalized,
        mode="fast_sql",
        signals=FAST_SQL_SIGNALS,
        scores=scores,
        matched_signals=matched_signals,
    )
    _apply_signals(
        query=normalized,
        mode="rag",
        signals=RAG_SIGNALS,
        scores=scores,
        matched_signals=matched_signals,
    )

    has_selection_intent = _has_any_signal(normalized, RECOMMENDATION_SELECTION_SIGNALS)
    if has_selection_intent:
        for signal in EMOTION_MODIFIER_SIGNALS:
            if _matches_signal(normalized, signal):
                scores["recommendation"] += 2
                matched_signals.append(f"recommendation:{signal}_modifier+2")

    if (
        scores["recommendation"] > 0
        and scores["recommendation"] >= scores["hybrid"]
        and scores["recommendation"] >= scores["fast_sql"]
    ):
        mode: ChatMode = "recommendation"
    elif scores["hybrid"] > 0 and scores["hybrid"] >= scores["fast_sql"]:
        mode = "hybrid"
    elif scores["fast_sql"] > 0:
        mode = "fast_sql"
    else:
        mode = "rag"

    decision = {
        "mode": mode,
        "scores": scores,
        "matched_signals": matched_signals,
    }
    LOGGER.info("routing_decision=%s, matched_keywords=%s", decision["mode"], decision["matched_signals"])
    return decision


def route_query(query: str) -> ChatMode:
    return route_query_debug(query)["mode"]
