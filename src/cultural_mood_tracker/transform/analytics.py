from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any


POSITIVE_WORDS = {
    "excellent",
    "great",
    "good",
    "amazing",
    "compelling",
    "smart",
    "funny",
    "moving",
    "beautiful",
    "inventive",
    "thrilling",
    "strong",
    "favorite",
    "impressive",
    "rich",
}
NEGATIVE_WORDS = {
    "bad",
    "poor",
    "weak",
    "boring",
    "dull",
    "mess",
    "awful",
    "flat",
    "confusing",
    "disappointing",
    "forgettable",
    "generic",
    "tedious",
    "worst",
    "empty",
}

THEME_KEYWORDS: dict[str, set[str]] = {
    "nostalgia": {"nostalgia", "nostalgic", "memory", "past", "retro", "throwback"},
    "escapism": {"escape", "escapist", "fantasy", "dream", "adventure", "immersive"},
    "anxiety": {"anxiety", "panic", "tense", "stress", "fear", "paranoia"},
    "loneliness": {"lonely", "alone", "isolated", "solitude", "abandonment"},
    "comfort": {"comfort", "cozy", "warm", "gentle", "hopeful", "healing"},
    "identity": {"identity", "self", "belonging", "becoming", "who am i", "authentic"},
}


def _source_group(source_name: str, document_type: str) -> str:
    if source_name == "tmdb" and document_type == "user_review":
        return "audience"
    if source_name in {"guardian", "rogerebert", "indiewire", "vulture", "slant", "slashfilm"}:
        return "editorial"
    if document_type in {"critic_review", "editorial_analysis", "tv_recap"}:
        return "editorial"
    return "reference"


def _sentiment_score(text: str) -> float:
    words = [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split()]
    if not words:
        return 0.0
    pos = sum(1 for word in words if word in POSITIVE_WORDS)
    neg = sum(1 for word in words if word in NEGATIVE_WORDS)
    score = (pos - neg) / max(len(words), 1) * 12
    return max(-1.0, min(1.0, round(score, 4)))


def _sentiment_label(score: float) -> str:
    if score >= 0.08:
        return "positive"
    if score <= -0.08:
        return "negative"
    return "neutral"


def _theme_labels(text: str) -> list[str]:
    normalized = text.lower()
    labels = [
        label
        for label, keywords in THEME_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return sorted(labels)


def annotate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("chunk_text") or ""
        sentiment = _sentiment_score(text)
        labels = _theme_labels(text)
        rows.append(
            {
                "annotation_id": f"{chunk['chunk_id']}:heuristic_v1",
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title_id": chunk["title_id"],
                "source_run_id": chunk["source_run_id"],
                "source_name": chunk.get("source_name"),
                "document_type": chunk.get("document_type"),
                "published_at": chunk.get("published_at"),
                "source_group": _source_group(
                    chunk.get("source_name") or "",
                    chunk.get("document_type") or "",
                ),
                "sentiment_score": sentiment,
                "sentiment_label": _sentiment_label(sentiment),
                "theme_labels": labels,
                "theme_count": len(labels),
            }
        )
    return rows


def build_title_theme_summary(
    titles: list[dict[str, Any]],
    chunk_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations_by_title: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_annotations:
        annotations_by_title[row["title_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for title in titles:
        title_id = title["title_id"]
        annotations = annotations_by_title.get(title_id, [])
        theme_counter = Counter()
        for row in annotations:
            theme_counter.update(row.get("theme_labels", []))
        avg_sentiment = mean([row["sentiment_score"] for row in annotations]) if annotations else 0.0
        rows.append(
            {
                "title_id": title_id,
                "source_run_id": title["source_run_id"],
                "dominant_themes": [label for label, _ in theme_counter.most_common(3)],
                "theme_counts": dict(theme_counter),
                "chunk_count": len(annotations),
                "avg_sentiment_score": round(avg_sentiment, 4),
                "positive_chunk_count": sum(1 for row in annotations if row["sentiment_label"] == "positive"),
                "negative_chunk_count": sum(1 for row in annotations if row["sentiment_label"] == "negative"),
            }
        )
    return rows


def build_genre_theme_summary(
    titles: list[dict[str, Any]],
    title_theme_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    title_lookup = {row["title_id"]: row for row in titles}
    buckets: dict[str, dict[str, Any]] = {}
    for summary in title_theme_summary:
        title = title_lookup.get(summary["title_id"], {})
        for genre in title.get("genres", []):
            bucket = buckets.setdefault(
                genre,
                {
                    "genre": genre,
                    "source_run_id": summary["source_run_id"],
                    "title_count": 0,
                    "theme_counter": Counter(),
                    "sentiments": [],
                },
            )
            bucket["title_count"] += 1
            bucket["theme_counter"].update(summary.get("theme_counts", {}))
            bucket["sentiments"].append(summary.get("avg_sentiment_score") or 0.0)

    rows: list[dict[str, Any]] = []
    for genre, bucket in buckets.items():
        rows.append(
            {
                "genre": genre,
                "source_run_id": bucket["source_run_id"],
                "title_count": bucket["title_count"],
                "dominant_themes": [label for label, _ in bucket["theme_counter"].most_common(3)],
                "theme_counts": dict(bucket["theme_counter"]),
                "avg_sentiment_score": round(mean(bucket["sentiments"]), 4) if bucket["sentiments"] else 0.0,
            }
        )
    return rows


def build_monthly_theme_trends(
    titles: list[dict[str, Any]],
    chunk_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    title_lookup = {row["title_id"]: row for row in titles}
    buckets: dict[tuple[str, str], dict[str, Any]] = {}

    for row in chunk_annotations:
        title = title_lookup.get(row["title_id"], {})
        published_at = row.get("published_at") if "published_at" in row else None
        month = None
        if isinstance(published_at, str) and len(published_at) >= 7:
            month = published_at[:7]
        elif title.get("release_date"):
            month = str(title["release_date"])[:7]
        elif title.get("release_year"):
            month = f"{title['release_year']}-01"
        else:
            month = "unknown"

        key = (month, row["title_id"])
        bucket = buckets.setdefault(
            key,
            {
                "month": month,
                "title_id": row["title_id"],
                "source_run_id": row["source_run_id"],
                "theme_counter": Counter(),
                "sentiments": [],
            },
        )
        bucket["theme_counter"].update(row.get("theme_labels", []))
        bucket["sentiments"].append(row["sentiment_score"])

    rows: list[dict[str, Any]] = []
    for (_, _), bucket in buckets.items():
        rows.append(
            {
                "month": bucket["month"],
                "title_id": bucket["title_id"],
                "source_run_id": bucket["source_run_id"],
                "theme_counts": dict(bucket["theme_counter"]),
                "dominant_themes": [label for label, _ in bucket["theme_counter"].most_common(3)],
                "avg_sentiment_score": round(mean(bucket["sentiments"]), 4) if bucket["sentiments"] else 0.0,
            }
        )
    return rows


def build_audience_vs_editorial_summary(
    chunk_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in chunk_annotations:
        key = (row["title_id"], row["source_group"])
        bucket = buckets.setdefault(
            key,
            {
                "title_id": row["title_id"],
                "source_group": row["source_group"],
                "source_run_id": row["source_run_id"],
                "theme_counter": Counter(),
                "sentiments": [],
                "chunk_count": 0,
            },
        )
        bucket["chunk_count"] += 1
        bucket["theme_counter"].update(row.get("theme_labels", []))
        bucket["sentiments"].append(row["sentiment_score"])

    rows: list[dict[str, Any]] = []
    for (_, _), bucket in buckets.items():
        rows.append(
            {
                "title_id": bucket["title_id"],
                "source_group": bucket["source_group"],
                "source_run_id": bucket["source_run_id"],
                "chunk_count": bucket["chunk_count"],
                "dominant_themes": [label for label, _ in bucket["theme_counter"].most_common(3)],
                "theme_counts": dict(bucket["theme_counter"]),
                "avg_sentiment_score": round(mean(bucket["sentiments"]), 4) if bucket["sentiments"] else 0.0,
            }
        )
    return rows


def build_attention_vs_reception(
    titles: list[dict[str, Any]],
    ratings: list[dict[str, Any]],
    attention_signals: list[dict[str, Any]],
    title_theme_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ratings_by_title: defaultdict[str, list[float]] = defaultdict(list)
    for row in ratings:
        try:
            value = float(row.get("rating_value"))
        except (TypeError, ValueError):
            continue
        ratings_by_title[row["title_id"]].append(value)

    attention_by_title: defaultdict[str, list[float]] = defaultdict(list)
    for row in attention_signals:
        try:
            value = float(row.get("signal_value"))
        except (TypeError, ValueError):
            continue
        attention_by_title[row["title_id"]].append(value)

    summary_lookup = {row["title_id"]: row for row in title_theme_summary}
    rows: list[dict[str, Any]] = []
    for title in titles:
        title_id = title["title_id"]
        rating_values = ratings_by_title.get(title_id, [])
        attention_values = attention_by_title.get(title_id, [])
        summary = summary_lookup.get(title_id, {})
        rows.append(
            {
                "title_id": title_id,
                "source_run_id": title["source_run_id"],
                "avg_rating_value": round(mean(rating_values), 4) if rating_values else None,
                "attention_total": round(sum(attention_values), 4) if attention_values else 0.0,
                "attention_peak": round(max(attention_values), 4) if attention_values else 0.0,
                "avg_sentiment_score": summary.get("avg_sentiment_score"),
                "dominant_themes": summary.get("dominant_themes", []),
                "chunk_count": summary.get("chunk_count", 0),
            }
        )
    return rows
