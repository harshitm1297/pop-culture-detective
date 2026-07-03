from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cultural_mood_tracker.sources import slugify

from .common import (
    clean_text,
    count_normalized_phrase_occurrences,
    find_matching_file,
    flatten_text_for_storage,
    maybe_load_json,
    normalize_datetime,
    quality_flags_for_text,
    strip_source_boilerplate,
    strip_tmdb_review_boilerplate,
)


def _normalize_document_text(raw_text: str, *, source_name: str, document_type: str) -> str:
    strip_urls = source_name == "tmdb" and document_type == "user_review"
    cleaned = clean_text(raw_text, strip_urls=strip_urls)
    if source_name == "tmdb" and document_type == "user_review":
        cleaned = strip_tmdb_review_boilerplate(cleaned)
    if source_name in {"vulture", "indiewire", "slashfilm"}:
        cleaned = strip_source_boilerplate(cleaned, source_name=source_name)
    return flatten_text_for_storage(cleaned)


def _build_document(
    *,
    document_id: str,
    title_id: str,
    source_run_id: str,
    title_name: str,
    content_type: str,
    source_name: str,
    document_type: str,
    source_record_id: str | None,
    source_url: str | None,
    published_at: str | None,
    author: str | None,
    language: str | None,
    raw_text: str,
    min_length: int,
    source_match_method: str,
    source_match_confidence: float,
    extra_flags: list[str] | None = None,
) -> dict[str, Any]:
    cleaned = _normalize_document_text(
        raw_text,
        source_name=source_name,
        document_type=document_type,
    )
    flags = quality_flags_for_text(cleaned, min_length=min_length)
    if source_match_confidence < 0.8:
        flags.append("weak_source_match")
    if extra_flags:
        flags.extend(extra_flags)
    unique_flags = sorted(set(flags))
    blocking_flags = {"empty_text", "too_short", "possible_encoding_noise", "weak_source_match"}
    return {
        "document_id": document_id,
        "title_id": title_id,
        "source_run_id": source_run_id,
        "title_name": title_name,
        "content_type": content_type,
        "source_name": source_name,
        "document_type": document_type,
        "source_record_id": source_record_id,
        "source_url": source_url,
        "published_at": normalize_datetime(published_at),
        "author": author,
        "language": language or "en",
        "text": cleaned,
        "text_length": len(cleaned),
        "source_match_method": source_match_method,
        "source_match_confidence": source_match_confidence,
        "quality_flags": unique_flags,
        "is_usable_for_rag": not any(flag in blocking_flags for flag in unique_flags),
    }


def _guardian_match(article: dict[str, Any], title_name: str) -> tuple[bool, str, float]:
    fields = article.get("fields", {})
    headline = clean_text(fields.get("headline") or article.get("webTitle") or "")
    trail = clean_text(fields.get("trailText") or "")
    body = clean_text(fields.get("bodyText") or "")
    article_id = clean_text(article.get("id") or "")
    web_url = clean_text(article.get("webUrl") or "")
    phrase = title_name
    slug = slugify(title_name).replace("-", " ")

    headline_hits = count_normalized_phrase_occurrences(headline, phrase)
    trail_hits = count_normalized_phrase_occurrences(trail, phrase)
    body_hits = count_normalized_phrase_occurrences(body, phrase)
    slug_in_id = slug and slug in article_id.lower().replace("-", " ")
    slug_in_url = slug and slug in web_url.lower().replace("-", " ")

    if headline_hits > 0:
        return True, "guardian_headline_exact", 0.98
    if trail_hits > 0 and body_hits > 0:
        return True, "guardian_trail_and_body_exact", 0.9
    if (slug_in_id or slug_in_url) and body_hits > 0:
        return True, "guardian_slug_and_body_exact", 0.85
    if body_hits >= 2:
        return True, "guardian_body_repeated_exact", 0.8
    return False, "guardian_incidental_mention", 0.25


def _gdelt_match(article: dict[str, Any], title_name: str) -> tuple[bool, str, float]:
    headline = clean_text(article.get("title") or "")
    if count_normalized_phrase_occurrences(headline, title_name) > 0:
        return True, "gdelt_headline_exact", 0.85
    return False, "gdelt_no_headline_match", 0.2


def _critic_source_match(article: dict[str, Any]) -> tuple[str, float]:
    confidence = float(article.get("match_confidence") or 0.0)
    method = clean_text(article.get("match_method") or "critic_source_match")
    return method or "critic_source_match", confidence or 0.75


def build_documents(
    anchors: list[dict[str, Any]],
    tmdb_run_dir: Path,
    tvmaze_run_dir: Path,
    guardian_run_dir: Path,
    gdelt_run_dir: Path,
    wikidata_run_dir: Path,
    critic_source_dirs: dict[str, Path] | None = None,
    *,
    source_run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    critic_source_dirs = critic_source_dirs or {}

    for anchor in anchors:
        title_id = f"{anchor['content_type']}_{anchor['tmdb_id']}"
        content_type = anchor["content_type"]
        title_name = clean_text(anchor["title_name"])
        type_tmdb_dir = tmdb_run_dir / content_type

        details_path = find_matching_file(type_tmdb_dir, f"{anchor['tmdb_id']}_", "_details.json")
        reviews_path = find_matching_file(type_tmdb_dir, f"{anchor['tmdb_id']}_", "_reviews.json")
        details = maybe_load_json(details_path) if details_path else {}
        reviews = maybe_load_json(reviews_path) if reviews_path else {}

        overview = clean_text((details or {}).get("overview") or "")
        if overview:
            rows.append(
                _build_document(
                    document_id=f"{title_id}:tmdb_overview",
                    title_id=title_id,
                    source_run_id=source_run_id,
                    title_name=title_name,
                    content_type=content_type,
                    source_name="tmdb",
                    document_type="overview",
                    source_record_id=str(anchor["tmdb_id"]),
                    source_url=None,
                    published_at=anchor.get("release_date"),
                    author=None,
                    language=anchor.get("original_language"),
                    raw_text=overview,
                    min_length=80,
                    source_match_method="anchor_file_binding",
                    source_match_confidence=1.0,
                )
            )

        for review in (reviews or {}).get("results", []):
            rows.append(
                _build_document(
                    document_id=f"{title_id}:tmdb_review:{review.get('id')}",
                    title_id=title_id,
                    source_run_id=source_run_id,
                    title_name=title_name,
                    content_type=content_type,
                    source_name="tmdb",
                    document_type="user_review",
                    source_record_id=review.get("id"),
                    source_url=review.get("url"),
                    published_at=review.get("updated_at") or review.get("created_at"),
                    author=review.get("author"),
                    language=anchor.get("original_language"),
                    raw_text=review.get("content") or "",
                    min_length=120,
                    source_match_method="anchor_file_binding",
                    source_match_confidence=1.0,
                )
            )

        tvmaze_path = find_matching_file(tvmaze_run_dir / content_type, f"{anchor['tmdb_id']}_", ".json")
        tvmaze = maybe_load_json(tvmaze_path) if tvmaze_path else {}
        if isinstance(tvmaze, dict) and "error" not in tvmaze and tvmaze.get("summary"):
            rows.append(
                _build_document(
                    document_id=f"{title_id}:tvmaze_summary",
                    title_id=title_id,
                    source_run_id=source_run_id,
                    title_name=title_name,
                    content_type=content_type,
                    source_name="tvmaze",
                    document_type="show_summary",
                    source_record_id=str(tvmaze.get("id")),
                    source_url=tvmaze.get("url"),
                    published_at=None,
                    author=None,
                    language=tvmaze.get("language") or anchor.get("original_language"),
                    raw_text=tvmaze.get("summary") or "",
                    min_length=80,
                    source_match_method="imdb_to_tvmaze_lookup",
                    source_match_confidence=0.95,
                )
            )

        guardian_path = find_matching_file(guardian_run_dir / content_type, f"{anchor['tmdb_id']}_", ".json")
        guardian = maybe_load_json(guardian_path) if guardian_path else {}
        for article in (guardian or {}).get("response", {}).get("results", []):
            is_match, match_method, match_confidence = _guardian_match(article, title_name)
            if not is_match:
                continue
            fields = article.get("fields", {})
            body = "\n\n".join(
                part
                for part in [
                    fields.get("headline"),
                    fields.get("trailText"),
                    fields.get("bodyText"),
                ]
                if part
            )
            rows.append(
                _build_document(
                    document_id=f"{title_id}:guardian:{article.get('id')}",
                    title_id=title_id,
                    source_run_id=source_run_id,
                    title_name=title_name,
                    content_type=content_type,
                    source_name="guardian",
                    document_type="critic_article",
                    source_record_id=article.get("id"),
                    source_url=article.get("webUrl"),
                    published_at=article.get("webPublicationDate"),
                    author=fields.get("byline"),
                    language="en",
                    raw_text=body,
                    min_length=200,
                    source_match_method=match_method,
                    source_match_confidence=match_confidence,
                )
            )

        gdelt_path = find_matching_file(gdelt_run_dir / content_type, f"{anchor['tmdb_id']}_", ".json")
        gdelt = maybe_load_json(gdelt_path) if gdelt_path else {}
        for article in (gdelt or {}).get("articles", []):
            is_match, match_method, match_confidence = _gdelt_match(article, title_name)
            if not is_match:
                continue
            body = "\n\n".join(
                part for part in [article.get("title"), article.get("seendate"), article.get("domain")] if part
            )
            rows.append(
                _build_document(
                    document_id=f"{title_id}:gdelt:{article.get('url')}",
                    title_id=title_id,
                    source_run_id=source_run_id,
                    title_name=title_name,
                    content_type=content_type,
                    source_name="gdelt",
                    document_type="news_article",
                    source_record_id=article.get("url"),
                    source_url=article.get("url"),
                    published_at=article.get("seendate"),
                    author=None,
                    language=article.get("language") or "en",
                    raw_text=body,
                    min_length=80,
                    source_match_method=match_method,
                    source_match_confidence=match_confidence,
                )
            )

        wikidata_path = find_matching_file(wikidata_run_dir / content_type, f"{anchor['tmdb_id']}_", ".json")
        wikidata = maybe_load_json(wikidata_path) if wikidata_path else {}
        if isinstance(wikidata, dict):
            entity = wikidata.get("entities", {}).get(anchor.get("wikidata_id") or "", {})
            description = entity.get("descriptions", {}).get("en", {}).get("value")
            if description:
                rows.append(
                    _build_document(
                        document_id=f"{title_id}:wikidata_description",
                        title_id=title_id,
                        source_run_id=source_run_id,
                        title_name=title_name,
                        content_type=content_type,
                        source_name="wikidata",
                        document_type="entity_description",
                        source_record_id=anchor.get("wikidata_id"),
                        source_url=None,
                        published_at=entity.get("modified"),
                        author=None,
                        language="en",
                        raw_text=description,
                        min_length=40,
                        source_match_method="wikidata_entity_binding",
                        source_match_confidence=1.0,
                    )
                )

        for source_name, source_root in critic_source_dirs.items():
            payload_path = find_matching_file(source_root / content_type, f"{anchor['tmdb_id']}_", ".json")
            payload = maybe_load_json(payload_path) if payload_path else {}
            for article in (payload or {}).get("articles", []):
                match_method, match_confidence = _critic_source_match(article)
                rows.append(
                    _build_document(
                        document_id=(
                            f"{title_id}:{source_name}:"
                            f"{slugify(article.get('headline') or article.get('source_url') or 'article')}"
                        ),
                        title_id=title_id,
                        source_run_id=source_run_id,
                        title_name=title_name,
                        content_type=content_type,
                        source_name=source_name,
                        document_type=article.get("document_type") or "critic_review",
                        source_record_id=article.get("source_url"),
                        source_url=article.get("source_url"),
                        published_at=article.get("published_at"),
                        author=article.get("author"),
                        language="en",
                        raw_text=article.get("text") or "",
                        min_length=180,
                        source_match_method=match_method,
                        source_match_confidence=match_confidence,
                    )
                )

    return rows
