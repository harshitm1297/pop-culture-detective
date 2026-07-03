from __future__ import annotations

import csv
import gzip
import os
import re
import stat
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from cultural_mood_tracker.sources import (
    detect_document_type,
    ensure_dir,
    fetch_article_text,
    fetch_candidate_entries,
    list_sources,
    save_json,
    slugify,
)
from cultural_mood_tracker.sources import gdelt as gdelt_source
from cultural_mood_tracker.sources import guardian as guardian_source
from cultural_mood_tracker.sources import imdb as imdb_source
from cultural_mood_tracker.sources import tmdb as tmdb_source
from cultural_mood_tracker.sources import tvmaze as tvmaze_source
from cultural_mood_tracker.sources import wikidata as wikidata_source
from cultural_mood_tracker.sources import wikipedia as wikipedia_source
from cultural_mood_tracker.sources.wikidata import extract_enwiki_title


def _remove_readonly(
    func: Callable[..., Any],
    path: str,
    exc_info: tuple[type[BaseException], BaseException, object],
) -> None:
    _ = exc_info
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clean_previous_outputs(raw_root: Path) -> None:
    critic_sources = [source.name for source in list_sources()]
    for name in ["tmdb_smoke", "anchors", "tmdb", "imdb", "tvmaze", "wikidata", "wikipedia", "guardian", "gdelt", *critic_sources]:
        target = raw_root / name
        if target.exists():
            shutil.rmtree(target, onerror=_remove_readonly)
            print(f"[cleanup] removed {target}")


def build_anchor_entries(
    api_key: str,
    language: str,
    start_date: str,
    end_date: str,
    movie_count: int,
    tv_count: int,
    anchor_dir: Path,
    tmdb_dir: Path,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    counts = {"movie": movie_count, "tv": tv_count}

    for content_type in ["movie", "tv"]:
        discover_results = tmdb_source.discover_titles(
            api_key=api_key,
            content_type=content_type,
            language=language,
            start_date=start_date,
            end_date=end_date,
            sample_size=counts[content_type],
        )
        type_anchor_dir = anchor_dir / content_type
        type_tmdb_dir = tmdb_dir / content_type
        ensure_dir(type_anchor_dir)
        ensure_dir(type_tmdb_dir)
        save_json(type_tmdb_dir / "discover.json", {"results": discover_results})

        for index, item in enumerate(discover_results, start=1):
            tmdb_id = int(item["id"])
            title_name = item.get("title") or item.get("name") or f"{content_type}-{tmdb_id}"
            print(f"[tmdb:{content_type}] fetching anchor {index}/{len(discover_results)}: {title_name}")
            details = tmdb_source.fetch_details(api_key, content_type, tmdb_id, language)
            reviews = tmdb_source.fetch_reviews(api_key, content_type, tmdb_id, language)
            safe_title = slugify(title_name)
            save_json(type_tmdb_dir / f"{tmdb_id}_{safe_title}_details.json", details)
            save_json(type_tmdb_dir / f"{tmdb_id}_{safe_title}_reviews.json", reviews)

            release_date = details.get("release_date") or details.get("first_air_date") or ""
            release_year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else None
            imdb_id = details.get("imdb_id") or details.get("external_ids", {}).get("imdb_id")
            wikidata_id = details.get("external_ids", {}).get("wikidata_id")

            anchors.append(
                {
                    "content_type": content_type,
                    "tmdb_id": tmdb_id,
                    "title_name": title_name,
                    "original_title_name": details.get("original_title") or details.get("original_name"),
                    "release_date": release_date,
                    "release_year": release_year,
                    "imdb_id": imdb_id or None,
                    "wikidata_id": wikidata_id or None,
                    "tmdb_popularity": item.get("popularity"),
                    "tmdb_vote_average": item.get("vote_average"),
                    "tmdb_vote_count": item.get("vote_count"),
                    "tmdb_review_count": len(reviews.get("results", [])),
                    "spoken_languages": details.get("spoken_languages", []),
                    "origin_country": details.get("origin_country", []),
                    "original_language": details.get("original_language"),
                }
            )

    save_json(anchor_dir / "anchor_titles.json", anchors)
    return anchors


def fetch_wikipedia_pageviews(
    anchor: dict[str, Any],
    wikipedia_dir: Path,
    wikidata_dir: Path,
    start_date: str,
    end_date: str,
) -> None:
    wikidata_payload = None
    if anchor.get("wikidata_id"):
        wikidata_path = wikidata_dir / anchor["content_type"] / f"{anchor['tmdb_id']}_{slugify(anchor['title_name'])}.json"
        wikidata_payload = maybe_load_json(wikidata_path) if wikidata_path.exists() else None
    article_title = None
    if isinstance(wikidata_payload, dict):
        article_title = extract_enwiki_title(wikidata_payload, anchor.get("wikidata_id"))
    try:
        payload = wikipedia_source.fetch_pageviews(
            anchor["title_name"],
            start_date,
            end_date,
            article_title=article_title,
        )
    except RuntimeError as exc:
        payload = {
            "error": str(exc),
            "title_name": anchor["title_name"],
            "tmdb_id": anchor["tmdb_id"],
            "wikipedia_article_title": article_title,
        }
    type_dir = wikipedia_dir / anchor["content_type"]
    ensure_dir(type_dir)
    save_json(type_dir / f"{anchor['tmdb_id']}_{slugify(anchor['title_name'])}.json", payload)


def fetch_wikidata_entity(anchor: dict[str, Any], wikidata_dir: Path) -> None:
    if not anchor.get("wikidata_id"):
        return
    try:
        payload = wikidata_source.fetch_entity(anchor["wikidata_id"])
    except RuntimeError as exc:
        payload = {"error": str(exc), "wikidata_id": anchor["wikidata_id"], "tmdb_id": anchor["tmdb_id"]}
    type_dir = wikidata_dir / anchor["content_type"]
    ensure_dir(type_dir)
    save_json(type_dir / f"{anchor['tmdb_id']}_{slugify(anchor['title_name'])}.json", payload)


def fetch_tvmaze(anchor: dict[str, Any], tvmaze_dir: Path) -> None:
    if anchor["content_type"] != "tv" or not anchor.get("imdb_id"):
        return
    try:
        payload = tvmaze_source.fetch_show(anchor["imdb_id"])
    except RuntimeError as exc:
        payload = {"error": str(exc), "imdb_id": anchor["imdb_id"], "tmdb_id": anchor["tmdb_id"]}
    type_dir = tvmaze_dir / "tv"
    ensure_dir(type_dir)
    save_json(type_dir / f"{anchor['tmdb_id']}_{slugify(anchor['title_name'])}.json", payload)


def fetch_guardian(
    anchor: dict[str, Any],
    guardian_dir: Path,
    api_key: str,
    start_date: str,
    end_date: str,
    page_size: int,
) -> None:
    try:
        payload = guardian_source.fetch_articles(
            anchor["title_name"],
            anchor["content_type"],
            api_key,
            start_date,
            end_date,
            page_size,
        )
    except RuntimeError as exc:
        payload = {"error": str(exc), "title_name": anchor["title_name"], "tmdb_id": anchor["tmdb_id"]}
    type_dir = guardian_dir / anchor["content_type"]
    ensure_dir(type_dir)
    save_json(type_dir / f"{anchor['tmdb_id']}_{slugify(anchor['title_name'])}.json", payload)


def fetch_gdelt(anchor: dict[str, Any], gdelt_dir: Path, max_records: int) -> None:
    try:
        payload = gdelt_source.fetch_articles(anchor["title_name"], max_records)
    except RuntimeError as exc:
        payload = {"error": str(exc), "title_name": anchor["title_name"], "tmdb_id": anchor["tmdb_id"]}
    type_dir = gdelt_dir / anchor["content_type"]
    ensure_dir(type_dir)
    save_json(type_dir / f"{anchor['tmdb_id']}_{slugify(anchor['title_name'])}.json", payload)


def call_source(source_name: str, fn: Callable[[], None], anchor: dict[str, Any]) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[{source_name}] failed for {anchor['content_type']} {anchor['title_name']} "
            f"(TMDB {anchor['tmdb_id']}): {exc}"
        )


def maybe_load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _count_title_hits(text: str, title_name: str) -> int:
    normalized_text = _normalize_phrase(text)
    normalized_title = _normalize_phrase(title_name)
    if not normalized_text or not normalized_title:
        return 0
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_title)}(?![a-z0-9])"
    return len(re.findall(pattern, normalized_text))


def _within_window(published_at: str | None, start_date: str, end_date: str) -> bool:
    if not published_at:
        return True
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return start_date <= dt.date().isoformat() <= end_date


def _build_article_payload(
    *,
    source_name: str,
    content_type: str,
    title_name: str,
    entry: dict[str, Any],
    article_text: str,
    match_method: str,
    match_confidence: float,
) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "content_type": content_type,
        "title_name": title_name,
        "headline": entry.get("title"),
        "source_url": entry.get("link"),
        "author": entry.get("author"),
        "published_at": entry.get("published_at"),
        "description": entry.get("description"),
        "document_type": detect_document_type(
            content_type=content_type,
            headline=entry.get("title") or "",
            url=entry.get("link") or "",
        ),
        "match_method": match_method,
        "match_confidence": match_confidence,
        "text": article_text,
    }


def fetch_critic_blog_sources(
    anchors: list[dict[str, Any]],
    *,
    raw_root: Path,
    run_id: str,
    start_date: str,
    end_date: str,
    entry_limit: int,
) -> None:
    article_cache: dict[str, str] = {}

    for source in list_sources():
        source_root = raw_root / source.name / run_id
        ensure_dir(source_root)
        try:
            entries = fetch_candidate_entries(
                source,
                start_date=start_date,
                end_date=end_date,
                entry_limit=entry_limit,
            )
        except RuntimeError as exc:
            payload = {"error": str(exc), "source_url": source.url, "strategy": source.strategy}
            for content_type in ("movie", "tv"):
                type_dir = source_root / content_type
                ensure_dir(type_dir)
                save_json(type_dir / "_feed_error.json", payload)
            print(f"[{source.name}] source fetch failed: {exc}")
            continue

        for anchor in anchors:
            type_dir = source_root / anchor["content_type"]
            ensure_dir(type_dir)
            matched_articles: list[dict[str, Any]] = []

            for entry in entries:
                if not _within_window(entry.get("published_at"), start_date, end_date):
                    continue
                title_hits = _count_title_hits(
                    " ".join(
                        part for part in [entry.get("title") or "", entry.get("description") or ""] if part
                    ),
                    anchor["title_name"],
                )
                if title_hits <= 0:
                    continue

                link = entry.get("link") or ""
                if not link:
                    continue
                if link not in article_cache:
                    try:
                        article_cache[link] = fetch_article_text(link)
                    except RuntimeError:
                        article_cache[link] = ""
                article_text = article_cache[link]
                body_hits = _count_title_hits(article_text, anchor["title_name"])
                if title_hits > 0 and body_hits > 0:
                    match_method = f"{source.name}_headline_and_body_exact"
                    match_confidence = 0.95
                elif title_hits >= 2:
                    match_method = f"{source.name}_headline_repeated_exact"
                    match_confidence = 0.85
                else:
                    match_method = f"{source.name}_headline_only_match"
                    match_confidence = 0.65
                matched_articles.append(
                    _build_article_payload(
                        source_name=source.name,
                        content_type=anchor["content_type"],
                        title_name=anchor["title_name"],
                        entry=entry,
                        article_text=article_text,
                        match_method=match_method,
                        match_confidence=match_confidence,
                    )
                )

            save_json(
                type_dir / f"{anchor['tmdb_id']}_{slugify(anchor['title_name'])}.json",
                {
                    "source_name": source.name,
                    "source_url": source.url,
                    "strategy": source.strategy,
                    "start_date": start_date,
                    "end_date": end_date,
                    "feed_entry_count": len(entries),
                    "articles": matched_articles,
                },
            )


def download_imdb_and_filter(anchors: list[dict[str, Any]], imdb_dir: Path) -> None:
    ensure_dir(imdb_dir)
    imdb_ids = {anchor["imdb_id"] for anchor in anchors if anchor.get("imdb_id")}

    (imdb_dir / "title.basics.tsv.gz").write_bytes(imdb_source.download_basics())
    (imdb_dir / "title.ratings.tsv.gz").write_bytes(imdb_source.download_ratings())
    (imdb_dir / "title.crew.tsv.gz").write_bytes(imdb_source.download_crew())
    (imdb_dir / "title.principals.tsv.gz").write_bytes(imdb_source.download_principals())
    (imdb_dir / "title.episode.tsv.gz").write_bytes(imdb_source.download_episode())

    matched_basics_path = imdb_dir / "matched_title_basics.tsv"
    matched_ratings_path = imdb_dir / "matched_title_ratings.tsv"
    matched_crew_path = imdb_dir / "matched_title_crew.tsv"
    matched_principals_path = imdb_dir / "matched_title_principals.tsv"
    matched_episode_path = imdb_dir / "matched_title_episode.tsv"

    with gzip.open(imdb_dir / "title.basics.tsv.gz", "rt", encoding="utf-8") as fh, matched_basics_path.open("w", encoding="utf-8", newline="") as out_fh:
        reader = csv.DictReader(fh, delimiter="\t")
        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("tconst") in imdb_ids:
                writer.writerow(row)

    with gzip.open(imdb_dir / "title.ratings.tsv.gz", "rt", encoding="utf-8") as fh, matched_ratings_path.open("w", encoding="utf-8", newline="") as out_fh:
        reader = csv.DictReader(fh, delimiter="\t")
        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("tconst") in imdb_ids:
                writer.writerow(row)

    crew_name_ids: set[str] = set()
    with gzip.open(imdb_dir / "title.crew.tsv.gz", "rt", encoding="utf-8") as fh, matched_crew_path.open("w", encoding="utf-8", newline="") as out_fh:
        reader = csv.DictReader(fh, delimiter="\t")
        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("tconst") not in imdb_ids:
                continue
            writer.writerow(row)
            for field in ("directors", "writers"):
                for raw_value in (row.get(field) or "").split(","):
                    if raw_value and raw_value != "\\N":
                        crew_name_ids.add(raw_value)

    principal_name_ids: set[str] = set()
    with gzip.open(imdb_dir / "title.principals.tsv.gz", "rt", encoding="utf-8") as fh, matched_principals_path.open("w", encoding="utf-8", newline="") as out_fh:
        reader = csv.DictReader(fh, delimiter="\t")
        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("tconst") not in imdb_ids:
                continue
            writer.writerow(row)
            if row.get("nconst") and row.get("nconst") != "\\N":
                principal_name_ids.add(row["nconst"])

    with gzip.open(imdb_dir / "title.episode.tsv.gz", "rt", encoding="utf-8") as fh, matched_episode_path.open("w", encoding="utf-8", newline="") as out_fh:
        reader = csv.DictReader(fh, delimiter="\t")
        writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("parentTconst") in imdb_ids or row.get("tconst") in imdb_ids:
                writer.writerow(row)

    all_name_ids = sorted(crew_name_ids | principal_name_ids)
    if all_name_ids:
        (imdb_dir / "name.basics.tsv.gz").write_bytes(imdb_source.download_name_basics())
        matched_names_path = imdb_dir / "matched_name_basics.tsv"
        with gzip.open(imdb_dir / "name.basics.tsv.gz", "rt", encoding="utf-8") as fh, matched_names_path.open("w", encoding="utf-8", newline="") as out_fh:
            reader = csv.DictReader(fh, delimiter="\t")
            writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            keep = set(all_name_ids)
            for row in reader:
                if row.get("nconst") in keep:
                    writer.writerow(row)
