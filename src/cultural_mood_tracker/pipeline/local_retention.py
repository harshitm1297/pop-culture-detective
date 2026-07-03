from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.transform.common import load_json, write_csv, write_json, write_jsonl


PER_TITLE_SOURCES = (
    "guardian",
    "gdelt",
    "tvmaze",
    "wikidata",
    "wikipedia",
    "rogerebert",
    "indiewire",
    "vulture",
    "slant",
    "slashfilm",
)
TABLES_BY_TITLE = (
    "titles",
    "documents",
    "document_chunks",
    "chunk_annotations",
    "ratings",
    "attention_signals",
    "title_cast",
    "title_crew",
    "episodes",
    "title_videos",
    "title_theme_summary",
    "monthly_theme_trends",
    "audience_vs_editorial_summary",
    "attention_vs_reception",
)


def _title_id(anchor: dict[str, Any]) -> str:
    return f"{anchor['content_type']}_{anchor['tmdb_id']}"


def _select_anchor_sample(
    anchors: list[dict[str, Any]],
    *,
    movie_count: int,
    tv_count: int,
) -> list[dict[str, Any]]:
    kept_movies = 0
    kept_tv = 0
    sampled: list[dict[str, Any]] = []

    for anchor in anchors:
        if anchor.get("content_type") == "movie" and kept_movies < movie_count:
            sampled.append(anchor)
            kept_movies += 1
            continue
        if anchor.get("content_type") == "tv" and kept_tv < tv_count:
            sampled.append(anchor)
            kept_tv += 1

    return sampled


def _delete_non_matching_files(directory: Path, kept_tmdb_ids: set[int], *, keep_discover: bool = False) -> int:
    if not directory.exists():
        return 0

    deleted = 0
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if keep_discover and path.name == "discover.json":
            continue
        prefix = path.name.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        if int(prefix) in kept_tmdb_ids:
            continue
        path.unlink()
        deleted += 1
    return deleted


def _retain_tmdb_run(tmdb_run_dir: Path, sampled_anchors: list[dict[str, Any]]) -> dict[str, int]:
    kept_by_type = {
        "movie": {int(anchor["tmdb_id"]) for anchor in sampled_anchors if anchor["content_type"] == "movie"},
        "tv": {int(anchor["tmdb_id"]) for anchor in sampled_anchors if anchor["content_type"] == "tv"},
    }
    stats = {"movie_files_deleted": 0, "tv_files_deleted": 0}

    for content_type, kept_ids in kept_by_type.items():
        type_dir = tmdb_run_dir / content_type
        if not type_dir.exists():
            continue
        discover_path = type_dir / "discover.json"
        if discover_path.exists():
            payload = load_json(discover_path)
            payload["results"] = [
                row for row in payload.get("results", []) if int(row.get("id", -1)) in kept_ids
            ]
            write_json(discover_path, payload)
        deleted = _delete_non_matching_files(type_dir, kept_ids, keep_discover=True)
        stats[f"{content_type}_files_deleted"] = deleted

    return stats


def _retain_per_title_source_run(
    run_dir: Path,
    sampled_anchors: list[dict[str, Any]],
) -> dict[str, int]:
    kept_by_type = {
        "movie": {int(anchor["tmdb_id"]) for anchor in sampled_anchors if anchor["content_type"] == "movie"},
        "tv": {int(anchor["tmdb_id"]) for anchor in sampled_anchors if anchor["content_type"] == "tv"},
    }
    stats = {"movie_files_deleted": 0, "tv_files_deleted": 0}

    for content_type, kept_ids in kept_by_type.items():
        deleted = _delete_non_matching_files(run_dir / content_type, kept_ids)
        stats[f"{content_type}_files_deleted"] = deleted

    return stats


def _filter_tsv_by_ids(path: Path, kept_ids: set[str]) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return 0
        rows = [row for row in reader if row.get("tconst") in kept_ids]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows)


def _retain_imdb_run(imdb_run_dir: Path, sampled_anchors: list[dict[str, Any]]) -> dict[str, int]:
    kept_imdb_ids = {
        anchor["imdb_id"]
        for anchor in sampled_anchors
        if isinstance(anchor.get("imdb_id"), str) and anchor["imdb_id"].strip()
    }

    matched_basics_path = imdb_run_dir / "matched_title_basics.tsv"
    matched_ratings_path = imdb_run_dir / "matched_title_ratings.tsv"
    kept_basics = _filter_tsv_by_ids(matched_basics_path, kept_imdb_ids)
    kept_ratings = _filter_tsv_by_ids(matched_ratings_path, kept_imdb_ids)

    deleted = 0
    for name in ("title.basics.tsv.gz", "title.ratings.tsv.gz"):
        path = imdb_run_dir / name
        if path.exists():
            path.unlink()
            deleted += 1

    return {
        "matched_title_basics_rows": kept_basics,
        "matched_title_ratings_rows": kept_ratings,
        "bulk_files_deleted": deleted,
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _retain_processed_run(processed_dir: Path, kept_title_ids: set[str]) -> dict[str, Any]:
    kept_document_ids: set[str] = set()
    kept_person_ids: set[str] = set()
    table_counts: dict[str, int] = {}

    for table_name in TABLES_BY_TITLE:
        path = processed_dir / f"{table_name}.jsonl"
        if not path.exists():
            continue
        rows = _load_jsonl_rows(path)
        filtered = [row for row in rows if row.get("title_id") in kept_title_ids]
        if table_name == "documents":
            kept_document_ids = {row["document_id"] for row in filtered if row.get("document_id")}
        if table_name in {"title_cast", "title_crew"}:
            kept_person_ids.update(
                row["person_id"] for row in filtered if isinstance(row.get("person_id"), str) and row["person_id"]
            )
        write_jsonl(path, filtered)
        csv_path = processed_dir / f"{table_name}.csv"
        if csv_path.exists():
            write_csv(csv_path, filtered)
        table_counts[table_name] = len(filtered)

    chunks_path = processed_dir / "document_chunks.jsonl"
    if chunks_path.exists():
        rows = _load_jsonl_rows(chunks_path)
        filtered = [row for row in rows if row.get("document_id") in kept_document_ids]
        write_jsonl(chunks_path, filtered)
        csv_path = processed_dir / "document_chunks.csv"
        if csv_path.exists():
            write_csv(csv_path, filtered)
        table_counts["document_chunks"] = len(filtered)

    people_path = processed_dir / "people.jsonl"
    if people_path.exists():
        rows = _load_jsonl_rows(people_path)
        filtered = [row for row in rows if row.get("person_id") in kept_person_ids]
        write_jsonl(people_path, filtered)
        csv_path = processed_dir / "people.csv"
        if csv_path.exists():
            write_csv(csv_path, filtered)
        table_counts["people"] = len(filtered)

    manifest_path = processed_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        manifest["local_retention"] = {
            "applied": True,
            "kept_title_count": len(kept_title_ids),
            "table_row_counts": table_counts,
        }
        write_json(manifest_path, manifest)

    return {
        "kept_title_count": len(kept_title_ids),
        "table_row_counts": table_counts,
    }


def apply_local_retention(
    *,
    project_root: Path,
    source_run_id: str,
    process_run_id: str,
    movie_count: int,
    tv_count: int,
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    anchor_dir = paths.raw_root / "anchors" / source_run_id
    tmdb_run_dir = paths.raw_root / "tmdb" / source_run_id
    imdb_run_dir = paths.raw_root / "imdb" / source_run_id
    processed_dir = paths.processed_root / process_run_id
    reports_dir = paths.reports_root / process_run_id

    anchors_path = anchor_dir / "anchor_titles.json"
    if not anchors_path.exists():
        raise RuntimeError(f"Missing anchor file for retention: {anchors_path}")
    if not processed_dir.exists():
        raise RuntimeError(f"Missing processed directory for retention: {processed_dir}")

    anchors = load_json(anchors_path)
    sampled_anchors = _select_anchor_sample(anchors, movie_count=movie_count, tv_count=tv_count)
    kept_title_ids = {_title_id(anchor) for anchor in sampled_anchors}

    write_json(anchors_path, sampled_anchors)

    source_stats: dict[str, Any] = {
        "tmdb": _retain_tmdb_run(tmdb_run_dir, sampled_anchors),
        "imdb": _retain_imdb_run(imdb_run_dir, sampled_anchors),
    }
    for source_name in PER_TITLE_SOURCES:
        source_stats[source_name] = _retain_per_title_source_run(
            paths.raw_root / source_name / source_run_id,
            sampled_anchors,
        )

    processed_stats = _retain_processed_run(processed_dir, kept_title_ids)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "applied": True,
        "source_run_id": source_run_id,
        "process_run_id": process_run_id,
        "kept_movie_count": sum(1 for anchor in sampled_anchors if anchor["content_type"] == "movie"),
        "kept_tv_count": sum(1 for anchor in sampled_anchors if anchor["content_type"] == "tv"),
        "kept_title_count": len(sampled_anchors),
        "source_stats": source_stats,
        "processed_stats": processed_stats,
    }
    write_json(reports_dir / "local_retention_manifest.json", manifest)
    return manifest
