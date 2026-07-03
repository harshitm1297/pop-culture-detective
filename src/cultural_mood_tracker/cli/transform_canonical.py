from __future__ import annotations

import argparse
from pathlib import Path

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.transform import (
    annotate_chunks,
    build_attention_signals,
    build_attention_vs_reception,
    build_audience_vs_editorial_summary,
    build_document_chunks,
    build_documents,
    build_episodes,
    build_genre_theme_summary,
    build_monthly_theme_trends,
    build_people_and_credits,
    build_ratings,
    build_title_theme_summary,
    build_title_videos,
    build_titles,
    build_validation_report,
    deduplicate_documents,
)
from cultural_mood_tracker.transform.common import (
    find_latest_run_id,
    load_json,
    write_csv,
    write_json,
    write_jsonl,
)
from cultural_mood_tracker.transform.reports import build_coverage_report


def parse_args(default_source_run_id: str | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform aligned raw source data into canonical processed tables."
    )
    parser.add_argument(
        "--source-run-id",
        default=default_source_run_id,
        help="Aligned raw run ID to transform. Defaults to the latest anchors run.",
    )
    parser.add_argument(
        "--output-run-id",
        default=None,
        help="Processed run ID. Defaults to a fresh UTC timestamp.",
    )
    return parser.parse_args()


def run_transform(project_root: Path, source_run_id: str, output_run_id: str | None = None) -> str:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()
    process_run_id = output_run_id or make_run_id()

    anchor_path = paths.raw_root / "anchors" / source_run_id / "anchor_titles.json"
    if not anchor_path.exists():
        raise RuntimeError(f"Missing anchor file: {anchor_path}")

    anchors = load_json(anchor_path)

    titles = build_titles(
        anchors,
        tmdb_run_dir=paths.raw_root / "tmdb" / source_run_id,
        imdb_run_dir=paths.raw_root / "imdb" / source_run_id,
        tvmaze_run_dir=paths.raw_root / "tvmaze" / source_run_id,
        wikidata_run_dir=paths.raw_root / "wikidata" / source_run_id,
        source_run_id=source_run_id,
    )
    raw_documents = build_documents(
        anchors,
        tmdb_run_dir=paths.raw_root / "tmdb" / source_run_id,
        tvmaze_run_dir=paths.raw_root / "tvmaze" / source_run_id,
        guardian_run_dir=paths.raw_root / "guardian" / source_run_id,
        gdelt_run_dir=paths.raw_root / "gdelt" / source_run_id,
        wikidata_run_dir=paths.raw_root / "wikidata" / source_run_id,
        critic_source_dirs={
            "rogerebert": paths.raw_root / "rogerebert" / source_run_id,
            "indiewire": paths.raw_root / "indiewire" / source_run_id,
            "vulture": paths.raw_root / "vulture" / source_run_id,
            "slant": paths.raw_root / "slant" / source_run_id,
            "slashfilm": paths.raw_root / "slashfilm" / source_run_id,
        },
        source_run_id=source_run_id,
    )
    documents, document_dedup_stats = deduplicate_documents(raw_documents)
    ratings = build_ratings(
        anchors,
        tmdb_run_dir=paths.raw_root / "tmdb" / source_run_id,
        imdb_run_dir=paths.raw_root / "imdb" / source_run_id,
        source_run_id=source_run_id,
    )
    attention_signals = build_attention_signals(
        anchors,
        wikipedia_run_dir=paths.raw_root / "wikipedia" / source_run_id,
        wikidata_run_dir=paths.raw_root / "wikidata" / source_run_id,
        source_run_id=source_run_id,
    )
    people, title_cast, title_crew = build_people_and_credits(
        anchors,
        tmdb_run_dir=paths.raw_root / "tmdb" / source_run_id,
        source_run_id=source_run_id,
    )
    episodes = build_episodes(
        anchors,
        tvmaze_run_dir=paths.raw_root / "tvmaze" / source_run_id,
        source_run_id=source_run_id,
    )
    title_videos = build_title_videos(
        anchors,
        tmdb_run_dir=paths.raw_root / "tmdb" / source_run_id,
        source_run_id=source_run_id,
    )

    titles_by_id = {row["title_id"]: row for row in titles}
    document_chunks = build_document_chunks(documents, titles_by_id)
    chunk_annotations = annotate_chunks(document_chunks)
    title_theme_summary = build_title_theme_summary(titles, chunk_annotations)
    genre_theme_summary = build_genre_theme_summary(titles, title_theme_summary)
    monthly_theme_trends = build_monthly_theme_trends(titles, chunk_annotations)
    audience_vs_editorial_summary = build_audience_vs_editorial_summary(chunk_annotations)
    attention_vs_reception = build_attention_vs_reception(
        titles,
        ratings,
        attention_signals,
        title_theme_summary,
    )

    for rows in (
        titles,
        documents,
        document_chunks,
        chunk_annotations,
        ratings,
        attention_signals,
        people,
        title_cast,
        title_crew,
        episodes,
        title_videos,
        title_theme_summary,
        genre_theme_summary,
        monthly_theme_trends,
        audience_vs_editorial_summary,
        attention_vs_reception,
    ):
        for row in rows:
            row["process_run_id"] = process_run_id

    processed_dir = paths.processed_root / process_run_id
    reports_dir = paths.reports_root / process_run_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "titles": titles,
        "documents": documents,
        "document_chunks": document_chunks,
        "chunk_annotations": chunk_annotations,
        "ratings": ratings,
        "attention_signals": attention_signals,
        "people": people,
        "title_cast": title_cast,
        "title_crew": title_crew,
        "episodes": episodes,
        "title_videos": title_videos,
        "title_theme_summary": title_theme_summary,
        "genre_theme_summary": genre_theme_summary,
        "monthly_theme_trends": monthly_theme_trends,
        "audience_vs_editorial_summary": audience_vs_editorial_summary,
        "attention_vs_reception": attention_vs_reception,
    }
    for name, rows in tables.items():
        write_jsonl(processed_dir / f"{name}.jsonl", rows)
        write_csv(processed_dir / f"{name}.csv", rows)

    coverage_report = build_coverage_report(
        source_run_id=source_run_id,
        process_run_id=process_run_id,
        titles=titles,
        documents=documents,
        ratings=ratings,
        attention_signals=attention_signals,
        document_chunks=document_chunks,
        people=people,
        title_cast=title_cast,
        title_crew=title_crew,
        episodes=episodes,
        title_videos=title_videos,
    )
    validation_report = build_validation_report(
        source_run_id=source_run_id,
        process_run_id=process_run_id,
        titles=titles,
        documents=documents,
        ratings=ratings,
        attention_signals=attention_signals,
        document_dedup_stats=document_dedup_stats,
        chunks=document_chunks,
        people=people,
        title_cast=title_cast,
        title_crew=title_crew,
        episodes=episodes,
        title_videos=title_videos,
    )
    manifest = {
        "source_run_id": source_run_id,
        "process_run_id": process_run_id,
        "outputs": sorted(f"{name}.jsonl" for name in tables),
    }
    write_json(processed_dir / "run_manifest.json", manifest)
    write_json(reports_dir / "coverage_report.json", coverage_report)
    write_json(reports_dir / "validation_report.json", validation_report)
    write_json(reports_dir / "document_deduplication.json", document_dedup_stats)

    print(f"Processed outputs written to: {processed_dir}")
    print(f"Coverage report written to: {reports_dir / 'coverage_report.json'}")
    print(f"Validation report written to: {reports_dir / 'validation_report.json'}")
    print(
        f"titles={len(titles)} documents={len(documents)} "
        f"chunks={len(document_chunks)} ratings={len(ratings)} "
        f"attention_signals={len(attention_signals)} people={len(people)} "
        f"title_cast={len(title_cast)} title_crew={len(title_crew)} "
        f"episodes={len(episodes)} title_videos={len(title_videos)} "
        f"chunk_annotations={len(chunk_annotations)} "
        f"title_theme_summary={len(title_theme_summary)} "
        f"genre_theme_summary={len(genre_theme_summary)} "
        f"monthly_theme_trends={len(monthly_theme_trends)} "
        f"audience_vs_editorial_summary={len(audience_vs_editorial_summary)} "
        f"attention_vs_reception={len(attention_vs_reception)}"
    )
    return process_run_id


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    default_source_run_id = None
    if paths.raw_root.joinpath("anchors").exists():
        default_source_run_id = find_latest_run_id(paths.raw_root / "anchors")

    args = parse_args(default_source_run_id)
    if not args.source_run_id:
        raise RuntimeError("No raw aligned run available. Run extraction first or pass --source-run-id.")

    run_transform(project_root, args.source_run_id, args.output_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
