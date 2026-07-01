from __future__ import annotations

from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import resolve_path
from cultural_mood_tracker.transform.common import load_json, write_json

from .bigquery import (
    TABLE_NAMES,
    create_bigquery_client,
    delete_existing_process_rows,
    ensure_dataset,
    load_jsonl_from_gcs,
    table_exists,
)
from .gcs import create_storage_client, upload_directory, upload_file


RAW_SOURCE_NAMES = ("anchors", "tmdb", "imdb", "tvmaze", "wikidata", "wikipedia", "guardian", "gdelt")


def _require_value(name: str, value: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise RuntimeError(f"Missing required cloud setting: {name}")
    return resolved


def _resolve_credentials(project_root: Path, raw_value: str) -> str | None:
    if not raw_value.strip():
        return None
    return str(resolve_path(project_root, raw_value))


def _load_process_manifest(processed_dir: Path) -> dict[str, Any]:
    manifest_path = processed_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing processed manifest: {manifest_path}")
    return load_json(manifest_path)


def _upload_raw_sources(*, client, paths, source_run_id: str, bucket_name: str) -> list[dict[str, Any]]:
    uploads: list[dict[str, Any]] = []
    for source_name in RAW_SOURCE_NAMES:
        local_dir = paths.raw_root / source_name / source_run_id
        if not local_dir.exists():
            continue
        uploads.extend(
            upload_directory(
                client=client,
                bucket_name=bucket_name,
                local_dir=local_dir,
                object_prefix=f"raw/{source_name}/{source_run_id}",
            )
        )
    return uploads


def _upload_processed_outputs(*, client, paths, process_run_id: str, bucket_name: str) -> dict[str, list[dict[str, Any]]]:
    processed_dir = paths.processed_root / process_run_id
    reports_dir = paths.reports_root / process_run_id
    return {
        "processed": upload_directory(
            client=client,
            bucket_name=bucket_name,
            local_dir=processed_dir,
            object_prefix=f"processed/{process_run_id}",
        ),
        "reports": upload_directory(
            client=client,
            bucket_name=bucket_name,
            local_dir=reports_dir,
            object_prefix=f"reports/{process_run_id}",
        ),
    }


def _load_bigquery_tables(
    *,
    project_id: str,
    dataset_name: str,
    location: str,
    credentials_path: str | None,
    bucket_name: str,
    process_run_id: str,
) -> list[dict[str, Any]]:
    client = create_bigquery_client(project_id, credentials_path)
    dataset_id = ensure_dataset(
        client=client,
        project_id=project_id,
        dataset_name=dataset_name,
        location=location,
    )

    results: list[dict[str, Any]] = []
    for table_name in TABLE_NAMES:
        table_id = f"{dataset_id}.{table_name}"
        if table_exists(client=client, table_id=table_id):
            delete_existing_process_rows(
                client=client,
                table_id=table_id,
                process_run_id=process_run_id,
            )
        results.append(
            load_jsonl_from_gcs(
                client=client,
                dataset_id=dataset_id,
                table_name=table_name,
                source_uri=f"gs://{bucket_name}/processed/{process_run_id}/{table_name}.jsonl",
            )
        )
    return results


def run_cloud_load(
    *,
    project_root: Path,
    source_run_id: str,
    process_run_id: str,
    pipeline_run_id: str | None = None,
    enable_gcs_upload: bool | None = None,
    enable_bigquery_load: bool | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    gcs_enabled = settings.enable_gcs_upload if enable_gcs_upload is None else enable_gcs_upload
    bq_enabled = settings.enable_bigquery_load if enable_bigquery_load is None else enable_bigquery_load
    if not gcs_enabled and not bq_enabled:
        return {
            "enabled": False,
            "source_run_id": source_run_id,
            "process_run_id": process_run_id,
            "message": "Cloud load skipped because both GCS and BigQuery flags are disabled.",
        }

    project_id = _require_value("GCP_PROJECT_ID", settings.gcp_project_id)
    raw_bucket = _require_value("GCS_BUCKET_RAW", settings.gcs_bucket_raw) if gcs_enabled else ""
    processed_bucket = _require_value("GCS_BUCKET_PROCESSED", settings.gcs_bucket_processed)
    credentials_path = _resolve_credentials(project_root, settings.google_application_credentials)

    processed_dir = paths.processed_root / process_run_id
    reports_dir = paths.reports_root / process_run_id
    if not processed_dir.exists():
        raise RuntimeError(f"Missing processed directory: {processed_dir}")
    if not reports_dir.exists():
        raise RuntimeError(f"Missing reports directory: {reports_dir}")

    process_manifest = _load_process_manifest(processed_dir)
    storage_client = create_storage_client(credentials_path, project_id=project_id)

    raw_uploads: list[dict[str, Any]] = []
    if gcs_enabled:
        raw_uploads = _upload_raw_sources(
            client=storage_client,
            paths=paths,
            source_run_id=source_run_id,
            bucket_name=raw_bucket,
        )

    processed_uploads = _upload_processed_outputs(
        client=storage_client,
        paths=paths,
        process_run_id=process_run_id,
        bucket_name=processed_bucket,
    )

    bigquery_results: list[dict[str, Any]] = []
    if bq_enabled:
        dataset_name = _require_value("BIGQUERY_DATASET", settings.bigquery_dataset)
        location = _require_value("BIGQUERY_LOCATION", settings.bigquery_location)
        bigquery_results = _load_bigquery_tables(
            project_id=project_id,
            dataset_name=dataset_name,
            location=location,
            credentials_path=credentials_path,
            bucket_name=processed_bucket,
            process_run_id=process_run_id,
        )

    manifest = {
        "enabled": True,
        "pipeline_run_id": pipeline_run_id,
        "source_run_id": source_run_id,
        "process_run_id": process_run_id,
        "gcs_upload_enabled": gcs_enabled,
        "bigquery_load_enabled": bq_enabled,
        "raw_bucket": raw_bucket,
        "processed_bucket": processed_bucket,
        "processed_outputs": process_manifest.get("outputs", []),
        "raw_upload_count": len(raw_uploads),
        "processed_upload_count": len(processed_uploads["processed"]),
        "report_upload_count": len(processed_uploads["reports"]),
        "raw_uploads": raw_uploads,
        "processed_uploads": processed_uploads["processed"],
        "report_uploads": processed_uploads["reports"],
        "bigquery_tables": bigquery_results,
    }

    manifest_path = reports_dir / "cloud_load_manifest.json"
    manifest["report_upload_count"] += 1
    manifest["report_uploads"].append(
        {
            "bucket": processed_bucket,
            "object_name": f"reports/{process_run_id}/cloud_load_manifest.json",
            "local_path": str(manifest_path),
        }
    )
    write_json(manifest_path, manifest)
    manifest["report_uploads"][-1] = upload_file(
        client=storage_client,
        bucket_name=processed_bucket,
        local_path=manifest_path,
        object_name=f"reports/{process_run_id}/cloud_load_manifest.json",
    )
    write_json(manifest_path, manifest)
    upload_file(
        client=storage_client,
        bucket_name=processed_bucket,
        local_path=manifest_path,
        object_name=f"reports/{process_run_id}/cloud_load_manifest.json",
    )
    return {
        "manifest_path": str(manifest_path),
        **manifest,
    }
