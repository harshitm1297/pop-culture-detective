from __future__ import annotations

import os


TABLE_NAMES = (
    "titles",
    "documents",
    "document_chunks",
    "ratings",
    "attention_signals",
)


def _import_bigquery():
    try:
        from google.cloud import bigquery
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError(
            "Missing dependency google-cloud-bigquery. Install requirements before running BigQuery load."
        ) from exc
    return bigquery


def create_bigquery_client(project_id: str, credentials_path: str | None = None):
    bigquery = _import_bigquery()
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return bigquery.Client(project=project_id)


def ensure_dataset(*, client, project_id: str, dataset_name: str, location: str) -> str:
    bigquery = _import_bigquery()
    dataset_id = f"{project_id}.{dataset_name}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = location
    client.create_dataset(dataset, exists_ok=True)
    return dataset_id


def table_exists(*, client, table_id: str) -> bool:
    try:
        client.get_table(table_id)
        return True
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ == "NotFound":
            return False
        raise


def delete_existing_process_rows(*, client, table_id: str, process_run_id: str) -> None:
    query = (
        f"DELETE FROM `{table_id}` "
        f"WHERE process_run_id = @process_run_id"
    )
    job_config = _import_bigquery().QueryJobConfig(
        query_parameters=[
            _import_bigquery().ScalarQueryParameter("process_run_id", "STRING", process_run_id),
        ]
    )
    client.query(query, job_config=job_config).result()


def load_jsonl_from_gcs(
    *,
    client,
    dataset_id: str,
    table_name: str,
    source_uri: str,
) -> dict[str, str | int]:
    bigquery = _import_bigquery()
    table_id = f"{dataset_id}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        ignore_unknown_values=False,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job = client.load_table_from_uri(source_uri, table_id, job_config=job_config)
    job.result()
    table = client.get_table(table_id)
    return {
        "table_id": table_id,
        "source_uri": source_uri,
        "row_count": table.num_rows,
    }
