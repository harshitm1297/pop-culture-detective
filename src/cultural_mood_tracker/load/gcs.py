from __future__ import annotations

import os
from pathlib import Path


def _import_storage():
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise RuntimeError(
            "Missing dependency google-cloud-storage. Install requirements before running cloud upload."
        ) from exc
    return storage


def build_gcs_uri(bucket_name: str, object_name: str) -> str:
    return f"gs://{bucket_name}/{object_name.replace('\\', '/')}"


def create_storage_client(credentials_path: str | None = None):
    storage = _import_storage()
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return storage.Client()


def upload_file(*, client, bucket_name: str, local_path: Path, object_name: str) -> dict[str, str | int]:
    blob = client.bucket(bucket_name).blob(object_name.replace("\\", "/"))
    blob.upload_from_filename(str(local_path))
    return {
        "bucket": bucket_name,
        "object_name": object_name.replace("\\", "/"),
        "local_path": str(local_path),
        "gcs_uri": build_gcs_uri(bucket_name, object_name),
        "size_bytes": local_path.stat().st_size,
    }


def upload_directory(
    *,
    client,
    bucket_name: str,
    local_dir: Path,
    object_prefix: str,
    allowed_suffixes: tuple[str, ...] | None = None,
) -> list[dict[str, str | int]]:
    if not local_dir.exists():
        raise RuntimeError(f"Cannot upload missing directory: {local_dir}")

    uploads: list[dict[str, str | int]] = []
    for path in sorted(item for item in local_dir.rglob("*") if item.is_file()):
        if allowed_suffixes and path.suffix.lower() not in allowed_suffixes:
            continue
        relative_path = path.relative_to(local_dir).as_posix()
        object_name = f"{object_prefix.rstrip('/')}/{relative_path}"
        uploads.append(
            upload_file(
                client=client,
                bucket_name=bucket_name,
                local_path=path,
                object_name=object_name,
            )
        )
    return uploads
