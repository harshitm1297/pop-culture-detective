from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import ProjectPaths, build_project_paths


@dataclass(frozen=True)
class Settings:
    tmdb_api_key: str
    tmdb_language: str
    tmdb_region: str
    tmdb_start_date: str
    tmdb_end_date: str
    tmdb_movie_sample_size: int
    tmdb_tv_sample_size: int
    guardian_api_key: str
    guardian_page_size: int
    gdelt_max_records: int
    gcp_project_id: str
    gcp_region: str
    gcs_bucket_raw: str
    gcs_bucket_processed: str
    bigquery_dataset: str
    bigquery_location: str
    google_application_credentials: str
    enable_gcs_upload: bool
    enable_bigquery_load: bool
    local_data_root: Path
    log_level: str

    def build_paths(self, project_root: Path) -> ProjectPaths:
        return build_project_paths(project_root=project_root, data_root=self.local_data_root)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    return Settings(
        tmdb_api_key=_require_env("TMDB_API_KEY"),
        tmdb_language=os.getenv("TMDB_LANGUAGE", "en-US").strip() or "en-US",
        tmdb_region=os.getenv("TMDB_REGION", "").strip(),
        tmdb_start_date=os.getenv("TMDB_START_DATE", "2025-06-29").strip() or "2025-06-29",
        tmdb_end_date=os.getenv("TMDB_END_DATE", "2026-06-29").strip() or "2026-06-29",
        tmdb_movie_sample_size=int(os.getenv("TMDB_MOVIE_SAMPLE_SIZE", "300")),
        tmdb_tv_sample_size=int(os.getenv("TMDB_TV_SAMPLE_SIZE", "200")),
        guardian_api_key=os.getenv("GUARDIAN_API_KEY", "test").strip() or "test",
        guardian_page_size=int(os.getenv("GUARDIAN_PAGE_SIZE", "5")),
        gdelt_max_records=int(os.getenv("GDELT_MAX_RECORDS", "5")),
        gcp_project_id=_get_env("GCP_PROJECT_ID"),
        gcp_region=os.getenv("GCP_REGION", "europe-west4").strip() or "europe-west4",
        gcs_bucket_raw=_get_env("GCS_BUCKET_RAW"),
        gcs_bucket_processed=_get_env("GCS_BUCKET_PROCESSED"),
        bigquery_dataset=_get_env("BIGQUERY_DATASET"),
        bigquery_location=os.getenv("BIGQUERY_LOCATION", "europe-west4").strip() or "europe-west4",
        google_application_credentials=_get_env("GOOGLE_APPLICATION_CREDENTIALS"),
        enable_gcs_upload=_get_bool("ENABLE_GCS_UPLOAD", default=False),
        enable_bigquery_load=_get_bool("ENABLE_BIGQUERY_LOAD", default=False),
        local_data_root=Path(os.getenv("LOCAL_DATA_ROOT", "data")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
    )
