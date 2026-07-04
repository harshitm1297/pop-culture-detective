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
    enable_critic_blog_sources: bool
    critic_feed_entry_limit: int
    motherduck_database: str
    motherduck_token: str
    enable_motherduck_load: bool
    enable_local_sample_retention: bool
    local_retain_movie_count: int
    local_retain_tv_count: int
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
        enable_critic_blog_sources=_get_bool("ENABLE_CRITIC_BLOG_SOURCES", default=True),
        critic_feed_entry_limit=int(os.getenv("CRITIC_FEED_ENTRY_LIMIT", "40")),
        motherduck_database=os.getenv("MOTHERDUCK_DATABASE", "cultural_mood_tracker").strip() or "cultural_mood_tracker",
        motherduck_token=_get_env("MOTHERDUCK_TOKEN"),
        enable_motherduck_load=_get_bool("ENABLE_MOTHERDUCK_LOAD", default=True),
        enable_local_sample_retention=_get_bool("ENABLE_LOCAL_SAMPLE_RETENTION", default=True),
        local_retain_movie_count=int(os.getenv("LOCAL_RETAIN_MOVIE_COUNT", "30")),
        local_retain_tv_count=int(os.getenv("LOCAL_RETAIN_TV_COUNT", "30")),
        local_data_root=Path(os.getenv("LOCAL_DATA_ROOT", "data")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
    )
