from .base import ensure_dir, http_download_binary, http_get_json, http_get_text, save_json, slugify
from .critic_blogs import (
    detect_document_type,
    fetch_article_text,
    fetch_candidate_entries,
    fetch_feed_entries,
    get_source,
    list_sources,
)

__all__ = [
    "detect_document_type",
    "ensure_dir",
    "fetch_article_text",
    "fetch_candidate_entries",
    "fetch_feed_entries",
    "get_source",
    "http_download_binary",
    "http_get_json",
    "http_get_text",
    "list_sources",
    "save_json",
    "slugify",
]
