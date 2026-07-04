from __future__ import annotations

from .base import http_download_binary


IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
IMDB_CREW_URL = "https://datasets.imdbws.com/title.crew.tsv.gz"
IMDB_PRINCIPALS_URL = "https://datasets.imdbws.com/title.principals.tsv.gz"
IMDB_EPISODE_URL = "https://datasets.imdbws.com/title.episode.tsv.gz"
IMDB_NAME_BASICS_URL = "https://datasets.imdbws.com/name.basics.tsv.gz"
IMDB_USER_AGENT = "cultural-mood-tracker-imdb/0.1"


def download_basics() -> bytes:
    return http_download_binary(IMDB_BASICS_URL, user_agent=IMDB_USER_AGENT)


def download_ratings() -> bytes:
    return http_download_binary(IMDB_RATINGS_URL, user_agent=IMDB_USER_AGENT)


def download_crew() -> bytes:
    return http_download_binary(IMDB_CREW_URL, user_agent=IMDB_USER_AGENT)


def download_principals() -> bytes:
    return http_download_binary(IMDB_PRINCIPALS_URL, user_agent=IMDB_USER_AGENT)


def download_episode() -> bytes:
    return http_download_binary(IMDB_EPISODE_URL, user_agent=IMDB_USER_AGENT)


def download_name_basics() -> bytes:
    return http_download_binary(IMDB_NAME_BASICS_URL, user_agent=IMDB_USER_AGENT)
