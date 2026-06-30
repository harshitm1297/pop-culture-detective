# Cultural Mood Tracker ETL

This repository contains the ETL implementation for the `movies + TV` version of the Cultural Mood Tracker project. The goal is to extract recent English-language title data, align multiple evidence sources to the same TMDB anchor titles, transform that data into canonical tables, and optionally publish the outputs to `GCS` and `BigQuery`.

## Scope

- Domain: English-language `movies` and `TV shows`
- Window: `last 1 year`, currently fixed in `.env`
- Primary use: local ETL now, RAG-ready processed data later
- Deployment target: local development first, `GCP` storage/warehouse optional

## Data Sources

The ETL currently uses these sources:

- `TMDB API`
  - Anchor titles
  - Title metadata
  - Overviews
  - TMDB user reviews
  - External IDs such as IMDb IDs
- `IMDb public datasets`
  - `title.basics.tsv.gz`
  - `title.ratings.tsv.gz`
  - Used for title normalization and ratings enrichment
- `TVMaze API`
  - Supplemental TV metadata aligned through IMDb ID
- `Wikidata API`
  - Entity linking and English Wikipedia article titles
- `Wikipedia Pageviews API`
  - Attention signals over time
- `Guardian Open Platform`
  - English editorial coverage for matched titles
- `GDELT`
  - Optional stretch source, disabled by default because it is noisy and rate-limited

## Repository Layout

```text
Project/
  data/
    raw/
    staging/
    processed/
    reports/
    logs/
  scripts/
    extract_tmdb_smoke_test.py
    extract_multisource_aligned.py
    transform_canonical.py
    embed_document_chunks.py
    ingest_chroma.py
    run_pipeline.py
    load_to_gcp.py
  src/
    cultural_mood_tracker/
      __init__.py
      cli/
      config/
      core/
      extract/
      load/
      rag/
      pipeline/
      sources/
      transform/
  .env
  .env.example
  .gitignore
  README.md
  requirements.txt
```

Notes:

- `scripts/` contains the runnable entry points.
- `src/cultural_mood_tracker/cli/` contains the actual CLI implementations.
- `data/` is ignored by git, so extracted and processed data should not be pushed.

## Files To Share With Teammates

You mentioned you will share these two files:

- `.env`
- `service-account.json`

That is enough to let teammates run the same pipeline, but each teammate still needs to place the JSON file on their own machine and update the local path if necessary.

## Teammate Setup

### 1. Clone The Repository

Clone the branch:

```powershell
git clone --branch ETL-implementation https://github.com/harshitm1297/pop-culture-detective.git
cd .\pop-culture-detective\Project
```

### 2. Python

Use `Python 3.11+`. This project has been tested with `Python 3.13` locally.

### 3. Create And Activate A Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

Local ETL itself uses the Python standard library. Cloud upload/load requires Google packages. RAG embeddings require `sentence-transformers`, and local vector storage requires `chromadb`.

```powershell
pip install -r .\requirements.txt
```

### 5. Copy The Shared Files

Place the shared files locally:

- `.env` in the `Project/` folder
- `service-account.json` anywhere on the local machine

Recommended local layout:

```text
Project/
  .env
  credentials/
    service-account.json
```

If you place the service account JSON there, set:

```text
GOOGLE_APPLICATION_CREDENTIALS=credentials\service-account.json
```

Do not commit the JSON file to git.

## `.env` Variables

These are the important settings your teammates should understand.

### TMDB Extraction

- `TMDB_API_KEY`
  - Required
  - TMDB API key
- `TMDB_LANGUAGE`
  - Current default: `en-US`
  - Controls TMDB response language
- `TMDB_REGION`
  - Optional regional bias
- `TMDB_START_DATE`
  - Lower bound for movie release date / TV first air date
- `TMDB_END_DATE`
  - Upper bound for movie release date / TV first air date
- `TMDB_MOVIE_SAMPLE_SIZE`
  - Number of movie anchors to fetch
- `TMDB_TV_SAMPLE_SIZE`
  - Number of TV anchors to fetch

### Secondary Sources

- `GUARDIAN_API_KEY`
  - `test` works for basic Guardian access
- `GUARDIAN_PAGE_SIZE`
  - Number of Guardian records per title query
- `GDELT_MAX_RECORDS`
  - GDELT cap per title

### GCP / Warehouse

- `GCP_PROJECT_ID`
  - GCP project containing the buckets and BigQuery resources
- `GCP_REGION`
  - Example: `europe-west4`
- `GCS_BUCKET_RAW`
  - Bucket for raw snapshots
- `GCS_BUCKET_PROCESSED`
  - Bucket for processed outputs and reports
- `BIGQUERY_DATASET`
  - Dataset for canonical tables
- `BIGQUERY_LOCATION`
  - Must be compatible with your dataset region
- `GOOGLE_APPLICATION_CREDENTIALS`
  - Local path to the service account JSON file
- `ENABLE_GCS_UPLOAD`
  - `true` or `false`
- `ENABLE_BIGQUERY_LOAD`
  - `true` or `false`

### Local Runtime

- `LOCAL_DATA_ROOT`
  - Usually `data`
- `LOG_LEVEL`
  - Usually `INFO`

## GCP Prerequisites

Before `load_to_gcp.py` will work, all of these must already exist or be correctly permissioned:

- `GCS_BUCKET_RAW`
- `GCS_BUCKET_PROCESSED`
- Service account JSON with access to those buckets
- BigQuery permissions for the configured dataset or project

### Required Bucket Permissions

The service account needs write access to both buckets. The simplest workable role is:

- `Storage Object Admin`

Grant it on:

- `cmt-tmdb-raw-data-engineering-course`
- `cmt-tmdb-processed-data-engineering-course`

### Required BigQuery Permissions

At minimum, the service account should be able to:

- create the dataset if needed
- create/load tables
- run query jobs

Typical roles:

- `BigQuery Data Editor`
- `BigQuery Job User`

If your team wants fewer permission issues during setup, a temporary broader role is acceptable during development, then tighten later.

## What Each Script Does

### `scripts/extract_tmdb_smoke_test.py`

Quick connectivity test for TMDB.

It:

- loads `.env`
- calls TMDB discover
- fetches details and reviews for a small sample
- optionally fetches Wikipedia pageviews
- saves raw JSON under `data/raw/tmdb_smoke/<run_id>/`

### `scripts/extract_multisource_aligned.py`

Full raw extraction for the ETL baseline.

It:

- creates TMDB anchor titles
- downloads IMDb metadata snapshots
- fetches aligned raw data from TVMaze, Wikidata, Wikipedia, Guardian, and optional GDELT
- saves source-specific raw files under `data/raw/<source>/<run_id>/`

### `scripts/transform_canonical.py`

Transforms a raw aligned run into canonical processed tables.

Outputs:

- `titles`
- `documents`
- `document_chunks`
- `ratings`
- `attention_signals`
- coverage/validation/dedup reports

### `scripts/embed_document_chunks.py`

Embeds processed document chunks for the local RAG layer.

It:

- reads `data/processed/<process_run_id>/document_chunks.jsonl`
- uses `SentenceTransformer("BAAI/bge-small-en-v1.5")`
- writes `data/processed/<process_run_id>/document_chunk_embeddings.jsonl`
- outputs ChromaDB-ready records with `id`, `document`, `metadata`, and `embedding`

### `scripts/ingest_chroma.py`

Loads precomputed embeddings into a persistent local ChromaDB database.

It:

- reads `document_chunk_embeddings.jsonl`
- uses `chromadb.PersistentClient`
- upserts batches into collection `movie_chunks`
- persists the database under `./chroma_db` by default

### `scripts/run_pipeline.py`

Runs the orchestrated ETL:

- extraction
- transform
- optional cloud load

### `scripts/load_to_gcp.py`

Uploads local ETL outputs to cloud:

- raw files to `GCS_BUCKET_RAW`
- processed files and reports to `GCS_BUCKET_PROCESSED`
- canonical JSONL tables into `BigQuery`

## Recommended Run Order

### 1. Smoke Test

```powershell
python .\scripts\extract_tmdb_smoke_test.py --content-type both --sample-size 3
```

### 2. Full Raw Extraction

```powershell
python .\scripts\extract_multisource_aligned.py --movie-count 100 --tv-count 100
```

### 3. Transform The Raw Run

```powershell
python .\scripts\transform_canonical.py --source-run-id <source_run_id>
```

### 4. Embed Document Chunks

```powershell
python .\scripts\embed_document_chunks.py --process-run-id <process_run_id>
```

The embedding output can be split directly into ChromaDB `ids`, `documents`, `metadatas`, and `embeddings` for collection `add` or `upsert`.

### 5. Ingest Embeddings Into ChromaDB

```powershell
python .\scripts\ingest_chroma.py --input-path data\processed\<process_run_id>\document_chunk_embeddings.jsonl
```

By default, this writes a persistent ChromaDB database under `chroma_db/` and uses collection `movie_chunks`.

### 6. Local End-To-End Pipeline

```powershell
python .\scripts\run_pipeline.py --cleanup-old-raw
```

### 7. Upload To GCP

```powershell
python .\scripts\load_to_gcp.py --process-run-id <process_run_id> --enable-gcs-upload --enable-bigquery-load
```

### 8. Full ETL + Cloud In One Command

```powershell
python .\scripts\run_pipeline.py --cleanup-old-raw --enable-gcs-upload --enable-bigquery-load
```

## Output Locations

### Raw

```text
data/raw/anchors/<run_id>/
data/raw/tmdb/<run_id>/
data/raw/imdb/<run_id>/
data/raw/tvmaze/<run_id>/
data/raw/wikidata/<run_id>/
data/raw/wikipedia/<run_id>/
data/raw/guardian/<run_id>/
data/raw/gdelt/<run_id>/
```

### Processed

```text
data/processed/<process_run_id>/titles.jsonl
data/processed/<process_run_id>/documents.jsonl
data/processed/<process_run_id>/document_chunks.jsonl
data/processed/<process_run_id>/document_chunk_embeddings.jsonl
data/processed/<process_run_id>/ratings.jsonl
data/processed/<process_run_id>/attention_signals.jsonl
data/processed/<process_run_id>/run_manifest.json
chroma_db/
```

### Reports

```text
data/reports/<process_run_id>/coverage_report.json
data/reports/<process_run_id>/validation_report.json
data/reports/<process_run_id>/document_deduplication.json
data/reports/<process_run_id>/cloud_load_manifest.json
```

## Common GCP Failure Modes

### `404 bucket does not exist`

Cause:

- bucket name in `.env` is wrong
- bucket was never created

Check:

- `GCS_BUCKET_RAW`
- `GCS_BUCKET_PROCESSED`

### `403 storage.objects.create denied`

Cause:

- service account does not have write permission to the bucket

Fix:

- grant `Storage Object Admin` on the target bucket

### `BigQuery permission denied`

Cause:

- service account can access Storage but not BigQuery

Fix:

- add `BigQuery Data Editor`
- add `BigQuery Job User`

## Notes

- Do not commit `data/` outputs.
- Do not commit service account JSON files.
- If teammates use a different local JSON path, they must update `GOOGLE_APPLICATION_CREDENTIALS` in `.env`.
- `GDELT` is optional and disabled by default because it introduces more noise and rate-limit issues than the other baseline sources.
