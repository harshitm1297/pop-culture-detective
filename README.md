# Cultural Mood Tracker ETL

This repository contains the ETL implementation for the `movies + TV` version of the Cultural Mood Tracker project. The pipeline extracts recent English-language title data, aligns multiple evidence sources to the same TMDB anchor titles, transforms that data into canonical tables, and publishes the processed outputs to `MotherDuck`.

## Scope

- Domain: English-language `movies` and `TV shows`
- Window: configured in `.env`, usually the last 1 year
- Primary use: local ETL plus shared online warehousing in `MotherDuck`
- RAG readiness: `documents` and `document_chunks` are the main text evidence layers

## Data Sources

The ETL currently uses these sources:

- `TMDB API`
  - anchor titles
  - title metadata
  - overviews
  - TMDB user reviews
  - external IDs such as IMDb IDs
- `IMDb public datasets`
  - `title.basics.tsv.gz`
  - `title.ratings.tsv.gz`
  - used for title normalization and ratings enrichment
- `TVMaze API`
  - supplemental TV metadata aligned through IMDb ID
- `Wikidata API`
  - entity linking and English Wikipedia article titles
- `Wikipedia Pageviews API`
  - attention signals over time
- `Guardian Open Platform`
  - English editorial coverage for matched titles
- `GDELT`
  - optional stretch source, disabled by default because it is noisy and rate-limited

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
    load_to_motherduck.py
    query_motherduck.py
    run_pipeline.py
  src/
    cultural_mood_tracker/
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
  README.md
  requirements.txt
```

## Teammate Setup

### 1. Clone the repository

```powershell
git clone --branch ETL-implementation https://github.com/harshitm1297/pop-culture-detective.git
cd .\pop-culture-detective\Project
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Local ETL itself uses the Python standard library. MotherDuck publishing requires `duckdb`. RAG embeddings require `sentence-transformers`, and local vector storage requires `chromadb`.

```powershell
pip install -r .\requirements.txt
```

### 4. Create `.env`

Start from `.env.example` and fill the required values:

```env
TMDB_API_KEY=your-tmdb-api-key
TMDB_LANGUAGE=en-US
TMDB_REGION=
TMDB_START_DATE=2025-06-29
TMDB_END_DATE=2026-06-29
TMDB_MOVIE_SAMPLE_SIZE=300
TMDB_TV_SAMPLE_SIZE=200

GUARDIAN_API_KEY=test
GUARDIAN_PAGE_SIZE=5
GDELT_MAX_RECORDS=5

MOTHERDUCK_DATABASE=cultural_mood_tracker
MOTHERDUCK_TOKEN=your-motherduck-token
ENABLE_MOTHERDUCK_LOAD=true
ENABLE_LOCAL_SAMPLE_RETENTION=true
LOCAL_RETAIN_MOVIE_COUNT=30
LOCAL_RETAIN_TV_COUNT=30

LOCAL_DATA_ROOT=data
LOG_LEVEL=INFO
```

## Main Command

The main entry point is:

```powershell
python .\scripts\run_pipeline.py
```

That one command does:

1. scrape aligned source data
2. transform it into canonical processed tables
3. upload those processed tables to `MotherDuck`
4. keep only a local sample of `30 movies + 30 TV titles` by default after a successful upload

If you want to stop after local transform and skip MotherDuck:

```powershell
python .\scripts\run_pipeline.py --skip-motherduck-load
```

If you want to keep the full local raw and processed outputs:

```powershell
python .\scripts\run_pipeline.py --keep-full-local
```

## What Each Script Does

### `scripts/extract_tmdb_smoke_test.py`

Quick connectivity test for TMDB. It loads `.env`, calls TMDB discover, fetches details and reviews for a small sample, optionally fetches Wikipedia pageviews, and saves raw JSON under `data/raw/tmdb_smoke/<run_id>/`.

### `scripts/extract_multisource_aligned.py`

Full raw extraction for the ETL baseline. It creates TMDB anchor titles, downloads IMDb metadata snapshots, fetches aligned raw data from TVMaze, Wikidata, Wikipedia, Guardian, and optional GDELT, and saves source-specific raw files under `data/raw/<source>/<run_id>/`.

### `scripts/transform_canonical.py`

Transforms a raw aligned run into canonical processed tables. The current core outputs are:

- `titles`
- `documents`
- `document_chunks`
- `ratings`
- `attention_signals`

If optional tables such as `people`, `title_cast`, or `title_crew` exist in a processed run, the MotherDuck loader will publish them too.

### `scripts/load_to_motherduck.py`

Publishes a processed run into the configured `MotherDuck` database. It defaults to the latest processed run if you do not pass a run ID.

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

Runs the full ETL flow end to end and is the recommended command for normal use.

### `scripts/query_motherduck.py`

Runs a MotherDuck query against the configured database.

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

### 6. Retrieval
```powershell
python .\scripts\retrieve.py --query "a love story" --top-k 5
```

### 7. Retrieval evaluation
``` powershel
python .\scripts\retrieval_eval.py
```

### 8. Prompt Construction
Offline (recommended first — proves prompting is independently testable, no ChromaDB/model needed)
```powershell
python .\scripts\retrieve.py --query "Disclosure Day" --top-k 5 --output-path data\eval\_scratch_retrieve.json
python .\scripts\build_prompt.py --input-path data\eval\_scratch_retrieve.json
```

Live (retrieval + prompt construction chained in one call):
```powershell
python .\scripts\build_prompt.py --query "a whistleblower exposes a corporate cover-up about extraterrestrial life" --top-k 5
```


### 6. Local End-To-End Pipeline

```powershell
python .\scripts\run_pipeline.py --skip-motherduck-load
```

### 7. Publish To MotherDuck

```powershell
python .\scripts\load_to_motherduck.py --process-run-id <process_run_id>
```

### 8. Full ETL + MotherDuck In One Command

```powershell
python .\scripts\run_pipeline.py
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

Optional processed tables may also appear, depending on the transform version:

```text
data/processed/<process_run_id>/people.jsonl
data/processed/<process_run_id>/title_cast.jsonl
data/processed/<process_run_id>/title_crew.jsonl
```

### Reports

```text
data/reports/<process_run_id>/coverage_report.json
data/reports/<process_run_id>/validation_report.json
data/reports/<process_run_id>/document_deduplication.json
data/reports/<process_run_id>/motherduck_load_manifest.json
```

## MotherDuck Notes

- `MOTHERDUCK_TOKEN` is required for upload
- `MOTHERDUCK_DATABASE` is created automatically if it does not already exist
- teammates should use their own `MotherDuck` tokens when possible
- the processed files remain the reproducible local source of truth; `MotherDuck` is the shared online warehouse layer

## Notes

- Do not commit `data/` outputs
- Do not commit `.env` with real tokens if the repository is public
- `GDELT` is optional and disabled by default because it introduces more noise and rate-limit issues than the baseline sources
