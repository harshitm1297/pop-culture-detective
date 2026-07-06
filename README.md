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
    retrieve.py
    retrieval_eval.py
    build_prompt.py
    chat.py
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
git clone --branch chiara https://github.com/harshitm1297/pop-culture-detective.git
cd .\pop-culture-detective\Project
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Local ETL itself uses the Python standard library. MotherDuck publishing and SQL chatbot queries require `duckdb`. RAG embeddings and retrieval require `sentence-transformers`, local vector storage requires `chromadb`, and the unified chatbot uses Groq-hosted `llama-3.1-8b-instant` inference through the `groq` SDK.

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

RAG_DATA_SOURCE=local
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

Embeds document chunks for the local RAG layer. The chunk source is controlled by `RAG_DATA_SOURCE` in `.env`.

It:

- reads local chunks from `data/processed/20260702T131134Z/document_chunks.jsonl` when `RAG_DATA_SOURCE=local`
- reads warehouse chunks from the MotherDuck `document_chunks` table when `RAG_DATA_SOURCE=motherduck`
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

### `scripts/retrieve.py`

Runs semantic retrieval over the local ChromaDB collection.

It:

- embeds the natural-language `--query` with the same SentenceTransformer model used for ingestion
- searches the persistent ChromaDB database under `./chroma_db` by default
- returns top matching chunks with similarity scores, metadata, source names, and text previews
- optionally writes full retrieval results to JSON with `--output-path`
- supports metadata filters such as `--content-type movie` or `--source-name guardian`

### `scripts/retrieval_eval.py`

Evaluates retrieval quality against the hand-labeled golden set in `data/eval/retrieval_golden_set.jsonl`.

It computes:

- `MRR`
- `recall@k`
- `precision@k`
- missing relevant chunk IDs, which usually indicate corpus or ChromaDB drift

By default, it writes a detailed JSON report under `data/reports/retrieval_eval/`.

### `scripts/build_prompt.py`

Builds a grounded RAG prompt from retrieved chunks.

It has two modes:

- offline mode: reads JSON previously written by `retrieve.py --output-path`
- live mode: retrieves from ChromaDB and builds the prompt in one command

The prompt builder outputs a system prompt, user prompt, included and excluded chunk IDs, and context size. Use `--max-context-chars` to control the context budget and `--min-similarity` to drop weak matches before prompt assembly.

### `scripts/chat.py`

Starts the unified chatbot that routes each question through the SQL, RAG, or hybrid pipeline.

It:

- uses MotherDuck SQL as the source of truth for ratings, rankings, cast, attention, and other structured facts
- uses ChromaDB retrieval for textual evidence from `document_chunks`
- uses compact hybrid prompts so SQL facts and RAG evidence stay separate
- answers with the Groq `llama-3.1-8b-instant` model through `src/cultural_mood_tracker/rag/llm.py`

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

Set `RAG_DATA_SOURCE=local` to embed from the local processed JSONL file. Set `RAG_DATA_SOURCE=motherduck` to embed from the MotherDuck `document_chunks` table with `MOTHERDUCK_TOKEN` and `MOTHERDUCK_DATABASE`.

The embedding output can be split directly into ChromaDB `ids`, `documents`, `metadatas`, and `embeddings` for collection `add` or `upsert`.

### 5. Ingest Embeddings Into ChromaDB

```powershell
python .\scripts\ingest_chroma.py --input-path data\processed\<process_run_id>\document_chunk_embeddings.jsonl
```

By default, this writes a persistent ChromaDB database under `chroma_db/` and uses collection `movie_chunks`.

### 6. Retrieve Relevant Chunks

```powershell
python .\scripts\retrieve.py --query "a whistleblower exposes a corporate cover-up about extraterrestrial life" --top-k 5
```

Useful filters:

```powershell
python .\scripts\retrieve.py --query "critical reviews about a dystopian TV show" --top-k 5 --content-type tv
python .\scripts\retrieve.py --query "press coverage for a popular movie" --top-k 5 --source-name guardian
```

To save retrieval results for offline prompt iteration:

```powershell
python .\scripts\retrieve.py --query "Disclosure Day" --top-k 5 --output-path data\eval\_scratch_retrieve.json
```

### 7. Evaluate Retrieval

```powershell
python .\scripts\retrieval_eval.py
```

To test specific `k` values or save the report to a known path:

```powershell
python .\scripts\retrieval_eval.py --k 1 3 5 10 --output-path data\reports\retrieval_eval\latest.json
```

### 8. Build A Grounded Prompt

Offline mode is recommended first because it proves prompt construction independently of ChromaDB and the embedding model:

```powershell
python .\scripts\build_prompt.py --input-path data\eval\_scratch_retrieve.json
```

Live mode chains retrieval and prompt construction in one call:

```powershell
python .\scripts\build_prompt.py --query "a whistleblower exposes a corporate cover-up about extraterrestrial life" --top-k 5
```

To keep only stronger matches and write the complete prompt payload:

```powershell
python .\scripts\build_prompt.py --query "a love story with public attention signals" --top-k 8 --min-similarity 0.3 --output-path data\eval\_scratch_prompt.json
```

### 9. Start The Unified Chatbot

Before starting the chatbot, make sure:

- `MOTHERDUCK_TOKEN` and `MOTHERDUCK_DATABASE` are set in `.env`
- ChromaDB has been populated under `chroma_db/`
- `pip install -r .\requirements.txt` has completed

Start an interactive session:

```powershell
python .\scripts\chat.py --top-k 8 --max-context-chars 1200
```

Then type questions directly into the prompt. Use `exit`, `quit`, or `:q` to close the session.

Single-question mode:

```powershell
python .\scripts\chat.py --query "Why is Obsession popular?" --json
```

Routing behavior:

- rating, ranking, cast, attention, and trend facts use MotherDuck SQL
- descriptions, summaries, and review interpretation use RAG
- comparison and popularity questions use hybrid SQL + RAG

Set `GROQ_API_KEY` in `.env` before starting the chatbot. Generation is handled by Groq, so no local LLM model is loaded into memory.

### 10. Start The Streamlit Chat UI

A web chat frontend (`app.py`) is available on top of the same orchestrator used by `scripts/chat.py`. It requires the same environment setup (`MOTHERDUCK_TOKEN`, `MOTHERDUCK_DATABASE`, `GROQ_API_KEY`, a populated `chroma_db/`).

```powershell
streamlit run app.py
```

It shows the retrieval mode (SQL / RAG / hybrid / recommendation), response time, the SQL statements used, and retrieved RAG chunks for each answer, and keeps chat history in the browser session. The terminal chatbot (`scripts/chat.py`) is unaffected and continues to work exactly as before.

### 10. Local End-To-End Pipeline

```powershell
python .\scripts\run_pipeline.py --skip-motherduck-load
```

### 11. Publish To MotherDuck

```powershell
python .\scripts\load_to_motherduck.py --process-run-id <process_run_id>
```

### 12. Full ETL + MotherDuck In One Command

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
data/eval/_scratch_retrieve.json
data/eval/_scratch_prompt.json
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
data/reports/retrieval_eval/<run_id>.json
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
