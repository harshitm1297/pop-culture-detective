# RAG Learning Guide — Cultural Mood Tracker

This document is a conceptual reference for the Retrieval-Augmented Generation (RAG) system being
built on top of the existing ETL. It explains *why* each piece exists, not just what the API calls
do. It is written against the real state of this repository, so the examples use actual data you
can go inspect yourself (e.g. `chroma_db/` currently holds 2,407 embedded chunks in the
`movie_chunks` collection, built from `BAAI/bge-small-en-v1.5`, 384 dimensions).

Pair this with the incremental, phase-by-phase build in chat — this doc is the "why", the chat
walkthroughs are the "how, one script at a time."

---

## 1. The Two Systems In This Repo

There are two pipelines living in the same repository, and it's worth being explicit about where
one ends and the other begins.

**ETL (already built, owned by the team):**

```
extract  -->  transform (canonical)  -->  chunk  -->  embed  -->  ingest into ChromaDB  -->  (optional) GCP
```

Concretely, in file terms:

```
scripts/extract_multisource_aligned.py   -> data/raw/<source>/<run_id>/*.json
scripts/transform_canonical.py           -> data/processed/<run_id>/{titles,documents,document_chunks}.jsonl
scripts/embed_document_chunks.py         -> data/processed/<run_id>/document_chunk_embeddings.jsonl
scripts/ingest_chroma.py                 -> chroma_db/  (persistent vector store, collection "movie_chunks")
scripts/load_to_gcp.py                   -> GCS + BigQuery (optional, disabled by default)
```

Everything up to and including `ingest_chroma.py` is the **RAG data pipeline** — it prepares a
searchable knowledge base. It runs offline, in batches, on a schedule or on demand. It does not
know that an LLM exists.

**RAG (what we're building now):**

```
user query --> embed query --> vector search --> retrieved chunks --> prompt --> LLM --> answer
```

This is the **online / serving** path. It runs per-request, in real time, and it is what actually
reads from the database the ETL built. The ETL and the RAG serving path share exactly one contract:
the ChromaDB collection. As long as that collection's schema (id, document text, metadata, 384-dim
vector) stays stable, the two sides can evolve independently — which is exactly why your teammates
can keep working on the ETL while you build retrieval on top of it.

```
        OFFLINE (batch)                          ONLINE (per request)
  ┌─────────────────────────┐              ┌───────────────────────────┐
  │ extract → transform →   │              │ query → embed → search →  │
  │ chunk → embed → ingest  │──chroma_db──▶│ retrieve → prompt → LLM   │
  └─────────────────────────┘   (shared)   └───────────────────────────┘
```

---

## 2. Embeddings: How Text Becomes a Vector

An embedding model (here, `SentenceTransformer("BAAI/bge-small-en-v1.5")`) maps a piece of text to
a fixed-length vector of numbers — in this project, 384 floats. The model was trained so that texts
with similar *meaning* end up as vectors that point in similar *directions*, regardless of exact
wording.

That's the whole trick: instead of matching keywords, you're comparing meaning as geometry.
"A cybersecurity expert becomes a whistleblower..." and "a hacker exposes a corporate cover-up" can
be lexically almost disjoint but land close together in vector space, because the model was trained
on millions of examples of semantically related sentence pairs (contrastive learning: pull similar
pairs together, push dissimilar pairs apart in vector space).

Two properties matter a lot in practice:

- **The vector space is only meaningful relative to itself.** You cannot compare a vector produced
  by `bge-small-en-v1.5` to one produced by a different model — they don't share a coordinate
  system. This is why the exact same model name must be used to embed the corpus (ingestion) and
  the query (retrieval). This repo pins `DEFAULT_EMBEDDING_MODEL` in `rag/embeddings.py` for
  exactly this reason.
- **Normalization matters for cosine similarity.** `embed_document_chunks_file` calls
  `model.encode(..., normalize_embeddings=True)`, which scales every vector to unit length. Once
  vectors are unit length, cosine similarity (the angle between two vectors) becomes equivalent to
  a simple dot product, which is fast and numerically stable. ChromaDB's collection was created
  with `metadata={"hnsw:space": "cosine"}` (see `rag/chroma_ingest.py`), so the index itself expects
  this.

Why `bge-small-en-v1.5` specifically? It's a small (~130MB), fast, CPU-friendly bi-encoder tuned for
retrieval tasks, with a well-documented convention: **passages are encoded as-is, but queries are
encoded with an instruction prefix** ("Represent this sentence for searching relevant passages: ").
This is a deliberate asymmetry the model was trained with — passages and queries are different
kinds of text (a search intent vs. a statement of fact) and the model performs better when told
which one it's looking at. This is a common silent failure point: forgetting the prefix doesn't
crash anything, it just quietly makes retrieval worse. Phase 1's retrieval code applies this prefix
by default.

---

## 3. Why Chunking Is Necessary

Two unrelated constraints both push toward chunking:

1. **Embedding models have a limited effective context.** A whole Guardian review or Wikipedia
   article summarized into one embedding vector loses resolution — the vector becomes an "average
   of everything," which is bad at representing any one specific fact well. Smaller, focused chunks
   embed more precisely.
2. **LLM prompts have a token budget.** Even if you *could* embed a 5,000-word article as one
   vector, you can't dump the whole article into every prompt. You need retrieval to return
   small, relevant units you can afford to include.

`transform/chunks.py` in this repo implements a sliding window: `target_words=180`,
`overlap_words=40`. The overlap exists so that a sentence sitting right at a chunk boundary isn't
orphaned — the next chunk repeats the tail of the previous one, so surrounding context survives the
split. Chunk size is a real trade-off, not a solved problem:

- **Too large** → each chunk covers many topics, its embedding is a blurry average, and precision
  drops (you retrieve a chunk that's "sort of" relevant to five different things).
- **Too small** → you lose surrounding context, and you multiply the number of vectors (more index
  size, more chunks to re-rank, more chance of retrieving fragments that are true but misleading in
  isolation).

180 words with 40-word overlap is a reasonable default for short-form review/summary text like this
project's documents. It's worth knowing it's tunable, and worth knowing *why* you'd tune it, but you
don't need to touch it for Phase 1.

---

## 4. Vector Search: How "Find Similar" Actually Works

Given a query vector, the naive approach is brute force: compute the cosine similarity between the
query and *every* stored vector, sort, take the top-k. This is exact, and for 2,407 vectors (this
repo's current size) it would even be fast enough. It stops being fast enough at hundreds of
thousands or millions of vectors — an O(n) scan doesn't scale.

Production vector databases use **Approximate Nearest Neighbor (ANN)** indexes instead. ChromaDB
uses **HNSW** (Hierarchical Navigable Small World) — a graph structure where each vector is a node,
connected to its approximate nearest neighbors, organized into layers (like a skip list). A search
starts at a coarse top layer and greedily walks toward the query, descending layers, narrowing in
on a small candidate set instead of touching every point. It trades a small amount of recall
(you might miss the *literal* single best match) for a massive speedup (logarithmic-ish instead of
linear). This is a universal pattern across vector databases (Pinecone, Weaviate, pgvector's HNSW
mode, Vertex AI Vector Search) — the concept transfers even if the implementation doesn't.

---

## 5. What ChromaDB Actually Is

`chromadb.PersistentClient(path="./chroma_db")` is an **embedded** database — it runs in-process,
no server, no network call, backed by files on disk. If you look inside `chroma_db/` in this repo
you'll find:

```
chroma_db/
  chroma.sqlite3                         <- metadata: collections, documents, filterable fields
  <uuid>/                                <- one directory per HNSW index segment
    data_level0.bin, header.bin, ...     <- the actual HNSW graph + raw vectors
```

SQLite stores the "relational" parts — collection names, document text, and metadata key/value
pairs (the stuff you filter on with `where={...}`). The HNSW segment files store the vectors and the
graph structure used for similarity search. A `collection` in Chroma is essentially a named,
independent HNSW index plus its associated metadata table — this repo uses one collection,
`movie_chunks`.

**Why a vector database instead of the relational database you already have (or BigQuery)?**
Relational engines index for *exact* or *range* lookups — B-trees are extremely good at "give me
rows where `release_year = 2026`" but have no efficient way to answer "give me the 5 rows whose
384-dimensional vector is closest to this one" — that's not an operation B-trees are built for, and
without an ANN index it degrades to a full table scan with a distance calculation per row.

You don't have to fully leave the relational world to get this, though — Postgres has a `pgvector`
extension that adds HNSW/IVF indexes on top of a normal table, and GCP's `AlloyDB` supports it. This
is directly relevant to this project's eventual cloud story: local Chroma today, `pgvector`
(AlloyDB) or `Vertex AI Vector Search` in production later, same underlying concept.

---

## 6. Retrieval: The Mechanism

Once the corpus is embedded and indexed, retrieval is:

```
query text
  │
  ▼ embed_query() — same model, query-instruction prefix, same normalization as ingestion
query vector (384 floats)
  │
  ▼ collection.query(query_embeddings=[...], n_results=k, where=filters)
top-k (id, document text, metadata, distance)
```

`distance` in a cosine-space collection is `1 - cosine_similarity`, so **lower distance = more
similar**. It's easy to get this backwards, so Phase 1's code exposes both `distance` and a derived
`similarity` so you can sanity-check results either way.

The `where` filter is metadata-based, exact-match filtering that runs *alongside* the vector search
(e.g. only search chunks where `content_type = "movie"`). This is cheap because it's backed by the
SQLite metadata table, and it's how you'd implement things like "only search reviews, not
overviews" or "only this year's titles" without touching the vector math at all.

---

## 7. Prompt Construction (Preview — built in a later phase)

Once you have retrieved chunks, prompt construction is the (surprisingly unglamorous) work of
turning `(user question, [chunk1, chunk2, ...])` into a single string handed to the LLM. The core
ideas, so you recognize them when we build this phase:

- **Grounding instruction**: tell the model explicitly to answer *using only* the retrieved context,
  and to say so if the context doesn't contain the answer. This is what separates RAG from "the
  model just knows things" — the retrieved text is the model's evidence, not a suggestion.
- **Context assembly**: concatenate chunks with light structure (e.g. `[Source: Guardian, 2026-04-02]
  ...text...`) so the model can cite or reason about provenance, and so near-duplicate chunks from
  different sources don't blur together.
- **Token budget management**: chunks + question + instructions must fit in the LLM's context
  window, with room left for the answer. This means `top_k` in retrieval and prompt construction are
  coupled — retrieving 20 chunks is pointless if you can only afford to include 5.
- **Empty-result handling**: what does the prompt say when retrieval finds nothing relevant? This is
  a real design decision, not an edge case — it's how you avoid the model hallucinating an answer
  from its own training data when your corpus genuinely doesn't cover the question.

---

## 8. Why the Embedding Model and the LLM Are Different Models

It's a fair question — why not use one model for everything? Because they optimize for different
objectives:

| | Embedding model (e.g. `bge-small-en-v1.5`) | LLM (e.g. local Llama/Mistral/Qwen) |
|---|---|---|
| Objective | Map text to a vector such that similar meaning → similar vector | Predict the next token, conditioned on everything before it |
| Architecture | Bi-encoder, often just an encoder stack | Decoder-only, autoregressive |
| Output | One fixed-length vector | A sequence of generated tokens |
| Size | Small (~100–400M params), CPU-friendly | Large (7B+ typical), usually wants a GPU |
| Used at | Both ingestion time and query time | Only at generation time |

This is the classic **retriever / reader** (or retriever / generator) split from the original RAG
literature (Lewis et al., 2020). The retriever's job is narrow and well-defined enough that a small,
cheap, frozen model does it well and *fast* — you need it to run on every ingested chunk and every
incoming query, so speed and cost matter enormously. The generator's job (synthesizing a fluent,
reasoned answer) genuinely needs the scale of a full LLM. Trying to make one model do both would
mean paying LLM-sized compute costs just to index your data, for no quality benefit on the retrieval
side.

---

## 9. How the Local LLM Talks to the Retrieval Pipeline

There's no framework magic here worth hiding: the "RAG orchestrator" is just a plain Python function
that calls retrieval, then calls generation, in order:

```python
def answer(question: str) -> str:
    chunks = query_collection(question, top_k=5)      # Phase 1
    prompt = build_prompt(question, chunks)             # Phase 3
    return generate(prompt)                              # Phase 4/5
```

Frameworks like LangChain or LlamaIndex wrap this same three-step shape in abstractions (chains,
retrievers, prompt templates). Building it by hand first means that when you eventually reach for
one of those frameworks, you'll recognize exactly what it's doing under the hood instead of trusting
a black box.

The local LLM itself will run either via a Python binding (e.g. `llama-cpp-python`,
`transformers` + a quantized model) or via a local server (e.g. **Ollama**, which exposes an
OpenAI-compatible HTTP API on `localhost`). Either way, from the orchestrator's point of view it's
just a function: `text in, text out`. This is a deliberate design choice worth keeping — if
`generate()` is a clean function boundary, swapping the local LLM for a hosted one later (Vertex AI,
Anthropic API, etc.) is a one-function change, not a rewrite.

---

## 10. Evaluating Retrieval (Preview — Phase 2)

Retrieval evaluation is **information-retrieval evaluation**, and it's independent of whether the
LLM is any good. You need a small labeled set: `(query, [relevant chunk_ids])`, curated by hand (30
queries is a fine starting size). Standard metrics:

- **Recall@k** — of the chunks you know are relevant, what fraction appear in the top-k results?
  Measures whether you're finding the right evidence at all.
- **Precision@k** — of the top-k results, what fraction are actually relevant? Measures how much
  noise you're feeding to the prompt.
- **MRR (Mean Reciprocal Rank)** — how high up the *first* relevant result ranks, averaged over
  queries. Useful when you mostly care about getting one good chunk near the top.
- **nDCG (normalized Discounted Cumulative Gain)** — like recall, but rewards relevant results
  appearing *earlier* in the ranking, with graded (not just binary) relevance if you have it.

The reason this is its own phase, before touching the LLM: if retrieval recall is bad, no amount of
prompt engineering or LLM quality fixes the answer — the evidence simply isn't there. You want to
be able to say "retrieval finds the right chunk 85% of the time" as a number you can track over
time, independent of generation.

---

## 11. Evaluating The Full RAG System (Preview — Phase 6)

Generation-level evaluation asks different questions than retrieval evaluation:

- **Faithfulness / groundedness** — does the generated answer only state things supported by the
  retrieved context, or did the model add unsupported claims (hallucination)? Typically checked with
  an LLM-as-judge prompt: "given this context and this answer, is every claim in the answer
  supported by the context?"
- **Answer relevance** — does the answer actually address the question asked (a faithful answer can
  still dodge the question)?
- **Context precision/recall** (as used by frameworks like RAGAS) — evaluates the *interaction*
  between retrieval and generation: did the generator make good use of the context it was given?

In practice this combines automatic LLM-as-judge scoring (cheap, fast, run often) with periodic
human spot-checks (slower, catches things automatic judges miss, especially domain-specific
correctness). Neither replaces the other.

---

## 12. From Local Prototype to Production (GCP)

None of this needs to happen now — it's here so the local design choices in front of you map
cleanly onto their production equivalents later, and so you can see why certain interfaces (a
`query_collection()` function, a `generate()` function) are worth keeping clean even locally.

| Local (today) | Production (GCP) | Why the mapping works |
|---|---|---|
| ChromaDB (`PersistentClient`, HNSW, single file on disk) | `Vertex AI Vector Search`, or `AlloyDB`/`Cloud SQL` + `pgvector` | Same ANN concept, managed, horizontally scalable, multi-writer |
| `SentenceTransformer` running in-process | Same model served behind `Vertex AI Prediction` / `Cloud Run` with GPU, or `Vertex AI` text-embedding API | Same bi-encoder concept, decoupled into its own scalable service |
| Local LLM (`Ollama` / `llama-cpp-python`) | `Vertex AI` (Gemini) or self-hosted via `vLLM` on `GKE` with GPUs | Same generate(text) → text contract, swapped implementation |
| Ad hoc Python scripts run by hand | `Cloud Run` job / `Cloud Functions` behind an API, triggered by `Cloud Scheduler` or `Pub/Sub` | Same ETL/RAG stages, orchestrated instead of manually invoked |
| Eval scripts printing to terminal | Scheduled eval job writing results into `BigQuery` (the dataset this repo's `load/bigquery.py` already knows how to write to) | Turns evaluation into a trend you can chart over time, not a one-off check |

The existing `load_to_gcp.py` / `load/gcs.py` / `load/bigquery.py` already show the team's chosen
pattern for cloud publishing (service account, bucket-per-layer, dataset-per-project) — when this
project does move to GCP, retrieval and evaluation outputs would follow that same established
pattern rather than inventing a new one.

---

## 13. Phase Roadmap At A Glance

| Phase | Purpose | Primary new files (planned) |
|---|---|---|
| 1. Retrieval | Turn a query into ranked, relevant chunks from ChromaDB | `rag/retrieval.py`, `cli/retrieve.py`, `scripts/retrieve.py` |
| 2. Retrieval evaluation | Measure recall/precision/MRR against a labeled query set | `rag/retrieval_eval.py`, a small hand-curated eval dataset file |
| 3. Prompt construction | Assemble grounded prompts from retrieved chunks | `rag/prompting.py` |
| 4. Local LLM | Wire up a local model as a `generate(text) -> text` function | `rag/generation.py` |
| 5. Chat inference | Single-turn Q&A, then multi-turn conversation state | `cli/chat.py`, `scripts/chat.py` |
| 6. Full RAG evaluation | Faithfulness / relevance scoring end to end | `rag/rag_eval.py` |
| 7. Deployment / cloud | Only after 1–6 work and are measured locally | (deferred) |

Each phase gets its own theory walkthrough and its own independently-runnable, independently-
testable script before the next one starts — see chat for the live, incremental build.

---

## 14. Glossary

- **Bi-encoder**: an embedding architecture that encodes query and document independently into
  vectors, compared afterward (fast, scalable — what this project uses). Contrast with a
  **cross-encoder**, which encodes query+document *together* for a more accurate but much slower
  relevance score (often used as a second-pass re-ranker, not for initial retrieval over a whole
  corpus).
- **ANN (Approximate Nearest Neighbor)**: search that trades a small amount of accuracy for large
  speed gains over exact brute-force search. HNSW is one ANN algorithm; there are others (IVF, PQ).
- **Cosine similarity**: the cosine of the angle between two vectors; measures directional
  similarity independent of magnitude. Ranges from -1 to 1; with normalized embeddings, values are
  typically close to 0–1 for related text.
- **Grounding**: constraining an LLM's answer to be based on supplied context rather than its
  parametric (training-time) knowledge.
- **Hallucination**: an LLM stating something not supported by its given context (or by fact).
- **RAG (Retrieval-Augmented Generation)**: retrieving relevant external documents at query time and
  supplying them to an LLM as context, so the model can answer using up-to-date or private
  information it wasn't trained on.
