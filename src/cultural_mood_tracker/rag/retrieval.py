from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from .embeddings import DEFAULT_EMBEDDING_MODEL


LOGGER = logging.getLogger(__name__)

# BGE models are trained asymmetrically: queries are encoded with an instruction
# prefix, passages are not. embed_document_chunks_file() (embeddings.py) embeds
# passages with no prefix, so this must stay in sync with that or retrieval
# quality degrades silently (no error, just worse rankings).
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    chunk_text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        # Collection was created with metadata={"hnsw:space": "cosine"} (chroma_ingest.py),
        # so Chroma reports distance = 1 - cosine_similarity. Lower distance = more similar.
        return 1.0 - self.distance


def _disable_chroma_telemetry_noise() -> None:
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    try:
        import posthog
    except ImportError:
        return
    posthog.disabled = True


def _load_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install sentence-transformers before running retrieval."
        ) from exc

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_query(
    query: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    normalize_embeddings: bool = True,
) -> list[float]:
    """Embed a single query string with the same model/normalization used at ingest time."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    model = _load_model(model_name)
    text = f"{query_instruction}{query.strip()}" if query_instruction else query.strip()
    embedding = model.encode(
        [text],
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )[0]
    return embedding.tolist()


def _open_collection(persist_dir: Path, collection_name: str) -> Any:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install chromadb before running retrieval.") from exc

    if not persist_dir.exists():
        raise RuntimeError(
            f"ChromaDB persist directory does not exist: {persist_dir}. Run ingest_chroma.py first."
        )

    _disable_chroma_telemetry_noise()
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{collection_name}' not found under {persist_dir}. Run ingest_chroma.py first."
        ) from exc


def query_collection(
    query: str,
    *,
    persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    normalize_embeddings: bool = True,
) -> list[RetrievedChunk]:
    """Embed `query` and return the top_k most similar chunks from a persisted ChromaDB collection."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    collection = _open_collection(persist_dir, collection_name)
    query_embedding = embed_query(
        query,
        model_name=model_name,
        query_instruction=query_instruction,
        normalize_embeddings=normalize_embeddings,
    )

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                chunk_text=document,
                metadata=metadata or {},
                distance=float(distance),
            )
        )

    LOGGER.info("Retrieved %s chunk(s) for query from %s/%s", len(chunks), persist_dir, collection_name)
    return chunks
