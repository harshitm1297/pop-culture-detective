"""RAG preparation utilities."""

from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    build_chroma_embedding_records,
    embed_document_chunks,
    embed_document_chunks_file,
)
from .document_chunks import load_document_chunks
from .chroma_ingest import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_DIR,
    ingest_embeddings_file,
)
from .chatbot import Chatbot, ChatbotResponse, EvidenceItem, RetrievalRoute, answer_question
from .llm import DEFAULT_MODEL, generate_answer

__all__ = [
    "Chatbot",
    "ChatbotResponse",
    "DEFAULT_CHROMA_COLLECTION",
    "DEFAULT_CHROMA_DB_DIR",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL",
    "EvidenceItem",
    "RetrievalRoute",
    "answer_question",
    "build_chroma_embedding_records",
    "embed_document_chunks",
    "embed_document_chunks_file",
    "generate_answer",
    "ingest_embeddings_file",
    "load_document_chunks",
]
