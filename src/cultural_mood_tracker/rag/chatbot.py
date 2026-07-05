from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from .embeddings import DEFAULT_EMBEDDING_MODEL
from .llm import DEFAULT_MODEL, generate_answer
from .prompting import DEFAULT_MAX_CONTEXT_CHARS, PromptResult, build_prompt
from .retrieval import DEFAULT_QUERY_INSTRUCTION, RetrievedChunk, query_collection


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    similarity: float
    title: Any
    source: Any
    document_type: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChatbotResponse:
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    retrieved_chunk_ids: list[str]
    similarity_scores: list[float]
    evidence: list[EvidenceItem]
    prompt: PromptResult


@dataclass(frozen=True)
class RetrievalRoute:
    """Routing seam for future SQL/MotherDuck or specialized prompt paths."""

    where: dict[str, Any] | None = None


class Chatbot:
    def __init__(
        self,
        *,
        persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_MODEL,
        top_k: int = 5,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        min_similarity: float | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.min_similarity = min_similarity
        self.query_instruction = query_instruction

    def answer_question(self, question: str, *, route: RetrievalRoute | None = None) -> ChatbotResponse:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        route = route or RetrievalRoute()
        chunks = query_collection(
            question,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            model_name=self.embedding_model_name,
            top_k=self.top_k,
            where=route.where,
            query_instruction=self.query_instruction,
        )
        prompt = build_prompt(
            question,
            chunks,
            max_context_chars=self.max_context_chars,
            min_similarity=self.min_similarity,
        )
        answer = generate_answer(prompt, model_name=self.llm_model_name)
        return ChatbotResponse(
            answer=answer,
            retrieved_chunks=chunks,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            similarity_scores=[chunk.similarity for chunk in chunks],
            evidence=[_evidence_item(chunk) for chunk in chunks],
            prompt=prompt,
        )


def _evidence_item(chunk: RetrievedChunk) -> EvidenceItem:
    metadata = chunk.metadata
    return EvidenceItem(
        chunk_id=chunk.chunk_id,
        similarity=chunk.similarity,
        title=metadata.get("title_name"),
        source=metadata.get("source_name"),
        document_type=metadata.get("document_type"),
        metadata=metadata,
    )


def answer_question(
    question: str,
    *,
    persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    min_similarity: float | None = None,
    where: dict[str, Any] | None = None,
) -> ChatbotResponse:
    chatbot = Chatbot(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        top_k=top_k,
        max_context_chars=max_context_chars,
        min_similarity=min_similarity,
    )
    return chatbot.answer_question(question, route=RetrievalRoute(where=where))
