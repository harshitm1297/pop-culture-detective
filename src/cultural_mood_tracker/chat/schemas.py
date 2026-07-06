from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from cultural_mood_tracker.rag.retrieval import RetrievedChunk


ChatMode = Literal["fast_sql", "rag", "sql", "hybrid", "recommendation"]


class ChatResponse(TypedDict):
    answer: str
    mode: ChatMode
    used_sql: bool
    retrieved_chunk_ids: list[str]


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: str
    similarity: float
    title: Any
    source: Any
    document_type: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OrchestratorResult:
    answer: str
    mode: ChatMode
    used_sql: bool
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    sql_results: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self.retrieved_chunks]

    @property
    def evidence(self) -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                similarity=chunk.similarity,
                title=chunk.metadata.get("title_name"),
                source=chunk.metadata.get("source_name"),
                document_type=chunk.metadata.get("document_type"),
                metadata=chunk.metadata,
            )
            for chunk in self.retrieved_chunks
        ]

    def to_response(self) -> ChatResponse:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "used_sql": self.used_sql,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
        }
