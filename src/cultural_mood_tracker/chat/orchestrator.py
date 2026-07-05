from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL, generate_answer
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS, PromptResult, build_prompt
from cultural_mood_tracker.rag.retrieval import RetrievedChunk, query_collection

from .router import route_query
from .schemas import ChatMode, ChatResponse, OrchestratorResult
from .sql_client import MotherDuckClient
from .sql_schemas import validate_sql_payload
from .prompt_compression import MAX_CONTEXT_CHARS, build_hybrid_user_prompt, estimate_tokens, format_sql_block
from .retrieval_rerank import chunk_type_distribution, filter_and_rerank_chunks


HYBRID_SYSTEM_PROMPT = (
    "You are a cultural intelligence assistant. Use ONLY provided structured data and evidence."
)

SQL_SYSTEM_PROMPT = (
    "You are a cultural intelligence assistant. Answer using only the structured SQL data "
    "provided. Do not invent numerical facts or perform unsupported calculations."
)

LOGGER = logging.getLogger(__name__)


class ChatOrchestrator:
    def __init__(
        self,
        *,
        persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_MODEL,
        top_k: int = 8,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        sql_client: MotherDuckClient | None = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.sql_client = sql_client or MotherDuckClient()

    def answer(self, query: str) -> OrchestratorResult:
        mode = route_query(query)
        if mode == "sql":
            return self._answer_sql(query)
        if mode == "hybrid":
            return self._answer_hybrid(query)
        return self._answer_rag(query)

    def _retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return query_collection(
            query,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            model_name=self.embedding_model_name,
            top_k=top_k or self.top_k,
        )

    def _answer_rag(self, query: str) -> OrchestratorResult:
        candidates = self._retrieve(query, top_k=min(max(self.top_k, 4), 8))
        chunks = filter_and_rerank_chunks(candidates)
        LOGGER.info("retrieval_count_before=%s", len(candidates))
        LOGGER.info("retrieval_count_after=%s", len(chunks))
        prompt = build_prompt(query, chunks, max_context_chars=self.max_context_chars)
        answer = generate_answer(prompt, model_name=self.llm_model_name)
        return OrchestratorResult(
            answer=answer,
            mode="rag",
            used_sql=False,
            retrieved_chunks=chunks,
        )

    def _answer_sql(self, query: str) -> OrchestratorResult:
        structured_data = self.sql_client.run_structured_query(query)
        LOGGER.info("sql_rows_returned=%s", len(structured_data.get("results", [])))
        LOGGER.info("sql_rows_used=%s", len(structured_data.get("results", [])))
        prompt = _build_sql_prompt(query=query, structured_data=structured_data)
        LOGGER.info("prompt_token_estimate=%s", estimate_tokens(prompt.system_prompt + prompt.user_prompt))
        answer = generate_answer(prompt, model_name=self.llm_model_name)
        return OrchestratorResult(
            answer=answer,
            mode="sql",
            used_sql=True,
            sql_results=structured_data,
        )

    def _answer_hybrid(self, query: str) -> OrchestratorResult:
        LOGGER.info("hybrid_mode_triggered=true")
        structured_data = self.sql_client.run_structured_query(query)
        LOGGER.info("sql_rows_returned=%s", len(structured_data.get("results", [])))
        LOGGER.info("sql_rows_used=%s", len(structured_data.get("results", [])))
        candidates = self._retrieve(query, top_k=min(max(self.top_k, 4), 8))
        chunks = filter_and_rerank_chunks(candidates)
        LOGGER.info("retrieval_count_before=%s", len(candidates))
        LOGGER.info("retrieval_count_after=%s", len(chunks))
        prompt = _build_hybrid_prompt(query=query, structured_data=structured_data, chunks=chunks)
        answer = generate_answer(prompt, model_name=self.llm_model_name)
        return OrchestratorResult(
            answer=answer,
            mode="hybrid",
            used_sql=True,
            retrieved_chunks=chunks,
            sql_results=structured_data,
        )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _build_sql_prompt(*, query: str, structured_data: dict[str, Any]) -> PromptResult:
    validate_sql_payload(structured_data)
    user_prompt = (
        f"STRUCTURED DATA (JSON):\n{_json_dumps(structured_data)}\n\n"
        f"QUESTION:\n{query}\n\n"
        "Answer using the structured data only. If the data does not contain the answer, say so."
    )
    return PromptResult(
        system_prompt=SQL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        included_chunk_ids=[],
        excluded_chunk_ids=[],
        context_char_count=len(user_prompt),
    )


def _build_hybrid_prompt(*, query: str, structured_data: dict[str, Any], chunks: list[RetrievedChunk]) -> PromptResult:
    validate_sql_payload(structured_data)
    _ = format_sql_block(structured_data)
    user_prompt, rag_chunks_used, token_estimate = build_hybrid_user_prompt(
        sql_data=structured_data,
        chunks=chunks[:5],
        question=query,
    )
    LOGGER.info("number_of_rag_chunks_used=%s", rag_chunks_used)
    LOGGER.info("chunk_type_distribution=%s", chunk_type_distribution(chunks[:rag_chunks_used]))
    LOGGER.info("final_prompt_size=%s", len(HYBRID_SYSTEM_PROMPT) + len(user_prompt))
    LOGGER.info("prompt_token_estimate=%s", token_estimate)
    return PromptResult(
        system_prompt=HYBRID_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        included_chunk_ids=[chunk.chunk_id for chunk in chunks[:rag_chunks_used]],
        excluded_chunk_ids=[],
        context_char_count=len(user_prompt),
    )


def answer_question(query: str) -> ChatResponse:
    return ChatOrchestrator().answer(query).to_response()
