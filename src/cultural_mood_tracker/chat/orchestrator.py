from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL, generate_answer
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS, PromptResult, build_prompt
from cultural_mood_tracker.rag.retrieval import RetrievedChunk, query_collection

from .router import route_query
from .schemas import ChatMode, ChatResponse, OrchestratorResult
from .sql_client import MotherDuckClient
from .sql_schemas import normalize_sql_output, validate_sql_payload
from .prompt_compression import MAX_CONTEXT_CHARS, build_hybrid_user_prompt, estimate_tokens, format_sql_block
from .retrieval_rerank import chunk_type_distribution, filter_and_rerank_chunks


HYBRID_SYSTEM_PROMPT = (
    "You are an evidence-grounded cultural intelligence assistant.\n\n"
    "Use the structured metrics as factual truth.\n"
    "Use analytical summaries to explain patterns.\n"
    "Use review excerpts only as supporting evidence.\n"
    "Never invent statistics.\n"
    "If structured metrics are missing, explicitly state that."
)

SQL_SYSTEM_PROMPT = (
    "You are a cultural intelligence assistant. Answer using only the structured SQL data "
    "provided. Do not invent numerical facts or perform unsupported calculations."
)

RECOMMENDATION_SYSTEM_PROMPT = (
    "You are an evidence-grounded movie and TV recommendation assistant. "
    "Recommend only using the retrieved summaries. Do not invent information. "
    "Explain why each recommendation matches the requested mood. Mention the supporting "
    "evidence. If the evidence is weak, explicitly say so."
)

LOGGER = logging.getLogger(__name__)

RECO_TITLE_LIMIT = 3
RECO_GENRE_LIMIT = 2
RECO_AUDIENCE_LIMIT = 3
RECO_MAX_PROMPT_TOKENS = 1200
FAST_HYBRID_MAX_NEW_TOKENS = 220
RAG_MAX_NEW_TOKENS = 260
RECOMMENDATION_MAX_NEW_TOKENS = 384
HybridQueryType = Literal[
    "POPULARITY_EXPLANATION",
    "ATTENTION_VS_RECEPTION",
    "ATTENTION_VS_RATING",
    "TITLE_COMPARISON",
    "TREND_EXPLANATION",
]


class ChatOrchestrator:
    def __init__(
        self,
        *,
        persist_dir: Path = Path(DEFAULT_CHROMA_DB_DIR),
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_MODEL,
        top_k: int = 3,
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
        started_at = time.perf_counter()
        mode = route_query(query)
        LOGGER.info("mode=%s", mode.upper())
        try:
            if mode == "recommendation":
                return self.answer_recommendation(query)
            if mode == "fast_sql":
                return self._answer_fast_sql(query)
            if mode == "sql":
                return self._answer_sql(query)
            if mode == "hybrid":
                return self._answer_hybrid(query)
            return self._answer_rag(query)
        finally:
            LOGGER.info("total_latency=%.3fs", time.perf_counter() - started_at)

    def _retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return query_collection(
            query,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            model_name=self.embedding_model_name,
            top_k=top_k or self.top_k,
        )

    def _answer_rag(self, query: str) -> OrchestratorResult:
        if _looks_like_title_comparison(query):
            titles = self._comparison_titles(query)
            if len(titles) >= 2:
                return self._answer_comparison_rag(query, titles)

        candidates = self._retrieve(query, top_k=min(self.top_k, 3))
        chunks = filter_and_rerank_chunks(candidates)[:2]
        LOGGER.info("retrieval_count_before=%s", len(candidates))
        LOGGER.info("retrieval_count_after=%s", len(chunks))
        prompt = build_prompt(query, chunks, max_context_chars=self.max_context_chars)
        answer = self._generate(prompt, max_new_tokens=RAG_MAX_NEW_TOKENS)
        return OrchestratorResult(
            answer=answer,
            mode="rag",
            used_sql=False,
            retrieved_chunks=chunks,
        )

    def _answer_fast_sql(self, query: str) -> OrchestratorResult:
        LOGGER.info("skipped_embedding=true")
        LOGGER.info("skipped_rag=true")
        sql_started_at = time.perf_counter()
        structured_data = self.sql_client.run_structured_query(query)
        LOGGER.info("sql_latency=%.3fs", time.perf_counter() - sql_started_at)
        answer = _format_fast_sql_answer(structured_data)
        LOGGER.info("llm_latency=0.000s")
        return OrchestratorResult(
            answer=answer,
            mode="fast_sql",
            used_sql=True,
            sql_results=structured_data,
        )

    def _answer_sql(self, query: str) -> OrchestratorResult:
        LOGGER.info("skipped_embedding=true")
        LOGGER.info("skipped_rag=true")
        sql_started_at = time.perf_counter()
        structured_data = self.sql_client.run_structured_query(query)
        LOGGER.info("sql_latency=%.3fs", time.perf_counter() - sql_started_at)
        answer = _format_fast_sql_answer(structured_data)
        LOGGER.info("llm_latency=0.000s")
        return OrchestratorResult(
            answer=answer,
            mode="sql",
            used_sql=True,
            sql_results=structured_data,
        )

    def _answer_hybrid(self, query: str) -> OrchestratorResult:
        LOGGER.info("hybrid_mode_triggered=true")
        hybrid_query_type = _classify_hybrid_query(query)
        LOGGER.info("hybrid_query_type=%s", hybrid_query_type)
        if hybrid_query_type == "TITLE_COMPARISON":
            titles = self._comparison_titles(query)
            if len(titles) >= 2:
                return self._answer_comparison_hybrid(query, titles, hybrid_query_type=hybrid_query_type)

        title = self._single_title(query)
        sql_started_at = time.perf_counter()
        metrics = self.sql_client.get_title_metrics(title)
        analytics = self.sql_client.get_title_analytical_summaries(title)
        LOGGER.info("sql_latency=%.3fs", time.perf_counter() - sql_started_at)
        structured_data = _build_hybrid_analytics_payload(
            query_type=hybrid_query_type,
            metrics=metrics,
            analytics=analytics,
        )
        LOGGER.info("metrics_rows=%s", 1 if metrics else 0)
        LOGGER.info("analytics_rows=%s", _analytics_row_count(analytics))
        candidates = self._retrieve(metrics.get("title") or title, top_k=2)
        chunks = filter_and_rerank_chunks(candidates)[:2]
        LOGGER.info("review_chunks=%s", len(chunks))
        prompt = _build_hybrid_analytics_prompt(
            query=query,
            metrics=structured_data["metrics"],
            analytics=structured_data["analytics"],
            chunks=chunks,
        )
        answer = self._generate(prompt, max_new_tokens=FAST_HYBRID_MAX_NEW_TOKENS)
        return OrchestratorResult(
            answer=answer,
            mode="hybrid",
            used_sql=True,
            retrieved_chunks=chunks,
            sql_results=structured_data,
        )

    def _single_title(self, query: str) -> str:
        from .sql_client import extract_title

        direct_title = extract_title(query)
        if direct_title:
            return direct_title
        titles = self.sql_client.extract_titles(query)
        if titles:
            return titles[-1]
        return query

    def _comparison_titles(self, query: str) -> list[str]:
        try:
            titles = self.sql_client.extract_titles(query)
        except RuntimeError as exc:
            LOGGER.info("comparison_title_detection_skipped=%s", exc)
            return []
        if len(titles) < 2:
            return titles
        titles = titles[:2]
        title_a, title_b = titles[0], titles[1]
        LOGGER.info("comparison_mode=true")
        LOGGER.info("detected_titles=%s", len(titles))
        LOGGER.info("title_a=%s", title_a)
        LOGGER.info("title_b=%s", title_b)
        return titles

    def _retrieve_for_titles(self, titles: list[str]) -> dict[str, list[RetrievedChunk]]:
        per_title_k = 2
        grouped: dict[str, list[RetrievedChunk]] = {}
        seen_chunk_ids: set[str] = set()
        seen_document_ids: set[str] = set()

        for index, title in enumerate(titles):
            candidates = self._retrieve(title, top_k=per_title_k)
            ranked = filter_and_rerank_chunks(candidates)[:per_title_k]
            title_chunks: list[RetrievedChunk] = []
            for chunk in sorted(ranked, key=lambda item: item.similarity, reverse=True):
                document_id = str(chunk.metadata.get("document_id") or "")
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                if document_id and document_id in seen_document_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                if document_id:
                    seen_document_ids.add(document_id)
                title_chunks.append(chunk)
            grouped[title] = title_chunks
            if index == 0:
                LOGGER.info("retrieved_chunks_title_a=%s", len(title_chunks))
            elif index == 1:
                LOGGER.info("retrieved_chunks_title_b=%s", len(title_chunks))
            LOGGER.info("retrieved_chunks_%s=%s", _log_key(title), len(title_chunks))

        LOGGER.info("final_chunks=%s", sum(len(chunks) for chunks in grouped.values()))
        return grouped

    def _answer_comparison_rag(self, query: str, titles: list[str]) -> OrchestratorResult:
        grouped_chunks = self._retrieve_for_titles(titles)
        chunks = _flatten_grouped_chunks(grouped_chunks)
        prompt = _build_comparison_rag_prompt(
            query=query,
            grouped_chunks=grouped_chunks,
            max_context_chars=self.max_context_chars,
        )
        answer = self._generate(prompt, max_new_tokens=RAG_MAX_NEW_TOKENS)
        return OrchestratorResult(
            answer=answer,
            mode="rag",
            used_sql=False,
            retrieved_chunks=chunks,
        )

    def _answer_comparison_hybrid(
        self,
        query: str,
        titles: list[str],
        *,
        hybrid_query_type: HybridQueryType = "TITLE_COMPARISON",
    ) -> OrchestratorResult:
        sql_started_at = time.perf_counter()
        structured_data = self.sql_client.compare_titles(titles[0], titles[1])
        LOGGER.info("sql_latency=%.3fs", time.perf_counter() - sql_started_at)
        structured_data["hybrid_query_type"] = hybrid_query_type
        LOGGER.info("metrics_rows=%s", len(structured_data.get("results", [])))
        LOGGER.info("analytics_rows=%s", sum(_analytics_row_count(row.get("analytics", {})) for row in structured_data.get("results", [])))
        grouped_chunks = self._retrieve_for_titles(titles)
        grouped_chunks = {title: chunks[:2] for title, chunks in grouped_chunks.items()}
        chunks = _flatten_grouped_chunks(grouped_chunks)
        LOGGER.info("review_chunks=%s", len(chunks))
        prompt = _build_comparison_hybrid_analytics_prompt(
            query=query,
            structured_data=structured_data,
            grouped_chunks=grouped_chunks,
        )
        answer = self._generate(prompt, max_new_tokens=FAST_HYBRID_MAX_NEW_TOKENS)
        return OrchestratorResult(
            answer=answer,
            mode="hybrid",
            used_sql=True,
            retrieved_chunks=chunks,
            sql_results=structured_data,
        )

    def answer_recommendation(self, query: str) -> OrchestratorResult:
        LOGGER.info("recommendation_mode=true")
        title_theme_rows = self.sql_client.get_title_theme_summary(query, limit=RECO_TITLE_LIMIT)
        genre_theme_rows = self.sql_client.get_genre_theme_summary(query, limit=RECO_GENRE_LIMIT)
        audience_editorial_rows = self.sql_client.get_audience_editorial_summary(query, limit=RECO_AUDIENCE_LIMIT)
        LOGGER.info("title_theme_rows=%s", len(title_theme_rows))
        LOGGER.info("genre_theme_rows=%s", len(genre_theme_rows))
        LOGGER.info("audience_editorial_rows=%s", len(audience_editorial_rows))

        candidates = _merge_recommendation_candidates(
            title_theme_rows=title_theme_rows,
            audience_editorial_rows=audience_editorial_rows,
        )
        for candidate in candidates:
            title = candidate.get("title")
            if not title:
                continue
            metrics = self.sql_client.get_title_metrics(str(title))
            candidate["avg_rating"] = metrics.get("avg_rating")
            candidate["attention_score"] = metrics.get("attention_score")
        candidates = rank_recommendation_candidates(candidates)[:RECO_TITLE_LIMIT]

        fallback_chunks: list[RetrievedChunk] = []
        if not (candidates or genre_theme_rows):
            fallback_candidates = self._retrieve(query, top_k=3)
            fallback_chunks = filter_and_rerank_chunks(fallback_candidates)[:2]

        LOGGER.info("fallback_review_chunks=%s", len(fallback_chunks))
        prompt = _build_recommendation_prompt(
            query=query,
            candidates=candidates,
            genre_theme_rows=genre_theme_rows,
            fallback_chunks=fallback_chunks,
        )
        LOGGER.info("final_prompt_tokens=%s", estimate_tokens(prompt.system_prompt + prompt.user_prompt))
        answer = self._generate(prompt, max_new_tokens=RECOMMENDATION_MAX_NEW_TOKENS)
        return OrchestratorResult(
            answer=answer,
            mode="recommendation",
            used_sql=True,
            retrieved_chunks=fallback_chunks,
            sql_results={
                "recommendation_candidates": candidates,
                "genre_theme_summary": genre_theme_rows,
            },
        )

    def _generate(self, prompt: PromptResult, *, max_new_tokens: int) -> str:
        llm_started_at = time.perf_counter()
        try:
            return generate_answer(
                prompt,
                model_name=self.llm_model_name,
                max_new_tokens=max_new_tokens,
            )
        finally:
            LOGGER.info("llm_latency=%.3fs", time.perf_counter() - llm_started_at)


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


def _format_fast_sql_answer(structured_data: dict[str, Any]) -> str:
    validate_sql_payload(structured_data)
    query_type = structured_data.get("query_type")
    rows = structured_data.get("results", [])
    if not rows:
        return "No structured data was found for that query."

    if query_type == "rating":
        row = rows[0]
        title = row.get("title") or structured_data.get("title") or "This title"
        fields = []
        for label, key in (
            ("average rating", "avg_rating"),
            ("aggregate rating", "rating_aggregate"),
            ("TMDB rating", "rating_tmdb"),
            ("IMDb rating", "rating_imdb"),
            ("rating count", "rating_count"),
            ("minimum rating", "min_rating"),
            ("maximum rating", "max_rating"),
        ):
            if key in row:
                fields.append(f"{label}: {row[key]}")
        return f"{title}: " + "; ".join(fields) if fields else f"No rating fields were available for {title}."

    if query_type == "cast":
        row = rows[0]
        title = row.get("title") or structured_data.get("title") or "This title"
        cast = row.get("cast") or []
        if not cast:
            return f"No cast data was found for {title}."
        return f"{title} cast: " + ", ".join(str(name) for name in cast[:15])

    if query_type == "attention":
        row = rows[0]
        title = row.get("title") or structured_data.get("title") or "This title"
        fields = []
        if "attention_score" in row:
            fields.append(f"attention score: {row['attention_score']}")
        if "time_window" in row:
            fields.append(f"time window: {row['time_window']}")
        return f"{title}: " + "; ".join(fields) if fields else f"No attention fields were available for {title}."

    if query_type == "comparison":
        row = rows[0]
        fields = [f"{key}: {value}" for key, value in row.items() if value not in (None, "", [])]
        return "; ".join(fields) if fields else "No comparison fields were available."

    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        compact = ", ".join(f"{key}: {value}" for key, value in row.items() if value not in (None, "", []))
        lines.append(f"{index}. {compact}")
    return "\n".join(lines)


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


def _classify_hybrid_query(query: str) -> HybridQueryType:
    normalized = query.casefold()
    if any(
        term in normalized
        for term in (
            "attention vs reception",
            "reception vs attention",
            "reception vs rating",
            "reception vs ratings",
            "audience reception vs rating",
            "audience reception vs ratings",
            "overperforming",
            "underperforming",
        )
    ):
        return "ATTENTION_VS_RECEPTION"
    if any(
        term in normalized
        for term in (
            "attention vs rating",
            "attention vs ratings",
            "attention compare to rating",
            "attention compares to rating",
            "attention compared to rating",
            "attention compare to ratings",
        )
    ):
        return "ATTENTION_VS_RATING"
    if any(
        term in normalized
        for term in (
            "compare",
            "comparison",
            "compare reception",
            "compare audience",
            "compare ratings",
            "more positively received",
            "stronger criticism",
            "lighter than",
            "darker than",
            " vs ",
            " versus ",
        )
    ):
        return "TITLE_COMPARISON"
    if any(term in normalized for term in ("why is", "popular", "popularity")):
        return "POPULARITY_EXPLANATION"
    if any(term in normalized for term in ("trend", "trending")):
        return "TREND_EXPLANATION"
    return "POPULARITY_EXPLANATION"


def _looks_like_title_comparison(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return any(
        term in normalized
        for term in (
            " compare ",
            " comparison ",
            " difference ",
            " vs ",
            " versus ",
            " more positively received ",
            " stronger criticism ",
            " lighter than ",
            " darker than ",
        )
    )


def _analytics_row_count(analytics: dict[str, Any]) -> int:
    if not isinstance(analytics, dict):
        return 0
    count = 0
    for value in analytics.values():
        if isinstance(value, list):
            count += len(value)
        elif isinstance(value, dict) and any(item not in (None, [], "") for item in value.values()):
            count += 1
    return count


def _remove_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _remove_empty(item) for key, item in value.items() if item not in (None, [], "")}
    if isinstance(value, list):
        return [_remove_empty(item) for item in value if item not in (None, [], "")]
    return value


def _build_hybrid_analytics_payload(
    *,
    query_type: HybridQueryType,
    metrics: dict[str, Any],
    analytics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "hybrid_query_type": query_type,
        "metrics": _remove_empty(metrics),
        "analytics": _remove_empty(analytics),
    }


def _format_review_evidence(chunks: list[RetrievedChunk], *, max_chunks: int = 3, snippet_chars: int = 260) -> str:
    if not chunks:
        return "No supporting review excerpts retrieved."
    lines: list[str] = []
    for index, chunk in enumerate(chunks[:max_chunks], start=1):
        metadata = chunk.metadata
        title = metadata.get("title_name") or "Unknown title"
        source = metadata.get("source_name") or "unknown source"
        document_type = metadata.get("document_type") or metadata.get("chunk_source_type") or "text"
        snippet = " ".join(chunk.chunk_text.split())
        if len(snippet) > snippet_chars:
            snippet = snippet[: snippet_chars - 3].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
        lines.append(f"[{index}] {title} ({source}/{document_type})\n- {snippet}")
    return "\n\n".join(lines)


def _build_hybrid_analytics_prompt(
    *,
    query: str,
    metrics: dict[str, Any],
    analytics: dict[str, Any],
    chunks: list[RetrievedChunk],
) -> PromptResult:
    metrics_json = json.dumps(_remove_empty(metrics), ensure_ascii=False, indent=2, default=str)
    analytics_json = json.dumps(_remove_empty(analytics), ensure_ascii=False, indent=2, default=str)
    review_evidence = _format_review_evidence(chunks, max_chunks=3)
    user_prompt = (
        f"QUESTION\n{query}\n\n"
        "----------------------------------\n\n"
        f"STRUCTURED METRICS\n{metrics_json}\n\n"
        "----------------------------------\n\n"
        f"ANALYTICAL SUMMARIES\n{analytics_json}\n\n"
        "----------------------------------\n\n"
        f"REVIEW EVIDENCE\n{review_evidence}\n\n"
        "----------------------------------\n\n"
        "Instructions\n\n"
        "Answer in three sections:\n\n"
        "1.\nKey facts\n\n"
        "2.\nInterpretation\n\n"
        "3.\nSupporting evidence"
    )
    LOGGER.info("prompt_tokens_estimate=%s", estimate_tokens(HYBRID_SYSTEM_PROMPT + user_prompt))
    return PromptResult(
        system_prompt=HYBRID_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        included_chunk_ids=[chunk.chunk_id for chunk in chunks[:3]],
        excluded_chunk_ids=[],
        context_char_count=len(review_evidence),
    )


def _format_rows(rows: list[dict[str, Any]], *, max_rows: int = 6) -> str:
    if not rows:
        return "No relevant summary rows found."

    lines: list[str] = []
    for index, row in enumerate(rows[:max_rows], start=1):
        fields: list[str] = []
        for key, value in row.items():
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                rendered = ", ".join(str(item) for item in value[:6])
            else:
                rendered = str(value)
            fields.append(f"{key}: {rendered}")
        lines.append(f"[{index}] " + "; ".join(fields))
    return "\n".join(lines)


def _theme_match_score(item: dict[str, Any]) -> float:
    score = 0.0
    for key in ("dominant_themes", "audience_themes", "editorial_themes"):
        value = item.get(key) or []
        if isinstance(value, list):
            score += min(len(value), 3) * 0.3
    evidence_count = item.get("evidence_count")
    if isinstance(evidence_count, (int, float)):
        score += min(float(evidence_count), 100.0) / 100.0
    return score


def _recommendation_score(item: dict[str, Any]) -> float:
    rating = item.get("avg_rating")
    attention = item.get("attention_score")
    score = _theme_match_score(item)
    if isinstance(rating, (int, float)):
        score += float(rating)
    if isinstance(attention, (int, float)) and attention > 0:
        score += min(float(attention) / 10000.0, 5.0)
    return score


def rank_recommendation_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rank using:
    - avg_rating
    - attention_score
    - theme match score (if available)
    """
    return sorted(items, key=_recommendation_score, reverse=True)


def _merge_recommendation_candidates(
    *,
    title_theme_rows: list[dict[str, Any]],
    audience_editorial_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_title: dict[str, dict[str, Any]] = {}
    for row in title_theme_rows + audience_editorial_rows:
        title = row.get("title")
        if not title:
            continue
        key = str(title).casefold()
        candidate = by_title.setdefault(key, {"title": title})
        for field in (
            "genres",
            "dominant_themes",
            "audience_themes",
            "editorial_themes",
            "emotional_tone",
            "evidence_count",
            "avg_sentiment_score",
        ):
            value = row.get(field)
            if value in (None, "", []):
                continue
            if field == "evidence_count" and isinstance(value, (int, float)):
                candidate[field] = max(int(candidate.get(field) or 0), int(value))
            elif field not in candidate:
                candidate[field] = value
    return list(by_title.values())


def _compact_theme_list(values: Any, *, limit: int = 3) -> str:
    if not isinstance(values, list) or not values:
        return "[]"
    return "[" + ", ".join(str(value) for value in values[:limit]) + "]"


def _short_number(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}m"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.0f}k"
    return f"{number:.1f}".rstrip("0").rstrip(".")


def _compact_recommendation_candidates(candidates: list[dict[str, Any]], *, theme_limit: int = 3) -> str:
    if not candidates:
        return "No structured recommendation candidates found."

    lines: list[str] = []
    for item in candidates[:RECO_TITLE_LIMIT]:
        title = item.get("title") or "Unknown title"
        theme_source = item.get("dominant_themes") or item.get("audience_themes") or item.get("editorial_themes") or []
        parts = [
            f"{title}:",
            f"themes={_compact_theme_list(theme_source, limit=theme_limit)}",
        ]
        mood = item.get("emotional_tone")
        if mood:
            parts.append(f"mood={mood}")
        rating = _short_number(item.get("avg_rating"))
        if rating:
            parts.append(f"rating={rating}")
        attention = _short_number(item.get("attention_score"))
        if attention:
            parts.append(f"attention={attention}")
        evidence_count = item.get("evidence_count")
        if isinstance(evidence_count, (int, float)):
            parts.append(f"evidence={int(evidence_count)}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _compact_genre_rows(rows: list[dict[str, Any]], *, theme_limit: int = 3) -> str:
    if not rows:
        return "No genre context."
    lines: list[str] = []
    for row in rows[:RECO_GENRE_LIMIT]:
        genre = row.get("genre") or "Unknown genre"
        parts = [
            f"{genre}:",
            f"themes={_compact_theme_list(row.get('dominant_themes'), limit=theme_limit)}",
        ]
        mood = row.get("emotional_tone")
        if mood:
            parts.append(f"mood={mood}")
        title_count = row.get("title_count")
        if isinstance(title_count, (int, float)):
            parts.append(f"titles={int(title_count)}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _format_fallback_chunks(chunks: list[RetrievedChunk], *, max_chunks: int = 4, max_text_chars: int = 260) -> str:
    if not chunks:
        return "No fallback review evidence used."

    lines: list[str] = []
    for index, chunk in enumerate(chunks[:max_chunks], start=1):
        metadata = chunk.metadata
        title = metadata.get("title_name") or metadata.get("title") or "Unknown title"
        source = metadata.get("source_name") or metadata.get("document_type") or "unknown source"
        snippet = " ".join(chunk.chunk_text.split())
        if len(snippet) > max_text_chars:
            snippet = snippet[: max_text_chars - 3].rstrip() + "..."
        lines.append(f"[{index}] {title} ({source}) - {snippet}")
    return "\n".join(lines)


def _build_recommendation_prompt(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    genre_theme_rows: list[dict[str, Any]],
    fallback_chunks: list[RetrievedChunk],
) -> PromptResult:
    trimmed = False
    user_prompt = _compose_recommendation_user_prompt(
        query=query,
        candidates=candidates,
        genre_theme_rows=genre_theme_rows,
        fallback_chunks=fallback_chunks,
        include_genres=True,
        theme_limit=3,
    )
    token_estimate = estimate_tokens(RECOMMENDATION_SYSTEM_PROMPT + user_prompt)
    if token_estimate > RECO_MAX_PROMPT_TOKENS:
        trimmed = True
        user_prompt = _compose_recommendation_user_prompt(
            query=query,
            candidates=candidates,
            genre_theme_rows=[],
            fallback_chunks=fallback_chunks,
            include_genres=False,
            theme_limit=3,
        )
        token_estimate = estimate_tokens(RECOMMENDATION_SYSTEM_PROMPT + user_prompt)
    if token_estimate > RECO_MAX_PROMPT_TOKENS:
        trimmed = True
        user_prompt = _compose_recommendation_user_prompt(
            query=query,
            candidates=candidates[:2],
            genre_theme_rows=[],
            fallback_chunks=[],
            include_genres=False,
            theme_limit=2,
        )
        token_estimate = estimate_tokens(RECOMMENDATION_SYSTEM_PROMPT + user_prompt)
    if token_estimate > RECO_MAX_PROMPT_TOKENS:
        trimmed = True
        user_prompt = _compose_recommendation_user_prompt(
            query=query,
            candidates=candidates[:2],
            genre_theme_rows=[],
            fallback_chunks=[],
            include_genres=False,
            theme_limit=1,
        )
        token_estimate = estimate_tokens(RECOMMENDATION_SYSTEM_PROMPT + user_prompt)

    if trimmed:
        LOGGER.info("recommendation_prompt_trimmed=true")
    LOGGER.info("recommendation_prompt_tokens_approx=%s", token_estimate)
    return PromptResult(
        system_prompt=RECOMMENDATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        included_chunk_ids=[chunk.chunk_id for chunk in fallback_chunks],
        excluded_chunk_ids=[],
        context_char_count=len(user_prompt),
    )


def _compose_recommendation_user_prompt(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    genre_theme_rows: list[dict[str, Any]],
    fallback_chunks: list[RetrievedChunk],
    include_genres: bool,
    theme_limit: int,
) -> str:
    candidate_block = _compact_recommendation_candidates(candidates, theme_limit=theme_limit)
    genre_block = _compact_genre_rows(genre_theme_rows, theme_limit=theme_limit) if include_genres else ""
    review_block = _format_fallback_chunks(fallback_chunks, max_chunks=2, max_text_chars=180)
    genre_section = f"\nGENRE SIGNALS\n{genre_block}\n" if genre_block else ""
    review_section = f"\nFALLBACK REVIEW EVIDENCE\n{review_block}\n" if fallback_chunks else ""
    user_prompt = (
        f"REQUEST\n{query}\n\n"
        f"CANDIDATES\n{candidate_block}\n"
        f"{genre_section}"
        f"{review_section}\n"
        "TASK\n"
        "Recommend exactly 3 titles. Use this format for each:\n"
        "- Title\n"
        "- Why it fits: max 2 sentences\n"
        "- Evidence: max 2 bullet points\n"
        "- Confidence: low/medium/high\n\n"
        "Keep the answer concise. Do not add extra sections or long explanations."
    )
    return user_prompt


def _log_key(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in cleaned.split("_") if part) or "title"


def _flatten_grouped_chunks(grouped_chunks: dict[str, list[RetrievedChunk]]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for title_chunks in grouped_chunks.values():
        chunks.extend(title_chunks)
    return chunks


def _format_comparison_evidence(
    grouped_chunks: dict[str, list[RetrievedChunk]],
    *,
    max_context_chars: int,
    snippet_chars: int = 360,
) -> tuple[str, list[str], int]:
    included_ids: list[str] = []
    sections: list[str] = []
    used_chars = 0

    for index, (title, chunks) in enumerate(grouped_chunks.items()):
        label = "TITLE A" if index == 0 else "TITLE B" if index == 1 else f"TITLE {index + 1}"
        lines = [label, title, "Evidence"]
        section_ids: list[str] = []
        if not chunks:
            lines.append("- No retrieved evidence found for this title.")
        for chunk_index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata
            source = metadata.get("source_name") or "unknown source"
            document_type = metadata.get("document_type") or metadata.get("chunk_source_type") or "text"
            snippet = " ".join(chunk.chunk_text.split())
            if len(snippet) > snippet_chars:
                snippet = snippet[: snippet_chars - 3].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
            lines.append(f"[{chunk_index}] {source}/{document_type}: {snippet}")
            section_ids.append(chunk.chunk_id)

        section = "\n".join(lines)
        if used_chars + len(section) > max_context_chars and sections:
            break
        sections.append(section)
        included_ids.extend(section_ids)
        used_chars += len(section)

    return "\n\n".join(sections), included_ids, used_chars


def _build_comparison_rag_prompt(
    *,
    query: str,
    grouped_chunks: dict[str, list[RetrievedChunk]],
    max_context_chars: int,
) -> PromptResult:
    evidence, included_ids, context_chars = _format_comparison_evidence(
        grouped_chunks,
        max_context_chars=max_context_chars,
    )
    user_prompt = (
        f"COMPARISON CONTEXT\n{evidence}\n\n"
        f"QUESTION\n{query}\n\n"
        "Use only the evidence grouped under each title. Compare the titles directly and do not "
        "infer evidence for a title from the other title's section."
    )
    return PromptResult(
        system_prompt=(
            "You are a helpful assistant answering comparison questions about movies and TV shows. "
            "Use only the title-grouped context provided by the user."
        ),
        user_prompt=user_prompt,
        included_chunk_ids=included_ids,
        excluded_chunk_ids=[],
        context_char_count=context_chars,
    )


def _build_comparison_sql_payload(sql_client: MotherDuckClient, titles: list[str]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for title in titles[:2]:
        profile = sql_client.get_title_profile(title)
        profile_row = profile["results"][0] if profile.get("results") else {}
        results.append(
            {
                "title": profile_row.get("title") or title,
                "rating_tmdb": profile_row.get("rating_tmdb"),
                "rating_imdb": profile_row.get("rating_imdb"),
                "rating_aggregate": profile_row.get("rating_aggregate"),
                "rating_count": profile_row.get("rating_count"),
                "attention_score": profile_row.get("attention_score"),
                "time_window": profile_row.get("time_window"),
                "themes": sql_client.get_title_theme_profile(title),
            }
        )
    return normalize_sql_output(
        results,
        "comparison",
        title=None,
        summary_metrics={"titles": titles[:2], "result_count": len(results)},
    )


def _build_comparison_hybrid_prompt(
    *,
    query: str,
    structured_data: dict[str, Any],
    grouped_chunks: dict[str, list[RetrievedChunk]],
) -> PromptResult:
    validate_sql_payload(structured_data)
    evidence, included_ids, context_chars = _format_comparison_evidence(
        grouped_chunks,
        max_context_chars=MAX_CONTEXT_CHARS,
        snippet_chars=260,
    )
    user_prompt = (
        f"STRUCTURED DATA BY TITLE (SQL FACTS ONLY):\n{json.dumps(structured_data, ensure_ascii=False, separators=(',', ':'), default=str)}\n\n"
        f"TEXTUAL EVIDENCE BY TITLE (RAG SUPPORTING CONTEXT):\n{evidence}\n\n"
        f"QUESTION\n{query}\n\n"
        "RULES:\n"
        "- Use SQL as source of truth for ratings, attention, and other structured values.\n"
        "- Use each title's text evidence only for interpretation of that same title.\n"
        "- Never infer which evidence belongs to which title.\n"
        "- If a structured value is missing, say it is not available in structured data."
    )
    token_estimate = estimate_tokens(HYBRID_SYSTEM_PROMPT + user_prompt)
    LOGGER.info("prompt_token_estimate=%s", token_estimate)
    return PromptResult(
        system_prompt=HYBRID_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        included_chunk_ids=included_ids,
        excluded_chunk_ids=[],
        context_char_count=context_chars,
    )


def _build_comparison_hybrid_analytics_prompt(
    *,
    query: str,
    structured_data: dict[str, Any],
    grouped_chunks: dict[str, list[RetrievedChunk]],
) -> PromptResult:
    metrics_by_title: list[dict[str, Any]] = []
    analytics_by_title: list[dict[str, Any]] = []
    for row in structured_data.get("results", []):
        title = row.get("title")
        metrics_by_title.append({"title": title, **(row.get("metrics") or {})})
        analytics_by_title.append({"title": title, **(row.get("analytics") or {})})

    evidence, included_ids, context_chars = _format_comparison_evidence(
        grouped_chunks,
        max_context_chars=MAX_CONTEXT_CHARS,
        snippet_chars=220,
    )
    user_prompt = (
        f"QUESTION\n{query}\n\n"
        "----------------------------------\n\n"
        f"STRUCTURED METRICS\n{json.dumps(_remove_empty(metrics_by_title), ensure_ascii=False, indent=2, default=str)}\n\n"
        "----------------------------------\n\n"
        f"ANALYTICAL SUMMARIES\n{json.dumps(_remove_empty(analytics_by_title), ensure_ascii=False, indent=2, default=str)}\n\n"
        "----------------------------------\n\n"
        f"REVIEW EVIDENCE\n{evidence}\n\n"
        "----------------------------------\n\n"
        "Instructions\n\n"
        "Answer in three sections:\n\n"
        "1.\nKey facts\n\n"
        "2.\nInterpretation\n\n"
        "3.\nSupporting evidence"
    )
    LOGGER.info("prompt_tokens_estimate=%s", estimate_tokens(HYBRID_SYSTEM_PROMPT + user_prompt))
    return PromptResult(
        system_prompt=HYBRID_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        included_chunk_ids=included_ids,
        excluded_chunk_ids=[],
        context_char_count=context_chars,
    )


def answer_question(query: str) -> ChatResponse:
    return ChatOrchestrator().answer(query).to_response()
