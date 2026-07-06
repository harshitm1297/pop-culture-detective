"""Streamlit chat frontend for the Cultural Mood Tracker orchestrator.

This file is a pure UI layer: it imports and calls the existing chatbot backend
(`cultural_mood_tracker.chat.orchestrator.ChatOrchestrator`) and never reimplements
routing, retrieval, SQL, or prompt/generation logic. `scripts/chat.py` (the terminal
entry point) is untouched and keeps working exactly as before -- this is an
additional frontend on top of the same backend, not a replacement.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.chat.orchestrator import ChatOrchestrator
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS


# Same mode -> color mapping used in the terminal (scripts/chat.py)'s rich output, so the two
# frontends stay visually consistent for anyone using both.
MODE_INFO: dict[str, dict[str, str]] = {
    "fast_sql": {"label": "SQL", "color": "#5fc9d4", "description": "Answered directly from MotherDuck (no LLM call)."},
    "sql": {"label": "SQL", "color": "#5fc9d4", "description": "Answered directly from MotherDuck (no LLM call)."},
    "rag": {"label": "RAG", "color": "#6fcf7d", "description": "Answered from retrieved review/summary text in ChromaDB."},
    "hybrid": {"label": "HYBRID", "color": "#d88ce8", "description": "Answered using MotherDuck facts + ChromaDB evidence together."},
    "recommendation": {"label": "RECOMMENDATION", "color": "#f2c14e", "description": "Answered from MotherDuck theme/genre analytics, with RAG as a fallback."},
}


# ---------------------------------------------------------------------------
# Page setup + theme
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Cultural Mood Tracker", page_icon="🎬", layout="centered")

CINEMA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@400;500;600&display=swap');

.stApp {
    background: radial-gradient(circle at top, #181b22 0%, #0a0b0f 60%);
    color: #eae6df;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Marquee-style header */
.marquee-header {
    text-align: center;
    padding: 0.4rem 0 1.1rem 0;
    border-bottom: 1px solid rgba(242, 193, 78, 0.25);
    margin-bottom: 1.4rem;
}
.marquee-title {
    font-family: 'Playfair Display', serif;
    font-weight: 800;
    font-size: 2.5rem;
    letter-spacing: 0.03em;
    background: linear-gradient(135deg, #f2c14e 20%, #e3903a 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.marquee-subtitle {
    color: #9a9eab;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 0.01em;
}

/* Metadata row under an assistant answer */
.meta-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.5rem 0 0.2rem 0;
    flex-wrap: wrap;
}
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.18rem 0.7rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid currentColor;
    background: rgba(255, 255, 255, 0.03);
}
.mode-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}
.elapsed-pill {
    font-size: 0.72rem;
    color: #9a9eab;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.sql-flag {
    font-size: 0.72rem;
    color: #9a9eab;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0e13;
    border-right: 1px solid rgba(242, 193, 78, 0.12);
}
.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #f2c14e;
    margin-bottom: 0.2rem;
}
.sidebar-caption {
    color: #9a9eab;
    font-size: 0.85rem;
    line-height: 1.4rem;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.3rem 0;
    font-size: 0.85rem;
}
</style>
"""

st.markdown(CINEMA_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="marquee-header">
        <p class="marquee-title">Cultural Mood Tracker</p>
        <p class="marquee-subtitle">Ask about movies &amp; TV shows &mdash; ratings, reviews, trends, and recommendations</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached backend resource
#
# st.cache_resource keeps a single ChatOrchestrator alive for the life of the Streamlit
# server process, shared across every rerun and every user session. That one instance is
# what makes connection reuse possible:
#   - MotherDuckClient (chat/sql_client.py) lazily opens one DuckDB connection and keeps it
#     on `self._connection` -- reused as long as the same client object is alive.
#   - ChatOrchestrator now lazily opens one ChromaDB collection and keeps it on
#     `self._chroma_collection` -- same idea, added specifically so this app doesn't reopen
#     a PersistentClient on every question.
#   - The SentenceTransformer embedding model and the Groq client are cached at the module
#     level inside rag/retrieval.py and rag/llm.py respectively (a dict keyed by model name),
#     so they're reused automatically regardless of how many ChatOrchestrator instances exist
#     -- nothing extra was needed there.
# Without st.cache_resource, Streamlit's return-to-top-of-script-on-every-interaction model
# would construct a brand new orchestrator (and eventually a new MotherDuck connection) on
# every single message.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to ChromaDB, MotherDuck, and Groq...")
def get_orchestrator() -> ChatOrchestrator:
    project_root = load_project_environment(Path(__file__))
    persist_dir = Path(DEFAULT_CHROMA_DB_DIR)
    if not persist_dir.is_absolute():
        persist_dir = project_root / persist_dir
    return ChatOrchestrator(
        persist_dir=persist_dir,
        collection_name=DEFAULT_CHROMA_COLLECTION,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        llm_model_name=DEFAULT_MODEL,
        top_k=3,
        max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
    )


def _friendly_error(exc: Exception) -> str:
    """Translate a raw backend exception into an actionable message for the chat window."""
    message = str(exc)
    lowered = message.casefold()
    if "groq_api_key" in lowered or "groq generation failed" in lowered or "groq returned" in lowered:
        return f"**The Groq LLM call failed.** Check that `GROQ_API_KEY` is set correctly in `.env`.\n\nDetails: {message}"
    if "motherduck" in lowered:
        return f"**The MotherDuck connection failed.** Check `MOTHERDUCK_TOKEN` and `MOTHERDUCK_DATABASE` in `.env`, and your network connection.\n\nDetails: {message}"
    if "chromadb" in lowered or "collection" in lowered or "persist" in lowered:
        return f"**The local ChromaDB retrieval step failed.** Make sure `chroma_db/` has been populated (see README step 5).\n\nDetails: {message}"
    return f"**Something went wrong while generating this answer.**\n\nDetails: {message}"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, Any]] = []


def _mode_badge_html(mode: str | None) -> str:
    info = MODE_INFO.get(mode or "", {"label": (mode or "unknown").upper(), "color": "#9a9eab"})
    return (
        f'<span class="mode-badge" style="color:{info["color"]};">'
        f'<span class="mode-dot"></span>{info["label"]}</span>'
    )


def _render_meta(entry: dict[str, Any]) -> None:
    badge = _mode_badge_html(entry.get("mode"))
    elapsed = entry.get("elapsed")
    elapsed_html = f'<span class="elapsed-pill">{elapsed:.2f}s</span>' if elapsed is not None else ""
    sql_flag_html = '<span class="sql-flag">uses SQL</span>' if entry.get("used_sql") else ""
    st.markdown(f'<div class="meta-row">{badge}{elapsed_html}{sql_flag_html}</div>', unsafe_allow_html=True)

    sql_queries = entry.get("sql_queries") or []
    if sql_queries:
        with st.expander(f"SQL queries used ({len(sql_queries)})"):
            for index, sql in enumerate(sql_queries, start=1):
                st.caption(f"Query {index}")
                st.code(sql, language="sql")

    evidence = entry.get("evidence") or []
    if evidence:
        with st.expander(f"Retrieved documents ({len(evidence)})"):
            for index, item in enumerate(evidence, start=1):
                title = item.get("title") or "Unknown title"
                source = item.get("source") or "unknown source"
                document_type = item.get("document_type") or "text"
                similarity = item.get("similarity")
                similarity_str = f"{similarity:.3f}" if isinstance(similarity, (int, float)) else "n/a"
                st.markdown(f"**{index}. {title}** &mdash; {source} / {document_type} (similarity {similarity_str})")
                st.caption(item.get("chunk_id", ""))


def _render_message(entry: dict[str, Any]) -> None:
    avatar = "🎬" if entry["role"] == "assistant" else "🙂"
    with st.chat_message(entry["role"], avatar=avatar):
        if entry.get("error"):
            st.error(entry["content"])
        else:
            st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("mode"):
            _render_meta(entry)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">Cultural Mood Tracker</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-caption">A chat interface over the project\'s SQL + RAG orchestrator. '
        "Every answer is routed to the retrieval strategy that fits the question.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

    with st.expander("How answers are routed"):
        for mode_key in ("fast_sql", "rag", "hybrid", "recommendation"):
            info = MODE_INFO[mode_key]
            st.markdown(
                f'<div class="legend-row">{_mode_badge_html(mode_key)}'
                f'<span class="sidebar-caption">{info["description"]}</span></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Backend connection (fails fast, with a readable message, instead of a raw traceback)
# ---------------------------------------------------------------------------
try:
    orchestrator = get_orchestrator()
except Exception as exc:  # noqa: BLE001 - deliberately broad: this is a top-level startup guard
    st.error(_friendly_error(exc))
    st.stop()


# ---------------------------------------------------------------------------
# Chat history + input
# ---------------------------------------------------------------------------
for entry in st.session_state.messages:
    _render_message(entry)

query = st.chat_input("Ask about a movie or TV show...")

if query:
    user_entry = {"role": "user", "content": query}
    st.session_state.messages.append(user_entry)
    _render_message(user_entry)

    with st.chat_message("assistant", avatar="🎬"):
        with st.spinner("Thinking..."):
            try:
                orchestrator.sql_client.clear_last_queries()
                started_at = time.perf_counter()
                result = orchestrator.answer(query)
                elapsed = time.perf_counter() - started_at
                sql_queries = orchestrator.sql_client.get_last_queries()
                evidence = [
                    {
                        "chunk_id": item.chunk_id,
                        "similarity": item.similarity,
                        "title": item.title,
                        "source": item.source,
                        "document_type": item.document_type,
                    }
                    for item in result.evidence
                ]
                assistant_entry = {
                    "role": "assistant",
                    "content": result.answer,
                    "mode": result.mode,
                    "used_sql": result.used_sql,
                    "elapsed": elapsed,
                    "sql_queries": sql_queries,
                    "evidence": evidence,
                    "error": False,
                }
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as a chat bubble, not a crash
                assistant_entry = {
                    "role": "assistant",
                    "content": _friendly_error(exc),
                    "mode": None,
                    "used_sql": False,
                    "elapsed": None,
                    "sql_queries": [],
                    "evidence": [],
                    "error": True,
                }

        if assistant_entry["error"]:
            st.error(assistant_entry["content"])
        else:
            st.markdown(assistant_entry["content"])
            _render_meta(assistant_entry)

    st.session_state.messages.append(assistant_entry)
