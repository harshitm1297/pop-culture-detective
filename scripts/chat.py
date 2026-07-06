from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cultural_mood_tracker.chat.orchestrator import ChatOrchestrator
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag.chroma_ingest import DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DB_DIR
from cultural_mood_tracker.rag.embeddings import DEFAULT_EMBEDDING_MODEL
from cultural_mood_tracker.rag.llm import DEFAULT_MODEL
from cultural_mood_tracker.rag.prompting import DEFAULT_MAX_CONTEXT_CHARS

# `rich` is an optional presentation dependency: if it isn't installed yet (e.g. a teammate
# hasn't re-run `pip install -r requirements.txt`), the CLI falls back to the original plain-text
# output instead of crashing. Nothing about routing, retrieval, SQL, or generation depends on it.
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# label + rich color per mode, so the same mode is always visually consistent across a session.
MODE_STYLES: dict[str, tuple[str, str]] = {
    "fast_sql": ("SQL", "cyan"),
    "sql": ("SQL", "cyan"),
    "rag": ("RAG", "green"),
    "hybrid": ("HYBRID", "magenta"),
    "recommendation": ("RECOMMENDATION", "yellow"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Cultural Mood Tracker chat orchestrator.")
    parser.add_argument(
        "--query",
        default=None,
        help="Natural-language question to answer. If omitted, starts an interactive chat loop.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of ChromaDB chunks for RAG/hybrid mode.")
    parser.add_argument(
        "--persist-dir",
        default=Path(DEFAULT_CHROMA_DB_DIR),
        type=Path,
        help=f"Directory of the persistent ChromaDB database. Defaults to ./{DEFAULT_CHROMA_DB_DIR}.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_CHROMA_COLLECTION,
        help=f"ChromaDB collection name. Defaults to {DEFAULT_CHROMA_COLLECTION}.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model used for retrieval. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--llm-model-name",
        default=DEFAULT_MODEL,
        help=f"Groq chat model. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Context character budget for RAG mode. Defaults to {DEFAULT_MAX_CONTEXT_CHARS}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the required structured response as JSON (always plain, never styled).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Force plain-text output even if rich is installed (useful for logs/piping/CI).",
    )
    return parser.parse_args()


def _use_rich(args: argparse.Namespace) -> bool:
    return _RICH_AVAILABLE and not args.json and not args.plain


def _print_response_plain(response: dict) -> None:
    print(f"Mode: {response['mode']}")
    print(f"Used SQL: {response['used_sql']}")
    if response["retrieved_chunk_ids"]:
        print(f"Retrieved chunks: {', '.join(response['retrieved_chunk_ids'])}")
    print()
    print(response["answer"])


def _print_response_rich(console: "Console", response: dict) -> None:
    label, color = MODE_STYLES.get(response["mode"], (response["mode"].upper(), "white"))
    sql_flag = "yes" if response["used_sql"] else "no"
    subtitle = f"used_sql={sql_flag}"

    console.print(
        Panel(
            Markdown(response["answer"]),
            title=f"[bold {color}]{label}[/bold {color}]",
            subtitle=subtitle,
            border_style=color,
            padding=(1, 2),
        )
    )

    chunk_ids = response["retrieved_chunk_ids"]
    if chunk_ids:
        table = Table(title="Retrieved chunks", show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", justify="right", style="dim")
        table.add_column("chunk_id")
        for index, chunk_id in enumerate(chunk_ids, start=1):
            table.add_row(str(index), chunk_id)
        console.print(table)


def _answer_once(orchestrator: ChatOrchestrator, query: str, *, args: argparse.Namespace) -> None:
    use_rich = _use_rich(args)

    if use_rich:
        console = Console()
        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            result = orchestrator.answer(query)
    else:
        result = orchestrator.answer(query)

    response = result.to_response()

    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    elif use_rich:
        _print_response_rich(Console(), response)
    else:
        _print_response_plain(response)


def _interactive_loop(orchestrator: ChatOrchestrator, *, args: argparse.Namespace) -> int:
    use_rich = _use_rich(args)
    console = Console() if use_rich else None

    if console is not None:
        console.print(
            "[bold]Unified Cultural Mood Tracker chat ready.[/bold] "
            "Type a question, or 'exit' to quit."
        )
    else:
        print("Unified Cultural Mood Tracker chat ready. Type a question, or 'exit' to quit.")

    while True:
        try:
            if console is not None:
                query = console.input("\n[bold cyan]Question:[/bold cyan] ").strip()
            else:
                query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            return 0

        try:
            _answer_once(orchestrator, query, args=args)
        except Exception as exc:
            if console is not None:
                console.print(Panel(str(exc), title="[bold red]Error[/bold red]", border_style="red"))
            else:
                print(f"Error: {exc}")


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    persist_dir = args.persist_dir if args.persist_dir.is_absolute() else project_root / args.persist_dir
    orchestrator = ChatOrchestrator(
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model_name,
        llm_model_name=args.llm_model_name,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )

    if args.query:
        _answer_once(orchestrator, args.query, args=args)
        return 0

    _interactive_loop(orchestrator, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
