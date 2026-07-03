from __future__ import annotations

import argparse
import json
from pathlib import Path

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.load.motherduck import (
    connect_to_database,
    list_database_tables,
    preview_table_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query or preview Cultural Mood Tracker tables in MotherDuck."
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Table name to preview. Required unless --list-tables or --sql is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of rows to preview. Defaults to 20.",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List tables in the configured MotherDuck database.",
    )
    parser.add_argument(
        "--sql",
        default=None,
        help="Run a read-only SQL query and return up to --limit rows.",
    )
    return parser.parse_args()


def _ensure_read_only_sql(sql: str) -> str:
    candidate = sql.strip().rstrip(";")
    if not candidate:
        raise RuntimeError("SQL query cannot be empty.")
    first_token = candidate.split(maxsplit=1)[0].lower()
    if first_token not in {"select", "with", "show", "describe"}:
        raise RuntimeError("Only read-only SELECT/WITH/SHOW/DESCRIBE queries are allowed.")
    return candidate


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    _ = project_root
    settings = load_settings()
    args = parse_args()

    connection = connect_to_database(
        token=settings.motherduck_token,
        database_name=settings.motherduck_database,
    )
    try:
        if args.list_tables:
            print(json.dumps({"tables": list_database_tables(connection=connection)}, indent=2))
            return 0

        if args.sql:
            sql = _ensure_read_only_sql(args.sql)
            cursor = connection.execute(f"{sql} LIMIT {int(args.limit)}")
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
            print(json.dumps({"query": sql, "row_count": len(rows), "rows": rows}, indent=2, default=str))
            return 0

        if not args.table:
            raise RuntimeError("Pass --table <name>, or use --list-tables, or use --sql.")

        rows = preview_table_rows(
            connection=connection,
            table_name=args.table,
            limit=args.limit,
        )
        print(json.dumps({"table": args.table, "row_count": len(rows), "rows": rows}, indent=2, default=str))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
