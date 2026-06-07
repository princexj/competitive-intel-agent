"""SQLite-backed report history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(".cache") / "intel_history.db"


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            targets TEXT NOT NULL,
            comparison TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    return connection


def save_run(
    results: list[dict[str, Any]],
    comparison: str,
    db_path: Path = DB_PATH,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    targets = [result["topic"] for result in results]
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute(
                "INSERT INTO reports VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(targets),
                    comparison,
                    json.dumps(results),
                ),
            )
    return run_id


def list_runs(limit: int = 20, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT id, created_at, targets FROM reports "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "targets": json.loads(row["targets"]),
        }
        for row in rows
    ]


def get_run(run_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    with closing(_connect(db_path)) as connection:
        row = connection.execute(
            "SELECT * FROM reports WHERE id = ?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "targets": json.loads(row["targets"]),
        "comparison": row["comparison"],
        "results": json.loads(row["payload"]),
    }
