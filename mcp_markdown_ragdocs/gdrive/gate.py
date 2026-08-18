"""Durable serialization for Google Drive provider requests."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from mcp_markdown_ragdocs.gdrive.errors import classify_provider_error

T = TypeVar("T")


class DriveRequestGate:
    """Bound concurrent Drive requests using a private, file-backed SQLite state.

    Up to ``max_concurrent`` requests may be in flight across processes at
    once; claims are additionally spaced by ``min_interval_seconds``. Each
    in-flight slot expires after ``request_timeout_seconds`` so a crashed
    holder cannot permanently occupy a slot.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        min_interval_seconds: float = 0.5,
        max_concurrent: int = 1,
        provider_cooldown_seconds: float = 5.0,
        request_timeout_seconds: float = 300.0,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if provider_cooldown_seconds < 0:
            raise ValueError("provider_cooldown_seconds must be non-negative")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.path = Path(db_path)
        self._min_interval_seconds = min_interval_seconds
        self._max_concurrent = max_concurrent
        self._provider_cooldown_seconds = provider_cooldown_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def run(self, operation: Callable[[], T]) -> T:
        """Run one provider operation after claiming a durable gate slot."""
        slot_id = self._claim()
        try:
            return operation()
        except Exception as error:
            if classify_provider_error(error).status_code == 429:
                self._extend_cooldown()
            raise
        finally:
            self._release(slot_id)

    def _claim(self) -> int:
        while True:
            now = time.time()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM drive_request_gate_slots WHERE expires_at <= ?",
                    (now,),
                )
                row = connection.execute(
                    "SELECT next_allowed_at FROM drive_request_gate WHERE id = 1"
                ).fetchone()
                next_allowed_at = float(row[0]) if row is not None else 0.0
                (slot_count,) = connection.execute(
                    "SELECT COUNT(*) FROM drive_request_gate_slots"
                ).fetchone()
                wait_for = max(next_allowed_at - now, 0.0)
                if wait_for == 0.0 and slot_count < self._max_concurrent:
                    cursor = connection.execute(
                        "INSERT INTO drive_request_gate_slots (expires_at) VALUES (?)",
                        (now + self._request_timeout_seconds,),
                    )
                    connection.execute(
                        """
                        INSERT INTO drive_request_gate (id, next_allowed_at)
                        VALUES (1, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            next_allowed_at = excluded.next_allowed_at
                        """,
                        (now + self._min_interval_seconds,),
                    )
                    assert cursor.lastrowid is not None
                    return cursor.lastrowid
            time.sleep(max(wait_for, 0.01) if wait_for > 0 else 0.01)

    def _release(self, slot_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM drive_request_gate_slots WHERE slot_id = ?", (slot_id,)
            )

    def _extend_cooldown(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE drive_request_gate
                SET next_allowed_at = MAX(next_allowed_at, ?)
                WHERE id = 1
                """,
                (time.time() + self._provider_cooldown_seconds,),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drive_request_gate (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    next_allowed_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drive_request_gate_slots (
                    slot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expires_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


__all__ = ["DriveRequestGate"]
