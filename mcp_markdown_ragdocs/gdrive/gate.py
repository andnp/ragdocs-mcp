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
    """Serialize Drive requests using a private, file-backed SQLite state."""

    def __init__(
        self,
        db_path: Path,
        *,
        min_interval_seconds: float = 0.5,
        provider_cooldown_seconds: float = 5.0,
        request_timeout_seconds: float = 300.0,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if provider_cooldown_seconds < 0:
            raise ValueError("provider_cooldown_seconds must be non-negative")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.path = Path(db_path)
        self._min_interval_seconds = min_interval_seconds
        self._provider_cooldown_seconds = provider_cooldown_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def run(self, operation: Callable[[], T]) -> T:
        """Run one provider operation after claiming the durable gate."""
        self._claim()
        try:
            return operation()
        except Exception as error:
            if classify_provider_error(error).status_code == 429:
                self._extend_cooldown()
            raise
        finally:
            self._release()

    def _claim(self) -> None:
        while True:
            now = time.time()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT next_allowed_at, in_flight_until FROM drive_request_gate WHERE id = 1"
                ).fetchone()
                next_allowed_at = float(row[0]) if row is not None else 0.0
                in_flight_until = float(row[1]) if row is not None else 0.0
                wait_for = max(next_allowed_at - now, 0.0)
                in_flight = in_flight_until > now
                if not in_flight and wait_for == 0.0:
                    connection.execute(
                        """
                        INSERT INTO drive_request_gate (id, next_allowed_at, in_flight_until)
                        VALUES (1, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            next_allowed_at = excluded.next_allowed_at,
                            in_flight_until = excluded.in_flight_until
                        """,
                        (now + self._min_interval_seconds, now + self._request_timeout_seconds),
                    )
                    return
            time.sleep(max(wait_for, 0.01) if not in_flight else 0.01)

    def _release(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE drive_request_gate SET in_flight_until = 0 WHERE id = 1"
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
                    next_allowed_at REAL NOT NULL,
                    in_flight_until REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


__all__ = ["DriveRequestGate"]
