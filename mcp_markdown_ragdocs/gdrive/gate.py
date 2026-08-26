"""Durable serialization for Google Drive provider requests."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from mcp_markdown_ragdocs.gdrive.errors import classify_provider_error

T = TypeVar("T")

# When the rate window is open but every concurrency slot is taken, there is
# no exact wake time (unlike the rate-limit wait, where next_allowed_at gives
# one). Poll with exponential backoff instead of a fixed short sleep, capped
# by GATE_SATURATION_POLL_CEILING_SECONDS so a long-lived saturation still
# notices a release promptly, and also capped by the soonest slot expiry
# (a slot cannot outlive it) so a claim never waits past its own guaranteed
# upper bound.
GATE_SATURATION_POLL_INITIAL_SECONDS = 0.01
GATE_SATURATION_POLL_CEILING_SECONDS = 0.5
GATE_SATURATION_POLL_BACKOFF_MULTIPLIER = 2.0


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
        time_source: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
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
        self._time = time_source
        self._sleep = sleep
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
        # Reused across retries within one claim so a saturated gate does not
        # open a fresh connection (and take a fresh write lock) every poll.
        connection = self._connect()
        try:
            saturation_poll_seconds = GATE_SATURATION_POLL_INITIAL_SECONDS
            while True:
                now = self._time()
                with connection:
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
                    if wait_for > 0.0:
                        # Rate-limited: next_allowed_at is an exact wake time.
                        sleep_for = wait_for
                        saturation_poll_seconds = GATE_SATURATION_POLL_INITIAL_SECONDS
                    else:
                        # Slots are saturated with no exact wake time. Back off,
                        # but never sleep past the soonest slot expiry -- a slot
                        # cannot outlive it, so it is a genuine upper bound.
                        (soonest_expiry,) = connection.execute(
                            "SELECT MIN(expires_at) FROM drive_request_gate_slots"
                        ).fetchone()
                        sleep_for = saturation_poll_seconds
                        if soonest_expiry is not None:
                            sleep_for = min(
                                sleep_for,
                                max(
                                    float(soonest_expiry) - now,
                                    GATE_SATURATION_POLL_INITIAL_SECONDS,
                                ),
                            )
                        saturation_poll_seconds = min(
                            saturation_poll_seconds * GATE_SATURATION_POLL_BACKOFF_MULTIPLIER,
                            GATE_SATURATION_POLL_CEILING_SECONDS,
                        )
                self._sleep(sleep_for)
        finally:
            connection.close()

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
                (self._time() + self._provider_cooldown_seconds,),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drive_request_gate (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    next_allowed_at REAL NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(drive_request_gate)"
                ).fetchall()
            }
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drive_request_gate_slots (
                    slot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expires_at REAL NOT NULL
                )
                """
            )
            if "in_flight_until" not in columns:
                return

            legacy_row = connection.execute(
                "SELECT next_allowed_at, in_flight_until "
                "FROM drive_request_gate WHERE id = 1"
            ).fetchone()
            if legacy_row is not None and float(legacy_row[1]) > self._time():
                connection.execute(
                    "INSERT INTO drive_request_gate_slots (expires_at) VALUES (?)",
                    (float(legacy_row[1]),),
                )

            connection.execute(
                """
                CREATE TABLE drive_request_gate_current (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    next_allowed_at REAL NOT NULL
                )
                """
            )
            if legacy_row is not None:
                connection.execute(
                    "INSERT INTO drive_request_gate_current (id, next_allowed_at) "
                    "VALUES (1, ?)",
                    (float(legacy_row[0]),),
                )
            connection.execute("DROP TABLE drive_request_gate")
            connection.execute(
                "ALTER TABLE drive_request_gate_current RENAME TO drive_request_gate"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        # WAL mode persists in the database header, but this data is ephemeral
        # rate-limiter bookkeeping that may be rebuilt at any path, so setting
        # it on every connect (before any transaction starts) is the more
        # robust choice rather than relying on a one-time initializer. Slot
        # rows are expiry-reclaimed by design, so relaxing durability to
        # NORMAL trades a crash losing the last commit for far less fsync
        # churn on the write-heavy claim/release path.
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection


__all__ = ["DriveRequestGate"]
