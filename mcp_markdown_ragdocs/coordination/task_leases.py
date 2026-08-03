"""Durable execution leases for dequeued Huey tasks.

The lease is claimed immediately after Huey dequeues a task. Claimed work is
at-least-once: an expired owner is reclaimed and its serialized task is
re-enqueued before a worker starts. The dequeue and lease claim are separate
SQLite boundaries, so a process crash in that narrow gap remains a known
at-most-once edge of the existing Huey dequeue contract.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path

ACTIVE_LEASE = "active"
COMPLETED_LEASE = "completed"
FAILED_LEASE = "failed"
RECLAIMED_LEASE = "reclaimed"
INDEX_WRITER_RESOURCE = "index-writer"
DEFAULT_LEASE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class TaskLease:
    task_id: str
    task_name: str | None
    state: str
    owner_token: str
    claimed_at: float
    heartbeat_at: float
    completed_at: float | None
    error: str | None
    payload: bytes | None
    attempt: int


class TaskLeaseStore:
    """Stores task ownership in the same SQLite database as Huey."""

    def __init__(
        self,
        db_path: Path,
        *,
        timeout_seconds: float = DEFAULT_LEASE_TIMEOUT_SECONDS,
    ) -> None:
        self._db_path = Path(db_path)
        self._timeout_seconds = timeout_seconds
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def claim(
        self,
        task_id: str,
        *,
        task_name: str | None,
        owner_token: str,
        payload: bytes,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state, heartbeat_at, attempt
                FROM task_leases
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if (
                row is not None
                and row["state"] == ACTIVE_LEASE
                and row["heartbeat_at"] > timestamp - self._timeout_seconds
            ):
                return False

            attempt = 1 if row is None else int(row["attempt"]) + 1
            connection.execute(
                """
                INSERT INTO task_leases (
                    task_id, task_name, state, owner_token, claimed_at,
                    heartbeat_at, completed_at, error, payload, attempt
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    task_name = excluded.task_name,
                    state = excluded.state,
                    owner_token = excluded.owner_token,
                    claimed_at = excluded.claimed_at,
                    heartbeat_at = excluded.heartbeat_at,
                    completed_at = NULL,
                    error = NULL,
                    payload = excluded.payload,
                    attempt = excluded.attempt
                """,
                (
                    task_id,
                    task_name,
                    ACTIVE_LEASE,
                    owner_token,
                    timestamp,
                    timestamp,
                    payload,
                    attempt,
                ),
            )
            return True

    def heartbeat(
        self,
        task_id: str,
        *,
        owner_token: str,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE task_leases
                SET heartbeat_at = ?
                WHERE task_id = ? AND owner_token = ? AND state = ?
                """,
                (timestamp, task_id, owner_token, ACTIVE_LEASE),
            )
            return cursor.rowcount == 1

    def complete(
        self,
        task_id: str,
        *,
        owner_token: str,
        now: float | None = None,
    ) -> bool:
        return self._finish(
            task_id,
            owner_token=owner_token,
            state=COMPLETED_LEASE,
            now=now,
        )

    def fail(
        self,
        task_id: str,
        *,
        owner_token: str,
        error: str,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE task_leases
                SET state = ?, completed_at = ?, error = ?
                WHERE task_id = ? AND owner_token = ? AND state = ?
                """,
                (
                    FAILED_LEASE,
                    timestamp,
                    error,
                    task_id,
                    owner_token,
                    ACTIVE_LEASE,
                ),
            )
            return cursor.rowcount == 1

    def reclaim_expired(self, *, now: float | None = None) -> list[TaskLease]:
        timestamp = time.time() if now is None else now
        cutoff = timestamp - self._timeout_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT *
                FROM task_leases
                WHERE state = ? AND heartbeat_at <= ?
                """,
                (ACTIVE_LEASE, cutoff),
            ).fetchall()
            if not rows:
                return []

            task_ids = [str(row["task_id"]) for row in rows]
            connection.executemany(
                """
                UPDATE task_leases
                SET state = ?, completed_at = ?, error = ?
                WHERE task_id = ? AND state = ?
                """,
                [
                    (
                        RECLAIMED_LEASE,
                        timestamp,
                        "lease expired before terminal state",
                        task_id,
                        ACTIVE_LEASE,
                    )
                    for task_id in task_ids
                ],
            )
            return [
                replace(
                    _row_to_lease(row),
                    state=RECLAIMED_LEASE,
                    completed_at=timestamp,
                    error="lease expired before terminal state",
                )
                for row in rows
            ]

    def active_count(self, *, now: float | None = None) -> int:
        timestamp = time.time() if now is None else now
        cutoff = timestamp - self._timeout_seconds
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM task_leases
                WHERE state = ? AND heartbeat_at > ?
                """,
                (ACTIVE_LEASE, cutoff),
            ).fetchone()
            assert row is not None
            return int(row[0])

    def get(self, task_id: str) -> TaskLease | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return None if row is None else _row_to_lease(row)

    def acquire_writer(self, owner_token: str, *, now: float | None = None) -> bool:
        timestamp = time.time() if now is None else now
        cutoff = timestamp - self._timeout_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT owner_token, heartbeat_at
                FROM writer_leases
                WHERE resource = ?
                """,
                (INDEX_WRITER_RESOURCE,),
            ).fetchone()
            if (
                row is not None
                and row["owner_token"] != owner_token
                and row["heartbeat_at"] > cutoff
            ):
                return False

            connection.execute(
                """
                INSERT INTO writer_leases (
                    resource, owner_token, acquired_at, heartbeat_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(resource) DO UPDATE SET
                    owner_token = excluded.owner_token,
                    acquired_at = excluded.acquired_at,
                    heartbeat_at = excluded.heartbeat_at
                """,
                (INDEX_WRITER_RESOURCE, owner_token, timestamp, timestamp),
            )
            return True

    def heartbeat_writer(
        self,
        owner_token: str,
        *,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE writer_leases
                SET heartbeat_at = ?
                WHERE resource = ? AND owner_token = ?
                """,
                (timestamp, INDEX_WRITER_RESOURCE, owner_token),
            )
            return cursor.rowcount == 1

    def release_writer(self, owner_token: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM writer_leases
                WHERE resource = ? AND owner_token = ?
                """,
                (INDEX_WRITER_RESOURCE, owner_token),
            )
            return cursor.rowcount == 1

    def writer_owner(self, *, now: float | None = None) -> str | None:
        timestamp = time.time() if now is None else now
        cutoff = timestamp - self._timeout_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT owner_token, heartbeat_at
                FROM writer_leases
                WHERE resource = ?
                """,
                (INDEX_WRITER_RESOURCE,),
            ).fetchone()
            if row is None:
                return None
            if row["heartbeat_at"] <= cutoff:
                connection.execute(
                    "DELETE FROM writer_leases WHERE resource = ?",
                    (INDEX_WRITER_RESOURCE,),
                )
                return None
            return str(row["owner_token"])

    def _finish(
        self,
        task_id: str,
        *,
        owner_token: str,
        state: str,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE task_leases
                SET state = ?, completed_at = ?
                WHERE task_id = ? AND owner_token = ? AND state = ?
                """,
                (state, timestamp, task_id, owner_token, ACTIVE_LEASE),
            )
            return cursor.rowcount == 1

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_leases (
                    task_id TEXT PRIMARY KEY,
                    task_name TEXT,
                    state TEXT NOT NULL,
                    owner_token TEXT NOT NULL,
                    claimed_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    completed_at REAL,
                    error TEXT,
                    payload BLOB,
                    attempt INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_leases_active_heartbeat
                ON task_leases (state, heartbeat_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS writer_leases (
                    resource TEXT PRIMARY KEY,
                    owner_token TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._db_path,
            timeout=self._timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


def _row_to_lease(row: sqlite3.Row) -> TaskLease:
    return TaskLease(
        task_id=str(row["task_id"]),
        task_name=row["task_name"],
        state=str(row["state"]),
        owner_token=str(row["owner_token"]),
        claimed_at=float(row["claimed_at"]),
        heartbeat_at=float(row["heartbeat_at"]),
        completed_at=(
            None if row["completed_at"] is None else float(row["completed_at"])
        ),
        error=row["error"],
        payload=row["payload"],
        attempt=int(row["attempt"]),
    )
