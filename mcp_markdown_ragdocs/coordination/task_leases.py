"""Durable execution leases for dequeued Huey tasks.

The lease is claimed immediately after Huey dequeues a task. Claimed work is
at-least-once: an expired owner is reclaimed and its serialized task is
re-enqueued before a worker starts. The dequeue and lease claim are separate
SQLite boundaries, so a process crash in that narrow gap remains a known
at-most-once edge of the existing Huey dequeue contract.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

ACTIVE_LEASE = "active"
COMPLETED_LEASE = "completed"
FAILED_LEASE = "failed"
RECLAIMED_LEASE = "reclaimed"
DEFAULT_LEASE_TIMEOUT_SECONDS = 30.0
# Terminal leases are only ever read for diagnostics (gdrive/leases.py.get(),
# daemon/queue_status.py) — claim()/reclaim_expired() never look past the
# active state, so a pruned row cannot affect crash recovery. 3 days covers a
# normal troubleshooting window while keeping the table from growing forever.
TASK_LEASE_RETENTION_SECONDS = 3 * 24 * 3600.0
PRUNE_BATCH_LIMIT = 500


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


class TaskLeasePort(Protocol):
    """Application capabilities required for durable task leases."""

    def claim(
        self,
        task_id: str,
        *,
        task_name: str | None,
        owner_token: str,
        payload: bytes,
        now: float | None = None,
    ) -> bool: ...

    def heartbeat(
        self,
        task_id: str,
        *,
        owner_token: str,
        now: float | None = None,
    ) -> bool: ...

    def complete(
        self,
        task_id: str,
        *,
        owner_token: str,
        now: float | None = None,
    ) -> bool: ...

    def fail(
        self,
        task_id: str,
        *,
        owner_token: str,
        error: str,
        now: float | None = None,
    ) -> bool: ...

    def get(self, task_id: str) -> TaskLease | None: ...

    def acquire_writer(
        self, resource: str, owner_token: str, *, now: float | None = None
    ) -> bool: ...

    def heartbeat_writer(
        self,
        resource: str,
        owner_token: str,
        *,
        now: float | None = None,
    ) -> bool: ...

    def release_writer(self, resource: str, owner_token: str) -> bool: ...

    def writer_owner(self, resource: str, *, now: float | None = None) -> str | None: ...


class TaskLeaseStore(TaskLeasePort):
    """SQLite adapter for the application task lease port."""

    def __init__(
        self,
        db_path: Path,
        *,
        timeout_seconds: float = DEFAULT_LEASE_TIMEOUT_SECONDS,
    ) -> None:
        self._db_path = Path(db_path)
        self._timeout_seconds = timeout_seconds
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = self._open_connection()
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

    def prune_terminal(
        self,
        *,
        now: float | None = None,
        retention_seconds: float = TASK_LEASE_RETENTION_SECONDS,
        limit: int = PRUNE_BATCH_LIMIT,
    ) -> list[str]:
        """Delete terminal leases past retention, bounded to `limit` rows.

        Only completed/failed/reclaimed rows older than retention are
        eligible. claim() and reclaim_expired() never read a lease once it
        has left the active state, so removing a terminal row here cannot
        affect crash recovery: a subsequent claim() for the same task_id
        simply sees no prior row and starts at attempt=1, which is correct
        once the old row has aged out.
        """
        timestamp = time.time() if now is None else now
        cutoff = timestamp - retention_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT task_id
                FROM task_leases
                WHERE state IN (?, ?, ?)
                  AND completed_at IS NOT NULL
                  AND completed_at <= ?
                LIMIT ?
                """,
                (COMPLETED_LEASE, FAILED_LEASE, RECLAIMED_LEASE, cutoff, limit),
            ).fetchall()
            if not rows:
                return []

            task_ids = [str(row["task_id"]) for row in rows]
            placeholders = ",".join("?" for _ in task_ids)
            connection.execute(
                f"DELETE FROM task_leases WHERE task_id IN ({placeholders})",
                task_ids,
            )
            return task_ids

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

    def acquire_writer(
        self, resource: str, owner_token: str, *, now: float | None = None
    ) -> bool:
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
                (resource,),
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
                (resource, owner_token, timestamp, timestamp),
            )
            return True

    def heartbeat_writer(
        self,
        resource: str,
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
                (timestamp, resource, owner_token),
            )
            return cursor.rowcount == 1

    def release_writer(self, resource: str, owner_token: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM writer_leases
                WHERE resource = ? AND owner_token = ?
                """,
                (resource, owner_token),
            )
            return cursor.rowcount == 1

    def writer_owner(self, resource: str, *, now: float | None = None) -> str | None:
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
                (resource,),
            ).fetchone()
            if row is None:
                return None
            if row["heartbeat_at"] <= cutoff:
                connection.execute(
                    "DELETE FROM writer_leases WHERE resource = ?",
                    (resource,),
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

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._db_path,
            timeout=self._timeout_seconds,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One connection is reused for the life of the store instead of one
        # per call (huey consumer threads plus a per-task heartbeat thread
        # otherwise churn ~1.5-2 connect/close cycles/sec, see
        # worker/consumer.py LEASE_HEARTBEAT_INTERVAL_SECONDS). A single
        # sqlite3.Connection is not safe for concurrent use across threads
        # even with check_same_thread=False, so a lock serializes access --
        # the same shape huey itself uses for this file's other connection
        # (huey/storage.py BaseSqlStorage.db()). Calls here are single small
        # statements, so lock contention on the 5s heartbeat path is minor.
        with self._lock, self._connection as connection:
            yield connection

    def close(self) -> None:
        """Close the underlying connection. Call once, on store disposal."""
        with self._lock:
            self._connection.close()


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
