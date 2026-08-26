"""Durable intent state for task producers and Huey workers."""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mcp_markdown_ragdocs.coordination.task_leases import (
    DEFAULT_LEASE_TIMEOUT_SECONDS,
)

PENDING = "pending"
CLAIMED = "claimed"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
MAX_AUTOMATIC_FAILURES = 3
# Terminal intents are re-read after termination: task_intents.py and
# indexing/tasks.py call find() when a claim is refused, to tell a permanent
# failure from "already pending", and a permanently-failed intent's
# failure_count backs the retry ceiling. Keep this longer than
# TASK_LEASE_RETENTION_SECONDS so a pruned row does not casually reopen a
# retry ceiling that was reached only a few days ago. submit() already
# unconditionally reopens a succeeded row to PENDING on resubmission, so
# pruning succeeded rows changes nothing about whether work gets redone.
WORK_INTENT_RETENTION_SECONDS = 7 * 24 * 3600.0
PRUNE_BATCH_LIMIT = 500

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkIntent:
    intent_id: str
    operation: str
    canonical_key: str
    payload: dict[str, Any]
    state: str
    claim_token: str | None
    claim_observed_at: float | None
    observed_at: float
    attempt: int
    failure_count: int
    error: str | None


class WorkIntentPort(Protocol):
    """Application capabilities required for durable work intents."""

    def submit(
        self,
        operation: str,
        canonical_key: str,
        payload: dict[str, Any],
        *,
        force_reopen: bool = False,
        now: float | None = None,
    ) -> WorkIntent: ...

    def claim(
        self,
        intent_id: str,
        *,
        now: float | None = None,
    ) -> tuple[WorkIntent, str] | None: ...

    def start(self, intent_id: str, claim_token: str) -> bool: ...

    def succeed(
        self,
        intent_id: str,
        claim_token: str,
        *,
        now: float | None = None,
    ) -> bool: ...

    def fail(
        self,
        intent_id: str,
        claim_token: str,
        error: str,
        *,
        now: float | None = None,
    ) -> bool: ...

    def release(
        self,
        intent_id: str,
        claim_token: str,
        *,
        now: float | None = None,
    ) -> bool: ...

    def get(self, intent_id: str) -> WorkIntent | None: ...

    def find(self, operation: str, canonical_key: str) -> WorkIntent | None: ...

    def list_active(self, *, limit: int = 100) -> list[WorkIntent]: ...


class WorkIntentStore(WorkIntentPort):
    """SQLite adapter for the application work intent port."""

    def __init__(
        self,
        db_path: Path,
        *,
        claim_timeout_seconds: float = DEFAULT_LEASE_TIMEOUT_SECONDS,
        retry_policy: Callable[[str, int], bool] | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._claim_timeout_seconds = claim_timeout_seconds
        self._retry_policy = retry_policy or (lambda operation, failure_count: True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def submit(
        self,
        operation: str,
        canonical_key: str,
        payload: dict[str, Any],
        *,
        force_reopen: bool = False,
        now: float | None = None,
    ) -> WorkIntent:
        timestamp = time.time() if now is None else now
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM work_intents
                WHERE operation = ? AND canonical_key = ?
                """,
                (operation, canonical_key),
            ).fetchone()
            if row is None:
                intent_id = secrets.token_hex(16)
                connection.execute(
                    """
                    INSERT INTO work_intents (
                        intent_id, operation, canonical_key, payload_json,
                        state, claim_token, claim_observed_at, observed_at,
                        attempt, error
                    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, 0, NULL)
                    """,
                    (
                        intent_id,
                        operation,
                        canonical_key,
                        encoded,
                        PENDING,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM work_intents WHERE intent_id = ?",
                    (intent_id,),
                ).fetchone()
            elif (
                row["state"] == FAILED
                and (
                    self._retry_policy(operation, int(row["failure_count"]))
                    or force_reopen
                )
            ) or row["state"] == SUCCEEDED or (
                force_reopen
                and row["state"] in {CLAIMED, RUNNING}
                and row["claim_observed_at"] is not None
                and float(row["claim_observed_at"])
                <= timestamp - self._claim_timeout_seconds
            ):
                connection.execute(
                    """
                    UPDATE work_intents
                    SET payload_json = ?, state = ?, claim_token = NULL,
                        claim_observed_at = NULL, observed_at = ?, error = NULL
                    WHERE intent_id = ?
                    """,
                    (encoded, PENDING, timestamp, row["intent_id"]),
                )
                row = connection.execute(
                    "SELECT * FROM work_intents WHERE intent_id = ?",
                    (row["intent_id"],),
                ).fetchone()
            assert row is not None
            return _row_to_intent(row)

    def claim(
        self,
        intent_id: str,
        *,
        now: float | None = None,
    ) -> tuple[WorkIntent, str] | None:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if row is None or row["state"] not in {PENDING, CLAIMED}:
                return None
            if (
                row["state"] == CLAIMED
                and row["claim_observed_at"] is not None
                and float(row["claim_observed_at"])
                > timestamp - self._claim_timeout_seconds
            ):
                return None
            token = secrets.token_hex(16)
            connection.execute(
                """
                UPDATE work_intents
                SET state = ?, claim_token = ?, claim_observed_at = ?,
                    observed_at = ?, attempt = attempt + 1, error = NULL
                WHERE intent_id = ?
                """,
                (CLAIMED, token, timestamp, timestamp, intent_id),
            )
            claimed = connection.execute(
                "SELECT * FROM work_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            assert claimed is not None
            return _row_to_intent(claimed), token

    def start(self, intent_id: str, claim_token: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE work_intents
                SET state = ?
                WHERE intent_id = ? AND claim_token = ? AND state = ?
                """,
                (RUNNING, intent_id, claim_token, CLAIMED),
            )
            return cursor.rowcount == 1

    def succeed(
        self,
        intent_id: str,
        claim_token: str,
        *,
        now: float | None = None,
    ) -> bool:
        return self._transition(
            intent_id,
            claim_token,
            from_states={CLAIMED, RUNNING},
            state=SUCCEEDED,
            now=now,
        )

    def fail(
        self,
        intent_id: str,
        claim_token: str,
        error: str,
        *,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE work_intents
                SET state = ?, claim_token = NULL, claim_observed_at = NULL,
                    observed_at = ?, failure_count = failure_count + 1, error = ?
                WHERE intent_id = ? AND claim_token = ?
                  AND state IN (?, ?)
                """,
                (FAILED, timestamp, error, intent_id, claim_token, CLAIMED, RUNNING),
            )
            failed = cursor.rowcount == 1
            if failed:
                row = connection.execute(
                    """
                    SELECT operation, failure_count
                    FROM work_intents
                    WHERE intent_id = ?
                    """,
                    (intent_id,),
                ).fetchone()
                if row is not None and not self._retry_policy(
                    str(row["operation"]), int(row["failure_count"])
                ):
                    logger.error(
                        "Automatic retry ceiling reached for work intent %s (operation=%s)",
                        intent_id,
                        row["operation"],
                    )
            return failed

    def release(self, intent_id: str, claim_token: str, *, now: float | None = None) -> bool:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE work_intents
                SET state = ?, claim_token = NULL, claim_observed_at = NULL,
                    observed_at = ?
                WHERE intent_id = ? AND claim_token = ? AND state IN (?, ?)
                """,
                (PENDING, timestamp, intent_id, claim_token, CLAIMED, RUNNING),
            )
            return cursor.rowcount == 1

    def recover_stale_claims(self, *, now: float | None = None) -> int:
        timestamp = time.time() if now is None else now
        cutoff = timestamp - self._claim_timeout_seconds
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE work_intents
                SET state = CASE
                        WHEN attempt >= ? THEN ?
                        ELSE ?
                    END,
                    claim_token = NULL, claim_observed_at = NULL,
                    observed_at = ?, error = ?
                WHERE state IN (?, ?) AND claim_observed_at <= ?
                """,
                (
                    MAX_AUTOMATIC_FAILURES * 2,
                    FAILED,
                    PENDING,
                    timestamp,
                    "claim expired before terminal state",
                    CLAIMED,
                    RUNNING,
                    cutoff,
                ),
            )
            return cursor.rowcount

    def reclaim_stale_claim(
        self,
        intent_id: str,
        claim_token: str,
        *,
        now: float | None = None,
    ) -> tuple[WorkIntent, str] | None:
        """Replace one expired claim with a fresh token for queue recovery."""
        timestamp = time.time() if now is None else now
        cutoff = timestamp - self._claim_timeout_seconds
        token = secrets.token_hex(16)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE work_intents
                SET state = ?, claim_token = ?, claim_observed_at = ?,
                    observed_at = ?, attempt = attempt + 1, error = NULL
                WHERE intent_id = ? AND claim_token = ?
                  AND state IN (?, ?) AND claim_observed_at <= ?
                """,
                (
                    CLAIMED,
                    token,
                    timestamp,
                    timestamp,
                    intent_id,
                    claim_token,
                    CLAIMED,
                    RUNNING,
                    cutoff,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM work_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            assert row is not None
            return _row_to_intent(row), token

    def prune_terminal(
        self,
        *,
        now: float | None = None,
        retention_seconds: float = WORK_INTENT_RETENTION_SECONDS,
        limit: int = PRUNE_BATCH_LIMIT,
    ) -> list[str]:
        """Delete succeeded/failed intents past retention, bounded to `limit` rows.

        PENDING/CLAIMED/RUNNING rows are never touched. A pruned row cannot
        be re-claimed, since claim() only accepts PENDING/CLAIMED states;
        find()/submit() simply see no prior row and behave as for a
        brand-new canonical_key.
        """
        timestamp = time.time() if now is None else now
        cutoff = timestamp - retention_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT intent_id
                FROM work_intents
                WHERE state IN (?, ?) AND observed_at <= ?
                LIMIT ?
                """,
                (SUCCEEDED, FAILED, cutoff, limit),
            ).fetchall()
            if not rows:
                return []

            intent_ids = [str(row["intent_id"]) for row in rows]
            placeholders = ",".join("?" for _ in intent_ids)
            connection.execute(
                f"DELETE FROM work_intents WHERE intent_id IN ({placeholders})",
                intent_ids,
            )
            return intent_ids

    def get(self, intent_id: str) -> WorkIntent | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM work_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
        return None if row is None else _row_to_intent(row)

    def find(self, operation: str, canonical_key: str) -> WorkIntent | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM work_intents
                WHERE operation = ? AND canonical_key = ?
                """,
                (operation, canonical_key),
            ).fetchone()
        return None if row is None else _row_to_intent(row)

    def list_active(self, *, limit: int = 100) -> list[WorkIntent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM work_intents
                WHERE state IN (?, ?, ?)
                ORDER BY observed_at ASC
                LIMIT ?
                """,
                (PENDING, CLAIMED, RUNNING, max(1, min(limit, 1000))),
            ).fetchall()
        return [_row_to_intent(row) for row in rows]

    def _transition(
        self,
        intent_id: str,
        claim_token: str,
        *,
        from_states: set[str],
        state: str,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        placeholders = ",".join("?" for _ in from_states)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE work_intents
                SET state = ?, claim_token = NULL, claim_observed_at = NULL,
                    observed_at = ?, failure_count = 0, error = NULL
                WHERE intent_id = ? AND claim_token = ?
                  AND state IN ({placeholders})
                """,
                (state, timestamp, intent_id, claim_token, *from_states),
            )
            return cursor.rowcount == 1

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS work_intents (
                    intent_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    canonical_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    claim_token TEXT,
                    claim_observed_at REAL,
                    observed_at REAL NOT NULL,
                    attempt INTEGER NOT NULL,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    UNIQUE(operation, canonical_key)
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(work_intents)")
            }
            if "failure_count" not in columns:
                connection.execute(
                    "ALTER TABLE work_intents ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_work_intents_state_observed
                ON work_intents (state, observed_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection


def _row_to_intent(row: sqlite3.Row) -> WorkIntent:
    return WorkIntent(
        intent_id=str(row["intent_id"]),
        operation=str(row["operation"]),
        canonical_key=str(row["canonical_key"]),
        payload=json.loads(str(row["payload_json"])),
        state=str(row["state"]),
        claim_token=(
            None if row["claim_token"] is None else str(row["claim_token"])
        ),
        claim_observed_at=(
            None
            if row["claim_observed_at"] is None
            else float(row["claim_observed_at"])
        ),
        observed_at=float(row["observed_at"]),
        attempt=int(row["attempt"]),
        failure_count=int(row["failure_count"]),
        error=None if row["error"] is None else str(row["error"]),
    )
