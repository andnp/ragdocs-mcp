"""Transactional schema migrations for the coordination database."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

CURRENT_COORDINATION_SCHEMA_VERSION = 1


class CoordinationSchemaError(RuntimeError):
    """Raised when a coordination database cannot be migrated safely."""


FailureInjector = Callable[[str], None]


def migrate_coordination_schema(
    db_path: Path,
    *,
    failure_injector: FailureInjector | None = None,
) -> bool:
    """Migrate a coordination database to the additive target schema.

    SQLite ``user_version`` is the sole schema-version authority. Legacy
    tables remain in place so this operation can be followed by an explicit
    cutover or restored from a database backup without an in-place downgrade.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CURRENT_COORDINATION_SCHEMA_VERSION:
            raise CoordinationSchemaError(
                f"unsupported coordination schema version: {version}"
            )
        if version == CURRENT_COORDINATION_SCHEMA_VERSION:
            return False

        try:
            connection.execute("BEGIN IMMEDIATE")
            _create_target_schema(connection)
            _inject(failure_injector, "target_schema_created")
            _copy_legacy_rows(connection)
            _inject(failure_injector, "legacy_rows_copied")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise CoordinationSchemaError(
                    f"coordination database integrity check failed: {integrity}"
                )
            connection.execute(
                f"PRAGMA user_version = {CURRENT_COORDINATION_SCHEMA_VERSION}"
            )
            _inject(failure_injector, "version_recorded")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return True


def _create_target_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS work_items (
            work_item_id TEXT PRIMARY KEY,
            operation TEXT NOT NULL,
            canonical_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            state TEXT NOT NULL,
            claim_token TEXT,
            fencing_token INTEGER NOT NULL DEFAULT 0,
            claim_observed_at REAL,
            observed_at REAL NOT NULL,
            attempt_count INTEGER NOT NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            max_retries INTEGER NOT NULL DEFAULT 3,
            dispatch_attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at REAL,
            dead_letter INTEGER NOT NULL DEFAULT 0,
            UNIQUE(operation, canonical_key)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_work_items_state_attempt
            ON work_items (state, next_attempt_at, observed_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS resource_leases (
            resource TEXT PRIMARY KEY,
            owner_token TEXT NOT NULL,
            fencing_token INTEGER NOT NULL DEFAULT 0,
            acquired_at REAL NOT NULL,
            heartbeat_at REAL NOT NULL
        )
        """
    )


def _copy_legacy_rows(connection: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "work_intents" in tables:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(work_intents)")
        }
        failure_count = "failure_count" if "failure_count" in columns else "0"
        connection.execute(
            f"""
            INSERT INTO work_items (
                work_item_id, operation, canonical_key, payload_json, state,
                claim_token, claim_observed_at, observed_at, attempt_count,
                failure_count, error
            )
            SELECT intent_id, operation, canonical_key, payload_json, state,
                   claim_token, claim_observed_at, observed_at, attempt,
                   {failure_count}, error
            FROM work_intents
            """
        )
    if "writer_leases" in tables:
        connection.execute(
            """
            INSERT INTO resource_leases (
                resource, owner_token, acquired_at, heartbeat_at
            )
            SELECT resource, owner_token, acquired_at, heartbeat_at
            FROM writer_leases
            """
        )


def _inject(
    failure_injector: FailureInjector | None,
    stage: str,
) -> None:
    if failure_injector is not None:
        failure_injector(stage)
