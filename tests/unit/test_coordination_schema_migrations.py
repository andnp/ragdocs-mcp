"""Contract tests for coordination schema migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.coordination.schema_migrations import (
    CURRENT_COORDINATION_SCHEMA_VERSION,
    CoordinationSchemaError,
    migrate_coordination_schema,
)


def test_migration_initializes_empty_database_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """An empty database receives the target schema exactly once."""
    path = tmp_path / "coordination.db"

    assert migrate_coordination_schema(path)
    assert not migrate_coordination_schema(path)

    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert {"work_items", "resource_leases"} <= tables
    assert version == CURRENT_COORDINATION_SCHEMA_VERSION


def test_migration_preserves_legacy_intents_and_resource_leases(
    tmp_path: Path,
) -> None:
    """Legacy durable state is copied while its original tables remain available."""
    path = tmp_path / "coordination.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE work_intents (
                intent_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                claim_token TEXT,
                claim_observed_at REAL,
                observed_at REAL NOT NULL,
                attempt INTEGER NOT NULL,
                failure_count INTEGER NOT NULL,
                error TEXT
            );
            CREATE TABLE writer_leases (
                resource TEXT PRIMARY KEY,
                owner_token TEXT NOT NULL,
                acquired_at REAL NOT NULL,
                heartbeat_at REAL NOT NULL
            );
            INSERT INTO work_intents VALUES
                ('item-1', 'index', 'doc-1', '{"path":"doc.md"}',
                 'failed', 'claim-1', 12.0, 13.0, 4, 2, 'boom');
            INSERT INTO writer_leases VALUES ('index', 'owner-1', 10.0, 11.0);
            """
        )

    migrate_coordination_schema(path)

    with sqlite3.connect(path) as connection:
        item = connection.execute(
            "SELECT work_item_id, payload_json, state, attempt_count, failure_count, error "
            "FROM work_items"
        ).fetchone()
        lease = connection.execute(
            "SELECT resource, owner_token, acquired_at, heartbeat_at "
            "FROM resource_leases"
        ).fetchone()
        legacy_count = connection.execute("SELECT count(*) FROM work_intents").fetchone()[0]

    assert item == ("item-1", '{"path":"doc.md"}', "failed", 4, 2, "boom")
    assert lease == ("index", "owner-1", 10.0, 11.0)
    assert legacy_count == 1


def test_migration_rolls_back_target_schema_when_copy_fails(tmp_path: Path) -> None:
    """A deterministic migration failure leaves the database at its prior version."""
    path = tmp_path / "coordination.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE work_intents (intent_id TEXT PRIMARY KEY, operation TEXT NOT NULL, "
            "canonical_key TEXT NOT NULL, payload_json TEXT NOT NULL, state TEXT NOT NULL, "
            "claim_token TEXT, claim_observed_at REAL, observed_at REAL NOT NULL, "
            "attempt INTEGER NOT NULL, failure_count INTEGER NOT NULL, error TEXT)"
        )
        connection.execute(
            "INSERT INTO work_intents VALUES "
            "('item-1', 'index', 'doc-1', '{}', 'pending', NULL, NULL, 1.0, 0, 0, NULL)"
        )

    def fail_after_copy(stage: str) -> None:
        if stage == "legacy_rows_copied":
            raise RuntimeError("injected migration failure")

    with pytest.raises(RuntimeError, match="injected migration failure"):
        migrate_coordination_schema(
            path,
            failure_injector=fail_after_copy,
        )

    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        count = connection.execute("SELECT count(*) FROM work_intents").fetchone()[0]

    assert "work_items" not in tables
    assert version == 0
    assert count == 1


def test_migration_rejects_newer_schema_without_mutation(tmp_path: Path) -> None:
    """A newer database is rejected instead of being downgraded or altered."""
    path = tmp_path / "coordination.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(CoordinationSchemaError, match="unsupported"):
        migrate_coordination_schema(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99
