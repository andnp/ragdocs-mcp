"""
Unit tests for get_system_status tool functionality.

Commit 4.4: Verifies system status reporting including journal counts
and lifecycle state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.memory.journal import System1Journal
from src.storage.db import DatabaseManager


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "test.db")


class TestSystemStatusData:
    def test_journal_count_by_status(self, db: DatabaseManager) -> None:
        """Journal provides correct status counts for status report."""
        journal = System1Journal(db)
        journal.record("thought 1")
        journal.record("thought 2")
        e3 = journal.record("thought 3")
        journal.mark_processed([e3.id])

        counts = journal.count_by_status()
        assert counts["pending"] == 2
        assert counts["processed"] == 1

    def test_journal_empty_counts(self, db: DatabaseManager) -> None:
        """Empty journal returns empty counts."""
        journal = System1Journal(db)
        assert journal.count_by_status() == {}

    def test_last_consolidation_not_set(self, db: DatabaseManager) -> None:
        """No consolidation record returns None."""
        conn = db.get_connection()
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = 'last_consolidation'"
        ).fetchone()
        assert row is None

    def test_last_consolidation_persists(self, db: DatabaseManager) -> None:
        """Consolidation timestamp can be stored and retrieved."""
        conn = db.get_connection()
        data = json.dumps({"timestamp": "2024-01-15T12:00:00Z", "entries_processed": 5})
        conn.execute(
            "INSERT INTO system_state (key, value) VALUES (?, ?)",
            ("last_consolidation", data),
        )
        conn.commit()

        row = conn.execute(
            "SELECT value FROM system_state WHERE key = 'last_consolidation'"
        ).fetchone()
        assert row is not None
        result = json.loads(row[0])
        assert result["timestamp"] == "2024-01-15T12:00:00Z"
        assert result["entries_processed"] == 5


class TestLifecycleStateReporting:
    def test_lifecycle_state_value(self) -> None:
        """LifecycleState has string values for status display."""
        from src.lifecycle import LifecycleState

        assert LifecycleState.READY.value == "ready"
        assert LifecycleState.READY_PRIMARY.value == "ready_primary"
        assert LifecycleState.READY_REPLICA.value == "ready_replica"
        assert LifecycleState.TERMINATED.value == "terminated"

    def test_lifecycle_coordinator_state(self) -> None:
        """LifecycleCoordinator exposes current state."""
        from src.lifecycle import LifecycleCoordinator, LifecycleState

        coord = LifecycleCoordinator()
        assert coord.state == LifecycleState.UNINITIALIZED


class TestStatusToolRegistration:
    def test_tool_in_metadata_tools_list(self) -> None:
        """get_system_status is included in metadata tools."""
        from src.mcp.tools.metadata_tools import get_metadata_tools

        tools = get_metadata_tools()
        names = [t.name for t in tools]
        assert "get_system_status" in names

    def test_handler_is_registered(self) -> None:
        """Handler for get_system_status exists in registry."""
        import src.mcp.tools.metadata_tools  # noqa: F401 - registers handlers

        from src.mcp.handlers import get_handler

        handler = get_handler("get_system_status")
        assert handler is not None
