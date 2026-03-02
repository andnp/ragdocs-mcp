"""
Unit tests for System 1 journal (record_thought persistence).

Commit 4.1: Verifies raw thoughts are persisted to system1_journal table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.journal import System1Journal
from src.storage.db import DatabaseManager


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "test.db")


@pytest.fixture()
def journal(db: DatabaseManager) -> System1Journal:
    return System1Journal(db)


class TestRecordThought:
    def test_record_creates_entry(self, journal: System1Journal) -> None:
        """record() creates a journal entry with correct fields."""
        entry = journal.record("This is a test thought")
        assert entry.id is not None
        assert entry.content == "This is a test thought"
        assert entry.status == "pending"
        assert entry.timestamp > 0

    def test_record_strips_whitespace(self, journal: System1Journal) -> None:
        """record() strips leading/trailing whitespace."""
        entry = journal.record("  padded thought  ")
        assert entry.content == "padded thought"

    def test_record_rejects_empty(self, journal: System1Journal) -> None:
        """record() raises ValueError for empty content."""
        with pytest.raises(ValueError, match="cannot be empty"):
            journal.record("")
        with pytest.raises(ValueError, match="cannot be empty"):
            journal.record("   ")

    def test_record_persistence(self, db: DatabaseManager) -> None:
        """Entries persist in database across journal instances."""
        j1 = System1Journal(db)
        j1.record("thought 1")
        j1.record("thought 2")

        j2 = System1Journal(db)
        entries = j2.get_pending()
        assert len(entries) == 2
        assert entries[0].content == "thought 1"
        assert entries[1].content == "thought 2"

    def test_record_auto_increments_ids(self, journal: System1Journal) -> None:
        """Each record gets a unique, increasing ID."""
        e1 = journal.record("first")
        e2 = journal.record("second")
        assert e2.id > e1.id


class TestPendingRetrieval:
    def test_get_pending_returns_oldest_first(self, journal: System1Journal) -> None:
        """get_pending() returns entries oldest-first."""
        journal.record("old")
        journal.record("new")
        entries = journal.get_pending()
        assert entries[0].content == "old"
        assert entries[1].content == "new"

    def test_get_pending_respects_limit(self, journal: System1Journal) -> None:
        """get_pending(limit=N) returns at most N entries."""
        for i in range(10):
            journal.record(f"thought {i}")
        entries = journal.get_pending(limit=3)
        assert len(entries) == 3

    def test_get_pending_excludes_processed(self, journal: System1Journal) -> None:
        """get_pending() only returns pending entries."""
        e1 = journal.record("will process")
        journal.record("still pending")
        journal.mark_processed([e1.id])
        entries = journal.get_pending()
        assert len(entries) == 1
        assert entries[0].content == "still pending"


class TestStatusTransitions:
    def test_mark_processed(self, journal: System1Journal) -> None:
        """mark_processed() transitions pending -> processed."""
        e1 = journal.record("thought")
        count = journal.mark_processed([e1.id])
        assert count == 1
        assert journal.get_pending() == []

    def test_mark_processed_idempotent(self, journal: System1Journal) -> None:
        """mark_processed() on already-processed entry returns 0."""
        e1 = journal.record("thought")
        journal.mark_processed([e1.id])
        count = journal.mark_processed([e1.id])
        assert count == 0

    def test_mark_archived(self, journal: System1Journal) -> None:
        """mark_archived() transitions processed -> archived."""
        e1 = journal.record("thought")
        journal.mark_processed([e1.id])
        count = journal.mark_archived([e1.id])
        assert count == 1

    def test_mark_archived_requires_processed(self, journal: System1Journal) -> None:
        """mark_archived() on pending entry returns 0 (must be processed first)."""
        e1 = journal.record("thought")
        count = journal.mark_archived([e1.id])
        assert count == 0


class TestCounts:
    def test_count_by_status(self, journal: System1Journal) -> None:
        """count_by_status() returns correct counts."""
        e1 = journal.record("a")
        journal.record("b")
        journal.record("c")
        journal.mark_processed([e1.id])

        counts = journal.count_by_status()
        assert counts["pending"] == 2
        assert counts["processed"] == 1

    def test_count_by_status_empty(self, journal: System1Journal) -> None:
        """count_by_status() returns empty dict when no entries."""
        assert journal.count_by_status() == {}


class TestRecentEntries:
    def test_get_recent_returns_newest_first(self, journal: System1Journal) -> None:
        """get_recent() returns entries newest-first."""
        journal.record("old")
        journal.record("new")
        entries = journal.get_recent()
        assert entries[0].content == "new"
        assert entries[1].content == "old"

    def test_get_recent_includes_all_statuses(self, journal: System1Journal) -> None:
        """get_recent() includes entries of all statuses."""
        journal.record("pending one")
        e2 = journal.record("will process")
        journal.mark_processed([e2.id])
        entries = journal.get_recent()
        assert len(entries) == 2


class TestToDict:
    def test_entry_to_dict(self, journal: System1Journal) -> None:
        """JournalEntry.to_dict() returns expected format."""
        entry = journal.record("test")
        d = entry.to_dict()
        assert d["content"] == "test"
        assert d["status"] == "pending"
        assert "id" in d
        assert "timestamp" in d
