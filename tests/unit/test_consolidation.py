"""
Unit tests for memory consolidation (Fast and Slow modes).

Commit 4.3: Verifies journal entries are consolidated into memory files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.consolidation import (
    FastConsolidation,
    SlowConsolidation,
    _group_related_entries,
    _slugify,
)
from src.memory.journal import System1Journal
from src.storage.db import DatabaseManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "test.db")


@pytest.fixture()
def journal(db: DatabaseManager) -> System1Journal:
    return System1Journal(db)


@pytest.fixture()
def memory_path(tmp_path: Path) -> Path:
    p = tmp_path / "memories"
    p.mkdir()
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """Fake AI provider returning canned responses."""

    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self._response = response or {"actions": []}
        self._error = error
        self.prompts: list[str] = []

    async def ask(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        if self._error:
            raise self._error
        return self._response


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self) -> None:
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self) -> None:
        assert _slugify("foo@bar#baz!") == "foobarbaz"

    def test_max_length(self) -> None:
        result = _slugify("a" * 100, max_len=10)
        assert len(result) == 10

    def test_empty_returns_untitled(self) -> None:
        assert _slugify("!!!") == "untitled"


class TestGroupRelatedEntries:
    def test_empty_returns_empty(self) -> None:
        assert _group_related_entries([]) == []

    def test_single_entry(self, journal: System1Journal) -> None:
        e = journal.record("hello world")
        groups = _group_related_entries([e])
        assert len(groups) == 1
        assert groups[0] == [e]

    def test_similar_entries_grouped(self, journal: System1Journal) -> None:
        e1 = journal.record("python async await patterns")
        e2 = journal.record("python async io patterns")
        e3 = journal.record("javascript fetch api")
        groups = _group_related_entries([e1, e2, e3], threshold=0.3)
        # e1 and e2 share "python", "async", "patterns" → likely grouped
        # e3 shares nothing → separate group
        assert len(groups) == 2

    def test_unrelated_entries_separate(self, journal: System1Journal) -> None:
        e1 = journal.record("alpha beta gamma")
        e2 = journal.record("delta epsilon zeta")
        groups = _group_related_entries([e1, e2], threshold=0.3)
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# Fast consolidation tests
# ---------------------------------------------------------------------------


class TestFastConsolidation:
    def test_no_pending_returns_empty(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """No entries → no memories created."""
        fc = FastConsolidation(journal, memory_path)
        result = fc.consolidate()
        assert result == []

    def test_creates_memory_file(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Pending entries are consolidated into a memory file."""
        journal.record("Discovered that FAISS needs normalization")
        journal.record("FAISS vectors should be L2 normalized")

        fc = FastConsolidation(journal, memory_path, similarity_threshold=0.2)
        created = fc.consolidate()

        assert len(created) >= 1
        # Verify file exists
        for filename in created:
            assert (memory_path / filename).exists()

    def test_marks_entries_processed(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Consolidated entries are marked as processed."""
        journal.record("thought one")
        journal.record("thought two")

        fc = FastConsolidation(journal, memory_path)
        fc.consolidate()

        # No more pending
        assert journal.get_pending() == []

    def test_memory_content_includes_entries(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Created memory file contains entry content."""
        journal.record("important observation about testing")

        fc = FastConsolidation(journal, memory_path)
        created = fc.consolidate()

        assert len(created) == 1
        content = (memory_path / created[0]).read_text()
        assert "important observation about testing" in content

    def test_memory_has_frontmatter(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Created memory file has YAML frontmatter."""
        journal.record("test thought")

        fc = FastConsolidation(journal, memory_path)
        created = fc.consolidate()

        content = (memory_path / created[0]).read_text()
        assert content.startswith("---\n")
        assert "type: observation" in content
        assert "auto-consolidated" in content

    def test_respects_batch_size(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Only processes up to batch_size entries."""
        for i in range(10):
            journal.record(f"unique thought number {i}")

        fc = FastConsolidation(journal, memory_path, batch_size=3)
        fc.consolidate()

        remaining = journal.get_pending()
        assert len(remaining) == 7  # 10 - 3 = 7


# ---------------------------------------------------------------------------
# Slow consolidation tests
# ---------------------------------------------------------------------------


class TestSlowConsolidation:
    @pytest.mark.asyncio
    async def test_ai_create_action(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """AI 'create' action creates a memory file."""
        journal.record("observation about caching")
        journal.record("redis vs memcached comparison")

        provider = FakeProvider(
            response={
                "actions": [
                    {
                        "type": "create",
                        "entry_indices": [0, 1],
                        "title": "Caching strategies",
                        "content": "Key insight: Redis is preferred for complex data structures.",
                    }
                ]
            }
        )

        sc = SlowConsolidation(journal, memory_path, provider)
        created = await sc.consolidate()

        assert len(created) == 1
        content = (memory_path / created[0]).read_text()
        assert "Key insight" in content

    @pytest.mark.asyncio
    async def test_ai_ignore_action(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """AI 'ignore' action marks entries processed without creating memory."""
        journal.record("random noise")

        provider = FakeProvider(
            response={"actions": [{"type": "ignore", "entry_indices": [0]}]}
        )

        sc = SlowConsolidation(journal, memory_path, provider)
        created = await sc.consolidate()

        assert created == []
        assert journal.get_pending() == []  # Still marked processed

    @pytest.mark.asyncio
    async def test_fallback_on_ai_failure(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Falls back to fast consolidation when AI fails."""
        journal.record("fallback test thought")

        provider = FakeProvider(error=RuntimeError("AI unavailable"))

        sc = SlowConsolidation(journal, memory_path, provider)
        created = await sc.consolidate()

        # Fast consolidation should have created something
        assert len(created) >= 1
        assert journal.get_pending() == []

    @pytest.mark.asyncio
    async def test_invalid_indices_ignored(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Actions with out-of-range indices are safely skipped."""
        journal.record("only entry")

        provider = FakeProvider(
            response={
                "actions": [
                    {"type": "create", "entry_indices": [99], "content": "bad"}
                ]
            }
        )

        sc = SlowConsolidation(journal, memory_path, provider)
        created = await sc.consolidate()
        assert created == []


# ---------------------------------------------------------------------------
# Soft delete safety test
# ---------------------------------------------------------------------------


class TestSoftDeleteSafety:
    def test_memory_written_to_correct_path(
        self, journal: System1Journal, memory_path: Path
    ) -> None:
        """Memories are written under memory_path, not elsewhere."""
        journal.record("safe path test")

        fc = FastConsolidation(journal, memory_path)
        created = fc.consolidate()

        for filename in created:
            full_path = memory_path / filename
            assert full_path.exists()
            # Verify it's actually under memory_path
            assert str(full_path).startswith(str(memory_path))
