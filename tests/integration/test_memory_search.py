"""
Integration tests for MemorySearchOrchestrator.

Tests hybrid search functionality for memories including recency boost,
tag/type filtering, and linked memory search via ghost nodes.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    MemoryConfig,
    SearchConfig,
    ServerConfig,
)
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.memory.manager import MemoryIndexManager
from src.memory.search import MemorySearchOrchestrator, apply_memory_recency_boost, apply_memory_decay
from src.memory.storage import ensure_memory_dirs


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def memory_config(tmp_path: Path):
    """
    Create test configuration with memory enabled and recency boost.
    """
    docs_path = tmp_path / "docs"
    docs_path.mkdir()

    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        memory=MemoryConfig(
            enabled=True,
            storage_strategy="project",
            score_threshold=0.001,  # Keep low for backward compatibility with tests
            recency_boost_days=7,
            recency_boost_factor=1.2,
        ),
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
        ),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    """
    Create and return the memory storage path.
    """
    path = tmp_path / ".memories"
    ensure_memory_dirs(path)
    return path


@pytest.fixture
def memory_indices():
    """
    Create fresh indices for memory testing.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def memory_manager(memory_config: Config, memory_path: Path, memory_indices):
    """
    Create MemoryIndexManager instance for testing.
    """
    vector, keyword, graph = memory_indices
    return MemoryIndexManager(memory_config, memory_path, vector, keyword, graph)


@pytest.fixture
def memory_search(memory_config: Config, memory_indices, memory_manager: MemoryIndexManager):
    """
    Create MemorySearchOrchestrator instance for testing.
    """
    vector, keyword, graph = memory_indices
    return MemorySearchOrchestrator(vector, keyword, graph, memory_config, memory_manager)


def create_memory_file(memory_path: Path, filename: str, content: str) -> Path:
    """
    Create a memory file with given content.
    """
    file_path = memory_path / f"{filename}.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ============================================================================
# Memory Decay Function Tests (Exponential Decay System)
# ============================================================================


class TestApplyMemoryDecay:
    """Test the new exponential decay system that replaced recency boost."""

    def test_zero_days_old_no_decay(self):
        """Verify memory created today has no decay (multiplier = 1.0)."""
        now = datetime.now(timezone.utc)
        decayed = apply_memory_decay(
            score=0.5,
            created_at=now,
            decay_rate=0.90,
            floor_multiplier=0.1,
        )
        assert decayed == pytest.approx(0.5)

    def test_exponential_decay_calculation(self):
        """Verify exponential decay formula: score × decay_rate^days."""
        now = datetime.now(timezone.utc)
        seven_days_ago = now - timedelta(days=7)

        decayed = apply_memory_decay(
            score=0.020,  # Typical post-RRF score
            created_at=seven_days_ago,
            decay_rate=0.90,
            floor_multiplier=0.1,
        )

        # Expected: 0.020 × 0.90^7 ≈ 0.020 × 0.478 ≈ 0.00956
        expected = 0.020 * (0.90 ** 7)
        assert decayed == pytest.approx(expected, abs=1e-6)

    def test_floor_multiplier_prevents_zero(self):
        """Verify floor prevents scores from decaying to zero."""
        now = datetime.now(timezone.utc)
        very_old = now - timedelta(days=180)

        decayed = apply_memory_decay(
            score=0.020,
            created_at=very_old,
            decay_rate=0.90,
            floor_multiplier=0.1,
        )

        # After 180 days, 0.90^180 ≈ 0.0000000002, floor kicks in
        assert decayed == pytest.approx(0.020 * 0.1)  # Floor applied

    def test_missing_created_at_no_penalty(self):
        """
        Verify None created_at returns original score without penalty.

        This handles legacy memories indexed before we added timestamp tracking.
        """
        decayed = apply_memory_decay(
            score=0.5,
            created_at=None,
            decay_rate=0.90,
            floor_multiplier=0.1,
        )
        assert decayed == pytest.approx(0.5)  # No penalty applied

    def test_handles_naive_datetime(self):
        """Verify naive datetime is converted to UTC."""
        naive_dt = datetime.now() - timedelta(days=7)

        decayed = apply_memory_decay(
            score=0.020,
            created_at=naive_dt,
            decay_rate=0.90,
            floor_multiplier=0.1,
        )

        # Should apply decay without errors
        assert 0.0 < decayed < 0.020

    def test_per_type_decay_rates(self):
        """Verify different memory types decay at different rates."""
        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        # Journal (fast decay)
        journal_decay = apply_memory_decay(
            0.020, thirty_days_ago, 0.90, 0.1
        )

        # Fact (slow decay)
        fact_decay = apply_memory_decay(
            0.020, thirty_days_ago, 0.98, 0.2
        )

        # Fact should retain more relevance
        assert fact_decay > journal_decay

    def test_thirty_day_old_fact(self):
        """Real-world scenario: 30-day-old fact memory."""
        now = datetime.now(timezone.utc)
        created = now - timedelta(days=30)

        decayed = apply_memory_decay(
            score=0.020,
            created_at=created,
            decay_rate=0.98,  # Fact decay rate
            floor_multiplier=0.2,
        )

        # 0.020 × 0.98^30 ≈ 0.011 (above floor)
        expected = 0.020 * (0.98 ** 30)
        assert decayed == pytest.approx(expected, abs=1e-6)

    def test_different_floor_multipliers(self):
        """Verify floor multipliers affect long-term retention."""
        now = datetime.now(timezone.utc)
        very_old = now - timedelta(days=365)

        low_floor = apply_memory_decay(
            0.020, very_old, 0.85, 0.05  # Plan: low floor
        )

        high_floor = apply_memory_decay(
            0.020, very_old, 0.98, 0.3  # Fact: high floor
        )

        # High floor preserves more
        assert high_floor > low_floor


class TestApplyMemoryRecencyBoost:
    """DEPRECATED: Tests for backward compatibility wrapper."""

    def test_backward_compatibility_wrapper(self):
        """
        Verify deprecated recency_boost function still works via wrapper.

        Note: Behavior changed - now approximates using decay system.
        """
        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=3)

        result = apply_memory_recency_boost(
            score=0.8,
            created_at=recent,
            boost_days=7,
            boost_factor=1.2,
        )

        # Wrapper now uses decay approximation, not exact boost
        # Just verify it returns a reasonable value
        assert 0.0 < result <= 0.8


# ============================================================================
# Basic Search Tests
# ============================================================================


class TestBasicMemorySearch:

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify basic search returns matching memories.
        """
        content = """---
type: "journal"
tags: ["auth"]
created_at: "2025-01-10T12:00:00Z"
---

# Authentication Implementation

Implemented OAuth2 authentication with JWT tokens.
"""
        file_path = create_memory_file(memory_path, "auth-impl", content)
        memory_manager.index_memory(str(file_path))

        results = await memory_search.search_memories("authentication OAuth", limit=5)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(
        self,
        memory_search: MemorySearchOrchestrator,
    ):
        """
        Verify empty query returns empty results.
        """
        results = await memory_search.search_memories("", limit=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(
        self,
        memory_search: MemorySearchOrchestrator,
    ):
        """
        Verify whitespace-only query returns empty results.
        """
        results = await memory_search.search_memories("   \n\t  ", limit=5)

        assert results == []


# ============================================================================
# Filter Tests
# ============================================================================


class TestMemorySearchFilters:

    @pytest.mark.asyncio
    async def test_filter_by_type(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify type filter narrows results to specified type.
        """
        create_memory_file(
            memory_path,
            "plan1",
            "---\ntype: \"plan\"\n---\n# Plan\n\nImplementation plan for feature."
        )
        create_memory_file(
            memory_path,
            "journal1",
            "---\ntype: \"journal\"\n---\n# Journal\n\nDaily notes about feature."
        )

        memory_manager.reindex_all()

        results = await memory_search.search_memories(
            "feature",
            limit=10,
            filter_type="plan"
        )

        for result in results:
            assert result.frontmatter.type == "plan"


# ============================================================================
# Linked Memory Search Tests
# ============================================================================


class TestLinkedMemorySearch:

    @pytest.mark.asyncio
    async def test_search_linked_finds_memories(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify search_linked_memories finds memories linking to target.
        """
        content = """# Server Bug Fix

Found and fixed bug in [[src/server.py]] causing timeout errors.
"""
        file_path = create_memory_file(memory_path, "bug-fix", content)
        memory_manager.index_memory(str(file_path))

        results = await memory_search.search_linked_memories(
            query="bug",
            target_document="src/server.py",
            limit=5
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_linked_returns_anchor_context(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify linked search results include anchor context.

        Anchor context explains the link relationship.
        """
        content = """# Refactoring Notes

Need to refactor [[src/handler.py]] for better error handling.
"""
        file_path = create_memory_file(memory_path, "refactor-notes", content)
        memory_manager.index_memory(str(file_path))

        results = await memory_search.search_linked_memories(
            query="refactor",
            target_document="src/handler.py",
            limit=5
        )

        if len(results) > 0:
            assert "refactor" in results[0].anchor_context.lower()

    @pytest.mark.asyncio
    async def test_search_linked_returns_edge_type(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify linked search results include inferred edge type.
        """
        content = """# Bug Report

Found a bug in [[src/auth.py]] causing login failures.
"""
        file_path = create_memory_file(memory_path, "bug-report", content)
        memory_manager.index_memory(str(file_path))

        results = await memory_search.search_linked_memories(
            query="bug",
            target_document="src/auth.py",
            limit=5
        )

        if len(results) > 0:
            assert results[0].edge_type == "debugs"

    @pytest.mark.asyncio
    async def test_search_linked_empty_when_no_links(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify search_linked_memories returns empty for unlinked targets.
        """
        content = """# Unrelated Note

This note links to [[other/file.py]] not the target.
"""
        file_path = create_memory_file(memory_path, "unrelated", content)
        memory_manager.index_memory(str(file_path))

        results = await memory_search.search_linked_memories(
            query="note",
            target_document="src/server.py",
            limit=5
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_linked_multiple_memories(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify search_linked_memories finds all memories linking to target.
        """
        create_memory_file(
            memory_path,
            "note1",
            "# Note 1\n\nInvestigating [[src/api.py]] performance."
        )
        create_memory_file(
            memory_path,
            "note2",
            "# Note 2\n\nRefactoring [[src/api.py]] endpoints."
        )
        create_memory_file(
            memory_path,
            "note3",
            "# Note 3\n\nPlanning [[src/api.py]] v2 implementation."
        )

        memory_manager.reindex_all()

        results = await memory_search.search_linked_memories(
            query="api",
            target_document="src/api.py",
            limit=10
        )

        assert len(results) == 3


# ============================================================================
# Result Quality Tests
# ============================================================================


class TestTimeRangeFiltering:
    """
    Tests for time-based filtering of memory search results.

    This class tests the newly implemented after_timestamp, before_timestamp,
    and relative_days filtering parameters.
    """

    @pytest.mark.asyncio
    async def test_filter_after_timestamp(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify only memories after timestamp are returned.
        """
        
        # Create memories with different timestamps
        old_memory = create_memory_file(
            memory_path,
            "old-memory",
            '---\ntype: "journal"\ncreated_at: "2025-01-01T12:00:00Z"\n---\n# Old Memory\n\nOld content about testing.'
        )
        new_memory = create_memory_file(
            memory_path,
            "new-memory",
            '---\ntype: "journal"\ncreated_at: "2025-01-14T12:00:00Z"\n---\n# New Memory\n\nNew content about testing.'
        )

        memory_manager.index_memory(str(old_memory))
        memory_manager.index_memory(str(new_memory))

        # Filter for memories after Jan 10, 2025
        cutoff = datetime(2025, 1, 10, tzinfo=timezone.utc)
        after_ts = int(cutoff.timestamp())

        results = await memory_search.search_memories(
            "testing",
            limit=10,
            after_timestamp=after_ts,
        )

        # Only new memory should be returned
        assert len(results) > 0
        for result in results:
            assert result.memory_id in ["memory:new-memory"]

    @pytest.mark.asyncio
    async def test_filter_before_timestamp(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify only memories before timestamp are returned.
        """
        # Create memories with different timestamps
        old_memory = create_memory_file(
            memory_path,
            "old-note",
            '---\ntype: "fact"\ncreated_at: "2025-01-05T12:00:00Z"\n---\n# Old Note\n\nDatabase configuration notes.'
        )
        new_memory = create_memory_file(
            memory_path,
            "new-note",
            '---\ntype: "fact"\ncreated_at: "2025-01-15T12:00:00Z"\n---\n# New Note\n\nDatabase configuration notes.'
        )

        memory_manager.index_memory(str(old_memory))
        memory_manager.index_memory(str(new_memory))

        # Filter for memories before Jan 10, 2025
        cutoff = datetime(2025, 1, 10, tzinfo=timezone.utc)
        before_ts = int(cutoff.timestamp())

        results = await memory_search.search_memories(
            "database configuration",
            limit=10,
            before_timestamp=before_ts,
        )

        # Only old memory should be returned
        assert len(results) > 0
        for result in results:
            assert result.memory_id in ["memory:old-note"]

    @pytest.mark.asyncio
    async def test_filter_time_range(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify combined after/before filtering (AND logic).
        """
        # Create memories at different dates
        memories = [
            ("mem-2024-12", '---\ntype: "plan"\ncreated_at: "2024-12-20T12:00:00Z"\n---\n# Plan 2024\n\nAPI redesign plan.'),
            ("mem-2025-01", '---\ntype: "plan"\ncreated_at: "2025-01-08T12:00:00Z"\n---\n# Plan Jan\n\nAPI redesign plan.'),
            ("mem-2025-02", '---\ntype: "plan"\ncreated_at: "2025-02-01T12:00:00Z"\n---\n# Plan Feb\n\nAPI redesign plan.'),
        ]

        for filename, content in memories:
            path = create_memory_file(memory_path, filename, content)
            memory_manager.index_memory(str(path))

        # Filter for Jan 2025 only
        after_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        before_ts = int(datetime(2025, 2, 1, tzinfo=timezone.utc).timestamp())

        results = await memory_search.search_memories(
            "API redesign",
            limit=10,
            after_timestamp=after_ts,
            before_timestamp=before_ts,
        )

        # Only January memory should be returned
        assert len(results) == 1
        assert results[0].memory_id == "memory:mem-2025-01"

    @pytest.mark.asyncio
    async def test_filter_relative_days(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify relative_days works correctly.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        
        # Create memories at different relative dates
        old_date = (now - timedelta(days=10)).isoformat()
        recent_date = (now - timedelta(days=3)).isoformat()

        old_memory = create_memory_file(
            memory_path,
            "old-observation",
            f'---\ntype: "observation"\ncreated_at: "{old_date}"\n---\n# Old Observation\n\nPerformance metrics observation.'
        )
        recent_memory = create_memory_file(
            memory_path,
            "recent-observation",
            f'---\ntype: "observation"\ncreated_at: "{recent_date}"\n---\n# Recent Observation\n\nPerformance metrics observation.'
        )

        memory_manager.index_memory(str(old_memory))
        memory_manager.index_memory(str(recent_memory))

        # Filter for last 7 days
        results = await memory_search.search_memories(
            "performance metrics",
            limit=10,
            relative_days=7,
        )

        # Only recent memory should be returned
        assert len(results) > 0
        for result in results:
            assert result.memory_id in ["memory:recent-observation"]

    @pytest.mark.asyncio
    async def test_relative_days_overrides_absolute(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify relative_days takes precedence over absolute timestamps.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        recent_date = (now - timedelta(days=2)).isoformat()

        memory = create_memory_file(
            memory_path,
            "recent-note",
            f'---\ntype: "journal"\ncreated_at: "{recent_date}"\n---\n# Recent Note\n\nImportant notes.'
        )

        memory_manager.index_memory(str(memory))

        # Provide both relative_days and absolute timestamps
        # relative_days should override
        results = await memory_search.search_memories(
            "important notes",
            limit=10,
            after_timestamp=0,  # Would match everything if used
            before_timestamp=1,  # Would match nothing if used
            relative_days=5,  # Should be used instead
        )

        # Should get results because relative_days=5 includes recent memory
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_invalid_timestamp_range_raises_error(
        self,
        memory_search: MemorySearchOrchestrator,
    ):
        """
        Verify ValueError for after >= before.
        """
        after_ts = 1000000
        before_ts = 500000  # before < after (invalid)

        with pytest.raises(ValueError, match="after_timestamp must be less than before_timestamp"):
            await memory_search.search_memories(
                "test query",
                limit=5,
                after_timestamp=after_ts,
                before_timestamp=before_ts,
            )

    @pytest.mark.asyncio
    async def test_negative_relative_days_raises_error(
        self,
        memory_search: MemorySearchOrchestrator,
    ):
        """
        Verify ValueError for relative_days < 0.
        """
        with pytest.raises(ValueError, match="relative_days must be non-negative"):
            await memory_search.search_memories(
                "test query",
                limit=5,
                relative_days=-5,
            )

    @pytest.mark.asyncio
    async def test_fallback_to_file_mtime(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify fallback when memory_created_at missing.
        """
        import os
        from datetime import timedelta

        # Create memory without created_at in frontmatter
        memory_no_timestamp = create_memory_file(
            memory_path,
            "no-timestamp",
            '---\ntype: "journal"\n---\n# No Timestamp\n\nContent without timestamp.'
        )

        # Set file mtime to a specific time
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=10)
        os.utime(memory_no_timestamp, (old_time.timestamp(), old_time.timestamp()))

        memory_manager.index_memory(str(memory_no_timestamp))

        # Filter for recent memories only
        cutoff = now - timedelta(days=5)
        after_ts = int(cutoff.timestamp())

        results = await memory_search.search_memories(
            "content",
            limit=10,
            after_timestamp=after_ts,
        )

        # Should get no results because file mtime is old
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_time_filter_with_other_filters(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify time filter works with type filters.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        recent_date = (now - timedelta(days=2)).isoformat()
        old_date = (now - timedelta(days=10)).isoformat()

        # Create memories with different types and dates
        memories = [
            ("recent-plan", f'---\ntype: "plan"\ncreated_at: "{recent_date}"\n---\n# Recent Plan\n\nAPI development.'),
            ("old-plan", f'---\ntype: "plan"\ncreated_at: "{old_date}"\n---\n# Old Plan\n\nAPI development.'),
            ("recent-fact", f'---\ntype: "fact"\ncreated_at: "{recent_date}"\n---\n# Recent Fact\n\nUI development.'),
        ]

        for filename, content in memories:
            path = create_memory_file(memory_path, filename, content)
            memory_manager.index_memory(str(path))

        # Filter for recent plans only
        results = await memory_search.search_memories(
            "development",
            limit=10,
            filter_type="plan",
            relative_days=7,
        )

        # Should only get recent-plan
        assert len(results) == 1
        assert results[0].memory_id == "memory:recent-plan"

    @pytest.mark.asyncio
    async def test_empty_results_with_time_filter(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify empty list when no memories match time filter.
        """
        # Create memory with old timestamp
        old_memory = create_memory_file(
            memory_path,
            "very-old",
            '---\ntype: "journal"\ncreated_at: "2020-01-01T12:00:00Z"\n---\n# Very Old\n\nAncient notes.'
        )

        memory_manager.index_memory(str(old_memory))

        # Filter for very recent memories (last 1 day)
        results = await memory_search.search_memories(
            "ancient notes",
            limit=10,
            relative_days=1,
        )

        # Should get no results
        assert results == []

    @pytest.mark.asyncio
    async def test_timezone_normalization(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify timezone handling for naive datetimes.
        """
        # Create memory with naive datetime (no timezone)
        memory = create_memory_file(
            memory_path,
            "naive-tz",
            '---\ntype: "fact"\ncreated_at: "2025-01-10T12:00:00"\n---\n# Naive TZ\n\nConfiguration details.'
        )

        memory_manager.index_memory(str(memory))

        # Filter using timezone-aware timestamp
        after_ts = int(datetime(2025, 1, 9, tzinfo=timezone.utc).timestamp())
        before_ts = int(datetime(2025, 1, 11, tzinfo=timezone.utc).timestamp())

        results = await memory_search.search_memories(
            "configuration details",
            limit=10,
            after_timestamp=after_ts,
            before_timestamp=before_ts,
        )

        # Should find the memory despite timezone differences
        assert len(results) > 0


class TestSearchResultQuality:

    @pytest.mark.asyncio
    async def test_results_are_sorted_by_score(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify search results are sorted by descending score.
        """
        for i in range(5):
            content = f"---\ntype: \"journal\"\n---\n# Note {i}\n\nAuthentication content."
            create_memory_file(memory_path, f"auth-note-{i}", content)

        memory_manager.reindex_all()

        results = await memory_search.search_memories("authentication", limit=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_results_respect_limit(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify search respects limit parameter.
        """
        for i in range(10):
            content = f"---\ntype: \"fact\"\n---\n# Fact {i}\n\nDatabase configuration."
            create_memory_file(memory_path, f"fact-{i}", content)

        memory_manager.reindex_all()

        results_3 = await memory_search.search_memories("database", limit=3)
        results_5 = await memory_search.search_memories("database", limit=5)

        assert len(results_3) <= 3
        assert len(results_5) <= 5

    @pytest.mark.asyncio
    async def test_result_contains_memory_metadata(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify search results include complete memory metadata.
        """
        content = """---
type: "observation"
status: "active"
tags: ["performance", "metrics"]
created_at: "2025-01-10T08:00:00Z"
---

# Performance Observation

CPU usage spikes during peak hours.
"""
        file_path = create_memory_file(memory_path, "perf-obs", content)
        memory_manager.index_memory(str(file_path))

        results = await memory_search.search_memories("CPU performance", limit=1)

        assert len(results) > 0
        result = results[0]
        assert result.memory_id.startswith("memory:")
        assert isinstance(result.score, float)
        assert result.content
        assert result.file_path
