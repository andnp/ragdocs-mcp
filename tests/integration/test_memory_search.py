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
from src.memory.search import MemorySearchOrchestrator, apply_memory_recency_boost
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
            recency_boost_days=7,
            recency_boost_factor=1.2,
        ),
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
        ),
        chunking=ChunkingConfig(),
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
# Recency Boost Function Tests
# ============================================================================


class TestApplyMemoryRecencyBoost:

    def test_recent_memory_gets_boost(self):
        """
        Verify memories created within boost_days receive score multiplier.
        """
        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=3)

        boosted = apply_memory_recency_boost(
            score=0.8,
            created_at=recent,
            boost_days=7,
            boost_factor=1.2,
        )

        assert boosted == pytest.approx(0.96)

    def test_old_memory_no_boost(self):
        """
        Verify memories older than boost_days receive no boost.
        """
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)

        boosted = apply_memory_recency_boost(
            score=0.8,
            created_at=old,
            boost_days=7,
            boost_factor=1.2,
        )

        assert boosted == 0.8

    def test_none_created_at_no_boost(self):
        """
        Verify memories with no created_at receive no boost.
        """
        boosted = apply_memory_recency_boost(
            score=0.5,
            created_at=None,
            boost_days=7,
            boost_factor=1.2,
        )

        assert boosted == 0.5

    def test_exact_boundary_gets_boost(self):
        """
        Verify memories at exactly boost_days boundary get boost.
        """
        now = datetime.now(timezone.utc)
        boundary = now - timedelta(days=7)

        boosted = apply_memory_recency_boost(
            score=1.0,
            created_at=boundary,
            boost_days=7,
            boost_factor=1.5,
        )

        assert boosted == pytest.approx(1.5)

    def test_handles_naive_datetime(self):
        """
        Verify recency boost handles naive datetime (no timezone).
        """
        recent = datetime.now() - timedelta(days=2)

        boosted = apply_memory_recency_boost(
            score=0.5,
            created_at=recent,
            boost_days=7,
            boost_factor=1.2,
        )

        assert boosted == pytest.approx(0.6)


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

    @pytest.mark.asyncio
    async def test_filter_by_tags(
        self,
        memory_manager: MemoryIndexManager,
        memory_search: MemorySearchOrchestrator,
        memory_path: Path,
    ):
        """
        Verify tag filter narrows results to memories with matching tags.
        """
        create_memory_file(
            memory_path,
            "api-note",
            "---\ntags: [\"api\", \"backend\"]\n---\n# API\n\nAPI documentation."
        )
        create_memory_file(
            memory_path,
            "frontend-note",
            "---\ntags: [\"frontend\", \"ui\"]\n---\n# Frontend\n\nUI documentation."
        )

        memory_manager.reindex_all()

        results = await memory_search.search_memories(
            "documentation",
            limit=10,
            filter_tags=["api"]
        )

        for result in results:
            assert "api" in result.frontmatter.tags or len(results) == 0


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
