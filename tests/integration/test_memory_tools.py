"""
Integration tests for Memory MCP Tools.

Tests the complete CRUD operations and search tools exposed via MCP,
including create, read, append, update, delete, search, and merge.
"""

from dataclasses import dataclass
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
from src.memory.search import MemorySearchOrchestrator
from src.memory.storage import ensure_memory_dirs, get_memory_file_path, get_trash_path
from src.memory.tools import (
    append_memory,
    create_memory,
    delete_memory,
    get_memory_stats,
    merge_memories,
    read_memory,
    search_linked_memories,
    search_memories,
    update_memory,
)


@dataclass
class FakeApplicationContext:
    memory_manager: MemoryIndexManager | None
    memory_search: MemorySearchOrchestrator | None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def memory_config(tmp_path: Path):
    """
    Create test configuration with memory enabled.
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
            score_threshold=0.001,
            recency_boost_days=7,
            recency_boost_factor=1.2,
        ),
        search=SearchConfig(),
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


@pytest.fixture
def app_context(memory_manager: MemoryIndexManager, memory_search: MemorySearchOrchestrator):
    """
    Create a FakeApplicationContext with memory components.

    Uses a simple dataclass instead of the full ApplicationContext to avoid
    complex dependencies.
    """
    return FakeApplicationContext(
        memory_manager=memory_manager,
        memory_search=memory_search,
    )


@pytest.fixture
def disabled_context():
    """
    Create a FakeApplicationContext with memory disabled.
    """
    return FakeApplicationContext(
        memory_manager=None,
        memory_search=None,
    )


# ============================================================================
# Create Memory Tests
# ============================================================================


class TestCreateMemory:

    @pytest.mark.asyncio
    async def test_create_memory_success(self, app_context, memory_path: Path):
        """
        Verify create_memory creates a new memory file with content.

        The file should have YAML frontmatter and be indexed.
        """
        result = await create_memory(
            app_context,
            filename="test-note",
            content="# Test Note\n\nThis is test content.",
            tags=["test", "example"],
            memory_type="journal",
        )

        assert result["status"] == "created"
        assert result["filename"] == "test-note"

        file_path = get_memory_file_path(memory_path, "test-note")
        assert file_path.exists()

        content = file_path.read_text()
        assert "type:" in content
        assert "tags:" in content
        assert "Test Note" in content

    @pytest.mark.asyncio
    async def test_create_memory_fails_if_exists(self, app_context, memory_path: Path):
        """
        Verify create_memory fails if file already exists.

        This prevents accidental overwrites.
        """
        file_path = memory_path / "existing.md"
        file_path.write_text("# Existing\n\nExisting content.")

        result = await create_memory(
            app_context,
            filename="existing",
            content="# New content",
            tags=[],
        )

        assert "error" in result
        assert "exists" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_memory_disabled_returns_error(self, disabled_context):
        """
        Verify create_memory returns error when memory is disabled.
        """
        result = await create_memory(
            disabled_context,
            filename="disabled-test",
            content="Content",
            tags=[],
        )

        assert "error" in result
        assert "not enabled" in result["error"].lower()


# ============================================================================
# Read Memory Tests
# ============================================================================


class TestReadMemory:

    @pytest.mark.asyncio
    async def test_read_memory_success(self, app_context, memory_path: Path):
        """
        Verify read_memory returns full file content.
        """
        file_path = memory_path / "readable.md"
        expected_content = "---\ntype: journal\n---\n\n# Readable Note\n\nContent here."
        file_path.write_text(expected_content)

        result = await read_memory(app_context, filename="readable")

        assert result["filename"] == "readable"
        assert result["content"] == expected_content
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_read_memory_not_found(self, app_context):
        """
        Verify read_memory returns error for non-existent file.
        """
        result = await read_memory(app_context, filename="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_read_memory_disabled_returns_error(self, disabled_context):
        """
        Verify read_memory returns error when memory is disabled.
        """
        result = await read_memory(disabled_context, filename="any")

        assert "error" in result


# ============================================================================
# Append Memory Tests
# ============================================================================


class TestAppendMemory:

    @pytest.mark.asyncio
    async def test_append_memory_success(self, app_context, memory_path: Path):
        """
        Verify append_memory adds content to existing file.
        """
        file_path = memory_path / "appendable.md"
        file_path.write_text("# Original\n\nOriginal content.")

        result = await append_memory(
            app_context,
            filename="appendable",
            content="## Appended Section\n\nNew content here."
        )

        assert result["status"] == "appended"

        updated_content = file_path.read_text()
        assert "Original content" in updated_content
        assert "Appended Section" in updated_content
        assert "New content here" in updated_content

    @pytest.mark.asyncio
    async def test_append_memory_not_found(self, app_context):
        """
        Verify append_memory returns error for non-existent file.
        """
        result = await append_memory(
            app_context,
            filename="nonexistent",
            content="Content to append"
        )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_append_memory_disabled_returns_error(self, disabled_context):
        """
        Verify append_memory returns error when memory is disabled.
        """
        result = await append_memory(
            disabled_context,
            filename="any",
            content="content"
        )

        assert "error" in result


# ============================================================================
# Update Memory Tests
# ============================================================================


class TestUpdateMemory:

    @pytest.mark.asyncio
    async def test_update_memory_success(self, app_context, memory_path: Path):
        """
        Verify update_memory replaces entire file content.
        """
        file_path = memory_path / "updatable.md"
        file_path.write_text("# Old Content\n\nThis will be replaced.")

        new_content = "---\ntype: plan\n---\n\n# New Content\n\nCompletely new."

        result = await update_memory(
            app_context,
            filename="updatable",
            content=new_content
        )

        assert result["status"] == "updated"

        actual_content = file_path.read_text()
        assert actual_content == new_content
        assert "Old Content" not in actual_content

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, app_context):
        """
        Verify update_memory returns error for non-existent file.
        """
        result = await update_memory(
            app_context,
            filename="nonexistent",
            content="New content"
        )

        assert "error" in result
        assert "not found" in result["error"].lower()


# ============================================================================
# Delete Memory Tests
# ============================================================================


class TestDeleteMemory:

    @pytest.mark.asyncio
    async def test_delete_memory_moves_to_trash(self, app_context, memory_path: Path):
        """
        Verify delete_memory moves file to .trash instead of hard delete.

        This is a safety feature to prevent accidental data loss.
        """
        file_path = memory_path / "deletable.md"
        file_path.write_text("# To Delete\n\nThis will be trashed.")

        result = await delete_memory(app_context, filename="deletable")

        assert result["status"] == "deleted"
        assert "moved_to" in result

        assert not file_path.exists()

        trash_path = get_trash_path(memory_path)
        trash_files = list(trash_path.glob("deletable_*.md"))
        assert len(trash_files) == 1

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, app_context):
        """
        Verify delete_memory returns error for non-existent file.
        """
        result = await delete_memory(app_context, filename="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()


# ============================================================================
# Search Memories Tests
# ============================================================================


class TestSearchMemories:

    @pytest.mark.asyncio
    async def test_search_memories_returns_results(self, app_context, memory_path: Path):
        """
        Verify search_memories finds indexed memories.
        """
        await create_memory(
            app_context,
            filename="searchable",
            content="# Authentication\n\nOAuth2 implementation details.",
            tags=["auth"],
            memory_type="fact",
        )

        results = await search_memories(
            app_context,
            query="OAuth authentication",
            limit=5,
        )

        assert len(results) > 0
        assert "error" not in results[0]

    @pytest.mark.asyncio
    async def test_search_memories_with_filters(self, app_context, memory_path: Path):
        """
        Verify search_memories respects filter parameters.
        """
        await create_memory(
            app_context,
            filename="plan-note",
            content="# Implementation Plan\n\nSteps for implementing authentication.",
            tags=["feature"],
            memory_type="plan",
        )
        await create_memory(
            app_context,
            filename="fact-note",
            content="# Feature Fact\n\nKnown behavior of feature.",
            tags=["feature"],
            memory_type="fact",
        )

        results = await search_memories(
            app_context,
            query="implementation authentication",
            limit=10,
            filter_type="plan",
        )

        assert len(results) > 0
        for result in results:
            assert "error" not in result
            assert result.get("type") == "plan"

    @pytest.mark.asyncio
    async def test_search_memories_disabled_returns_error(self, disabled_context):
        """
        Verify search_memories returns error when memory is disabled.
        """
        results = await search_memories(
            disabled_context,
            query="test",
            limit=5,
        )

        assert len(results) == 1
        assert "error" in results[0]


# ============================================================================
# Search Linked Memories Tests
# ============================================================================


class TestSearchLinkedMemories:

    @pytest.mark.asyncio
    async def test_search_linked_finds_memories(self, app_context, memory_path: Path):
        """
        Verify search_linked_memories finds memories with [[links]].
        """
        await create_memory(
            app_context,
            filename="linked-note",
            content="# Bug Fix\n\nFixed issue in [[src/server.py]] causing errors.",
            tags=["bugfix"],
            memory_type="journal",
        )

        results = await search_linked_memories(
            app_context,
            query="bug",
            target_document="src/server.py",
            limit=5,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_linked_returns_empty_for_unlinked(
        self, app_context, memory_path: Path
    ):
        """
        Verify search_linked_memories returns empty for unlinked targets.
        """
        await create_memory(
            app_context,
            filename="other-note",
            content="# Other Note\n\nNo link to the target file.",
            tags=[],
            memory_type="journal",
        )

        results = await search_linked_memories(
            app_context,
            query="note",
            target_document="src/nonexistent.py",
            limit=5,
        )

        assert results == []


# ============================================================================
# Get Memory Stats Tests
# ============================================================================


class TestGetMemoryStats:

    @pytest.mark.asyncio
    async def test_get_memory_stats_returns_counts(self, app_context, memory_path: Path):
        """
        Verify get_memory_stats returns accurate statistics.
        """
        await create_memory(
            app_context,
            filename="stat1",
            content="# Stat 1",
            tags=["tag1"],
            memory_type="plan",
        )
        await create_memory(
            app_context,
            filename="stat2",
            content="# Stat 2",
            tags=["tag1", "tag2"],
            memory_type="fact",
        )

        stats = await get_memory_stats(app_context)

        assert "error" not in stats
        assert stats["count"] == 2
        assert "total_size" in stats
        assert "tags" in stats
        assert "types" in stats

    @pytest.mark.asyncio
    async def test_get_memory_stats_empty_initially(self, app_context):
        """
        Verify get_memory_stats returns zero counts initially.
        """
        stats = await get_memory_stats(app_context)

        assert stats["count"] == 0

    @pytest.mark.asyncio
    async def test_get_memory_stats_disabled_returns_error(self, disabled_context):
        """
        Verify get_memory_stats returns error when memory is disabled.
        """
        stats = await get_memory_stats(disabled_context)

        assert "error" in stats


# ============================================================================
# Merge Memories Tests
# ============================================================================


class TestMergeMemories:

    @pytest.mark.asyncio
    async def test_merge_memories_consolidates_files(
        self, app_context, memory_path: Path
    ):
        """
        Verify merge_memories creates target and moves sources to trash.
        """
        await create_memory(
            app_context,
            filename="source1",
            content="# Source 1\n\nContent from source 1.",
            tags=["merge"],
        )
        await create_memory(
            app_context,
            filename="source2",
            content="# Source 2\n\nContent from source 2.",
            tags=["merge"],
        )

        summary = """---
type: "fact"
tags: ["merged"]
---

# Merged Summary

Combined content from sources.
"""

        result = await merge_memories(
            app_context,
            source_files=["source1", "source2"],
            target_file="merged",
            summary_content=summary,
        )

        assert result["status"] == "merged"
        assert result["sources_merged"] == "2"

        target_path = get_memory_file_path(memory_path, "merged")
        assert target_path.exists()

        source1_path = get_memory_file_path(memory_path, "source1")
        source2_path = get_memory_file_path(memory_path, "source2")
        assert not source1_path.exists()
        assert not source2_path.exists()

        trash = get_trash_path(memory_path)
        trash_files = list(trash.glob("source*.md"))
        assert len(trash_files) == 2

    @pytest.mark.asyncio
    async def test_merge_memories_fails_if_target_exists(
        self, app_context, memory_path: Path
    ):
        """
        Verify merge_memories fails if target file already exists.
        """
        await create_memory(
            app_context,
            filename="source",
            content="# Source",
            tags=[],
        )
        await create_memory(
            app_context,
            filename="existing-target",
            content="# Existing Target",
            tags=[],
        )

        result = await merge_memories(
            app_context,
            source_files=["source"],
            target_file="existing-target",
            summary_content="# Summary",
        )

        assert "error" in result
        assert "exists" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_merge_memories_fails_if_source_missing(
        self, app_context, memory_path: Path
    ):
        """
        Verify merge_memories fails if any source file is missing.
        """
        await create_memory(
            app_context,
            filename="source",
            content="# Source",
            tags=[],
        )

        result = await merge_memories(
            app_context,
            source_files=["source", "nonexistent"],
            target_file="merged",
            summary_content="# Summary",
        )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_merge_memories_empty_sources_fails(self, app_context):
        """
        Verify merge_memories fails with empty source list.
        """
        result = await merge_memories(
            app_context,
            source_files=[],
            target_file="merged",
            summary_content="# Summary",
        )

        assert "error" in result


# ============================================================================
# Search Memories with Time Filtering Tests
# ============================================================================


class TestSearchMemoriesTimeFiltering:
    """
    Test time filtering parameters at the tool level.

    These tests validate that the MCP tool interface correctly
    passes time filtering parameters to the search orchestrator.
    """

    @pytest.mark.asyncio
    async def test_search_invalid_range_returns_error(self, app_context):
        """
        Verify search_memories tool returns error for invalid timestamp range.
        """
        # after > before (invalid)
        results = await search_memories(
            app_context,
            query="test",
            limit=5,
            after_timestamp=2000000,
            before_timestamp=1000000,
        )

        # Should return error
        assert len(results) > 0
        assert "error" in results[0]
        assert "after_timestamp" in results[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_search_negative_relative_days_returns_error(self, app_context):
        """
        Verify search_memories tool returns error for negative relative_days.
        """
        results = await search_memories(
            app_context,
            query="test",
            limit=5,
            relative_days=-10,
        )

        # Should return error
        assert len(results) > 0
        assert "error" in results[0]
        assert "non-negative" in results[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_search_disabled_context_returns_error(self, disabled_context):
        """
        Verify time filtering parameters still work with disabled context.
        """
        results = await search_memories(
            disabled_context,
            query="test",
            limit=5,
            after_timestamp=1000000,
            before_timestamp=2000000,
            relative_days=7,
        )

        # Should return error about disabled memory system
        assert len(results) == 1
        assert "error" in results[0]
        assert "not enabled" in results[0]["error"].lower()


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


class TestMemoryToolsWorkflow:

    @pytest.mark.asyncio
    async def test_create_read_update_delete_cycle(
        self, app_context, memory_path: Path
    ):
        """
        Verify complete CRUD lifecycle works correctly.

        This is an end-to-end test of the most common workflow.
        """
        create_result = await create_memory(
            app_context,
            filename="lifecycle-test",
            content="# Initial\n\nInitial content.",
            tags=["lifecycle"],
            memory_type="journal",
        )
        assert create_result["status"] == "created"

        read_result = await read_memory(app_context, filename="lifecycle-test")
        assert "Initial content" in read_result["content"]

        update_result = await update_memory(
            app_context,
            filename="lifecycle-test",
            content="---\ntype: plan\n---\n\n# Updated\n\nUpdated content.",
        )
        assert update_result["status"] == "updated"

        read_again = await read_memory(app_context, filename="lifecycle-test")
        assert "Updated content" in read_again["content"]
        assert "Initial content" not in read_again["content"]

        delete_result = await delete_memory(app_context, filename="lifecycle-test")
        assert delete_result["status"] == "deleted"

        read_deleted = await read_memory(app_context, filename="lifecycle-test")
        assert "error" in read_deleted
