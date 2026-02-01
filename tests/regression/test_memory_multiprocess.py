"""
Regression tests for memory system in multiprocess mode.

Bug reference: Memory search returned 0 results in multiprocess mode because
ReadOnlyContext was looking for memory indices in the snapshot directory
(which only contains document indices) instead of the memory storage path.

These tests verify that:
1. ReadOnlyContext properly loads memory indices from their storage location
2. Memory indices are rebuilt when missing
3. Memory search returns results after loading
4. New memory files are detected via reconciliation
"""

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


def create_memory_file(memory_path: Path, filename: str, content: str) -> Path:
    """Helper to create a properly formatted memory file."""
    file_path = memory_path / f"{filename}.md"
    file_path.write_text(content)
    return file_path


def create_test_memory(
    memory_path: Path,
    filename: str,
    title: str,
    body: str,
    memory_type: str = "observation",
    tags: list[str] | None = None,
) -> Path:
    """Create a memory file with proper frontmatter."""
    tags = tags or ["test"]
    tags_str = ", ".join(tags)
    content = f"""---
type: {memory_type}
status: active
tags: [{tags_str}]
---

# {title}

{body}
"""
    return create_memory_file(memory_path, filename, content)


class TestReadOnlyContextMemoryLoading:
    """
    Tests verifying ReadOnlyContext loads memory indices correctly.

    Regression: ReadOnlyContext was looking in snapshot directory for memory
    indices, but they are stored in memory_path/indices/.
    """

    @pytest.mark.asyncio
    async def test_memory_rebuilt_when_no_indices_exist(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Verify memories are indexed when no persisted indices exist.

        This is the core regression test - when memory indices don't exist,
        ReadOnlyContext should call reindex_all() to build them from files.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        memory_path = tmp_path / "memories"
        memory_path.mkdir()
        (memory_path / "indices").mkdir()

        # Create multiple memory files
        create_test_memory(
            memory_path,
            "architecture-decisions",
            "Architecture Decisions",
            "We chose PostgreSQL for ACID compliance.",
            memory_type="fact",
            tags=["architecture", "database"],
        )
        create_test_memory(
            memory_path,
            "auth-implementation",
            "Auth Implementation Notes",
            "JWT tokens with 1-hour expiry implemented in auth_middleware.py.",
            memory_type="observation",
            tags=["auth", "security"],
        )

        monkeypatch.setattr(
            "src.config.resolve_memory_path",
            lambda config, detected_project, projects: memory_path,
        )

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
            memory=MemoryConfig(enabled=True, score_threshold=0.001),
        )

        ctx = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)

        # Critical assertion: memories should be indexed
        assert ctx.memory_manager is not None
        assert ctx.memory_manager.get_memory_count() == 2, (
            "Expected 2 memories to be indexed from files when no indices exist"
        )

    @pytest.mark.asyncio
    async def test_memory_indices_persisted_and_reloaded(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Verify persisted indices are loaded on subsequent startups.

        After initial indexing, indices should be persisted and reloaded
        without needing to re-parse memory files.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        memory_path = tmp_path / "memories"
        memory_path.mkdir()
        (memory_path / "indices").mkdir()

        create_test_memory(
            memory_path,
            "test-memory",
            "Test Memory",
            "Content for persistence testing.",
        )

        monkeypatch.setattr(
            "src.config.resolve_memory_path",
            lambda config, detected_project, projects: memory_path,
        )

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
            memory=MemoryConfig(enabled=True),
        )

        # First context creation - should index from files
        ctx1 = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)
        assert ctx1.memory_manager is not None
        assert ctx1.memory_manager.get_memory_count() == 1

        # Second context creation - should load from persisted indices
        ctx2 = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)
        assert ctx2.memory_manager is not None
        assert ctx2.memory_manager.get_memory_count() == 1

    @pytest.mark.asyncio
    async def test_new_memories_detected_via_reconciliation(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Verify new memory files are indexed via reconciliation.

        When a new memory file is added after initial indexing,
        reconcile() should detect and index it.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        memory_path = tmp_path / "memories"
        memory_path.mkdir()
        (memory_path / "indices").mkdir()

        # Create initial memory
        create_test_memory(
            memory_path,
            "initial-memory",
            "Initial Memory",
            "This memory exists at startup.",
        )

        monkeypatch.setattr(
            "src.config.resolve_memory_path",
            lambda config, detected_project, projects: memory_path,
        )

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
            memory=MemoryConfig(enabled=True),
        )

        # First context - indexes initial memory
        ctx1 = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)
        assert ctx1.memory_manager is not None
        initial_count = ctx1.memory_manager.get_memory_count()
        assert initial_count == 1

        # Add new memory file (simulating file created between restarts)
        create_test_memory(
            memory_path,
            "new-memory",
            "New Memory",
            "This memory was added after initial indexing.",
        )

        # Second context - should detect and index new memory
        ctx2 = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)
        assert ctx2.memory_manager is not None
        assert ctx2.memory_manager.get_memory_count() == 2, (
            "Expected reconciliation to detect and index new memory file"
        )


class TestMemorySearchAfterLoading:
    """
    End-to-end tests for memory search functionality after loading.

    These tests verify the complete flow from context creation through
    search result retrieval.
    """

    @pytest.mark.asyncio
    async def test_search_returns_results_in_multiprocess_context(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Verify search_memories returns results in ReadOnlyContext.

        This is the core regression test for the reported bug - search
        was returning 0 results despite memories existing.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        memory_path = tmp_path / "memories"
        memory_path.mkdir()
        (memory_path / "indices").mkdir()

        # Create searchable memories
        create_test_memory(
            memory_path,
            "database-decision",
            "Database Selection",
            "PostgreSQL was selected for its ACID compliance and JSON support.",
            memory_type="fact",
            tags=["database", "architecture"],
        )
        create_test_memory(
            memory_path,
            "api-design",
            "REST API Design",
            "RESTful endpoints follow resource-based naming conventions.",
            memory_type="observation",
            tags=["api", "design"],
        )

        monkeypatch.setattr(
            "src.config.resolve_memory_path",
            lambda config, detected_project, projects: memory_path,
        )

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
            memory=MemoryConfig(enabled=True, score_threshold=0.001),
            search=SearchConfig(),
            document_chunking=ChunkingConfig(),
            memory_chunking=ChunkingConfig(),
        )

        ctx = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)
        assert ctx.memory_search is not None

        # Search for database-related memories
        search_results = await ctx.memory_search.search_memories("database PostgreSQL")
        assert isinstance(search_results, list)  # Type guard
        results = search_results

        assert len(results) > 0, "Expected search to return at least one result"
        assert any("PostgreSQL" in r.content for r in results), (
            "Expected to find memory about PostgreSQL"
        )

    @pytest.mark.asyncio
    async def test_search_stats_show_indexed_count(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Verify search stats correctly report indexed memory count.

        The original bug showed "Total indexed memories: 0" in stats.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        memory_path = tmp_path / "memories"
        memory_path.mkdir()
        (memory_path / "indices").mkdir()

        # Create memories
        for i in range(3):
            create_test_memory(
                memory_path,
                f"memory-{i}",
                f"Test Memory {i}",
                f"Content for test memory number {i}.",
            )

        monkeypatch.setattr(
            "src.config.resolve_memory_path",
            lambda config, detected_project, projects: memory_path,
        )

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
            memory=MemoryConfig(enabled=True, score_threshold=0.001),
        )

        ctx = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)
        assert ctx.memory_search is not None

        # Search with stats
        search_result = await ctx.memory_search.search_memories(
            "test memory", include_stats=True
        )
        assert isinstance(search_result, tuple)  # Type guard
        results, stats = search_result

        assert stats.total_indexed == 3, (
            f"Expected stats to show 3 indexed memories, got {stats.total_indexed}"
        )


class TestMemorySystemSeparation:
    """
    Tests verifying document and memory indices are properly separated.

    The bug occurred because memory indices were expected in the document
    snapshot directory. These tests verify the separation.
    """

    @pytest.mark.asyncio
    async def test_document_snapshots_do_not_affect_memory(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Verify document snapshot changes don't affect memory system.

        Memory indices should be independent of document snapshots.
        """
        import struct

        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # Create document snapshot (simulating worker process)
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 5))
        snapshot_dir = snapshot_base / "v5"
        snapshot_dir.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        memory_path = tmp_path / "memories"
        memory_path.mkdir()
        (memory_path / "indices").mkdir()

        create_test_memory(
            memory_path,
            "test-memory",
            "Test Memory",
            "Memory should load regardless of document snapshot state.",
        )

        monkeypatch.setattr(
            "src.config.resolve_memory_path",
            lambda config, detected_project, projects: memory_path,
        )

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
            memory=MemoryConfig(enabled=True),
        )

        ctx = await ReadOnlyContext.create(config=config, snapshot_base=snapshot_base)

        # Memory should still be indexed despite document snapshot existing
        assert ctx.memory_manager is not None
        assert ctx.memory_manager.get_memory_count() == 1
