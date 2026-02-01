"""
Unit tests for ReadOnlyContext.

Tests the read-only context used by main process for search-only operations.
"""

import struct
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, ServerConfig


class TestReadOnlyContextCreation:
    """Tests for ReadOnlyContext.create() factory method."""

    @pytest.mark.asyncio
    async def test_create_with_no_snapshot(self, tmp_path: Path):
        """
        Verify create() works with empty snapshot directory.

        When no snapshot exists, indices start empty.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        # Should create successfully with empty indices
        assert ctx.config == config
        assert ctx.vector is not None
        assert ctx.keyword is not None
        assert ctx.graph is not None
        assert ctx.orchestrator is not None

    @pytest.mark.asyncio
    async def test_create_with_snapshot(self, tmp_path: Path):
        """
        Verify create() loads indices from existing snapshot.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # Create version file pointing to v1
        version_file = snapshot_base / "version"
        version_file.write_bytes(struct.pack("<I", 1))

        # Create minimal snapshot directory
        snapshot_dir = snapshot_base / "v1"
        snapshot_dir.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        assert ctx is not None
        assert ctx.sync_receiver is not None


class TestReadOnlyContextIsReady:
    """Tests for ReadOnlyContext.is_ready() method."""

    @pytest.mark.asyncio
    async def test_is_ready_false_with_no_snapshot(self, tmp_path: Path):
        """
        Verify is_ready returns False when no snapshot loaded.

        When no snapshot exists at startup, sync_receiver.current_version is 0,
        so is_ready() should return False (waiting for worker to publish).
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        # No snapshot loaded, so is_ready() should be False
        assert ctx.is_ready() is False
        assert ctx.sync_receiver.current_version == 0

    @pytest.mark.asyncio
    async def test_is_ready_true_after_loading_snapshot(self, tmp_path: Path):
        """
        Regression test: is_ready returns True after loading existing snapshot.

        This prevents the timeout bug where wait_ready() would hang for 60s
        even when indices were successfully loaded from an existing snapshot.

        Bug scenario (before fix):
        1. Worker publishes snapshot v5
        2. MCP server restarts
        3. ReadOnlyContext.create() loads snapshot v5
        4. sync_receiver.current_version stayed at 0
        5. is_ready() returned False
        6. wait_ready() timed out after 60s

        After fix, loading snapshot should set current_version correctly.
        """
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # Create version file pointing to v5
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 5))

        # Create snapshot directory with empty indices
        snapshot_dir = snapshot_base / "v5"
        snapshot_dir.mkdir()

        # Create minimal index files to satisfy load
        vector_dir = snapshot_dir / "vector"
        vector_dir.mkdir()
        keyword_dir = snapshot_dir / "keyword"
        keyword_dir.mkdir()
        graph_dir = snapshot_dir / "graph"
        graph_dir.mkdir()

        # Create empty index files that load() expects
        (vector_dir / "doc_id_to_node_ids.json").write_text("{}")
        (keyword_dir / "index_exists").write_text("")
        (graph_dir / "graph_data.json").write_text("{}")

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        # After loading snapshot v5, is_ready() should return True
        assert ctx.is_ready() is True, (
            f"is_ready() returned False with sync_receiver.current_version={ctx.sync_receiver.current_version}. "
            "This would cause query_documents to timeout waiting for ready state."
        )
        assert ctx.sync_receiver.current_version == 5

    @pytest.mark.asyncio
    async def test_is_ready_after_create(self, tmp_path: Path):
        """Verify is_ready returns True after successful creation with vector index loaded."""
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        # Verify context was created with all components
        assert ctx.orchestrator is not None
        assert ctx.vector is not None
        assert ctx.keyword is not None
        assert ctx.graph is not None


class TestReadOnlyContextAttributes:
    """Tests for ReadOnlyContext dataclass attributes."""

    @pytest.mark.asyncio
    async def test_required_attributes(self, tmp_path: Path):
        """Verify ReadOnlyContext has expected attributes."""
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        assert hasattr(ctx, "config")
        assert hasattr(ctx, "orchestrator")
        assert hasattr(ctx, "sync_receiver")
        assert hasattr(ctx, "vector")
        assert hasattr(ctx, "keyword")
        assert hasattr(ctx, "graph")
        assert hasattr(ctx, "memory_manager")
        assert hasattr(ctx, "memory_search")

    @pytest.mark.asyncio
    async def test_memory_attributes_default_to_none(self, tmp_path: Path):
        """Verify memory attributes are None when memory disabled in config."""
        from src.config import MemoryConfig
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
            memory=MemoryConfig(enabled=False),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        assert ctx.memory_manager is None
        assert ctx.memory_search is None

    @pytest.mark.asyncio
    async def test_memory_loaded_when_enabled(self, tmp_path: Path, monkeypatch):
        """Verify memory indices are loaded/rebuilt when enabled."""
        from src.config import MemoryConfig
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        memory_path = tmp_path / "memories"
        memory_path.mkdir()
        (memory_path / "indices").mkdir()

        # Create a memory file to be indexed
        memory_file = memory_path / "test-memory.md"
        memory_file.write_text(
            "---\ntype: observation\nstatus: active\ntags: [test]\n---\n\n"
            "# Test Memory\n\nThis is test content for memory indexing."
        )

        # Patch resolve_memory_path to return our test directory
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

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        assert ctx.memory_manager is not None
        assert ctx.memory_search is not None
        # Verify memory was indexed (rebuilt from files since no indices existed)
        assert ctx.memory_manager.get_memory_count() == 1


class TestFindLatestSnapshotResilience:
    """Tests for _find_latest_snapshot fallback behavior."""

    def test_find_latest_snapshot_normal_case(self, tmp_path: Path):
        """Verify normal case where version.bin points to existing snapshot."""
        from src.reader.context import _find_latest_snapshot

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # version.bin points to v5
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 5))

        # v5 exists
        (snapshot_base / "v5").mkdir()

        result = _find_latest_snapshot(snapshot_base)

        assert result == snapshot_base / "v5"

    def test_find_latest_snapshot_falls_back_to_highest_available(self, tmp_path: Path):
        """
        Verify fallback to highest available snapshot when
        version.bin points to non-existent directory.
        """
        from src.reader.context import _find_latest_snapshot

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # version.bin points to v999 (doesn't exist)
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 999))

        # But v100 and v150 exist on disk
        (snapshot_base / "v100").mkdir()
        (snapshot_base / "v150").mkdir()

        result = _find_latest_snapshot(snapshot_base)

        # Should fall back to v150 (highest available)
        assert result == snapshot_base / "v150"

    def test_find_latest_snapshot_returns_none_when_no_snapshots(self, tmp_path: Path):
        """
        Verify returns None when version.bin points to missing snapshot
        and no other snapshots exist.
        """
        from src.reader.context import _find_latest_snapshot

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # version.bin points to v42 (doesn't exist)
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 42))

        result = _find_latest_snapshot(snapshot_base)

        assert result is None

    def test_find_latest_snapshot_uses_fallback_when_no_version_file(
        self, tmp_path: Path
    ):
        """Verify uses highest snapshot when version.bin is missing."""
        from src.reader.context import _find_latest_snapshot

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # No version.bin, but snapshots exist
        (snapshot_base / "v10").mkdir()
        (snapshot_base / "v20").mkdir()
        (snapshot_base / "v5").mkdir()

        result = _find_latest_snapshot(snapshot_base)

        # Should use v20 (highest available)
        assert result == snapshot_base / "v20"


class TestFindAvailableSnapshots:
    """Tests for _find_available_snapshots helper."""

    def test_find_available_snapshots_returns_sorted_list(self, tmp_path: Path):
        """Verify returns versions sorted descending."""
        from src.reader.context import _find_available_snapshots

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        (snapshot_base / "v10").mkdir()
        (snapshot_base / "v5").mkdir()
        (snapshot_base / "v200").mkdir()
        (snapshot_base / "v50").mkdir()
        (snapshot_base / "not-a-version").mkdir()  # Should be ignored

        available = _find_available_snapshots(snapshot_base)

        versions = [v for v, _ in available]
        assert versions == [200, 50, 10, 5]

    def test_find_available_snapshots_empty_when_no_snapshots(self, tmp_path: Path):
        """Verify returns empty list when no snapshots exist."""
        from src.reader.context import _find_available_snapshots

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        available = _find_available_snapshots(snapshot_base)

        assert available == []

    def test_find_available_snapshots_empty_when_base_missing(self, tmp_path: Path):
        """Verify returns empty list when snapshot base doesn't exist."""
        from src.reader.context import _find_available_snapshots

        snapshot_base = tmp_path / "nonexistent"

        available = _find_available_snapshots(snapshot_base)

        assert available == []
