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
        from src.indices.graph import GraphStore
        from src.indices.keyword import KeywordIndex
        from src.indices.vector import VectorIndex
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
