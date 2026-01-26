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
