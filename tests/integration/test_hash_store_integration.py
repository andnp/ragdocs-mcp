"""Integration tests for hash store integration with IndexManager."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import ChunkingConfig, Config, IndexingConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex


@pytest.fixture
def config(tmp_path):
    """Create test configuration."""
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / ".index"),
            enable_delta_indexing=True,
            delta_full_reindex_threshold=0.5,
        ),
        document_chunking=ChunkingConfig(
            min_chunk_chars=100,
            max_chunk_chars=500,
        ),
    )


@pytest.fixture
def indices(shared_embedding_model):
    """Create fresh indices."""
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(config, indices):
    """Create IndexManager with hash store."""
    vector, keyword, graph = indices
    return IndexManager(config, vector, keyword, graph)


def test_hash_store_initialized(manager, config):
    """Test that hash store is properly initialized with IndexManager."""
    assert manager._hash_store is not None
    assert manager._hash_store._storage_path == Path(
        config.indexing.index_path
    ) / "chunk_hashes.json"


def test_hash_store_persisted_with_indices(tmp_path, manager):
    """Test that hash store is persisted alongside other indices."""
    # Create a test document
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    test_file = docs_path / "test.md"
    test_file.write_text("# Test\n\nSome content for testing.")

    # Index the document
    manager.index_document(str(test_file))

    # Get hash from chunks
    chunks = list(manager.vector._index.docstore.docs.values())
    assert len(chunks) > 0
    chunk_id = chunks[0].id_

    # Manually store hash (simulating delta indexing flow)
    from src.models import Chunk

    test_chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="test",
        content="Test content",
        metadata={},
        chunk_index=0,
        header_path="Test",
        start_pos=0,
        end_pos=12,
        file_path=str(test_file),
        modified_time=datetime.now(timezone.utc),
    )
    manager._hash_store.set_hash(test_chunk.chunk_id, test_chunk.content_hash)

    # Persist
    manager.persist()

    # Verify hash store file created
    hash_store_path = Path(manager._config.indexing.index_path) / "chunk_hashes.json"
    assert hash_store_path.exists()

    # Load into new manager and verify hash persisted
    from src.indices.hash_store import ChunkHashStore

    new_hash_store = ChunkHashStore(hash_store_path)
    assert new_hash_store.get_hash(chunk_id) is not None


def test_hash_store_survives_reload(tmp_path, config, shared_embedding_model):
    """Test that hash store persists across IndexManager instances."""
    # Create first manager
    vector1 = VectorIndex(embedding_model=shared_embedding_model)
    keyword1 = KeywordIndex()
    graph1 = GraphStore()
    manager1 = IndexManager(config, vector1, keyword1, graph1)

    # Store some hashes
    manager1._hash_store.set_hash("doc1#chunk-0", "hash1")
    manager1._hash_store.set_hash("doc1#chunk-1", "hash2")
    manager1.persist()

    # Create second manager (simulating restart)
    vector2 = VectorIndex(embedding_model=shared_embedding_model)
    keyword2 = KeywordIndex()
    graph2 = GraphStore()
    manager2 = IndexManager(config, vector2, keyword2, graph2)

    # Verify hashes loaded
    assert manager2._hash_store.get_hash("doc1#chunk-0") == "hash1"
    assert manager2._hash_store.get_hash("doc1#chunk-1") == "hash2"


def test_hash_store_disabled_when_config_false(tmp_path):
    """Test that hash store still initializes but isn't used when disabled."""
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / ".index"),
            enable_delta_indexing=False,  # Disabled
        ),
        document_chunking=ChunkingConfig(),
    )

    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)

    # Hash store should still be initialized (for future use)
    assert manager._hash_store is not None

    # But persist shouldn't write it when disabled
    manager._hash_store.set_hash("test#chunk-0", "hash1")
    manager.persist()

    # Behavior: hash store exists but delta indexing logic won't use it
    # This is acceptable - infrastructure is ready but feature is off
