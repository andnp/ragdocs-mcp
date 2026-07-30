"""Integration tests for hash store integration with IndexManager."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchkernel.config import ChunkingConfig, Config, IndexingConfig
from searchkernel.indexing.manager import IndexManager
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex


def _with_hash(chunk):
    """Finalize a freshly-built domain.Chunk (test helper).

    domain.Chunk, unlike the legacy models.Chunk, does not auto-compute
    content_hash in __post_init__, and its metadata dict must stay JSON
    serializable (it flows into index/docstore persistence), so a raw
    datetime `modified_time` is normalized to ISO text.
    """
    if not chunk.content_hash:
        chunk.content_hash = chunk.compute_content_hash()
    modified_time = chunk.metadata.get("modified_time")
    if hasattr(modified_time, "isoformat"):
        chunk.metadata["modified_time"] = modified_time.isoformat()
    return chunk



@pytest.fixture
def config(tmp_path):
    """Create test configuration."""
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / ".index"),
            delta_full_reindex_threshold=0.5,
        ),
        chunking=ChunkingConfig(
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
    assert (
        manager._hash_store._storage_path
        == Path(config.indexing.index_path) / "chunk_hashes.json"
    )


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
    from searchkernel.domain import Chunk

    test_chunk = _with_hash(Chunk(chunk_id=chunk_id, record_id="test", content="Test content", metadata={**({}), "header_path": "Test", "start_pos": 0, "end_pos": 12, "file_path": str(test_file), "modified_time": datetime.now(UTC)}, chunk_index=0))
    manager._hash_store.set_hash(test_chunk.chunk_id, test_chunk.content_hash)

    # Persist
    manager.persist()

    # Verify hash store file created
    hash_store_path = Path(manager._config.indexing.index_path) / "chunk_hashes.json"
    assert hash_store_path.exists()

    # Load into new manager and verify hash persisted
    from searchkernel.indices.hash_store import ChunkHashStore

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
