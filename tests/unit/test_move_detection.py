"""Unit tests for file move detection functionality."""

from datetime import UTC, datetime

import pytest
from searchkernel.domain import Chunk
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.hash_store import ChunkHashStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex

from mcp_markdown_ragdocs.config import ChunkingConfig, Config, IndexingConfig
from mcp_markdown_ragdocs.indexing.manager import IndexManager


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
def hash_store(tmp_path):
    return ChunkHashStore(tmp_path / "hashes.json")


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        _with_hash(Chunk(chunk_id="docs/test_chunk_0", record_id="docs/test", content="First chunk content", metadata={"header": "Introduction", "header_path": "# Introduction", "start_pos": 0, "end_pos": 50, "file_path": "/docs/test.md", "modified_time": datetime.now(UTC)}, chunk_index=0)),
        _with_hash(Chunk(chunk_id="docs/test_chunk_1", record_id="docs/test", content="Second chunk content", metadata={"header": "Details", "header_path": "# Details", "start_pos": 51, "end_pos": 100, "file_path": "/docs/test.md", "modified_time": datetime.now(UTC)}, chunk_index=1)),
    ]


# ============================================================================
# Hash Store Tests
# ============================================================================


def test_hash_store_reverse_lookup(hash_store, sample_chunks):
    """Test get_chunk_id_by_hash returns first chunk with matching hash."""
    chunk = sample_chunks[0]
    hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    retrieved_id = hash_store.get_chunk_id_by_hash(chunk.content_hash)
    assert retrieved_id == chunk.chunk_id


def test_hash_store_reverse_lookup_not_found(hash_store):
    """Test get_chunk_id_by_hash returns None for unknown hash."""
    result = hash_store.get_chunk_id_by_hash("nonexistent_hash")
    assert result is None


def test_hash_store_get_chunks_by_document(hash_store, sample_chunks):
    """Test get_chunks_by_document returns all chunks for a document."""
    for chunk in sample_chunks:
        hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    chunks = hash_store.get_chunks_by_document("docs/test")
    assert len(chunks) == 2

    chunk_ids = {chunk_id for chunk_id, _ in chunks}
    assert "docs/test_chunk_0" in chunk_ids
    assert "docs/test_chunk_1" in chunk_ids


def test_hash_store_get_chunks_by_document_not_found(hash_store):
    """Test get_chunks_by_document returns empty list for unknown doc."""
    chunks = hash_store.get_chunks_by_document("nonexistent_doc")
    assert chunks == []


def test_hash_store_maintains_reverse_lookup_on_update(hash_store):
    """Test reverse lookup is maintained when updating hash."""
    old_hash = "old_hash_value"
    new_hash = "new_hash_value"

    hash_store.set_hash("chunk_1", old_hash)
    assert hash_store.get_chunk_id_by_hash(old_hash) == "chunk_1"

    # Update to new hash
    hash_store.set_hash("chunk_1", new_hash)

    # Old hash should not resolve
    assert hash_store.get_chunk_id_by_hash(old_hash) is None
    # New hash should resolve
    assert hash_store.get_chunk_id_by_hash(new_hash) == "chunk_1"


def test_hash_store_maintains_reverse_lookup_on_remove(hash_store, sample_chunks):
    """Test reverse lookup is maintained when removing chunks."""
    chunk = sample_chunks[0]
    hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    # Verify it exists
    assert hash_store.get_chunk_id_by_hash(chunk.content_hash) == chunk.chunk_id

    # Remove chunk
    hash_store.remove_chunk(chunk.chunk_id)

    # Should no longer resolve
    assert hash_store.get_chunk_id_by_hash(chunk.content_hash) is None


def test_hash_store_maintains_reverse_lookup_on_remove_document(
    hash_store, sample_chunks
):
    """Test reverse lookup is maintained when removing document."""
    for chunk in sample_chunks:
        hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    # Remove document
    hash_store.remove_document("docs/test")

    # All chunks should be removed from reverse lookup
    for chunk in sample_chunks:
        assert hash_store.get_chunk_id_by_hash(chunk.content_hash) is None


def test_hash_store_persist_and_load_reverse_lookup(tmp_path, sample_chunks):
    """Test reverse lookup is rebuilt after persist/load."""
    store_path = tmp_path / "hashes.json"
    hash_store = ChunkHashStore(store_path)

    for chunk in sample_chunks:
        hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    hash_store.persist()

    # Load into new instance
    hash_store2 = ChunkHashStore(store_path)

    # Verify reverse lookup works
    for chunk in sample_chunks:
        assert hash_store2.get_chunk_id_by_hash(chunk.content_hash) == chunk.chunk_id


# ============================================================================
# VectorIndex Tests
# ============================================================================


def test_vector_index_update_chunk_path(shared_embedding_model):
    """Test VectorIndex.update_chunk_path creates new chunk with reused content."""
    vector = VectorIndex(embedding_model=shared_embedding_model)

    chunk = _with_hash(Chunk(chunk_id="old_path_chunk_0", record_id="old_path", content="Test content", metadata={"tags": ["test"], "header_path": "# Header", "start_pos": 0, "end_pos": 50, "file_path": "/old/path.md", "modified_time": datetime.now(UTC)}, chunk_index=0))

    vector.add_chunk(chunk)

    # Update path
    new_metadata = {
        "doc_id": "new_path",
        "chunk_id": "new_path_chunk_0",
        "file_path": "/new/path.md",
        "header_path": "# Header",
    }

    success = vector.update_chunk_path(
        "old_path_chunk_0", "new_path_chunk_0", new_metadata
    )
    assert success is True

    # New chunk should be in mappings
    assert "new_path_chunk_0" in vector._chunk_id_to_node_id
    # Old chunk should be removed from mappings
    assert "old_path_chunk_0" not in vector._chunk_id_to_node_id

    # New doc_id should have the node
    assert "new_path" in vector._doc_id_to_node_ids
    assert len(vector._doc_id_to_node_ids["new_path"]) > 0


def test_vector_index_update_chunk_path_not_found(shared_embedding_model):
    """Test update_chunk_path returns False for nonexistent chunk."""
    vector = VectorIndex(embedding_model=shared_embedding_model)

    result = vector.update_chunk_path(
        "nonexistent_chunk", "new_chunk", {"doc_id": "test", "file_path": "/test.md"}
    )
    assert result is False


# ============================================================================
# KeywordIndex Tests
# ============================================================================


def test_keyword_index_move_chunk():
    """Test KeywordIndex.move_chunk copies document with new ID."""
    keyword = KeywordIndex()

    chunk = _with_hash(Chunk(chunk_id="old_path_chunk_0", record_id="old_path", content="Test content for keyword search", metadata={ "header_path": "", "start_pos": 0, "end_pos": 50, "file_path": "/old/path.md", "modified_time": datetime.now(UTC)}, chunk_index=0))

    keyword.add_chunk(chunk)

    # Create new chunk with updated path
    new_chunk = _with_hash(Chunk(chunk_id="new_path_chunk_0", record_id="new_path", content="Test content for keyword search", metadata={ "header_path": "", "start_pos": 0, "end_pos": 50, "file_path": "/new/path.md", "modified_time": datetime.now(UTC)}, chunk_index=0))

    success = keyword.move_chunk("old_path_chunk_0", new_chunk)
    assert success is True

    # Verify new chunk is searchable
    results = keyword.search("content", top_k=5)
    chunk_ids = [r["chunk_id"] for r in results]
    assert "new_path_chunk_0" in chunk_ids
    # Old chunk should not appear
    assert "old_path_chunk_0" not in chunk_ids


def test_keyword_index_move_chunk_not_found():
    """Test move_chunk returns False for nonexistent chunk."""
    keyword = KeywordIndex()

    new_chunk = _with_hash(Chunk(chunk_id="new_chunk", record_id="test", content="test", metadata={ "header_path": "", "start_pos": 0, "end_pos": 10, "file_path": "/test.md", "modified_time": datetime.now(UTC)}, chunk_index=0))

    result = keyword.move_chunk("nonexistent_chunk", new_chunk)
    assert result is False


# ============================================================================
# GraphStore Tests
# ============================================================================


def test_graph_rename_node():
    """Test GraphStore.rename_node preserves edges and metadata."""
    graph = GraphStore()

    # Create nodes and edges
    graph.add_node("old_node", {"meta": "data"})
    graph.add_node("other_node", {})
    graph.add_edge(
        "old_node", "other_node", edge_type="links_to", edge_context="context1"
    )
    graph.add_edge(
        "other_node", "old_node", edge_type="links_to", edge_context="context2"
    )

    # Rename node
    success = graph.rename_node("old_node", "new_node")
    assert success is True

    # Verify old node gone, new node exists
    assert not graph.has_node("old_node")
    assert graph.has_node("new_node")

    # Verify edges preserved
    out_edges = graph.get_edges_from("new_node")
    assert len(out_edges) == 1
    assert out_edges[0]["target"] == "other_node"
    assert out_edges[0]["edge_context"] == "context1"

    in_edges = graph.get_edges_to("new_node")
    assert len(in_edges) == 1
    assert in_edges[0]["source"] == "other_node"
    assert in_edges[0]["edge_context"] == "context2"


def test_graph_rename_node_not_found():
    """Test rename_node returns False for nonexistent node."""
    graph = GraphStore()

    result = graph.rename_node("nonexistent", "new_node")
    assert result is False


# ============================================================================
# IndexManager Move Detection Tests
# ============================================================================


@pytest.fixture
def config(tmp_path):
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "index"),
            move_detection_threshold=0.8,
        ),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def manager(config, shared_embedding_model):
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()

    return IndexManager(config, vector, keyword, graph)


def test_detect_file_moves_simple_rename(manager, sample_chunks):
    """Test _detect_file_moves identifies perfect content match."""
    # Simulate removed doc with old chunks
    for chunk in sample_chunks:
        manager._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    removed_docs = {"docs/test"}

    # Create new chunks with same content but different path
    new_chunks = [
        _with_hash(Chunk(chunk_id="docs/renamed_chunk_0", record_id="docs/renamed", content=sample_chunks[0].content, metadata={ "header_path": sample_chunks[0].metadata.get("header_path"), "start_pos": 0, "end_pos": 50, "file_path": "/docs/renamed.md", "modified_time": datetime.now(UTC)}, chunk_index=0)),
        _with_hash(Chunk(chunk_id="docs/renamed_chunk_1", record_id="docs/renamed", content=sample_chunks[1].content, metadata={ "header_path": sample_chunks[1].metadata.get("header_path"), "start_pos": 51, "end_pos": 100, "file_path": "/docs/renamed.md", "modified_time": datetime.now(UTC)}, chunk_index=1)),
    ]

    added_docs = {"docs/renamed": new_chunks}

    moves = manager._detect_file_moves(removed_docs, added_docs)

    assert "docs/test" in moves
    assert moves["docs/test"] == "docs/renamed"


def test_detect_file_moves_with_edit(manager, sample_chunks):
    """Test move detection with partial content changes."""
    # Store old chunks
    for chunk in sample_chunks:
        manager._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    removed_docs = {"docs/test"}

    # Create new chunks: one changed, one unchanged
    new_chunks = [
        _with_hash(Chunk(chunk_id="docs/moved_chunk_0", record_id="docs/moved", content="CHANGED CONTENT", metadata={ "header_path": "# Introduction", "start_pos": 0, "end_pos": 50, "file_path": "/docs/moved.md", "modified_time": datetime.now(UTC)}, chunk_index=0)),
        _with_hash(Chunk(chunk_id="docs/moved_chunk_1", record_id="docs/moved", content=sample_chunks[1].content, metadata={ "header_path": "# Details", "start_pos": 51, "end_pos": 100, "file_path": "/docs/moved.md", "modified_time": datetime.now(UTC)}, chunk_index=1)),
    ]

    added_docs = {"docs/moved": new_chunks}

    moves = manager._detect_file_moves(removed_docs, added_docs)

    # 50% match should NOT trigger move (threshold is 80%)
    assert "docs/test" not in moves


def test_detect_file_moves_threshold(manager):
    """Test move detection respects threshold setting."""
    # Create 10 chunks, 8 matching (80% threshold)
    old_chunks = []
    for i in range(10):
        chunk = _with_hash(Chunk(chunk_id=f"old_doc_chunk_{i}", record_id="old_doc", content=f"Chunk {i} content", metadata={ "header_path": f"# Section {i}", "start_pos": i * 100, "end_pos": (i + 1) * 100, "file_path": "/old/doc.md", "modified_time": datetime.now(UTC)}, chunk_index=i))
        manager._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)
        old_chunks.append(chunk)

    removed_docs = {"old_doc"}

    # New doc with 8 matching chunks + 2 new chunks
    new_chunks = []
    for i in range(8):
        new_chunks.append(
            _with_hash(Chunk(chunk_id=f"new_doc_chunk_{i}", record_id="new_doc", content=old_chunks[i].content, metadata={ "header_path": f"# Section {i}", "start_pos": i * 100, "end_pos": (i + 1) * 100, "file_path": "/new/doc.md", "modified_time": datetime.now(UTC)}, chunk_index=i))
        )

    # Add 2 new chunks
    for i in range(8, 10):
        new_chunks.append(
            _with_hash(Chunk(chunk_id=f"new_doc_chunk_{i}", record_id="new_doc", content=f"NEW CONTENT {i}", metadata={ "header_path": f"# Section {i}", "start_pos": i * 100, "end_pos": (i + 1) * 100, "file_path": "/new/doc.md", "modified_time": datetime.now(UTC)}, chunk_index=i))
        )

    added_docs = {"new_doc": new_chunks}

    moves = manager._detect_file_moves(removed_docs, added_docs)

    # 80% match should trigger move (exactly at threshold)
    assert "old_doc" in moves
    assert moves["old_doc"] == "new_doc"
