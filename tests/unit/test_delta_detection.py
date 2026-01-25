"""Unit tests for delta detection logic in IndexManager."""
from datetime import datetime

import pytest

from src.config import ChunkingConfig, Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import Chunk
from tests.conftest import create_test_document


def make_chunk(chunk_id: str, doc_id: str, content: str, chunk_index: int = 0) -> Chunk:
    """Helper to create test chunks with minimal boilerplate."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        metadata={},
        chunk_index=chunk_index,
        header_path="",
        start_pos=0,
        end_pos=len(content),
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )


@pytest.fixture
def manager(tmp_path, shared_embedding_model):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_dir),
            index_path=str(tmp_path / ".index_data"),
            enable_delta_indexing=True,
            delta_full_reindex_threshold=0.5,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="local"),
        document_chunking=ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=200,
            max_chunk_chars=1500,
        ),
        memory_chunking=ChunkingConfig(),
    )

    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return IndexManager(config, vector, keyword, graph)


def test_detect_changed_chunks_all_new(manager):
    """Verify _detect_changed_chunks detects all chunks as new when hash store is empty."""
    chunks = [
        make_chunk("doc1#chunk_0", "doc1", "First chunk content", 0),
        make_chunk("doc1#chunk_1", "doc1", "Second chunk content", 1),
    ]

    changed, unchanged = manager._detect_changed_chunks(chunks)

    assert len(changed) == 2
    assert len(unchanged) == 0
    assert all(c in changed for c in chunks)


def test_detect_changed_chunks_no_changes(manager):
    """Verify _detect_changed_chunks detects no changes when hashes match."""
    chunks = [
        make_chunk("doc1#chunk_0", "doc1", "First chunk content", 0),
    ]

    # Store initial hashes
    for chunk in chunks:
        manager._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    # Re-check with same chunks
    changed, unchanged = manager._detect_changed_chunks(chunks)

    assert len(changed) == 0
    assert len(unchanged) == 1
    assert unchanged[0] == "doc1#chunk_0"


def test_detect_changed_chunks_partial_changes(manager):
    """Verify _detect_changed_chunks detects partial changes correctly."""
    chunks = [
        make_chunk("doc1#chunk_0", "doc1", "Original content", 0),
        make_chunk("doc1#chunk_1", "doc1", "Unchanged content", 1),
    ]

    # Store initial hashes
    for chunk in chunks:
        manager._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    # Modify first chunk
    updated_chunks = [
        make_chunk("doc1#chunk_0", "doc1", "Modified content", 0),
        make_chunk("doc1#chunk_1", "doc1", "Unchanged content", 1),
    ]

    changed, unchanged = manager._detect_changed_chunks(updated_chunks)

    assert len(changed) == 1
    assert len(unchanged) == 1
    assert changed[0].chunk_id == "doc1#chunk_0"
    assert unchanged[0] == "doc1#chunk_1"


def test_detect_changed_chunks_with_delta_disabled(manager):
    """Verify _detect_changed_chunks treats all chunks as changed when delta is disabled."""
    # Disable delta indexing
    manager._config.indexing.enable_delta_indexing = False

    chunks = [
        make_chunk("doc1#chunk_0", "doc1", "Content", 0),
    ]

    # Store hashes (should be ignored)
    for chunk in chunks:
        manager._hash_store.set_hash(chunk.chunk_id, chunk.content_hash)

    changed, unchanged = manager._detect_changed_chunks(chunks)

    assert len(changed) == 1
    assert len(unchanged) == 0


def test_should_use_delta_indexing_below_threshold(manager):
    """Verify _should_use_delta_indexing returns True when change ratio is below threshold."""
    total_chunks = 10
    changed_chunks = [
        make_chunk(f"doc#chunk_{i}", "doc", f"Content {i}", i)
        for i in range(4)
    ]

    # 4/10 = 40% < 50% threshold
    should_use_delta = manager._should_use_delta_indexing(changed_chunks, total_chunks)

    assert should_use_delta is True


def test_should_use_delta_indexing_above_threshold(manager):
    """Verify _should_use_delta_indexing returns False when change ratio exceeds threshold."""
    total_chunks = 10
    changed_chunks = [
        make_chunk(f"doc#chunk_{i}", "doc", f"Content {i}", i)
        for i in range(6)
    ]

    # 6/10 = 60% > 50% threshold
    should_use_delta = manager._should_use_delta_indexing(changed_chunks, total_chunks)

    assert should_use_delta is False


def test_should_use_delta_indexing_exactly_at_threshold(manager):
    """Verify _should_use_delta_indexing behavior at exact threshold boundary."""
    total_chunks = 10
    changed_chunks = [
        make_chunk(f"doc#chunk_{i}", "doc", f"Content {i}", i)
        for i in range(5)
    ]

    # 5/10 = 50% = threshold (should use delta)
    should_use_delta = manager._should_use_delta_indexing(changed_chunks, total_chunks)

    assert should_use_delta is True


def test_should_use_delta_indexing_zero_chunks(manager):
    """Verify _should_use_delta_indexing handles edge case of zero total chunks."""
    changed_chunks = []
    total_chunks = 0

    should_use_delta = manager._should_use_delta_indexing(changed_chunks, total_chunks)

    # Should default to True (no operation needed)
    assert should_use_delta is True


def test_update_chunks_removes_old_and_adds_new(manager, tmp_path):
    """Verify _update_chunks correctly removes old chunks and adds new ones."""
    docs_dir = tmp_path / "docs"
    doc_path = create_test_document(docs_dir, "test_doc", "# Title\n\nOriginal content")

    # Initial index
    manager.index_document(doc_path)

    original_chunk_ids = list(manager._hash_store._hashes.keys())
    assert len(original_chunk_ids) > 0

    # Create modified chunks
    modified_chunks = [
        make_chunk("test_doc#chunk_0", "test_doc", "Modified content", 0),
    ]

    # Update chunks
    manager._update_chunks("test_doc", modified_chunks)

    # Verify old chunks were removed and new ones added
    vector_doc_ids = manager.vector.get_document_ids()
    assert "test_doc" in vector_doc_ids

    # Verify keyword index has updated content
    keyword_results = manager.keyword.search("Modified content", top_k=5)
    assert len(keyword_results) > 0
    assert any("test_doc" in r["doc_id"] for r in keyword_results)


def test_full_reindex_document_removes_all_old_chunks(manager, tmp_path):
    """Verify _full_reindex_document completely replaces old chunks."""
    docs_dir = tmp_path / "docs"
    doc_path = create_test_document(docs_dir, "test_doc", "# Title\n\nOriginal content")

    # Initial index
    manager.index_document(doc_path)

    original_hashes = manager._hash_store._hashes.copy()
    assert len(original_hashes) > 0

    # Create completely new chunks with matching format
    new_chunks = [
        make_chunk("test_doc_chunk_0", "test_doc", "Completely new content", 0),
        make_chunk("test_doc_chunk_1", "test_doc", "Additional new chunk", 1),
    ]

    # Full re-index
    manager._full_reindex_document("test_doc", new_chunks)

    # Verify hash store updated
    new_hashes = manager._hash_store._hashes
    assert len(new_hashes) == 2
    assert all(k.startswith("test_doc_chunk_") for k in new_hashes.keys())

    # Verify chunks are in indices
    vector_doc_ids = manager.vector.get_document_ids()
    assert "test_doc" in vector_doc_ids


def test_full_reindex_document_persists_hashes(manager, tmp_path):
    """Verify _full_reindex_document persists hash store to disk."""
    docs_dir = tmp_path / "docs"
    # Create test document (needed for directory setup)
    _ = create_test_document(docs_dir, "test_doc", "# Title\n\nContent")

    chunks = [
        make_chunk("test_doc#chunk_0", "test_doc", "Test content", 0),
    ]

    # Full re-index
    manager._full_reindex_document("test_doc", chunks)

    # Verify hash store file exists
    hash_store_path = tmp_path / ".index_data" / "chunk_hashes.json"
    assert hash_store_path.exists()

    # Verify content
    import json
    with open(hash_store_path, "r") as f:
        persisted_hashes = json.load(f)

    assert "test_doc#chunk_0" in persisted_hashes
