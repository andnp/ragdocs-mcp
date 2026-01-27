"""Unit tests for ChunkHashStore."""

import json
from datetime import datetime, timezone

import pytest

from src.indices.hash_store import ChunkHashStore
from src.models import Chunk


@pytest.fixture
def temp_hash_store(tmp_path):
    """Create a temporary hash store for testing."""
    storage_path = tmp_path / "chunk_hashes.json"
    return ChunkHashStore(storage_path)


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="Sample content",
        metadata={},
        chunk_index=0,
        header_path="Introduction",
        start_pos=0,
        end_pos=14,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )


def test_hash_store_initialization(tmp_path):
    """Test hash store initializes with empty state."""
    storage_path = tmp_path / "chunk_hashes.json"
    store = ChunkHashStore(storage_path)

    assert store.get_hash("nonexistent") is None


def test_hash_store_set_and_get(temp_hash_store):
    """Test storing and retrieving a hash."""
    temp_hash_store.set_hash("chunk1", "abc123")

    assert temp_hash_store.get_hash("chunk1") == "abc123"


def test_hash_store_persist_and_load(tmp_path):
    """Test persistence across store instances."""
    storage_path = tmp_path / "chunk_hashes.json"

    # Create first store and add data
    store1 = ChunkHashStore(storage_path)
    store1.set_hash("chunk1", "hash1")
    store1.set_hash("chunk2", "hash2")
    store1.persist()

    # Create second store and verify data loaded
    store2 = ChunkHashStore(storage_path)
    assert store2.get_hash("chunk1") == "hash1"
    assert store2.get_hash("chunk2") == "hash2"


def test_hash_store_remove_document(temp_hash_store):
    """Test removing all chunks for a document."""
    temp_hash_store.set_hash("doc1#chunk-0", "hash1")
    temp_hash_store.set_hash("doc1#chunk-1", "hash2")
    temp_hash_store.set_hash("doc2#chunk-0", "hash3")

    temp_hash_store.remove_document("doc1")

    assert temp_hash_store.get_hash("doc1#chunk-0") is None
    assert temp_hash_store.get_hash("doc1#chunk-1") is None
    assert temp_hash_store.get_hash("doc2#chunk-0") == "hash3"


def test_hash_store_has_changed_new_chunk(temp_hash_store, sample_chunk):
    """Test has_changed returns True for new chunks."""
    assert temp_hash_store.has_changed(sample_chunk) is True


def test_hash_store_has_changed_unchanged_chunk(temp_hash_store, sample_chunk):
    """Test has_changed returns False when content hasn't changed."""
    # Store the chunk's hash
    temp_hash_store.set_hash(sample_chunk.chunk_id, sample_chunk.content_hash)

    assert temp_hash_store.has_changed(sample_chunk) is False


def test_hash_store_has_changed_modified_chunk(temp_hash_store):
    """Test has_changed returns True when content has changed."""
    # Create chunk with original content
    chunk_v1 = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="Original content",
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=16,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    # Store original hash
    temp_hash_store.set_hash(chunk_v1.chunk_id, chunk_v1.content_hash)

    # Create chunk with modified content (same ID)
    chunk_v2 = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="Modified content",
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=16,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    assert temp_hash_store.has_changed(chunk_v2) is True


def test_hash_store_corrupted_json_recovery(tmp_path):
    """Test recovery from corrupted JSON file."""
    storage_path = tmp_path / "chunk_hashes.json"

    # Write corrupted JSON
    with open(storage_path, "w") as f:
        f.write("{invalid json content}")

    # Should initialize with empty state instead of crashing
    store = ChunkHashStore(storage_path)
    assert store.get_hash("any_chunk") is None


def test_hash_store_missing_file(tmp_path):
    """Test initialization when storage file doesn't exist."""
    storage_path = tmp_path / "nonexistent" / "chunk_hashes.json"

    # Should not raise error
    store = ChunkHashStore(storage_path)
    assert store.get_hash("any_chunk") is None


def test_hash_store_persist_creates_directory(tmp_path):
    """Test persist creates parent directory if it doesn't exist."""
    storage_path = tmp_path / "subdir" / "nested" / "chunk_hashes.json"
    store = ChunkHashStore(storage_path)

    store.set_hash("chunk1", "hash1")
    store.persist()

    assert storage_path.exists()
    assert storage_path.parent.exists()


def test_hash_store_persist_overwrites(tmp_path):
    """Test persist overwrites existing file."""
    storage_path = tmp_path / "chunk_hashes.json"

    store1 = ChunkHashStore(storage_path)
    store1.set_hash("chunk1", "hash1")
    store1.persist()

    store2 = ChunkHashStore(storage_path)
    store2.set_hash("chunk1", "updated_hash")
    store2.set_hash("chunk2", "hash2")
    store2.persist()

    # Verify updated data
    with open(storage_path, "r") as f:
        data = json.load(f)

    assert data["chunk1"] == "updated_hash"
    assert data["chunk2"] == "hash2"


def test_hash_store_large_dataset(tmp_path):
    """Test hash store handles large number of chunks."""
    storage_path = tmp_path / "chunk_hashes.json"
    store = ChunkHashStore(storage_path)

    # Add 10,000 hashes
    num_chunks = 10_000
    for i in range(num_chunks):
        store.set_hash(f"doc{i // 100}#chunk-{i % 100}", f"hash{i}")

    store.persist()

    # Load and verify
    store2 = ChunkHashStore(storage_path)
    assert store2.get_hash("doc0#chunk-0") == "hash0"
    assert store2.get_hash("doc50#chunk-50") == "hash5050"
    assert store2.get_hash("doc99#chunk-99") == "hash9999"


def test_hash_store_remove_document_partial_match(temp_hash_store):
    """Test remove_document only matches full document IDs."""
    temp_hash_store.set_hash("doc1#chunk-0", "hash1")
    temp_hash_store.set_hash("doc10#chunk-0", "hash2")
    temp_hash_store.set_hash("doc100#chunk-0", "hash3")

    temp_hash_store.remove_document("doc1")

    # Should only remove exact prefix match
    assert temp_hash_store.get_hash("doc1#chunk-0") is None
    assert temp_hash_store.get_hash("doc10#chunk-0") == "hash2"
    assert temp_hash_store.get_hash("doc100#chunk-0") == "hash3"


def test_hash_store_remove_document_with_special_chars(temp_hash_store):
    """Test remove_document handles special characters in doc IDs."""
    temp_hash_store.set_hash("docs/guide-v1.0#chunk-0", "hash1")
    temp_hash_store.set_hash("docs/guide-v1.0#chunk-1", "hash2")
    temp_hash_store.set_hash("docs/other#chunk-0", "hash3")

    temp_hash_store.remove_document("docs/guide-v1.0")

    assert temp_hash_store.get_hash("docs/guide-v1.0#chunk-0") is None
    assert temp_hash_store.get_hash("docs/guide-v1.0#chunk-1") is None
    assert temp_hash_store.get_hash("docs/other#chunk-0") == "hash3"


def test_hash_store_unicode_chunk_ids(temp_hash_store):
    """Test hash store handles Unicode in chunk IDs."""
    temp_hash_store.set_hash("文档#chunk-0", "hash1")
    temp_hash_store.set_hash("документ#chunk-0", "hash2")

    assert temp_hash_store.get_hash("文档#chunk-0") == "hash1"
    assert temp_hash_store.get_hash("документ#chunk-0") == "hash2"

def test_hash_store_persist_skips_when_not_dirty(tmp_path):
    """Test persist skips write when nothing has changed."""
    storage_path = tmp_path / "chunk_hashes.json"
    store = ChunkHashStore(storage_path)

    # Add some data and persist
    store.set_hash("chunk1", "hash1")
    store.persist()
    assert storage_path.exists()
    initial_mtime = storage_path.stat().st_mtime

    # Touch file with delay to ensure mtime would change
    import time
    time.sleep(0.01)

    # Call persist again without changes - should skip
    store.persist()

    # mtime should be unchanged (no write occurred)
    assert storage_path.stat().st_mtime == initial_mtime


def test_hash_store_dirty_tracking_after_operations(tmp_path):
    """Test dirty tracking across set, remove, and persist operations."""
    storage_path = tmp_path / "chunk_hashes.json"
    store = ChunkHashStore(storage_path)

    # Initial persist with data - use proper chunk ID format (doc_id#chunk-N)
    store.set_hash("doc1#chunk-0", "hash1")
    store.set_hash("doc2#chunk-0", "hash2")
    store.persist()

    # After persist, dirty should be empty, so persist should be no-op
    assert not store._dirty

    # set_hash marks dirty
    store.set_hash("doc3#chunk-0", "hash3")
    assert "doc3#chunk-0" in store._dirty

    # remove_document marks dirty
    store.persist()
    store.remove_document("doc1")
    assert store._dirty  # Should have entries for removed chunks

    # remove_chunk marks dirty
    store.persist()
    store.remove_chunk("doc2#chunk-0")
    assert "doc2#chunk-0" in store._dirty


def test_hash_store_persist_io_error_handling(tmp_path, monkeypatch):
    """Test persist handles I/O errors gracefully."""
    storage_path = tmp_path / "chunk_hashes.json"
    store = ChunkHashStore(storage_path)
    store.set_hash("chunk1", "hash1")

    # Make the file read-only to trigger permission error
    storage_path.parent.chmod(0o444)

    # persist should log error but not crash
    try:
        store.persist()
    except Exception:
        pytest.fail("persist() should not raise exception on I/O error")
    finally:
        # Restore permissions
        storage_path.parent.chmod(0o755)
