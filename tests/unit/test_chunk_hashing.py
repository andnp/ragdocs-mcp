"""Unit tests for chunk content hashing."""

import hashlib
from datetime import datetime, timezone

from src.models import Chunk


def test_chunk_hash_computed_on_init():
    """Test that content hash is automatically computed during initialization."""
    chunk = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="This is test content",
        metadata={},
        chunk_index=0,
        header_path="Introduction",
        start_pos=0,
        end_pos=20,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    assert chunk.content_hash != ""
    assert len(chunk.content_hash) == 64  # SHA256 hex digest length


def test_chunk_hash_consistency():
    """Test that same content produces same hash."""
    content = "Identical content for both chunks"

    chunk1 = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content=content,
        metadata={},
        chunk_index=0,
        header_path="Section A",
        start_pos=0,
        end_pos=33,
        file_path="test1.md",
        modified_time=datetime.now(timezone.utc),
    )

    chunk2 = Chunk(
        chunk_id="doc2#chunk-0",
        doc_id="doc2",
        content=content,
        metadata={"different": "metadata"},
        chunk_index=5,
        header_path="Section B",
        start_pos=100,
        end_pos=133,
        file_path="test2.md",
        modified_time=datetime.now(timezone.utc),
    )

    assert chunk1.content_hash == chunk2.content_hash


def test_chunk_hash_uniqueness():
    """Test that different content produces different hashes."""
    chunk1 = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="First content",
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=13,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    chunk2 = Chunk(
        chunk_id="doc1#chunk-1",
        doc_id="doc1",
        content="Second content",
        metadata={},
        chunk_index=1,
        header_path="Section",
        start_pos=14,
        end_pos=28,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    assert chunk1.content_hash != chunk2.content_hash


def test_chunk_hash_matches_manual_computation():
    """Test that computed hash matches manual SHA256 computation."""
    content = "Test content for manual hash verification"

    chunk = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content=content,
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=41,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert chunk.content_hash == expected_hash


def test_chunk_hash_whitespace_sensitivity():
    """Test that hash is sensitive to whitespace differences."""
    chunk1 = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="Content without trailing space",
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=30,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    chunk2 = Chunk(
        chunk_id="doc1#chunk-1",
        doc_id="doc1",
        content="Content without trailing space ",  # Note trailing space
        metadata={},
        chunk_index=1,
        header_path="Section",
        start_pos=31,
        end_pos=62,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    assert chunk1.content_hash != chunk2.content_hash


def test_chunk_hash_empty_content():
    """Test hash computation for empty content."""
    chunk = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="",
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=0,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    # Empty string should produce a valid hash
    assert chunk.content_hash != ""
    expected_hash = hashlib.sha256(b"").hexdigest()
    assert chunk.content_hash == expected_hash


def test_chunk_hash_unicode_content():
    """Test hash computation for Unicode content."""
    content = "Test with émojis 🎉 and spëcial çharacters"

    chunk = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content=content,
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=len(content),
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    # Should handle UTF-8 encoding correctly
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert chunk.content_hash == expected_hash


def test_chunk_hash_newline_sensitivity():
    """Test that hash is sensitive to newline differences."""
    chunk1 = Chunk(
        chunk_id="doc1#chunk-0",
        doc_id="doc1",
        content="Line one\nLine two",
        metadata={},
        chunk_index=0,
        header_path="Section",
        start_pos=0,
        end_pos=17,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    chunk2 = Chunk(
        chunk_id="doc1#chunk-1",
        doc_id="doc1",
        content="Line one\r\nLine two",  # CRLF instead of LF
        metadata={},
        chunk_index=1,
        header_path="Section",
        start_pos=18,
        end_pos=36,
        file_path="test.md",
        modified_time=datetime.now(timezone.utc),
    )

    assert chunk1.content_hash != chunk2.content_hash


def test_chunk_compute_content_hash_method():
    """Test the compute_content_hash method can be called explicitly."""
    chunk = Chunk(
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

    original_hash = chunk.content_hash

    # Manually modify content (not recommended, but test method directly)
    chunk.content = "Modified content"
    recomputed_hash = chunk.compute_content_hash()

    assert recomputed_hash != original_hash
    expected_hash = hashlib.sha256(b"Modified content").hexdigest()
    assert recomputed_hash == expected_hash
