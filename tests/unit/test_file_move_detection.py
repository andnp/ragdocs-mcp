"""
Unit tests for IndexManager file move detection.

Tests the hash-based file move detection that avoids re-embedding when
a file is renamed but content remains the same.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import Chunk


def make_chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    chunk_index: int = 0,
    content_hash: str | None = None,
):
    """Helper to create a Chunk with minimal required fields."""
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        metadata={},
        chunk_index=chunk_index,
        header_path="",
        start_pos=0,
        end_pos=len(content),
        file_path=f"/docs/{doc_id}.md",
        modified_time=datetime.now(timezone.utc),
    )
    # Override computed hash if needed for test control
    if content_hash is not None:
        object.__setattr__(chunk, "content_hash", content_hash)
    return chunk


@pytest.fixture
def config_with_move_detection(tmp_path: Path):
    """Config with move detection and delta indexing enabled."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    index_path = tmp_path / "index"
    index_path.mkdir()

    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
            enable_delta_indexing=True,
            enable_move_detection=True,
            move_detection_threshold=0.8,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        llm=LLMConfig(embedding_model="local"),
    )


@pytest.fixture
def config_without_move_detection(tmp_path: Path):
    """Config with delta indexing but move detection disabled."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    index_path = tmp_path / "index"
    index_path.mkdir()

    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
            enable_delta_indexing=True,
            enable_move_detection=False,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        llm=LLMConfig(embedding_model="local"),
    )


@pytest.fixture
def manager_with_move_detection(config_with_move_detection: Config, shared_embedding_model):
    """IndexManager with move detection enabled."""
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return IndexManager(config_with_move_detection, vector, keyword, graph)


class TestDetectFileMoves:
    """Tests for _detect_file_moves() hash comparison logic."""

    def test_detect_move_identical_content(self, manager_with_move_detection: IndexManager):
        """
        Verify file move detection when content is identical.

        When a file is renamed without content changes, all chunk hashes
        should match, resulting in 100% match ratio.
        """
        manager = manager_with_move_detection

        # Simulate existing chunks in hash store with specific hashes
        # Note: chunk_id format is {doc_id}_chunk_{index}
        manager._hash_store.set_hash("old_doc_chunk_0", "hash_a")
        manager._hash_store.set_hash("old_doc_chunk_1", "hash_b")
        manager._hash_store.persist()

        # New chunks with same hashes (content identical)
        new_chunks = [
            make_chunk("new_doc_chunk_0", "new_doc", "Content A", 0, content_hash="hash_a"),
            make_chunk("new_doc_chunk_1", "new_doc", "Content B", 1, content_hash="hash_b"),
        ]

        removed_docs = {"old_doc"}
        added_docs = {"new_doc": new_chunks}

        moves = manager._detect_file_moves(removed_docs, added_docs)

        assert "old_doc" in moves
        assert moves["old_doc"] == "new_doc"

    def test_detect_move_partial_content_change(self, manager_with_move_detection: IndexManager):
        """
        Verify move detection with threshold (content partially changed).

        If 80%+ of chunks match, it's still considered a move.
        """
        manager = manager_with_move_detection

        # 5 chunks in old document
        for i in range(5):
            manager._hash_store.set_hash(f"old_doc_chunk_{i}", f"hash_{i}")
        manager._hash_store.persist()

        # 4 of 5 chunks match (80% = threshold)
        new_chunks = [
            make_chunk(
                f"new_doc_chunk_{i}",
                "new_doc",
                f"Content {i}",
                i,
                content_hash=f"hash_{i}" if i < 4 else "hash_new",
            )
            for i in range(5)
        ]

        moves = manager._detect_file_moves({"old_doc"}, {"new_doc": new_chunks})

        assert "old_doc" in moves, "80% match should trigger move detection"

    def test_no_move_detected_below_threshold(self, manager_with_move_detection: IndexManager):
        """
        Verify no move detected when match ratio is below threshold.

        If content has changed significantly (< 80% match), treat as
        delete + add rather than move.
        """
        manager = manager_with_move_detection

        # 5 chunks in old document
        for i in range(5):
            manager._hash_store.set_hash(f"old_doc_chunk_{i}", f"hash_{i}")
        manager._hash_store.persist()

        # Only 3 of 5 chunks match (60% < 80% threshold)
        new_chunks = [
            make_chunk(
                f"new_doc_chunk_{i}",
                "new_doc",
                f"Content {i}",
                i,
                content_hash=f"hash_{i}" if i < 3 else f"new_hash_{i}",
            )
            for i in range(5)
        ]

        moves = manager._detect_file_moves({"old_doc"}, {"new_doc": new_chunks})

        assert "old_doc" not in moves, "60% match should NOT trigger move detection"

    def test_move_detection_disabled(self, config_without_move_detection: Config, shared_embedding_model):
        """
        Verify move detection returns empty when disabled in config.
        """
        vector = VectorIndex(embedding_model=shared_embedding_model)
        manager = IndexManager(config_without_move_detection, vector, KeywordIndex(), GraphStore())

        manager._hash_store.set_hash("old_doc_chunk_0", "hash_a")
        manager._hash_store.persist()

        new_chunks = [make_chunk("new_doc_chunk_0", "new_doc", "Content", 0, content_hash="hash_a")]

        moves = manager._detect_file_moves({"old_doc"}, {"new_doc": new_chunks})

        assert moves == {}, "Move detection should return empty when disabled"

    def test_move_detection_best_match_selection(self, manager_with_move_detection: IndexManager):
        """
        Verify move detection selects best matching removed document.

        When new doc could match multiple removed docs, pick highest ratio.
        """
        manager = manager_with_move_detection

        # Two old documents with different content
        manager._hash_store.set_hash("old_a_chunk_0", "hash_1")
        manager._hash_store.set_hash("old_a_chunk_1", "hash_2")

        manager._hash_store.set_hash("old_b_chunk_0", "hash_1")
        manager._hash_store.set_hash("old_b_chunk_1", "hash_3")
        manager._hash_store.set_hash("old_b_chunk_2", "hash_4")
        manager._hash_store.persist()

        # New doc matches old_a better (100% vs 33%)
        new_chunks = [
            make_chunk("new_doc_chunk_0", "new_doc", "Content", 0, content_hash="hash_1"),
            make_chunk("new_doc_chunk_1", "new_doc", "Content", 1, content_hash="hash_2"),
        ]

        moves = manager._detect_file_moves({"old_a", "old_b"}, {"new_doc": new_chunks})

        assert "old_a" in moves
        assert moves["old_a"] == "new_doc"
        assert "old_b" not in moves


class TestApplyFileMove:
    """Tests for _apply_file_move() that updates indices without re-embedding."""

    def test_apply_move_returns_false_when_no_old_chunks(self, manager_with_move_detection: IndexManager):
        """
        Verify _apply_file_move returns False when old doc has no stored hashes.

        This should trigger fallback to full re-index.
        """
        manager = manager_with_move_detection

        new_chunks = [make_chunk("new_doc_chunk_0", "new_doc", "Content", 0, content_hash="hash_a")]

        # No hashes stored for old_doc
        result = manager._apply_file_move("old_doc", "new_doc", new_chunks)

        assert result is False, "Should return False when no old chunks found"


class TestFileMoveIntegration:
    """Integration tests for complete file move workflow."""

    def test_index_then_rename_creates_hash_store_entries(
        self,
        config_with_move_detection: Config,
        shared_embedding_model,
    ):
        """
        End-to-end test: index file, verify hash store is populated.

        This ensures the prerequisite for move detection (stored hashes) works.
        """
        docs_path = Path(config_with_move_detection.indexing.documents_path)

        # Create and index original file
        original_path = docs_path / "original.md"
        original_path.write_text("# Test Document\n\nThis is test content.")

        vector = VectorIndex(embedding_model=shared_embedding_model)
        keyword = KeywordIndex()
        graph = GraphStore()
        manager = IndexManager(config_with_move_detection, vector, keyword, graph)

        manager.index_document(str(original_path))
        manager.persist()

        # Verify hash store has entries for the document
        old_chunks = manager._hash_store.get_chunks_by_document("original")
        assert len(old_chunks) > 0, "Original document should have stored hashes"

        # Verify indexed
        assert manager.get_document_count() == 1
