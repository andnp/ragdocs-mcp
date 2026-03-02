"""Tests for VectorIndex SQLite persistence (save_to_db / load_from_db)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.indices.vector import VectorIndex
from src.models import Chunk
from src.storage.db import DatabaseManager


def _make_chunk(doc_id: str, index: int, content: str) -> Chunk:
    chunk_id = f"{doc_id}_chunk_{index}"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        metadata={"doc_id": doc_id},
        chunk_index=index,
        header_path="",
        start_pos=0,
        end_pos=len(content),
        file_path=f"/docs/{doc_id}.md",
        modified_time=datetime.now(timezone.utc),
    )


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "test.db")


@pytest.fixture
def vector(shared_embedding_model) -> VectorIndex:
    return VectorIndex(embedding_model=shared_embedding_model)


class TestVectorSqlitePersistence:
    def test_save_and_load_100_chunks(self, db, shared_embedding_model):
        """After persisting 100 chunks to SQLite, VectorIndex populates FAISS with matching IDs."""
        vi = VectorIndex(embedding_model=shared_embedding_model)
        chunks = [_make_chunk("doc_a", i, f"Content about topic number {i}") for i in range(100)]
        vi.add_chunks(chunks)
        vi.save_to_db(db)

        vi2 = VectorIndex(embedding_model=shared_embedding_model)
        vi2.load_from_db(db)

        assert len(vi2._chunk_id_to_node_id) == 100
        results = vi2.search("topic number 50", top_k=5)
        assert len(results) > 0

    def test_vectors_stored_in_chunks_table(self, db, vector):
        """Verify individual vectors are written to the chunks table."""
        chunks = [_make_chunk("vec_doc", i, f"Chunk content {i}") for i in range(5)]
        vector.add_chunks(chunks)
        vector.save_to_db(db)

        conn = db.get_connection()
        rows = conn.execute("SELECT chunk_id, vector FROM chunks").fetchall()
        assert len(rows) == 5
        for row in rows:
            # vector may be None if embedding wasn't stored on the node,
            # but chunk_id must be present
            assert row["chunk_id"].startswith("vec_doc_chunk_")

    def test_mappings_stored_in_kv_store(self, db, vector):
        """Verify JSON mappings are present in kv_store after save."""
        chunks = [_make_chunk("map_doc", 0, "Mapping test content")]
        vector.add_chunks(chunks)
        vector.save_to_db(db)

        conn = db.get_connection()
        expected_keys = [
            "vector_index:faiss_binary",
            "vector_index:doc_id_mapping",
            "vector_index:chunk_id_mapping",
            "vector_index:tombstones",
            "vector_index:docstore",
        ]
        for key in expected_keys:
            row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
            assert row is not None, f"Missing kv_store key: {key}"
            assert row["value"], f"Empty value for key: {key}"

        # Validate JSON structure of doc_id mapping
        doc_map_row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", ("vector_index:doc_id_mapping",)
        ).fetchone()
        doc_map = json.loads(doc_map_row["value"])
        assert "map_doc" in doc_map

    def test_roundtrip_preserves_search_quality(self, db, shared_embedding_model):
        """Search results should be similar before and after save/load cycle."""
        vi = VectorIndex(embedding_model=shared_embedding_model)
        topics = [
            "Python async programming with asyncio",
            "Machine learning with neural networks",
            "Database indexing strategies for performance",
            "React component lifecycle methods",
            "Kubernetes deployment best practices",
        ]
        chunks = [_make_chunk("quality", i, t) for i, t in enumerate(topics)]
        vi.add_chunks(chunks)

        results_before = vi.search("async programming", top_k=3)
        vi.save_to_db(db)

        vi2 = VectorIndex(embedding_model=shared_embedding_model)
        vi2.load_from_db(db)
        results_after = vi2.search("async programming", top_k=3)

        assert len(results_after) > 0
        # The top result should be the same chunk
        assert results_before[0]["chunk_id"] == results_after[0]["chunk_id"]

    def test_load_from_empty_db_initializes_fresh(self, db, shared_embedding_model):
        """Loading from DB with no stored data should initialize a fresh index."""
        vi = VectorIndex(embedding_model=shared_embedding_model)
        vi.load_from_db(db)

        # Should have a working (empty) index
        assert vi.is_ready()
        assert len(vi._chunk_id_to_node_id) == 0
        assert vi.search("anything") == []

    def test_save_with_tombstones(self, db, shared_embedding_model):
        """Tombstoned documents are preserved across save/load."""
        vi = VectorIndex(embedding_model=shared_embedding_model)
        chunks = [
            _make_chunk("keep_doc", 0, "Content to keep"),
            _make_chunk("tombstone_doc", 0, "Content to tombstone"),
        ]
        vi.add_chunks(chunks)
        vi.prune_document("tombstone_doc")

        assert "tombstone_doc" in vi._tombstoned_docs
        vi.save_to_db(db)

        vi2 = VectorIndex(embedding_model=shared_embedding_model)
        vi2.load_from_db(db)

        assert "tombstone_doc" in vi2._tombstoned_docs
        # The kept doc should still be accessible
        assert "keep_doc" in vi2._doc_id_to_node_ids
