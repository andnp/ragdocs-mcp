"""Integration tests for pgvector store adapter.

Tests VectorStore, KeywordStore, GraphStore, and CacheStore implementations
against a live Postgres database with pgvector extension.
"""

import os
from datetime import UTC, datetime

import numpy as np
import pytest

from searchkernel.adapters.stores.pgvector import (
    PGCacheStore,
    PGGraphStore,
    PGKeywordStore,
    PGVectorStore,
    PostgresConnection,
    _create_schema,
    _vector_table_name,
)
from searchkernel.domain import Record, RecordStatus, Vector
from tests.integration.conftest import pg_dsn_for_schema, pg_worker_schema


@pytest.fixture(scope="session")
def pg_dsn():
    """Get PostgreSQL DSN from environment."""
    dsn = os.environ.get("SEARCHKERNEL_PG_DSN")
    if not dsn:
        pytest.skip("SEARCHKERNEL_PG_DSN not set")
    return dsn


@pytest.fixture(scope="function")
def pg_conn(pg_dsn, request):
    """Create a test connection pool scoped to this xdist worker's own schema.

    Each xdist worker gets a private Postgres schema (pinned via search_path
    on the connection DSN), so this file's DELETE-everything cleanup below
    only ever touches this worker's own tables -- concurrent workers running
    the same file's tests can never collide, regardless of --dist mode.
    """
    schema = pg_worker_schema(request.config)
    scoped_dsn = pg_dsn_for_schema(pg_dsn, schema)

    bootstrap_pool = PostgresConnection(pg_dsn, min_connections=1, max_connections=1)
    bootstrap_conn = bootstrap_pool.get_connection()
    bootstrap_cursor = bootstrap_conn.cursor()
    bootstrap_cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    bootstrap_conn.commit()
    bootstrap_cursor.close()
    bootstrap_pool.put_connection(bootstrap_conn)
    bootstrap_pool.close()

    conn_pool = PostgresConnection(scoped_dsn)
    _create_schema(conn_pool)

    # Clean slate for this worker's schema before every test.
    conn = conn_pool.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT table_name FROM vector_tables;")
    for (table_name,) in cursor.fetchall():
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')

    cursor.execute("DELETE FROM vector_tables;")
    cursor.execute("DELETE FROM records;")
    cursor.execute("DELETE FROM graph_edges;")
    cursor.execute("DELETE FROM cache_store;")
    cursor.execute("UPDATE index_epoch SET epoch = 0;")
    conn.commit()
    cursor.close()
    conn_pool.put_connection(conn)

    yield conn_pool

    # Cleanup
    conn_pool.close()


@pytest.fixture
def fixture_records():
    """Create fixture records for testing."""
    now = datetime.now(UTC)
    return [
        Record(
            source_kind="test",
            source_id="test:1",
            title="Machine Learning Basics",
            body="Machine learning is a subset of AI. It enables systems to learn from data.",
            created_at=now,
            updated_at=now,
            metadata={"category": "ai"},
            uri="http://example.com/ml",
            status=RecordStatus.ACTIVE,
            embedding=[1.0, 0.0, 0.0, 0.0],
        ),
        Record(
            source_kind="test",
            source_id="test:2",
            title="Deep Learning Neural Networks",
            body="Neural networks are inspired by biological neurons. Deep learning uses many layers.",
            created_at=now,
            updated_at=now,
            metadata={"category": "ai"},
            uri="http://example.com/dl",
            status=RecordStatus.ACTIVE,
            embedding=[0.9, 0.1, 0.0, 0.0],
        ),
        Record(
            source_kind="test",
            source_id="test:3",
            title="Database Systems",
            body="Relational databases use SQL. PostgreSQL is a popular open-source database.",
            created_at=now,
            updated_at=now,
            metadata={"category": "database"},
            uri="http://example.com/db",
            status=RecordStatus.ACTIVE,
            embedding=[0.0, 0.0, 1.0, 0.0],
        ),
    ]


class TestVectorStore:
    """Tests for VectorStore port implementation."""

    def test_upsert_and_search_parity(self, pg_conn, fixture_records):
        """Test that pgvector ANN search returns same top-k as brute-force cosine.

        This is the key acceptance test: verify that pgvector HNSW gives
        same (or very similar) results as a reference numpy implementation.
        """
        store = PGVectorStore(pg_conn)

        # Upsert fixture records
        store.upsert(fixture_records, model_name="test-model", dim=4)

        # Query vector similar to record 1 and 2
        query_vec: Vector = [0.95, 0.05, 0.0, 0.0]

        # Get results from pgvector
        pgvector_results = store.search(query_vec, k=3, model_name="test-model", dim=4)

        # Compute brute-force reference using numpy
        embeddings = np.array([r.embedding for r in fixture_records])
        query_array = np.array(query_vec)

        # Cosine similarity: (A · B) / (||A|| ||B||)
        similarities = []
        for emb in embeddings:
            dot = np.dot(query_array, emb)
            norm_q = np.linalg.norm(query_array)
            norm_e = np.linalg.norm(emb)
            cosine_sim = dot / (norm_q * norm_e)
            similarities.append(cosine_sim)

        # Sort by similarity descending
        sorted_indices = np.argsort(similarities)[::-1]
        reference_ids = [fixture_records[i].source_id for i in sorted_indices[:3]]

        # Verify pgvector results match reference (order may differ slightly for ties)
        pgvector_ids = [r[0] for r in pgvector_results]
        assert len(pgvector_ids) == 3
        assert set(pgvector_ids) == set(reference_ids[:3])
        # First result should be closest
        assert pgvector_ids[0] in reference_ids[:2]

    def test_mixed_dimension_rejection(self, pg_conn, fixture_records):
        """Test that mixed-dimension writes are rejected."""
        store = PGVectorStore(pg_conn)

        # Upsert with dim=4
        store.upsert(fixture_records, model_name="model-v1", dim=4)

        # Try to upsert with dim=5 (should fail)
        bad_record = Record(
            source_kind="test",
            source_id="test:bad",
            title="Bad Embedding",
            body="This has wrong dimension",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0],  # 5 dims
        )

        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.upsert([bad_record], model_name="model-v1", dim=5)

    def test_delete_records(self, pg_conn, fixture_records):
        """Test delete operation."""
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="test-model", dim=4)

        # Delete first two records
        store.delete([fixture_records[0].source_id, fixture_records[1].source_id])

        # Search should only return third record
        query_vec = [0.0, 0.0, 1.0, 0.0]
        results = store.search(query_vec, k=10, model_name="test-model", dim=4)

        result_ids = [r[0] for r in results]
        assert fixture_records[0].source_id not in result_ids
        assert fixture_records[1].source_id not in result_ids
        assert fixture_records[2].source_id in result_ids

    def test_epoch_tracking(self, pg_conn, fixture_records):
        """Test that epoch is incremented on upsert/delete."""
        store = PGVectorStore(pg_conn)

        epoch0 = store.epoch()
        store.upsert(fixture_records, model_name="test-model", dim=4)
        epoch1 = store.epoch()

        assert epoch1 > epoch0

        store.delete([fixture_records[0].source_id])
        epoch2 = store.epoch()

        assert epoch2 > epoch1

    def test_per_model_isolation(self, pg_conn, fixture_records):
        """Test that different embedding models are isolated."""
        store = PGVectorStore(pg_conn)

        # Upsert same records with different models
        store.upsert(fixture_records, model_name="model-v1", dim=4)
        store.upsert(fixture_records, model_name="model-v2", dim=4)

        # Search in model-v1; should work
        results = store.search(
            [1.0, 0.0, 0.0, 0.0], k=3, model_name="model-v1", dim=4
        )
        assert len(results) == 3

    def test_hnsw_index_exists_per_model_table(self, pg_conn, fixture_records):
        """Test that an HNSW index is created on the per-model vector table."""
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="test-model", dim=4)

        table_name = _vector_table_name("test-model", 4)

        conn = pg_conn.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename = %s;",
                (table_name,),
            )
            indexdefs = [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            pg_conn.put_connection(conn)

        assert any("hnsw" in indexdef.lower() for indexdef in indexdefs), (
            f"Expected an HNSW index on {table_name}, found: {indexdefs}"
        )

    def test_ann_recall_at_10(self, pg_conn):
        """Test that HNSW ANN search achieves recall@10 >= 0.9 vs brute-force cosine.

        HNSW is approximate, so exact top-k equality is not guaranteed;
        this verifies the index returns a highly-overlapping result set
        with a numpy brute-force reference over a larger, randomized corpus.
        """
        rng = np.random.default_rng(42)
        n_records = 500
        dim = 1024

        raw_vectors = rng.normal(size=(n_records, dim))
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        vectors = raw_vectors / norms

        now = datetime.now(UTC)
        records = [
            Record(
                source_kind="test",
                source_id=f"recall:{i}",
                title=f"Recall fixture {i}",
                body="Randomized recall corpus entry.",
                created_at=now,
                updated_at=now,
                embedding=vectors[i].tolist(),
            )
            for i in range(n_records)
        ]

        store = PGVectorStore(pg_conn)
        store.upsert(records, model_name="recall-model", dim=dim)

        query_vec = rng.normal(size=dim)
        query_vec = query_vec / np.linalg.norm(query_vec)

        k = 10
        pgvector_results = store.search(
            query_vec.tolist(), k=k, model_name="recall-model", dim=dim
        )
        pgvector_ids = {r[0] for r in pgvector_results}

        # Brute-force cosine similarity reference (vectors are unit-norm).
        similarities = vectors @ query_vec
        top_k_indices = np.argsort(similarities)[::-1][:k]
        reference_ids = {f"recall:{i}" for i in top_k_indices}

        recall = len(pgvector_ids & reference_ids) / k
        assert recall >= 0.9, (
            f"recall@{k} = {recall} below threshold; "
            f"pgvector={pgvector_ids} reference={reference_ids}"
        )


class TestKeywordStore:
    """Tests for KeywordStore port implementation."""

    def test_keyword_search(self, pg_conn, fixture_records):
        """Test full-text search returns expected results."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        # First upsert records (populates records table)
        vector_store.upsert(fixture_records, model_name="test-model", dim=4)

        # Index for keyword search
        keyword_store.index(fixture_records)

        # Search for "machine learning" should return top result
        results = keyword_store.search("machine learning", k=3)
        assert len(results) > 0

        # First result should be about machine learning
        top_id = results[0][0]
        top_record = next(r for r in fixture_records if r.source_id == top_id)
        assert "machine" in top_record.body.lower()

    def test_keyword_search_with_filters(self, pg_conn, fixture_records):
        """Test keyword search with source_kind filter."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        vector_store.upsert(fixture_records, model_name="test-model", dim=4)
        keyword_store.index(fixture_records)

        # Search with filter for "database" category but only in ai source_kind
        results = keyword_store.search(
            "database",
            k=10,
            filters={"source_kinds": ["test"]},
        )

        # Should find the database record
        assert len(results) > 0
        assert results[0][0] == "test:3"


class TestGraphStore:
    """Tests for GraphStore port implementation."""

    def test_upsert_and_retrieve_edges(self, pg_conn):
        """Test edge upsert and neighbor retrieval."""
        store = PGGraphStore(pg_conn)

        edges = [
            ("test:1", "test:2", "related", 0.9),
            ("test:1", "test:3", "related", 0.5),
            ("test:2", "test:3", "derived_from", 0.7),
        ]

        store.upsert_edges(edges)

        # Get neighbors of test:1
        neighbors = store.neighbors("test:1")
        neighbor_ids = [n[0] for n in neighbors]

        assert "test:2" in neighbor_ids
        assert "test:3" in neighbor_ids

    def test_neighbors_with_edge_type_filter(self, pg_conn):
        """Test neighbor retrieval with edge type filter."""
        store = PGGraphStore(pg_conn)

        edges = [
            ("test:1", "test:2", "related", 0.9),
            ("test:1", "test:3", "derived_from", 0.5),
        ]

        store.upsert_edges(edges)

        # Get neighbors only of type "related"
        neighbors = store.neighbors("test:1", edge_types=["related"])
        neighbor_ids = [n[0] for n in neighbors]

        assert "test:2" in neighbor_ids
        assert "test:3" not in neighbor_ids


class TestCacheStore:
    """Tests for CacheStore port implementation."""

    def test_get_set(self, pg_conn):
        """Test basic cache get/set."""
        store = PGCacheStore(pg_conn)

        store.set("key1", {"data": "value1"}, epoch=0)

        result = store.get("key1")
        assert result == {"data": "value1"}

    def test_cache_miss(self, pg_conn):
        """Test cache miss returns None."""
        store = PGCacheStore(pg_conn)

        result = store.get("nonexistent")
        assert result is None

    def test_epoch_invalidation(self, pg_conn):
        """Test epoch-based invalidation."""
        store = PGCacheStore(pg_conn)

        # Set values with different epochs
        store.set("key1", {"data": "value1"}, epoch=0)
        store.set("key2", {"data": "value2"}, epoch=1)
        store.set("key3", {"data": "value3"}, epoch=2)

        # Both should exist
        assert store.get("key1") is not None
        assert store.get("key2") is not None
        assert store.get("key3") is not None

        # Invalidate epochs 0 and 1
        store.invalidate_epoch(1)

        # key1 and key2 should be gone; key3 should remain
        assert store.get("key1") is None
        assert store.get("key2") is None
        assert store.get("key3") is not None

    def test_cache_update(self, pg_conn):
        """Test that set overwrites existing values."""
        store = PGCacheStore(pg_conn)

        store.set("key1", {"data": "value1"}, epoch=0)
        store.set("key1", {"data": "value2"}, epoch=1)

        result = store.get("key1")
        assert result == {"data": "value2"}


class TestRoundTrip:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, pg_conn, fixture_records):
        """Test a complete upsert -> search -> delete workflow."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)
        cache_store = PGCacheStore(pg_conn)

        # Upsert vectors
        vector_store.upsert(fixture_records, model_name="test-model", dim=4)
        keyword_store.index(fixture_records)

        epoch_before = vector_store.epoch()

        # Cache a result
        cache_store.set("search:1", {"results": ["test:1"]}, epoch=epoch_before)

        # Search
        results = vector_store.search(
            [1.0, 0.0, 0.0, 0.0], k=2, model_name="test-model", dim=4
        )
        assert len(results) > 0

        # Delete a record
        vector_store.delete([fixture_records[0].source_id])

        epoch_after = vector_store.epoch()
        assert epoch_after > epoch_before

        # Cache should still be there (epoch hasn't changed for it)
        cached = cache_store.get("search:1")
        assert cached is not None

        # After invalidation, cache should be gone
        cache_store.invalidate_epoch(epoch_before)
        cached = cache_store.get("search:1")
        assert cached is None
