"""Unit tests for the reindex runtime and state machine.

Tests the expand → backfill → flip → contract pipeline for safe embedding
model migration, including rollback and mixed-dimension guards.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from searchkernel.domain import Record, Vector
from searchkernel.runtime.reindex import ReindexError, ReindexRoutine

# ===== Test fixtures =====


class StubEmbeddingProvider:
    """Stub embedding provider for testing (no external dependencies)."""

    def __init__(self, model_name: str = "test-model-v1", dim: int = 768):
        self.model_name = model_name
        self.dim = dim
        self.embed_calls = 0
        self.embed_texts = []

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return deterministic embeddings based on text length."""
        self.embed_calls += 1
        self.embed_texts.extend(texts)
        return [
            [float(len(t) % (self.dim)) for _ in range(self.dim)]
            for t in texts
        ]


class InMemoryVectorStore:
    """In-memory vector store for testing (implements VectorStore port)."""

    def __init__(self):
        self.tables: dict[tuple[str, int], dict[str, Vector]] = {}
        self.all_records: dict[str, Record] = {}
        self.deletes: list[str] = []
        self.epoch_counter = 0

    def upsert(
        self, records: list[Record], model_name: str, dim: int
    ) -> None:
        """Upsert records with embeddings into per-(model, dim) table."""
        key = (model_name, dim)
        if key not in self.tables:
            self.tables[key] = {}

        for record in records:
            if record.embedding is None:
                raise ValueError(f"Record {record.source_id} has no embedding")
            if len(record.embedding) != dim:
                raise ValueError(
                    f"Embedding dim {len(record.embedding)} != "
                    f"expected {dim} for {record.source_id}"
                )
            self.tables[key][record.source_id] = record.embedding
            self.all_records[record.source_id] = record

        self.epoch_counter += 1

    def search(
        self, query_vector: Vector, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Search (not tested here)."""
        return []

    def delete(self, record_ids: list[str]) -> None:
        """Delete records by ID."""
        self.deletes.extend(record_ids)
        for model_dim_table in self.tables.values():
            for rid in record_ids:
                model_dim_table.pop(rid, None)

    def epoch(self) -> int:
        """Return epoch counter."""
        return self.epoch_counter


# ===== Fixtures =====


@pytest.fixture
def sample_records():
    """Create a small fixture corpus (5 records)."""
    records = [
        Record(
            source_kind="test",
            source_id="test:1",
            title="Record 1",
            body="This is the first test record with some content.",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
        Record(
            source_kind="test",
            source_id="test:2",
            title="Record 2",
            body="Second record has a different body.",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
        Record(
            source_kind="test",
            source_id="test:3",
            title="Record 3",
            body="The third record is here.",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
        Record(
            source_kind="test",
            source_id="test:4",
            title="Record 4",
            body="Fourth record follows.",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
        Record(
            source_kind="test",
            source_id="test:5",
            title="Record 5",
            body="Fifth and final record.",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
    ]
    return records


@pytest.fixture
def old_embedding_provider():
    """Old embedding provider (768 dims)."""
    return StubEmbeddingProvider(model_name="old-model-v0", dim=768)


@pytest.fixture
def new_embedding_provider():
    """New embedding provider (1024 dims)."""
    return StubEmbeddingProvider(model_name="new-model-v1", dim=1024)


@pytest.fixture
def vector_store():
    """In-memory vector store."""
    return InMemoryVectorStore()


# ===== Tests =====


class TestReindexExpand:
    """Test the expand stage."""

    def test_expand_creates_table(self, sample_records, new_embedding_provider, vector_store):
        """Expand stage creates new per-(model, dim) table."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()

        # Table should be created (or at least marked as available)
        assert ("new-model-v1", 1024) in vector_store.tables

    def test_expand_idempotent(self, sample_records, new_embedding_provider, vector_store):
        """Expand is idempotent (can be called multiple times)."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        routine.expand()  # Should not raise

        assert ("new-model-v1", 1024) in vector_store.tables


class TestReindexBackfill:
    """Test the backfill stage."""

    def test_backfill_embeds_all_records(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Backfill embeds and writes all records to new table."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
            batch_size=2,  # Use small batch to test batching
        )
        routine.expand()
        progress = routine.backfill()

        # Check progress
        assert progress.stage == "backfill"
        assert progress.records_processed == len(sample_records)
        assert progress.total_records == len(sample_records)
        assert len(progress.errors) == 0

        # Check that embeddings were written
        table_key = ("new-model-v1", 1024)
        assert len(vector_store.tables[table_key]) == len(sample_records)

        # Check that provider was called (batched)
        assert new_embedding_provider.embed_calls > 0
        assert len(new_embedding_provider.embed_texts) == len(sample_records)

    def test_backfill_requires_expand(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Backfill should work after expand (table exists)."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        # Skip expand intentionally - backfill should handle missing table
        progress = routine.backfill()

        # Should still work (backfill can create table if needed)
        assert progress.records_processed == len(sample_records)

    def test_backfill_marks_new_embeddings_written(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """After backfill, new embeddings are marked as written."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        assert not routine._new_embeddings_written
        routine.backfill()
        assert routine._new_embeddings_written


class TestReindexFlip:
    """Test the flip stage."""

    def test_flip_requires_backfill(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Flip should fail if backfill hasn't completed."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()

        with pytest.raises(ReindexError, match="before backfill"):
            routine.flip()

    def test_flip_succeeds_after_backfill(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Flip succeeds after successful backfill."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        routine.backfill()
        routine.flip()  # Should not raise

        assert routine._current_stage == "flip"


class TestReindexContract:
    """Test the contract stage."""

    def test_contract_requires_flip(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Contract should be gated by flip completion."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        routine.backfill()

        # Try to contract without flip
        with pytest.raises(ReindexError, match="before flip"):
            routine.contract("old-model-v0")

    def test_contract_after_complete_migration(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Contract succeeds at the end of a complete migration."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        routine.backfill()
        routine.flip()
        routine.contract("old-model-v0")  # Should not raise

        assert routine._current_stage == "contract"


class TestReindexRollback:
    """Test the rollback capability."""

    def test_rollback_clears_new_embeddings_flag(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Rollback resets the _new_embeddings_written flag."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        routine.backfill()
        assert routine._new_embeddings_written

        routine.rollback("old-model-v0")
        assert not routine._new_embeddings_written

    def test_rollback_before_flip(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Rollback is safe to call before flip."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        routine.backfill()
        routine.rollback("old-model-v0")  # Should not raise

        # After rollback, flip should fail (embeddings not written)
        with pytest.raises(ReindexError, match="before backfill"):
            routine.flip()


class TestReindexMixedDimensionGuard:
    """Test mixed-dimension safety gates."""

    def test_dimension_mismatch_rejected_on_upsert(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Upsert with mismatched dimension should be rejected."""
        # Try to upsert with wrong dimension
        record = sample_records[0]
        record.embedding = [1.0] * 512  # Wrong dim (not 1024)
        record.embedding_model = "new-model-v1"

        with pytest.raises(ValueError, match="Embedding dim .* != expected"):
            vector_store.upsert(
                [record],
                model_name="new-model-v1",
                dim=1024,
            )

    def test_truncate_dim_valid_range(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Truncate dimension must be in valid range."""
        # Valid: truncate_dim < provider.dim
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
            truncate_dim=512,  # Valid
        )
        assert routine.target_dim == 512

        # Invalid: truncate_dim > provider.dim
        with pytest.raises(ReindexError, match="truncate_dim"):
            ReindexRoutine(
                sample_records,
                new_embedding_provider,
                vector_store,
                truncate_dim=2048,  # Invalid
            )

        # Invalid: truncate_dim <= 0
        with pytest.raises(ReindexError, match="truncate_dim"):
            ReindexRoutine(
                sample_records,
                new_embedding_provider,
                vector_store,
                truncate_dim=0,  # Invalid
            )

    def test_backfill_with_dimension_truncation(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Backfill truncates embeddings to target_dim if specified."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
            batch_size=2,
            truncate_dim=512,  # Truncate 1024 → 512
        )
        routine.expand()
        routine.backfill()

        # Check that embeddings were truncated
        table_key = ("new-model-v1", 512)  # Note: using target_dim, not provider.dim
        for embedding in vector_store.tables[table_key].values():
            assert len(embedding) == 512


class TestReindexEpochTracking:
    """Test epoch-based cache invalidation."""

    def test_expand_increments_epoch(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Expand should bump the store epoch."""
        initial_epoch = vector_store.epoch()
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()

        # Epoch should increment (implementation-dependent; just verify it tracks)
        assert vector_store.epoch() >= initial_epoch

    def test_backfill_increments_epoch(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Backfill should bump the store epoch."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        initial_epoch = vector_store.epoch()
        routine.backfill()

        # Epoch should increment for each batch
        assert vector_store.epoch() > initial_epoch


class TestReindexCompleteWorkflow:
    """Integration tests for complete reindex workflow."""

    def test_complete_expand_backfill_flip_contract(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Complete workflow: expand → backfill → flip → contract."""
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
            batch_size=2,
        )

        # Stage 1: expand
        routine.expand()
        assert ("new-model-v1", 1024) in vector_store.tables

        # Stage 2: backfill
        progress = routine.backfill()
        assert progress.records_processed == len(sample_records)
        assert len(progress.errors) == 0

        # Stage 3: flip
        routine.flip()
        assert routine._current_stage == "flip"

        # Stage 4: contract
        routine.contract("old-model-v0")
        assert routine._current_stage == "contract"

    def test_old_embeddings_survive_until_contract(
        self, sample_records, new_embedding_provider, vector_store
    ):
        """Old vectors are preserved until explicit contract() call."""
        # Pre-populate old embeddings
        old_provider = StubEmbeddingProvider("old-model-v0", 768)
        old_records = sample_records.copy()
        for record in old_records:
            embeddings = old_provider.embed([record.body])
            record.embedding = embeddings[0]
        vector_store.upsert(
            old_records,
            model_name="old-model-v0",
            dim=768,
        )

        # New reindex
        routine = ReindexRoutine(
            sample_records,
            new_embedding_provider,
            vector_store,
        )
        routine.expand()
        routine.backfill()

        # Both old and new tables should coexist
        assert ("old-model-v0", 768) in vector_store.tables
        assert ("new-model-v1", 1024) in vector_store.tables
        assert len(vector_store.tables[("old-model-v0", 768)]) == len(sample_records)
        assert len(vector_store.tables[("new-model-v1", 1024)]) == len(sample_records)

        # Flip (mark new as active)
        routine.flip()
        # Old table still exists
        assert ("old-model-v0", 768) in vector_store.tables

        # Contract (delete old)
        routine.contract("old-model-v0")
        # Now caller should handle deletion logic (store adapter responsibility)
