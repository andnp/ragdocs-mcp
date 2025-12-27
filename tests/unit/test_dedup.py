"""
Unit tests for deduplication functions.
"""

import numpy as np
import pytest

from src.search.dedup import cosine_similarity, deduplicate_by_similarity


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors_returns_one(self):
        """Identical vectors have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        """Orthogonal vectors have similarity 0.0."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors_returns_negative_one(self):
        """Opposite vectors have similarity -1.0."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(-1.0)

    def test_known_similarity_value(self):
        """Test with known similarity calculation."""
        # vec_a = [1, 2, 3], vec_b = [4, 5, 6]
        # dot = 4 + 10 + 18 = 32
        # norm_a = sqrt(14) = 3.7417
        # norm_b = sqrt(77) = 8.7750
        # similarity = 32 / (3.7417 * 8.7750) = 0.9746
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([4.0, 5.0, 6.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == pytest.approx(0.9746, abs=0.001)

    def test_zero_vector_returns_zero(self):
        """Zero vector returns similarity 0.0 (division by zero handled)."""
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0

    def test_both_zero_vectors_returns_zero(self):
        """Two zero vectors return similarity 0.0."""
        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0

    def test_high_dimensional_vectors(self):
        """Works with high-dimensional vectors (typical embedding size)."""
        np.random.seed(42)
        vec_a = np.random.randn(384)  # BAAI/bge-small-en-v1.5 dimension
        vec_b = np.random.randn(384)

        similarity = cosine_similarity(vec_a, vec_b)

        # Should be a valid similarity between -1 and 1
        assert -1.0 <= similarity <= 1.0


class TestDeduplicateBySimilarity:
    """Tests for deduplicate_by_similarity function."""

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list and 0 clusters merged."""
        def get_embedding(chunk_id: str):
            return None

        result, clusters_merged = deduplicate_by_similarity([], get_embedding)

        assert result == []
        assert clusters_merged == 0

    def test_single_item_returns_unchanged(self):
        """Single item returns unchanged, 0 clusters merged."""
        def get_embedding(chunk_id: str):
            return [0.1, 0.2, 0.3]

        results = [("chunk_a", 0.9)]
        deduped, clusters_merged = deduplicate_by_similarity(results, get_embedding)

        assert deduped == [("chunk_a", 0.9)]
        assert clusters_merged == 0

    def test_removes_duplicates_above_threshold(self):
        """Removes chunks with similarity above threshold."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.99, 0.1, 0.0],  # Very similar to chunk_a
            "chunk_c": [0.0, 1.0, 0.0],   # Different from both
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),  # Should be removed (similar to chunk_a)
            ("chunk_c", 0.7),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.9
        )

        # chunk_b removed as duplicate of chunk_a
        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_c", 0.7) in deduped
        assert ("chunk_b", 0.8) not in deduped
        assert clusters_merged == 1

    def test_keeps_highest_scored_representative(self):
        """First occurrence (highest score) is kept as representative."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [1.0, 0.0, 0.0],  # Identical to chunk_a
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        # Results already sorted by score (descending)
        results = [
            ("chunk_a", 0.9),  # Higher score, should be kept
            ("chunk_b", 0.7),  # Lower score, should be removed
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.85
        )

        assert deduped == [("chunk_a", 0.9)]
        assert clusters_merged == 1

    def test_handles_missing_embeddings_gracefully(self):
        """Chunks with missing embeddings are kept (not removed as duplicates)."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            # chunk_b has no embedding
            "chunk_c": [0.0, 1.0, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),  # No embedding, should be kept
            ("chunk_c", 0.7),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.85
        )

        # All chunks kept (no duplicates detected, chunk_b has no embedding)
        assert len(deduped) == 3
        assert clusters_merged == 0

    def test_threshold_one_keeps_all(self):
        """Threshold 1.0 only removes exact duplicates."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.99, 0.1, 0.0],  # Very similar but not identical
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=1.0
        )

        # Both kept (not exactly identical)
        assert len(deduped) == 2
        assert clusters_merged == 0

    def test_low_threshold_removes_most(self):
        """Low threshold removes more duplicates."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.8, 0.6, 0.0],   # Somewhat similar
            "chunk_c": [0.1, 0.9, 0.4],   # Different
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.5
        )

        # chunk_b removed (similarity with chunk_a > 0.5)
        assert ("chunk_a", 0.9) in deduped
        assert clusters_merged >= 1

    def test_multiple_clusters_merged(self):
        """Correctly counts multiple clusters being merged."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [1.0, 0.0, 0.0],   # Duplicate of a
            "chunk_c": [1.0, 0.0, 0.0],   # Duplicate of a
            "chunk_d": [0.0, 1.0, 0.0],   # Different
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
            ("chunk_d", 0.6),
        ]

        deduped, clusters_merged = deduplicate_by_similarity(
            results, get_embedding, similarity_threshold=0.99
        )

        assert len(deduped) == 2
        assert ("chunk_a", 0.9) in deduped
        assert ("chunk_d", 0.6) in deduped
        assert clusters_merged == 2  # chunk_b and chunk_c merged into chunk_a's cluster

    def test_default_threshold_is_085(self):
        """Default threshold is 0.85."""
        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.9, 0.44, 0.0],  # cosine ~0.9 > 0.85
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        results = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        # Use default threshold
        deduped, clusters_merged = deduplicate_by_similarity(results, get_embedding)

        # chunk_b should be removed at default threshold 0.85
        assert len(deduped) == 1
        assert clusters_merged == 1
