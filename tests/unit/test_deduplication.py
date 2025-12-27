"""
Unit tests for semantic deduplication in context compression.

Tests cover:
- Similarity matrix computation (shape, values, symmetry)
- Clustering by similarity threshold
- Representative selection (highest score per cluster)
- End-to-end deduplication pipeline

Test strategies:
- Use numpy to create embeddings with known similarity relationships
- Test boundary conditions for similarity thresholds
- Verify cluster formation and representative selection logic
"""

import numpy as np
import pytest

from src.compression.deduplication import (
    DeduplicationResult,
    cluster_by_similarity,
    compute_similarity_matrix,
    deduplicate_results,
    select_representatives,
)
from src.models import ChunkResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def identical_embeddings() -> np.ndarray:
    """
    Create embeddings that are all identical.

    All vectors are [1, 0, 0], resulting in similarity = 1.0 between all pairs.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])


@pytest.fixture
def orthogonal_embeddings() -> np.ndarray:
    """
    Create embeddings that are orthogonal (completely distinct).

    Unit vectors along x, y, z axes have similarity = 0.0 between different axes.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def mixed_similarity_embeddings() -> np.ndarray:
    """
    Create embeddings with mixed similarity relationships.

    - chunk_0 and chunk_1: very similar (cosine ~0.99)
    - chunk_2: distinct from both
    - chunk_3 and chunk_4: moderately similar (cosine ~0.7)
    """
    return np.array([
        [1.0, 0.0, 0.0],       # chunk_0
        [0.99, 0.14, 0.0],     # chunk_1: similar to chunk_0
        [0.0, 1.0, 0.0],       # chunk_2: orthogonal to chunk_0/1
        [0.5, 0.5, 0.707],     # chunk_3
        [0.5, 0.5, 0.6],       # chunk_4: similar to chunk_3
    ])


@pytest.fixture
def chunk_results_five() -> list[ChunkResult]:
    """
    Create five ChunkResult objects with varying scores.

    Scores: 0.9, 0.8, 0.7, 0.6, 0.5 (descending order by chunk index)
    """
    return [
        ChunkResult(
            chunk_id=f"chunk_{i}",
            doc_id="doc",
            score=0.9 - (i * 0.1),
            header_path=f"# Section {i}",
            file_path=f"/docs/file_{i}.md",
            content=f"Content for chunk {i}",
        )
        for i in range(5)
    ]


# ============================================================================
# Similarity Matrix Tests
# ============================================================================


class TestComputeSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_similarity_matrix_shape(
        self,
        mixed_similarity_embeddings: np.ndarray,
    ) -> None:
        """
        Verifies similarity matrix has correct shape (n x n).

        For n embeddings, should produce an n x n symmetric matrix.
        """
        matrix = compute_similarity_matrix(mixed_similarity_embeddings)

        assert matrix.shape == (5, 5)

    def test_similarity_matrix_diagonal_is_one(
        self,
        mixed_similarity_embeddings: np.ndarray,
    ) -> None:
        """
        Verifies diagonal elements are 1.0 (self-similarity).

        Each vector has cosine similarity of 1.0 with itself.
        """
        matrix = compute_similarity_matrix(mixed_similarity_embeddings)

        diagonal = np.diag(matrix)
        np.testing.assert_array_almost_equal(diagonal, np.ones(5))

    def test_similarity_matrix_symmetry(
        self,
        mixed_similarity_embeddings: np.ndarray,
    ) -> None:
        """
        Verifies similarity matrix is symmetric.

        sim(a, b) should equal sim(b, a) for all pairs.
        """
        matrix = compute_similarity_matrix(mixed_similarity_embeddings)

        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_identical_embeddings_all_ones(
        self,
        identical_embeddings: np.ndarray,
    ) -> None:
        """
        Tests that identical embeddings produce all-ones matrix.

        When all vectors are the same, all pairwise similarities should be 1.0.
        """
        matrix = compute_similarity_matrix(identical_embeddings)

        expected = np.ones((3, 3))
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_orthogonal_embeddings_identity_like(
        self,
        orthogonal_embeddings: np.ndarray,
    ) -> None:
        """
        Tests that orthogonal embeddings produce identity-like matrix.

        Orthogonal unit vectors should have similarity 0.0 with each other,
        resulting in an identity matrix.
        """
        matrix = compute_similarity_matrix(orthogonal_embeddings)

        expected = np.eye(3)
        np.testing.assert_array_almost_equal(matrix, expected, decimal=5)

    def test_similarity_values_in_range(
        self,
        mixed_similarity_embeddings: np.ndarray,
    ) -> None:
        """
        Verifies all similarity values are in [-1, 1] range.

        Cosine similarity should always be bounded by -1 and 1.
        Uses small tolerance for floating-point precision.
        """
        matrix = compute_similarity_matrix(mixed_similarity_embeddings)

        # Allow small floating-point tolerance
        assert np.all(matrix >= -1.0 - 1e-6)
        assert np.all(matrix <= 1.0 + 1e-6)

    def test_handles_zero_vectors(self) -> None:
        """
        Tests graceful handling of zero vectors.

        Zero vectors should not cause division by zero errors.
        """
        embeddings = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        # Should not raise an exception
        matrix = compute_similarity_matrix(embeddings)

        assert matrix.shape == (2, 2)


# ============================================================================
# Clustering Tests
# ============================================================================


class TestClusterBySimilarity:
    """Tests for similarity-based clustering."""

    def test_all_identical_single_cluster(
        self,
        identical_embeddings: np.ndarray,
    ) -> None:
        """
        Tests that identical embeddings form a single cluster.

        All vectors with similarity 1.0 should be grouped together.
        """
        matrix = compute_similarity_matrix(identical_embeddings)
        clusters = cluster_by_similarity(matrix, threshold=0.85)

        assert len(clusters) == 1
        assert clusters[0] == {0, 1, 2}

    def test_all_orthogonal_separate_clusters(
        self,
        orthogonal_embeddings: np.ndarray,
    ) -> None:
        """
        Tests that orthogonal embeddings form separate clusters.

        Each distinct vector should be its own cluster.
        """
        matrix = compute_similarity_matrix(orthogonal_embeddings)
        clusters = cluster_by_similarity(matrix, threshold=0.85)

        assert len(clusters) == 3
        cluster_sizes = {len(c) for c in clusters}
        assert cluster_sizes == {1}

    def test_mixed_similarity_clustering(
        self,
        mixed_similarity_embeddings: np.ndarray,
    ) -> None:
        """
        Tests clustering with mixed similarity relationships.

        Similar chunks should cluster together, distinct ones separate.
        With threshold 0.85:
        - chunk_0 and chunk_1 should cluster (sim ~0.99)
        - chunk_2 should be alone
        - chunk_3 and chunk_4 may or may not cluster depending on their similarity
        """
        matrix = compute_similarity_matrix(mixed_similarity_embeddings)
        clusters = cluster_by_similarity(matrix, threshold=0.85)

        # chunk_0 and chunk_1 should be in the same cluster
        cluster_with_0 = next(c for c in clusters if 0 in c)
        assert 1 in cluster_with_0

        # chunk_2 should be in its own cluster
        cluster_with_2 = next(c for c in clusters if 2 in c)
        assert cluster_with_2 == {2}

    def test_threshold_affects_clustering(self) -> None:
        """
        Tests that different thresholds produce different clusters.

        Lower threshold should merge more items into fewer clusters.
        """
        # Two vectors with similarity ~0.7
        embeddings = np.array([
            [1.0, 0.0],
            [0.7, 0.714],  # cosine similarity with [1,0] is ~0.7
        ])
        matrix = compute_similarity_matrix(embeddings)

        # High threshold: separate clusters
        clusters_high = cluster_by_similarity(matrix, threshold=0.9)
        assert len(clusters_high) == 2

        # Low threshold: merged into one cluster
        clusters_low = cluster_by_similarity(matrix, threshold=0.5)
        assert len(clusters_low) == 1

    def test_exact_threshold_boundary(self) -> None:
        """
        Tests clustering when similarity exactly equals threshold.

        Items with similarity exactly at threshold should be clustered together.
        """
        # Create vectors with known similarity
        embeddings = np.array([
            [1.0, 0.0],
            [0.8, 0.6],  # cosine with [1,0] = 0.8
        ])
        matrix = compute_similarity_matrix(embeddings)

        # Threshold exactly 0.8 should cluster them
        clusters = cluster_by_similarity(matrix, threshold=0.8)
        assert len(clusters) == 1

        # Threshold just above 0.8 should keep them separate
        clusters_higher = cluster_by_similarity(matrix, threshold=0.81)
        assert len(clusters_higher) == 2

    def test_single_item_clustering(self) -> None:
        """
        Tests clustering with a single item.

        Should return a single cluster containing the one item.
        """
        matrix = np.array([[1.0]])
        clusters = cluster_by_similarity(matrix, threshold=0.85)

        assert len(clusters) == 1
        assert clusters[0] == {0}


# ============================================================================
# Representative Selection Tests
# ============================================================================


class TestSelectRepresentatives:
    """Tests for selecting cluster representatives."""

    def test_selects_highest_score_per_cluster(
        self,
        chunk_results_five: list[ChunkResult],
    ) -> None:
        """
        Verifies highest-scoring chunk is selected from each cluster.

        For each cluster, the representative should be the chunk
        with the maximum score within that cluster.
        """
        # Cluster {0, 1} has scores 0.9 and 0.8 -> select chunk_0
        # Cluster {2} has score 0.7 -> select chunk_2
        clusters = [{0, 1}, {2}]

        representatives = select_representatives(chunk_results_five, clusters)

        rep_ids = {r.chunk_id for r in representatives}
        assert rep_ids == {"chunk_0", "chunk_2"}

    def test_representatives_sorted_by_score(
        self,
        chunk_results_five: list[ChunkResult],
    ) -> None:
        """
        Verifies representatives are sorted by score in descending order.

        Final list should have highest scores first.
        """
        clusters = [{2}, {0, 1}, {3, 4}]

        representatives = select_representatives(chunk_results_five, clusters)

        scores = [r.score for r in representatives]
        assert scores == sorted(scores, reverse=True)

    def test_single_item_clusters(
        self,
        chunk_results_five: list[ChunkResult],
    ) -> None:
        """
        Tests selection when all clusters are singletons.

        Should return all chunks, sorted by score.
        """
        clusters = [{0}, {1}, {2}, {3}, {4}]

        representatives = select_representatives(chunk_results_five, clusters)

        assert len(representatives) == 5
        # Should be sorted by score descending
        assert representatives[0].chunk_id == "chunk_0"
        assert representatives[4].chunk_id == "chunk_4"

    def test_all_in_one_cluster(
        self,
        chunk_results_five: list[ChunkResult],
    ) -> None:
        """
        Tests selection when all items are in a single cluster.

        Should return only the highest-scoring item.
        """
        clusters = [{0, 1, 2, 3, 4}]

        representatives = select_representatives(chunk_results_five, clusters)

        assert len(representatives) == 1
        assert representatives[0].chunk_id == "chunk_0"
        assert representatives[0].score == 0.9


# ============================================================================
# End-to-End Deduplication Tests
# ============================================================================


class TestDeduplicateResults:
    """Tests for complete deduplication pipeline."""

    def test_no_duplicates_all_returned(
        self,
        chunk_results_five: list[ChunkResult],
        orthogonal_embeddings: np.ndarray,
    ) -> None:
        """
        Tests that distinct chunks are all returned.

        When no chunks are similar enough to deduplicate,
        all original chunks should be returned.
        """
        # Use only first 3 chunks to match orthogonal embeddings
        results = chunk_results_five[:3]

        dedup_result = deduplicate_results(
            results,
            orthogonal_embeddings,
            similarity_threshold=0.85,
        )

        assert len(dedup_result.results) == 3
        assert dedup_result.original_count == 3
        assert dedup_result.clusters_merged == 0

    def test_all_duplicates_one_returned(
        self,
        chunk_results_five: list[ChunkResult],
        identical_embeddings: np.ndarray,
    ) -> None:
        """
        Tests that identical chunks collapse to one.

        When all chunks are identical, only the highest-scoring
        representative should be returned.
        """
        # Use only first 3 chunks to match identical embeddings
        results = chunk_results_five[:3]

        dedup_result = deduplicate_results(
            results,
            identical_embeddings,
            similarity_threshold=0.85,
        )

        assert len(dedup_result.results) == 1
        assert dedup_result.original_count == 3
        assert dedup_result.clusters_merged == 2
        assert dedup_result.results[0].chunk_id == "chunk_0"  # highest score

    def test_partial_duplicates(
        self,
        chunk_results_five: list[ChunkResult],
        mixed_similarity_embeddings: np.ndarray,
    ) -> None:
        """
        Tests deduplication with mixed similarity relationships.

        Should correctly identify and merge similar chunks while
        preserving distinct ones.
        """
        dedup_result = deduplicate_results(
            chunk_results_five,
            mixed_similarity_embeddings,
            similarity_threshold=0.85,
        )

        # Expect fewer results due to merging similar chunks
        assert len(dedup_result.results) < 5
        assert dedup_result.clusters_merged > 0

        # Original count should be preserved
        assert dedup_result.original_count == 5

    def test_single_result_passthrough(self) -> None:
        """
        Tests that single result is returned unchanged.

        Edge case: deduplication of one item should be a no-op.
        """
        single_result = [
            ChunkResult(
                chunk_id="single",
                doc_id="doc",
                score=0.9,
                header_path="# Single",
                file_path="/docs/single.md",
                content="Single chunk",
            )
        ]
        single_embedding = np.array([[1.0, 0.0, 0.0]])

        dedup_result = deduplicate_results(
            single_result,
            single_embedding,
            similarity_threshold=0.85,
        )

        assert len(dedup_result.results) == 1
        assert dedup_result.original_count == 1
        assert dedup_result.clusters_merged == 0
        assert dedup_result.results[0].chunk_id == "single"

    def test_empty_results_passthrough(self) -> None:
        """
        Tests that empty input returns empty result.

        Edge case: no results should produce empty deduplication result.
        """
        empty_results: list[ChunkResult] = []
        empty_embeddings = np.array([]).reshape(0, 3)

        dedup_result = deduplicate_results(
            empty_results,
            empty_embeddings,
            similarity_threshold=0.85,
        )

        assert len(dedup_result.results) == 0
        assert dedup_result.original_count == 0
        assert dedup_result.clusters_merged == 0

    def test_threshold_affects_deduplication(
        self,
        chunk_results_five: list[ChunkResult],
    ) -> None:
        """
        Tests that threshold parameter affects deduplication aggressiveness.

        Lower threshold should merge more results, higher threshold fewer.
        """
        # Create embeddings with moderate similarity
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],   # sim ~0.8 with chunk_0
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.6],   # sim ~0.8 with chunk_2
            [0.0, 0.0, 1.0],
        ])

        # High threshold: less merging
        result_high = deduplicate_results(
            chunk_results_five,
            embeddings,
            similarity_threshold=0.9,
        )

        # Low threshold: more merging
        result_low = deduplicate_results(
            chunk_results_five,
            embeddings,
            similarity_threshold=0.7,
        )

        assert len(result_low.results) <= len(result_high.results)

    def test_returns_deduplication_result_type(
        self,
        chunk_results_five: list[ChunkResult],
        orthogonal_embeddings: np.ndarray,
    ) -> None:
        """
        Verifies return type is DeduplicationResult dataclass.

        The result should have all expected attributes.
        """
        results = chunk_results_five[:3]

        dedup_result = deduplicate_results(
            results,
            orthogonal_embeddings,
            similarity_threshold=0.85,
        )

        assert isinstance(dedup_result, DeduplicationResult)
        assert hasattr(dedup_result, "results")
        assert hasattr(dedup_result, "original_count")
        assert hasattr(dedup_result, "clusters_merged")


# ============================================================================
# Integration Tests with Realistic Data
# ============================================================================


class TestDeduplicationRealistic:
    """Tests with more realistic embedding patterns."""

    def test_two_clusters_of_three_plus_singletons(self) -> None:
        """
        Tests scenario: 2 clusters of 3 similar items + 2 singletons.

        Expected: 4 results returned (one rep per cluster + 2 singletons).
        """
        # Create results with scores that make representative selection predictable
        results = [
            # Cluster 1: chunks 0, 1, 2 (very similar embeddings)
            ChunkResult(
                chunk_id="c0", doc_id="doc", score=0.95,
                header_path="", file_path="", content="Similar A1",
            ),
            ChunkResult(
                chunk_id="c1", doc_id="doc", score=0.85,
                header_path="", file_path="", content="Similar A2",
            ),
            ChunkResult(
                chunk_id="c2", doc_id="doc", score=0.75,
                header_path="", file_path="", content="Similar A3",
            ),
            # Cluster 2: chunks 3, 4, 5 (very similar embeddings, different direction)
            ChunkResult(
                chunk_id="c3", doc_id="doc", score=0.90,
                header_path="", file_path="", content="Similar B1",
            ),
            ChunkResult(
                chunk_id="c4", doc_id="doc", score=0.80,
                header_path="", file_path="", content="Similar B2",
            ),
            ChunkResult(
                chunk_id="c5", doc_id="doc", score=0.70,
                header_path="", file_path="", content="Similar B3",
            ),
            # Singletons: chunks 6, 7 (orthogonal to everything)
            ChunkResult(
                chunk_id="c6", doc_id="doc", score=0.60,
                header_path="", file_path="", content="Distinct C",
            ),
            ChunkResult(
                chunk_id="c7", doc_id="doc", score=0.50,
                header_path="", file_path="", content="Distinct D",
            ),
        ]

        # Embeddings: clusters along x and y axes, singletons along z
        embeddings = np.array([
            # Cluster 1 (along x-axis)
            [1.0, 0.0, 0.0],
            [0.99, 0.14, 0.0],
            [0.98, 0.2, 0.0],
            # Cluster 2 (along y-axis)
            [0.0, 1.0, 0.0],
            [0.14, 0.99, 0.0],
            [0.2, 0.98, 0.0],
            # Singletons (unique directions)
            [0.0, 0.0, 1.0],
            [0.57, 0.57, 0.57],  # diagonal, distinct from all clusters
        ])

        dedup_result = deduplicate_results(results, embeddings, similarity_threshold=0.9)

        # Should have 4 results: 1 rep from each of 2 clusters + 2 singletons
        assert len(dedup_result.results) == 4
        assert dedup_result.clusters_merged == 4  # 8 - 4 = 4 merged

        # Check that highest-scoring reps are selected
        rep_ids = {r.chunk_id for r in dedup_result.results}
        assert "c0" in rep_ids  # highest in cluster 1
        assert "c3" in rep_ids  # highest in cluster 2
        assert "c6" in rep_ids  # singleton
        assert "c7" in rep_ids  # singleton
