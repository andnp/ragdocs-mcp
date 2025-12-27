"""
Unit tests for Score Normalization (normalize_scores).

Tests the min-max score normalization function that converts raw RRF+recency
scores into the [0, 1] range for user-facing result presentation.
"""

import pytest

from src.search.fusion import normalize_scores


class TestNormalizeScores:
    """Tests for min-max score normalization."""

    def test_normalize_scores_min_max_scaling(self):
        """
        Verify correct min-max normalization formula application.

        Given a list of scores, the highest should become 1.0,
        the lowest should become 0.0, and intermediate values
        should be linearly interpolated.

        Formula: (score - min) / (max - min)
        """
        fused = [("doc1", 0.05), ("doc2", 0.03), ("doc3", 0.01)]
        normalized = normalize_scores(fused)

        # Highest score (0.05) becomes 1.0
        assert normalized[0][0] == "doc1"
        assert normalized[0][1] == pytest.approx(1.0)

        # Middle score (0.03) becomes 0.5
        # (0.03 - 0.01) / (0.05 - 0.01) = 0.02 / 0.04 = 0.5
        assert normalized[1][0] == "doc2"
        assert normalized[1][1] == pytest.approx(0.5)

        # Lowest score (0.01) becomes 0.0
        assert normalized[2][0] == "doc3"
        assert normalized[2][1] == pytest.approx(0.0)

    def test_normalize_scores_single_result(self):
        """
        Test that single result always gets perfect score of 1.0.

        Edge case: When there's only one result, there's no relative
        comparison possible, so it should be treated as a perfect match.
        """
        fused = [("doc1", 0.0391)]
        normalized = normalize_scores(fused)

        assert len(normalized) == 1
        assert normalized[0][0] == "doc1"
        assert normalized[0][1] == 1.0

        # Test with different score values
        fused_high = [("doc_high", 100.5)]
        normalized_high = normalize_scores(fused_high)
        assert normalized_high[0][1] == 1.0

        fused_low = [("doc_low", 0.001)]
        normalized_low = normalize_scores(fused_low)
        assert normalized_low[0][1] == 1.0

    def test_normalize_scores_empty_results(self):
        """
        Test that empty input returns empty output.

        Edge case: Graceful handling of no search results.
        Should return empty list without errors.
        """
        normalized = normalize_scores([])
        assert normalized == []

    def test_normalize_scores_identical_scores(self):
        """
        Test that identical scores all become 1.0.

        Edge case: When all scores are the same, there's no way to
        differentiate them, so all should be treated as perfect matches.
        This prevents division by zero: (max - min) = 0.
        """
        fused = [("doc1", 0.05), ("doc2", 0.05), ("doc3", 0.05)]
        normalized = normalize_scores(fused)

        assert len(normalized) == 3
        assert all(score == 1.0 for _, score in normalized)
        assert normalized[0] == ("doc1", 1.0)
        assert normalized[1] == ("doc2", 1.0)
        assert normalized[2] == ("doc3", 1.0)

        # Test with different identical values
        fused_zeros = [("a", 0.0), ("b", 0.0)]
        normalized_zeros = normalize_scores(fused_zeros)
        assert all(score == 1.0 for _, score in normalized_zeros)

    def test_normalize_scores_preserves_order(self):
        """
        Test that document order is preserved after normalization.

        The input is already sorted descending by RRF fusion.
        Normalization should maintain this order, not re-sort.
        """
        fused = [("doc1", 0.05), ("doc2", 0.03), ("doc3", 0.01)]
        normalized = normalize_scores(fused)

        doc_ids = [doc_id for doc_id, _ in normalized]
        assert doc_ids == ["doc1", "doc2", "doc3"]

        # Test with many documents
        fused_many = [
            ("a", 1.0),
            ("b", 0.8),
            ("c", 0.6),
            ("d", 0.4),
            ("e", 0.2),
        ]
        normalized_many = normalize_scores(fused_many)
        doc_ids_many = [doc_id for doc_id, _ in normalized_many]
        assert doc_ids_many == ["a", "b", "c", "d", "e"]

    def test_normalize_scores_range_invariant(self):
        """
        Test that all normalized scores fall within [0, 1] range.

        Regardless of input score magnitude, output must be bounded.
        Tests with various input ranges to verify invariant holds.
        """
        # Large scores
        fused_large = [("doc1", 100.5), ("doc2", 50.2), ("doc3", 1.1)]
        normalized_large = normalize_scores(fused_large)

        assert all(0.0 <= score <= 1.0 for _, score in normalized_large)
        assert normalized_large[0][1] == 1.0  # Highest becomes 1.0
        assert normalized_large[-1][1] == 0.0  # Lowest becomes 0.0

        # Small scores
        fused_small = [("doc1", 0.003), ("doc2", 0.002), ("doc3", 0.001)]
        normalized_small = normalize_scores(fused_small)

        assert all(0.0 <= score <= 1.0 for _, score in normalized_small)
        assert normalized_small[0][1] == 1.0
        assert normalized_small[-1][1] == 0.0

        # Mixed positive/negative (shouldn't happen in practice, but defensive)
        fused_mixed = [("doc1", 10.0), ("doc2", 0.0), ("doc3", -5.0)]
        normalized_mixed = normalize_scores(fused_mixed)

        assert all(0.0 <= score <= 1.0 for _, score in normalized_mixed)
        assert normalized_mixed[0][1] == 1.0
        assert normalized_mixed[-1][1] == 0.0

    def test_normalize_scores_linear_interpolation(self):
        """
        Test that intermediate scores are correctly interpolated.

        Verifies the linearity of the normalization: evenly spaced
        input scores should produce evenly spaced output scores.
        """
        # Evenly spaced input: 0.1, 0.2, 0.3
        fused = [("doc1", 0.3), ("doc2", 0.2), ("doc3", 0.1)]
        normalized = normalize_scores(fused)

        # Expected: doc1=1.0, doc2=0.5, doc3=0.0
        assert normalized[0][1] == pytest.approx(1.0)
        assert normalized[1][1] == pytest.approx(0.5)
        assert normalized[2][1] == pytest.approx(0.0)

        # Test with more gradations
        fused_many = [
            ("a", 1.0),
            ("b", 0.75),
            ("c", 0.5),
            ("d", 0.25),
            ("e", 0.0),
        ]
        normalized_many = normalize_scores(fused_many)

        # (1.0 - 0.0) / (1.0 - 0.0) = 1.0
        assert normalized_many[0][1] == pytest.approx(1.0)
        # (0.75 - 0.0) / (1.0 - 0.0) = 0.75
        assert normalized_many[1][1] == pytest.approx(0.75)
        # (0.5 - 0.0) / (1.0 - 0.0) = 0.5
        assert normalized_many[2][1] == pytest.approx(0.5)
        # (0.25 - 0.0) / (1.0 - 0.0) = 0.25
        assert normalized_many[3][1] == pytest.approx(0.25)
        # (0.0 - 0.0) / (1.0 - 0.0) = 0.0
        assert normalized_many[4][1] == pytest.approx(0.0)

    def test_normalize_scores_two_results(self):
        """
        Test normalization with exactly two results.

        Edge case between single result (always 1.0) and
        multiple results (full range [0, 1]). Should produce
        scores of 1.0 and 0.0.
        """
        fused = [("doc1", 0.05), ("doc2", 0.03)]
        normalized = normalize_scores(fused)

        assert len(normalized) == 2
        assert normalized[0] == ("doc1", 1.0)
        assert normalized[1] == ("doc2", 0.0)

        # Test with very close scores
        fused_close = [("high", 0.1001), ("low", 0.1000)]
        normalized_close = normalize_scores(fused_close)

        assert normalized_close[0][1] == 1.0
        assert normalized_close[1][1] == 0.0

    def test_normalize_scores_with_real_rrf_scores(self):
        """
        Test normalization with realistic RRF+recency scores.

        Integration test using score ranges typical of actual
        RRF fusion output (around 0.01-0.05 range).
        """
        # Typical RRF scores from fusion.py
        # RRF(0) with k=60 = 1/60 ≈ 0.0167
        # RRF(1) with k=60 = 1/61 ≈ 0.0164
        fused_realistic = [
            ("doc1", 0.0331),  # Appears in 2 strategies
            ("doc2", 0.0325),  # Appears in 2 strategies
            ("doc3", 0.0197),  # Single strategy, recency boost
            ("doc4", 0.0167),  # Single strategy, rank 0
            ("doc5", 0.0164),  # Single strategy, rank 1
        ]

        normalized = normalize_scores(fused_realistic)

        # Verify all scores in [0, 1]
        assert all(0.0 <= score <= 1.0 for _, score in normalized)

        # Highest score is 1.0
        assert normalized[0][1] == 1.0
        assert normalized[0][0] == "doc1"

        # Lowest score is 0.0
        assert normalized[-1][1] == 0.0
        assert normalized[-1][0] == "doc5"

        # Scores decrease monotonically
        scores = [score for _, score in normalized]
        assert scores == sorted(scores, reverse=True)

        # Verify approximate relative positions
        # doc2 should be very close to doc1 (0.0325 vs 0.0331)
        assert normalized[1][1] > 0.9  # Close to top

        # doc3 should be roughly in the middle
        assert 0.1 < normalized[2][1] < 0.9
