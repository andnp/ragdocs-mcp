"""
Unit tests for Score Calibration (normalize_scores).

Tests the sigmoid calibration function that converts raw RRF+recency
scores into absolute confidence scores [0, 1].
"""


from src.search.calibration import calibrate_results as normalize_scores


class TestNormalizeScores:
    """Tests for sigmoid score calibration."""

    def test_normalize_scores_calibration(self):
        """
        Verify correct sigmoid calibration.

        High scores map to high confidence, low scores to low confidence,
        and threshold score (0.04) maps to ~0.5.
        """
        fused = [("doc1", 0.06), ("doc2", 0.04), ("doc3", 0.01)]
        normalized = normalize_scores(fused)

        # Highest score (0.06) should have high confidence
        assert normalized[0][0] == "doc1"
        assert 0.90 < normalized[0][1] <= 1.0

        # Middle score (0.04) is at threshold, should be ~0.5
        assert normalized[1][0] == "doc2"
        assert 0.45 < normalized[1][1] < 0.55

        # Lowest score (0.01) is well below threshold, should have very low confidence
        assert normalized[2][0] == "doc3"
        assert 0.0 <= normalized[2][1] < 0.02

    def test_normalize_scores_single_result(self):
        """
        Test that single result gets calibrated based on absolute score.

        Calibration provides absolute confidence, not relative ranking.
        """
        fused = [("doc1", 0.06)]
        normalized = normalize_scores(fused)

        assert len(normalized) == 1
        assert normalized[0][0] == "doc1"
        # 0.06 is above threshold (0.04), so should be high confidence
        assert 0.90 < normalized[0][1] < 1.0

        # Test with high score
        fused_high = [("doc_high", 0.100)]
        normalized_high = normalize_scores(fused_high)
        assert normalized_high[0][1] > 0.99

        # Test with low score
        fused_low = [("doc_low", 0.001)]
        normalized_low = normalize_scores(fused_low)
        assert normalized_low[0][1] < 0.01

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
        Test that identical scores get identical calibrated values.

        All identical scores map to the same confidence level.
        """
        fused = [("doc1", 0.05), ("doc2", 0.05), ("doc3", 0.05)]
        normalized = normalize_scores(fused)

        assert len(normalized) == 3
        # All should have same high confidence
        first_score = normalized[0][1]
        assert all(abs(score - first_score) < 0.001 for _, score in normalized)
        assert 0.75 < first_score <= 1.0

        # Test with low identical values
        fused_low = [("a", 0.01), ("b", 0.01)]
        normalized_low = normalize_scores(fused_low)
        assert abs(normalized_low[0][1] - normalized_low[1][1]) < 0.001
        # 0.01 is well below threshold (0.04), should be very low
        assert normalized_low[0][1] < 0.02

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
        Test that all calibrated scores fall within [0, 1] range.

        Regardless of input score magnitude, output must be bounded.
        Tests with various input ranges to verify invariant holds.
        """
        # Large scores
        fused_large = [("doc1", 100.5), ("doc2", 50.2), ("doc3", 1.1)]
        normalized_large = normalize_scores(fused_large)

        assert all(0.0 <= score <= 1.0 for _, score in normalized_large)
        assert abs(normalized_large[0][1] - 1.0) < 0.01  # Very high score saturates at 1.0
        assert abs(normalized_large[-1][1] - 1.0) < 0.01  # Still well above threshold

        # Small scores
        fused_small = [("doc1", 0.003), ("doc2", 0.002), ("doc3", 0.001)]
        normalized_small = normalize_scores(fused_small)

        assert all(0.0 <= score <= 1.0 for _, score in normalized_small)
        # All very low (well below 0.035 threshold), should be very low confidence
        assert all(score < 0.02 for _, score in normalized_small)

        # Negative scores (defensive - shouldn't happen)
        fused_negative = [("doc1", 0.05), ("doc2", 0.0), ("doc3", -0.05)]
        normalized_negative = normalize_scores(fused_negative)

        assert all(0.0 <= score <= 1.0 for _, score in normalized_negative)
        assert abs(normalized_negative[2][1]) < 0.01  # Negative very close to 0

    def test_normalize_scores_monotonic_increasing(self):
        """
        Test that calibration preserves monotonic order.

        Higher input scores always produce higher output scores.
        """
        # Realistic RRF score range
        fused = [
            ("doc1", 0.08),
            ("doc2", 0.06),
            ("doc3", 0.04),
            ("doc4", 0.03),
            ("doc5", 0.02),
        ]
        normalized = normalize_scores(fused)

        # Verify monotonic decreasing (since sorted descending)
        scores = [score for _, score in normalized]
        assert scores == sorted(scores, reverse=True)

        # Verify each score is lower than previous
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_normalize_scores_two_results(self):
        """
        Test calibration with exactly two results.

        Both should be calibrated independently based on absolute values.
        """
        fused = [("doc1", 0.06), ("doc2", 0.04)]
        normalized = normalize_scores(fused)

        assert len(normalized) == 2
        # 0.06 should have high confidence
        assert 0.90 < normalized[0][1] <= 1.0
        # 0.04 is at threshold, should be ~0.5
        assert 0.45 < normalized[1][1] < 0.55

        # Test with very close scores - both get similar calibration
        fused_close = [("high", 0.0501), ("low", 0.0500)]
        normalized_close = normalize_scores(fused_close)

        # Both should be very close in calibrated value
        assert abs(normalized_close[0][1] - normalized_close[1][1]) < 0.01
        # Both should be high confidence
        assert normalized_close[0][1] > 0.75
        assert normalized_close[1][1] > 0.75

    def test_normalize_scores_with_real_rrf_scores(self):
        """
        Test calibration with realistic RRF+recency scores.

        Integration test using score ranges typical of actual
        RRF fusion output (around 0.01-0.08 range).
        """
        # Typical RRF scores from fusion.py
        # RRF(0) with k=60 = 1/60 ≈ 0.0167
        # RRF(1) with k=60 = 1/61 ≈ 0.0164
        fused_realistic = [
            ("doc1", 0.0831),  # High multi-strategy + boost
            ("doc2", 0.0667),  # High multi-strategy
            ("doc3", 0.0500),  # Above threshold
            ("doc4", 0.0400),  # At threshold
            ("doc5", 0.0250),  # Below threshold
            ("doc6", 0.0167),  # Well below threshold
        ]

        normalized = normalize_scores(fused_realistic)

        # Verify all scores in [0, 1]
        assert all(0.0 <= score <= 1.0 for _, score in normalized)

        # Scores decrease monotonically
        scores = [score for _, score in normalized]
        assert scores == sorted(scores, reverse=True)

        # Verify calibration ranges
        # doc1 (very high) should have very high confidence
        assert normalized[0][1] > 0.99

        # doc4 (0.04 = threshold) should be ~0.5
        assert 0.45 < normalized[3][1] < 0.55

        # doc6 (0.0167) is well below threshold, should be very low
        assert normalized[5][1] < 0.05
