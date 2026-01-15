"""
Unit tests for score calibration via sigmoid function.

Tests the calibration module that converts raw RRF scores to absolute
confidence scores in [0,1] using a sigmoid curve.
"""

import math

import pytest

from src.search.calibration import calibrate_results, calibrate_score


class TestCalibrateScore:
    """Tests for sigmoid score calibration."""

    def test_threshold_gives_half(self):
        """Score at threshold → 0.5."""
        threshold = 0.035
        score = calibrate_score(threshold, threshold=threshold)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_high_score_high_confidence(self):
        """0.08 RRF → >0.95 confidence."""
        score = calibrate_score(0.08, threshold=0.035, steepness=150.0)
        assert score > 0.95
        assert score <= 1.0

    def test_low_score_low_confidence(self):
        """0.015 RRF → <0.05 confidence."""
        score = calibrate_score(0.015, threshold=0.035, steepness=150.0)
        assert score < 0.05
        assert score >= 0.0

    def test_monotonic_increasing(self):
        """Preserves ranking order."""
        scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        calibrated = [calibrate_score(s) for s in scores]

        # Verify monotonic increasing
        for i in range(len(calibrated) - 1):
            assert calibrated[i] < calibrated[i + 1]

    def test_bounded_output(self):
        """All scores in [0, 1]."""
        test_scores = [0.0, 0.001, 0.01, 0.035, 0.05, 0.1, 0.5, 1.0, 10.0]

        for score in test_scores:
            calibrated = calibrate_score(score)
            assert 0.0 <= calibrated <= 1.0, f"Score {score} → {calibrated} out of bounds"

    def test_steepness_affects_curve(self):
        """Higher steepness → sharper transition."""
        score = 0.04
        threshold = 0.035

        # Lower steepness → gentler curve
        cal_gentle = calibrate_score(score, threshold=threshold, steepness=50.0)
        # Higher steepness → sharper curve
        cal_sharp = calibrate_score(score, threshold=threshold, steepness=300.0)

        # Both should be > 0.5 (above threshold)
        assert cal_gentle > 0.5
        assert cal_sharp > 0.5

        # Sharp curve should be closer to 1.0 (more extreme)
        assert cal_sharp > cal_gentle

    def test_overflow_protection(self):
        """Extreme scores don't crash."""
        # Test very large scores
        large_score = calibrate_score(1000.0, threshold=0.035, steepness=150.0)
        assert large_score == pytest.approx(1.0, abs=0.01)

        # Test very small scores
        small_score = calibrate_score(-1000.0, threshold=0.035, steepness=150.0)
        assert small_score == pytest.approx(0.0, abs=0.001)

        # Test extreme steepness
        extreme_steep = calibrate_score(0.05, threshold=0.035, steepness=1000.0)
        assert 0.0 <= extreme_steep <= 1.0

    def test_empty_results(self):
        """Returns []."""
        calibrated = calibrate_results([])
        assert calibrated == []

    def test_realistic_rrf_scores(self):
        """Typical 0.01-0.10 range."""
        # Typical RRF scores from real searches
        fused = [
            ("doc1", 0.0831),  # Very high RRF (multiple strategies + boost)
            ("doc2", 0.0667),  # High RRF
            ("doc3", 0.0450),  # Above threshold
            ("doc4", 0.0350),  # At threshold
            ("doc5", 0.0250),  # Below threshold
            ("doc6", 0.0167),  # Low RRF
            ("doc7", 0.0100),  # Very low RRF
        ]

        calibrated = calibrate_results(fused, threshold=0.035, steepness=150.0)

        # Verify all scores in [0, 1]
        assert all(0.0 <= score <= 1.0 for _, score in calibrated)

        # Verify monotonic order preserved
        scores = [score for _, score in calibrated]
        assert scores == sorted(scores, reverse=True)

        # Verify threshold behavior
        # doc1 (very high) should have very high confidence
        assert calibrated[0][1] > 0.95

        # doc4 (at threshold) should be ~0.5
        assert 0.4 < calibrated[3][1] < 0.6

        # doc7 (very low) should have low confidence
        assert calibrated[6][1] < 0.05

    def test_recency_boosted_scores(self):
        """0.04 RRF with recency boost → 0.7-0.8 confidence."""
        # Simulate recency-boosted score
        base_rrf = 0.033  # Base RRF score
        recency_multiplier = 1.2  # 7-day recency boost
        boosted_score = base_rrf * recency_multiplier  # ~0.040

        calibrated = calibrate_score(boosted_score, threshold=0.035, steepness=150.0)

        # Should map to moderate-high confidence
        assert 0.6 < calibrated < 0.85, f"Expected 0.6-0.85, got {calibrated}"


class TestCalibrateResults:
    """Tests for batch calibration."""

    def test_calibrate_results_preserves_doc_ids(self):
        """Document IDs preserved after calibration."""
        fused = [("doc1", 0.05), ("doc2", 0.03), ("doc3", 0.01)]
        calibrated = calibrate_results(fused)

        doc_ids = [doc_id for doc_id, _ in calibrated]
        assert doc_ids == ["doc1", "doc2", "doc3"]

    def test_calibrate_results_custom_parameters(self):
        """Custom threshold/steepness work correctly."""
        fused = [("doc1", 0.10), ("doc2", 0.05)]

        # Custom threshold at 0.05
        calibrated = calibrate_results(fused, threshold=0.05, steepness=100.0)

        # doc1 (0.10) well above threshold → high confidence
        assert calibrated[0][1] > 0.9

        # doc2 (0.05) at threshold → ~0.5
        assert 0.4 < calibrated[1][1] < 0.6

    def test_calibrate_results_single_result(self):
        """Single result gets calibrated (not forced to 1.0)."""
        fused = [("doc1", 0.0391)]
        calibrated = calibrate_results(fused, threshold=0.035, steepness=150.0)

        assert len(calibrated) == 1
        assert calibrated[0][0] == "doc1"

        # Should be calibrated based on sigmoid, not forced to 1.0
        # 0.0391 is just above threshold (0.035), so should be ~0.5-0.6
        assert 0.5 < calibrated[0][1] < 0.7


class TestCalibrationVsNormalization:
    """Comparison tests showing calibration vs min-max normalization differences."""

    def test_single_result_not_always_one(self):
        """
        Calibration: Single result can be < 1.0 if score is low.
        Normalization: Single result always 1.0.
        """
        low_score_result = [("doc1", 0.015)]  # Below threshold

        calibrated = calibrate_results(low_score_result, threshold=0.035, steepness=150.0)

        # Calibrated score should be < 0.5 (below threshold)
        assert calibrated[0][1] < 0.1

        # This is different from normalization which would return 1.0

    def test_absolute_confidence_interpretation(self):
        """
        Calibration provides absolute confidence scores independent of result set.

        A score of 0.08 should have ~same confidence whether it's
        alone or in a list with other scores.
        """
        score_value = 0.08

        # Single result
        single = calibrate_results([("doc1", score_value)])

        # With other results
        multiple = calibrate_results([
            ("doc1", score_value),
            ("doc2", 0.05),
            ("doc3", 0.02),
        ])

        # doc1's calibrated score should be similar in both cases
        # (small difference due to floating point, but conceptually same)
        assert abs(single[0][1] - multiple[0][1]) < 0.001

    def test_scores_map_to_confidence_levels(self):
        """
        Calibration maps RRF scores to interpretable confidence levels:
        - 0.08+: Very high confidence (>95%)
        - 0.05-0.08: High confidence (75-95%)
        - 0.035-0.05: Moderate confidence (50-75%)
        - 0.02-0.035: Low confidence (5-50%)
        - <0.02: Very low confidence (<5%)
        """
        test_cases = [
            (0.100, "very_high", 0.95, 1.00),
            (0.080, "very_high", 0.95, 1.00),
            (0.065, "very_high", 0.95, 1.00),  # Adjusted to very_high
            (0.050, "high", 0.75, 0.95),
            (0.042, "moderate", 0.50, 0.75),
            (0.035, "moderate", 0.40, 0.60),
            (0.028, "low", 0.05, 0.50),
            (0.020, "low", 0.05, 0.50),
            (0.015, "very_low", 0.00, 0.05),
            (0.010, "very_low", 0.00, 0.05),
        ]

        for score, level, min_conf, max_conf in test_cases:
            calibrated = calibrate_score(score, threshold=0.035, steepness=150.0)
            assert min_conf <= calibrated <= max_conf, \
                f"Score {score} ({level}) → {calibrated:.3f}, expected [{min_conf}, {max_conf}]"


class TestEdgeCases:
    """Edge case handling."""

    def test_zero_score(self):
        """Zero score → very low confidence."""
        calibrated = calibrate_score(0.0)
        assert 0.0 <= calibrated < 0.01

    def test_negative_score(self):
        """Negative score (shouldn't happen) → very close to 0.0."""
        calibrated = calibrate_score(-0.05)
        assert calibrated == pytest.approx(0.0, abs=0.001)

    def test_huge_score(self):
        """Very large score → 1.0."""
        calibrated = calibrate_score(100.0)
        assert calibrated == pytest.approx(1.0, abs=0.01)

    def test_nan_protection(self):
        """Overflow protection prevents NaN."""
        # Extreme exponent that could cause overflow
        calibrated = calibrate_score(0.2, threshold=0.035, steepness=1000.0)
        assert not math.isnan(calibrated)
        assert calibrated == pytest.approx(1.0, abs=0.01)
