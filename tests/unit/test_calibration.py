import math

import pytest

from src.search.calibration import calibrate_results, calibrate_score


class TestCalibrateScore:

    def test_threshold_gives_half(self):
        """
        Score at threshold yields ~0.5 confidence.
        """
        threshold = 0.035
        score = calibrate_score(threshold, threshold=threshold)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_high_score_high_confidence(self):
        """
        High RRF scores map to strong confidence.
        """
        score = calibrate_score(0.08, threshold=0.035, steepness=150.0)
        assert score > 0.95
        assert score <= 1.0

    def test_low_score_low_confidence(self):
        """
        Low RRF scores map to very low confidence.
        """
        score = calibrate_score(0.015, threshold=0.035, steepness=150.0)
        assert score < 0.06
        assert score >= 0.0

    def test_monotonic_increasing(self):
        """
        Preserves ranking order.
        """
        scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        calibrated = [calibrate_score(s) for s in scores]

        for i in range(len(calibrated) - 1):
            assert calibrated[i] < calibrated[i + 1]

    def test_bounded_output(self):
        """
        All scores in [0, 1].
        """
        test_scores = [0.0, 0.001, 0.01, 0.035, 0.05, 0.1, 0.5, 1.0, 10.0]

        for score in test_scores:
            calibrated = calibrate_score(score)
            assert 0.0 <= calibrated <= 1.0, f"Score {score} → {calibrated} out of bounds"

    def test_steepness_affects_curve(self):
        """
        Higher steepness yields a sharper transition.
        """
        score = 0.05
        threshold = 0.035

        cal_gentle = calibrate_score(score, threshold=threshold, steepness=50.0)
        cal_sharp = calibrate_score(score, threshold=threshold, steepness=300.0)

        assert cal_gentle > 0.5
        assert cal_sharp > 0.5

        assert cal_sharp > cal_gentle

    def test_overflow_protection(self):
        """
        Extreme scores do not overflow or crash.
        """
        large_score = calibrate_score(1000.0, threshold=0.035, steepness=150.0)
        assert large_score == pytest.approx(1.0, abs=0.01)

        small_score = calibrate_score(-1000.0, threshold=0.035, steepness=150.0)
        assert small_score == pytest.approx(0.0, abs=0.001)

        extreme_steep = calibrate_score(0.06, threshold=0.035, steepness=1000.0)
        assert 0.0 <= extreme_steep <= 1.0

    def test_empty_results(self):
        """
        Returns [].
        """
        calibrated = calibrate_results([])
        assert calibrated == []

    def test_realistic_rrf_scores(self):
        """
        Typical 0.01-0.10 range behaves as expected.
        """
        fused = [
            ("doc1", 0.0831),
            ("doc2", 0.0667),
            ("doc3", 0.0500),
            ("doc4", 0.0350),
            ("doc5", 0.0250),
            ("doc6", 0.0167),
            ("doc7", 0.0100),
        ]

        calibrated = calibrate_results(fused, threshold=0.035, steepness=150.0)

        assert all(0.0 <= score <= 1.0 for _, score in calibrated)

        scores = [score for _, score in calibrated]
        assert scores == sorted(scores, reverse=True)

        assert calibrated[0][1] > 0.95

        assert 0.4 < calibrated[3][1] < 0.6

        assert calibrated[6][1] < 0.03

    def test_recency_boosted_scores(self):
        """
        Recency-boosted scores map to moderate-high confidence.
        """
        base_rrf = 0.038
        recency_multiplier = 1.2
        boosted_score = base_rrf * recency_multiplier

        calibrated = calibrate_score(boosted_score, threshold=0.035, steepness=150.0)

        assert 0.75 < calibrated < 0.9, f"Expected 0.75-0.9, got {calibrated}"


class TestCalibrateResults:

    def test_calibrate_results_preserves_doc_ids(self):
        """
        Document IDs preserved after calibration.
        """
        fused = [("doc1", 0.05), ("doc2", 0.03), ("doc3", 0.01)]
        calibrated = calibrate_results(fused)

        doc_ids = [doc_id for doc_id, _ in calibrated]
        assert doc_ids == ["doc1", "doc2", "doc3"]

    def test_calibrate_results_custom_parameters(self):
        """
        Custom threshold/steepness work correctly.
        """
        fused = [("doc1", 0.10), ("doc2", 0.05)]

        calibrated = calibrate_results(fused, threshold=0.05, steepness=100.0)

        assert calibrated[0][1] > 0.9

        assert 0.4 < calibrated[1][1] < 0.6

    def test_calibrate_results_single_result(self):
        """
        Single result gets calibrated (not forced to 1.0).
        """
        fused = [("doc1", 0.0391)]
        calibrated = calibrate_results(fused, threshold=0.035, steepness=150.0)

        assert len(calibrated) == 1
        assert calibrated[0][0] == "doc1"

        assert 0.5 < calibrated[0][1] < 0.7


class TestCalibrationVsNormalization:

    def test_single_result_not_always_one(self):
        """
        Calibration: Single result can be < 1.0 if score is low.
        Normalization: Single result always 1.0.
        """
        low_score_result = [("doc1", 0.015)]

        calibrated = calibrate_results(low_score_result, threshold=0.035, steepness=150.0)

        assert calibrated[0][1] < 0.1

    def test_absolute_confidence_interpretation(self):
        """
        Calibration provides absolute confidence scores independent of result set.

        A score of 0.08 should have ~same confidence whether it's
        alone or in a list with other scores.
        """
        score_value = 0.08

        single = calibrate_results([("doc1", score_value)])

        multiple = calibrate_results([
            ("doc1", score_value),
            ("doc2", 0.05),
            ("doc3", 0.02),
        ])

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
            (0.065, "very_high", 0.95, 1.00),
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

    def test_zero_score(self):
        """
        Verify a zero score produces a very low calibrated value.
        """
        calibrated = calibrate_score(0.0)
        assert 0.0 <= calibrated < 0.01

    def test_negative_score(self):
        """
        Verify negative scores produce very low calibrated values.
        """
        calibrated = calibrate_score(-0.05)
        assert calibrated == pytest.approx(0.0, abs=0.01)

    def test_huge_score(self):
        """
        Very large score → 1.0.
        """
        calibrated = calibrate_score(100.0)
        assert calibrated == pytest.approx(1.0, abs=0.01)

    def test_nan_protection(self):
        """
        Overflow protection prevents NaN.
        """
        calibrated = calibrate_score(0.2, threshold=0.035, steepness=1000.0)
        assert not math.isnan(calibrated)
        assert calibrated == pytest.approx(1.0, abs=0.01)
