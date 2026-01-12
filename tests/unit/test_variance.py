"""
Unit tests for variance calculation and dynamic weight adjustment.

Tests cover:
- Variance calculation for score distributions
- Standard deviation calculation
- Dynamic weight computation based on variance
- Edge cases: single score, zero variance, empty lists
- Weight adjustment bounds and normalization
"""

import pytest
import math

from src.search.variance import (
    calculate_variance,
    calculate_std_dev,
    compute_dynamic_weights,
)


# ============================================================================
# Variance Calculation Tests
# ============================================================================


class TestCalculateVariance:
    """Tests for the calculate_variance function."""

    def test_basic_variance_calculation(self):
        """
        Variance should be calculated as mean of squared deviations from mean.
        For [1, 2, 3]: mean=2, variance = ((1-2)² + (2-2)² + (3-2)²) / 3 = 2/3
        """
        scores = [1.0, 2.0, 3.0]
        variance = calculate_variance(scores)

        expected = 2 / 3
        assert variance == pytest.approx(expected)

    def test_identical_scores_zero_variance(self):
        """
        Identical scores should have zero variance.
        No deviation from mean when all values are the same.
        """
        scores = [0.5, 0.5, 0.5, 0.5]
        variance = calculate_variance(scores)

        assert variance == 0.0

    def test_high_variance_distribution(self):
        """
        Widely spread scores should have high variance.
        """
        scores = [0.0, 1.0]
        variance = calculate_variance(scores)

        expected = 0.25
        assert variance == pytest.approx(expected)

    def test_normalized_scores_variance(self):
        """
        Normalized [0, 1] scores should have reasonable variance.
        """
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        variance = calculate_variance(scores)

        assert 0 < variance < 1


class TestCalculateVarianceEdgeCases:
    """Tests for edge cases in variance calculation."""

    def test_empty_list_returns_zero(self):
        """
        Empty score list should return zero variance.
        No data means no variance.
        """
        scores = []
        variance = calculate_variance(scores)

        assert variance == 0.0

    def test_single_score_returns_zero(self):
        """
        Single score should return zero variance.
        Cannot calculate variance with fewer than 2 values.
        """
        scores = [0.75]
        variance = calculate_variance(scores)

        assert variance == 0.0

    def test_two_scores_variance(self):
        """
        Two scores should calculate variance correctly.
        """
        scores = [0.0, 1.0]
        variance = calculate_variance(scores)

        assert variance == pytest.approx(0.25)

    def test_negative_scores(self):
        """
        Negative scores should work correctly for variance.
        """
        scores = [-1.0, 0.0, 1.0]
        variance = calculate_variance(scores)

        expected = 2 / 3
        assert variance == pytest.approx(expected)


# ============================================================================
# Standard Deviation Tests
# ============================================================================


class TestCalculateStdDev:
    """Tests for the calculate_std_dev function."""

    def test_std_dev_is_sqrt_variance(self):
        """
        Standard deviation should be square root of variance.
        """
        scores = [1.0, 2.0, 3.0]
        std_dev = calculate_std_dev(scores)
        variance = calculate_variance(scores)

        assert std_dev == pytest.approx(math.sqrt(variance))

    def test_zero_variance_zero_std_dev(self):
        """
        Zero variance should result in zero standard deviation.
        """
        scores = [0.5, 0.5, 0.5]
        std_dev = calculate_std_dev(scores)

        assert std_dev == 0.0

    def test_empty_list_zero_std_dev(self):
        """
        Empty list should return zero standard deviation.
        """
        scores = []
        std_dev = calculate_std_dev(scores)

        assert std_dev == 0.0


# ============================================================================
# Dynamic Weight Computation Tests
# ============================================================================


class TestComputeDynamicWeights:
    """Tests for dynamic weight computation based on variance."""

    def test_high_variance_preserves_weights(self):
        """
        High variance (confident results) should preserve base weights.
        When variance exceeds threshold, weights should remain close to base.
        """
        vector_scores = [0.1, 0.5, 0.9]
        keyword_scores = [0.2, 0.4, 0.8]

        vector_w, keyword_w = compute_dynamic_weights(
            vector_scores,
            keyword_scores,
            base_vector_weight=1.0,
            base_keyword_weight=0.8,
            variance_threshold=0.01,
        )

        assert vector_w == pytest.approx(1.0, rel=0.1)
        assert keyword_w == pytest.approx(0.8, rel=0.1)

    def test_low_variance_reduces_weight(self):
        """
        Low variance (muddy/uncertain results) should reduce that strategy's weight.
        Flat scores indicate poor discrimination, so weight is reduced.
        """
        flat_vector_scores = [0.5, 0.51, 0.49, 0.5]
        high_variance_keyword = [0.1, 0.5, 0.9]

        vector_w, keyword_w = compute_dynamic_weights(
            flat_vector_scores,
            high_variance_keyword,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=0.1,
        )

        assert vector_w < 1.0
        assert keyword_w > vector_w

    def test_zero_variance_gets_minimum_weight(self):
        """
        Zero variance should result in minimum weight factor applied.
        """
        identical_scores = [0.5, 0.5, 0.5, 0.5]
        varied_scores = [0.1, 0.5, 0.9]

        vector_w, keyword_w = compute_dynamic_weights(
            identical_scores,
            varied_scores,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=0.1,
            min_weight_factor=0.5,
        )

        assert vector_w < keyword_w

    def test_both_low_variance_both_reduced(self):
        """
        Both strategies with low variance should both have reduced weights.
        """
        flat_vector = [0.5, 0.5, 0.5]
        flat_keyword = [0.6, 0.6, 0.6]

        vector_w, keyword_w = compute_dynamic_weights(
            flat_vector,
            flat_keyword,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=0.1,
            min_weight_factor=0.5,
        )

        total = vector_w + keyword_w
        assert total == pytest.approx(2.0, rel=0.01)

    def test_empty_vector_scores(self):
        """
        Empty vector scores should be handled gracefully.
        """
        vector_scores = []
        keyword_scores = [0.1, 0.5, 0.9]

        vector_w, keyword_w = compute_dynamic_weights(
            vector_scores,
            keyword_scores,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=0.1,
        )

        assert keyword_w > 0

    def test_empty_keyword_scores(self):
        """
        Empty keyword scores should be handled gracefully.
        """
        vector_scores = [0.1, 0.5, 0.9]
        keyword_scores = []

        vector_w, keyword_w = compute_dynamic_weights(
            vector_scores,
            keyword_scores,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=0.1,
        )

        assert vector_w > 0

    def test_both_empty_scores(self):
        """
        Both empty score lists should return normalized base weights.
        """
        vector_w, keyword_w = compute_dynamic_weights(
            [],
            [],
            base_vector_weight=1.0,
            base_keyword_weight=0.5,
            variance_threshold=0.1,
        )

        total = vector_w + keyword_w
        assert total == pytest.approx(1.5, rel=0.01)


class TestComputeDynamicWeightsNormalization:
    """Tests for weight normalization in dynamic weight computation."""

    def test_weights_sum_preserved(self):
        """
        Total weight sum should be preserved after dynamic adjustment.
        """
        vector_scores = [0.1, 0.5, 0.9]
        keyword_scores = [0.2, 0.6, 0.8]

        base_v = 1.0
        base_k = 0.8
        expected_sum = base_v + base_k

        vector_w, keyword_w = compute_dynamic_weights(
            vector_scores,
            keyword_scores,
            base_vector_weight=base_v,
            base_keyword_weight=base_k,
            variance_threshold=0.1,
        )

        assert vector_w + keyword_w == pytest.approx(expected_sum, rel=0.01)

    def test_different_base_weights(self):
        """
        Different base weights should be respected in normalization.
        """
        scores = [0.1, 0.5, 0.9]

        v_w_equal, k_w_equal = compute_dynamic_weights(
            scores, scores,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=0.01,
        )

        v_w_unequal, k_w_unequal = compute_dynamic_weights(
            scores, scores,
            base_vector_weight=2.0,
            base_keyword_weight=1.0,
            variance_threshold=0.01,
        )

        assert v_w_unequal + k_w_unequal == pytest.approx(3.0, rel=0.01)


class TestComputeDynamicWeightsParameters:
    """Tests for variance_threshold and min_weight_factor parameters."""

    def test_variance_threshold_boundary(self):
        """
        Variance exactly at threshold should not reduce weight.
        """
        threshold = 0.1
        scores_at_threshold = [0.0, 1.0]

        vector_w, keyword_w = compute_dynamic_weights(
            scores_at_threshold,
            scores_at_threshold,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=threshold,
        )

        total = vector_w + keyword_w
        assert total == pytest.approx(2.0, rel=0.01)

    def test_min_weight_factor_floor(self):
        """
        Weight should not drop below min_weight_factor * base_weight.
        """
        identical_scores = [0.5, 0.5, 0.5]
        varied_scores = [0.0, 1.0]

        vector_w, keyword_w = compute_dynamic_weights(
            identical_scores,
            varied_scores,
            base_vector_weight=1.0,
            base_keyword_weight=1.0,
            variance_threshold=0.1,
            min_weight_factor=0.3,
        )

        assert vector_w >= 0.3
