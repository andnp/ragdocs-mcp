"""
Unit tests for score normalization functions.

Tests cover:
- Min-max normalization to [0, 1] range
- Edge cases: empty lists, single elements, identical scores
- Negative score handling
- Result tuple normalization
"""

import pytest

from src.search.normalization import normalize_scores, normalize_result_scores


# ============================================================================
# Basic Normalization Tests
# ============================================================================


class TestNormalizeScores:
    """Tests for the normalize_scores function."""

    def test_basic_min_max_normalization(self):
        """
        Scores should be normalized to [0, 1] range using min-max scaling.
        Minimum score maps to 0, maximum to 1.
        """
        scores = [0.2, 0.5, 0.8]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_preserves_relative_ordering(self):
        """
        Normalization should preserve the relative ordering of scores.
        """
        scores = [10.0, 30.0, 20.0, 50.0, 40.0]
        normalized = normalize_scores(scores)

        assert normalized[0] < normalized[2] < normalized[1] < normalized[4] < normalized[3]

    def test_arbitrary_range_normalizes_correctly(self):
        """
        Scores in arbitrary ranges should normalize to [0, 1].
        """
        scores = [100.0, 200.0, 300.0]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_already_normalized_range(self):
        """
        Scores already in [0, 1] should still be re-normalized.
        """
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.25)
        assert normalized[2] == pytest.approx(0.5)
        assert normalized[3] == pytest.approx(0.75)
        assert normalized[4] == pytest.approx(1.0)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestNormalizeScoresEdgeCases:
    """Tests for edge cases in score normalization."""

    def test_empty_list_returns_empty(self):
        """
        Empty input should return empty output.
        No scores to normalize means no output.
        """
        scores = []
        normalized = normalize_scores(scores)

        assert normalized == []

    def test_single_element_returns_one(self):
        """
        Single element should normalize to 1.0.
        With no range to scale, the only score becomes maximum.
        """
        scores = [0.5]
        normalized = normalize_scores(scores)

        assert normalized == [1.0]

    def test_single_element_any_value(self):
        """
        Any single score value should normalize to 1.0.
        """
        assert normalize_scores([100]) == [1.0]
        assert normalize_scores([0]) == [1.0]
        assert normalize_scores([-50]) == [1.0]

    def test_identical_scores_all_return_one(self):
        """
        All identical scores should normalize to 1.0.
        No variance means all scores are effectively maximum.
        """
        scores = [0.5, 0.5, 0.5, 0.5]
        normalized = normalize_scores(scores)

        assert all(s == 1.0 for s in normalized)

    def test_two_identical_scores(self):
        """
        Two identical scores should both normalize to 1.0.
        """
        scores = [0.7, 0.7]
        normalized = normalize_scores(scores)

        assert normalized == [1.0, 1.0]

    def test_identical_scores_different_value(self):
        """
        Any identical value should result in all 1.0s.
        """
        assert normalize_scores([0, 0, 0]) == [1.0, 1.0, 1.0]
        assert normalize_scores([100, 100]) == [1.0, 1.0]


# ============================================================================
# Negative Score Tests
# ============================================================================


class TestNormalizeScoresNegative:
    """Tests for handling negative scores."""

    def test_negative_scores_normalize_correctly(self):
        """
        Negative scores should normalize correctly to [0, 1].
        Min-max works regardless of sign.
        """
        scores = [-10.0, 0.0, 10.0]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_all_negative_scores(self):
        """
        All negative scores should still normalize to [0, 1].
        """
        scores = [-30.0, -20.0, -10.0]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_mixed_negative_positive(self):
        """
        Mixed negative and positive scores normalize correctly.
        """
        scores = [-5.0, 0.0, 5.0, 10.0]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(1 / 3)
        assert normalized[2] == pytest.approx(2 / 3)
        assert normalized[3] == pytest.approx(1.0)


# ============================================================================
# Result Tuple Normalization Tests
# ============================================================================


class TestNormalizeResultScores:
    """Tests for normalizing (doc_id, score) tuples."""

    def test_basic_result_normalization(self):
        """
        Result tuples should have scores normalized while preserving doc IDs.
        """
        results = [("doc1", 0.2), ("doc2", 0.5), ("doc3", 0.8)]
        normalized = normalize_result_scores(results)

        assert normalized[0] == ("doc1", pytest.approx(0.0))
        assert normalized[1] == ("doc2", pytest.approx(0.5))
        assert normalized[2] == ("doc3", pytest.approx(1.0))

    def test_preserves_doc_id_order(self):
        """
        Doc IDs should remain in original order after normalization.
        """
        results = [("a", 100.0), ("b", 50.0), ("c", 75.0)]
        normalized = normalize_result_scores(results)

        doc_ids = [doc_id for doc_id, _ in normalized]
        assert doc_ids == ["a", "b", "c"]

    def test_empty_results_returns_empty(self):
        """
        Empty result list should return empty list.
        """
        results = []
        normalized = normalize_result_scores(results)

        assert normalized == []

    def test_single_result_normalizes_to_one(self):
        """
        Single result should have score normalized to 1.0.
        """
        results = [("only_doc", 0.5)]
        normalized = normalize_result_scores(results)

        assert normalized == [("only_doc", 1.0)]

    def test_identical_scores_all_one(self):
        """
        Identical scores should all normalize to 1.0.
        """
        results = [("doc1", 0.5), ("doc2", 0.5), ("doc3", 0.5)]
        normalized = normalize_result_scores(results)

        for doc_id, score in normalized:
            assert score == 1.0


# ============================================================================
# Precision Tests
# ============================================================================


class TestNormalizationPrecision:
    """Tests for numerical precision in normalization."""

    def test_very_small_range(self):
        """
        Very small score ranges should normalize without floating point issues.
        """
        scores = [0.0001, 0.0002, 0.0003]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_very_large_values(self):
        """
        Very large score values should normalize correctly.
        """
        scores = [1e10, 2e10, 3e10]
        normalized = normalize_scores(scores)

        assert normalized[0] == pytest.approx(0.0)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(1.0)

    def test_float_precision_edge(self):
        """
        Floating point edge cases should be handled.
        """
        scores = [0.1 + 0.2, 0.3, 0.6]
        normalized = normalize_scores(scores)

        assert len(normalized) == 3
        assert all(0.0 <= s <= 1.0 for s in normalized)
