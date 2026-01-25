"""
Unit tests for RRF (Reciprocal Rank Fusion) search result fusion.

Tests cover:
- RRF score calculation
- Recency boost tier application
"""

from datetime import datetime, timezone

import pytest

from src.search.fusion import apply_recency_boost, rrf_score


class TestRRFScore:
    """Tests for RRF score calculation."""

    def test_rrf_score_calculation(self):
        """
        Validates RRF score formula: 1 / (k + rank).
        Ensures correct score computation for various ranks and k values.
        """
        # k=60 is the default constant
        assert rrf_score(0, 60) == 1 / 60  # Rank 0 (first position)
        assert rrf_score(1, 60) == 1 / 61  # Rank 1 (second position)
        assert rrf_score(5, 60) == 1 / 65  # Rank 5
        assert rrf_score(10, 60) == 1 / 70  # Rank 10

        # Different k values
        assert rrf_score(0, 10) == 1 / 10
        assert rrf_score(0, 100) == 1 / 100

    def test_rrf_score_decreases_with_rank(self):
        """
        Verifies that RRF scores decrease monotonically with increasing rank.
        Higher ranks should always produce lower scores.
        """
        k = 60
        score_0 = rrf_score(0, k)
        score_1 = rrf_score(1, k)
        score_5 = rrf_score(5, k)
        score_10 = rrf_score(10, k)

        assert score_0 > score_1 > score_5 > score_10


class TestRecencyBoost:
    """Tests for recency-based score boosting."""

    def test_recency_boost_tiers(self):
        """
        Tests the three recency boost tiers:
        - 7 days: 1.2x multiplier
        - 30 days: 1.1x multiplier
        - >30 days: 1.0x (no boost)
        """
        base_score = 1.0
        now = datetime.now(timezone.utc).timestamp()

        # 7 days tier (1.2x boost)
        modified_times_7days = {"doc1": now - (5 * 86400)}  # 5 days ago
        tiers = [(7, 1.2), (30, 1.1)]
        boosted = apply_recency_boost("doc1", base_score, modified_times_7days, tiers)
        assert boosted == pytest.approx(1.2)

        # 30 days tier (1.1x boost)
        modified_times_30days = {"doc2": now - (15 * 86400)}  # 15 days ago
        boosted = apply_recency_boost("doc2", base_score, modified_times_30days, tiers)
        assert boosted == pytest.approx(1.1)

        # No boost for old documents
        modified_times_old = {"doc3": now - (60 * 86400)}  # 60 days ago
        boosted = apply_recency_boost("doc3", base_score, modified_times_old, tiers)
        assert boosted == pytest.approx(1.0)

    def test_recency_boost_missing_doc(self):
        """
        Verifies graceful handling when document timestamp is missing.
        Should return original score without modification.
        """
        base_score = 1.0
        modified_times = {"doc1": datetime.now(timezone.utc).timestamp()}
        tiers = [(7, 1.2), (30, 1.1)]

        # Document not in modified_times
        boosted = apply_recency_boost("doc_unknown", base_score, modified_times, tiers)
        assert boosted == base_score

    def test_recency_boost_edge_boundaries(self):
        """
        Tests boundary conditions for tier transitions.
        Ensures correct tier selection near 7 and 30 day boundaries.
        """
        base_score = 1.0
        now = datetime.now(timezone.utc).timestamp()
        tiers = [(7, 1.2), (30, 1.1)]

        # Just under 7 days ago (should get 1.2x boost)
        modified_times_7 = {"doc1": now - (6.5 * 86400)}
        boosted = apply_recency_boost("doc1", base_score, modified_times_7, tiers)
        assert boosted == pytest.approx(1.2)

        # Between 7 and 30 days (should get 1.1x boost)
        modified_times_30 = {"doc2": now - (15 * 86400)}
        boosted = apply_recency_boost("doc2", base_score, modified_times_30, tiers)
        assert boosted == pytest.approx(1.1)

        # Just over 30 days (no boost)
        modified_times_31 = {"doc3": now - (31 * 86400)}
        boosted = apply_recency_boost("doc3", base_score, modified_times_31, tiers)
        assert boosted == pytest.approx(1.0)
