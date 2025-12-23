"""
Unit tests for RRF (Reciprocal Rank Fusion) search result fusion.

Tests cover:
- RRF score calculation
- Recency boost tier application
- Basic result merging across multiple strategies
- Weighted strategy fusion
- Edge cases (empty results, single strategy, score ordering)
"""

from datetime import datetime, timezone

import pytest

from src.search.fusion import apply_recency_boost, fuse_results, rrf_score


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


class TestFuseResults:
    """Tests for multi-strategy result fusion."""

    def test_fuse_results_basic_merging(self):
        """
        Tests basic merging of results from multiple strategies.
        Verifies RRF scoring aggregates across strategies correctly.
        """
        results = {
            "semantic": ["doc1", "doc2", "doc3"],
            "keyword": ["doc2", "doc3", "doc4"],
        }
        k = 60
        weights = {"semantic": 1.0, "keyword": 1.0}
        modified_times = {}  # No recency boost

        fused = fuse_results(results, k, weights, modified_times)

        # doc2 and doc3 appear in both lists, should rank highest
        # doc2: RRF(1) + RRF(0) = 1/61 + 1/60 ≈ 0.0164 + 0.0167 = 0.0331
        # doc3: RRF(2) + RRF(1) = 1/62 + 1/61 ≈ 0.0161 + 0.0164 = 0.0325
        # doc1: RRF(0) = 1/60 ≈ 0.0167
        # doc4: RRF(2) = 1/62 ≈ 0.0161

        doc_ids = [doc_id for doc_id, _ in fused]
        scores = [score for _, score in fused]

        # doc2 should be first (appears in both lists at high ranks)
        assert doc_ids[0] == "doc2"

        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)

    def test_fuse_results_with_weights(self):
        """
        Tests weighted strategy fusion.
        Higher weight should increase influence of that strategy's results.
        """
        results = {
            "semantic": ["doc1", "doc2"],
            "keyword": ["doc3", "doc4"],
        }
        k = 60
        weights = {"semantic": 2.0, "keyword": 1.0}  # Semantic weighted 2x
        modified_times = {}

        fused = fuse_results(results, k, weights, modified_times)

        # doc1: 2.0 * RRF(0) = 2.0 * 1/60 ≈ 0.0333
        # doc2: 2.0 * RRF(1) = 2.0 * 1/61 ≈ 0.0328
        # doc3: 1.0 * RRF(0) = 1.0 * 1/60 ≈ 0.0167
        # doc4: 1.0 * RRF(1) = 1.0 * 1/61 ≈ 0.0164

        doc_ids = [doc_id for doc_id, _ in fused]

        # doc1 should rank highest due to 2x weight
        assert doc_ids[0] == "doc1"
        assert doc_ids[1] == "doc2"
        assert doc_ids[2] == "doc3"
        assert doc_ids[3] == "doc4"

    def test_fuse_results_with_recency_boost(self):
        """
        Tests integration of recency boost with RRF scoring.
        Recent documents should receive score multipliers.
        """
        now = datetime.now(timezone.utc).timestamp()

        results = {
            "semantic": ["doc_old", "doc_recent"],
        }
        k = 60
        weights = {"semantic": 1.0}
        modified_times = {
            "doc_old": now - (60 * 86400),  # 60 days ago (no boost)
            "doc_recent": now - (3 * 86400),  # 3 days ago (1.2x boost)
        }

        fused = fuse_results(results, k, weights, modified_times)

        # doc_old: RRF(0) * 1.0 = 1/60 ≈ 0.0167
        # doc_recent: RRF(1) * 1.2 = 1/61 * 1.2 ≈ 0.0197

        doc_ids = [doc_id for doc_id, _ in fused]

        # doc_recent should rank first due to recency boost
        assert doc_ids[0] == "doc_recent"
        assert doc_ids[1] == "doc_old"

    def test_fuse_results_empty_input(self):
        """
        Tests graceful handling of empty results.
        Should return empty list without errors.
        """
        results = {}
        k = 60
        weights = {}
        modified_times = {}

        fused = fuse_results(results, k, weights, modified_times)

        assert fused == []

    def test_fuse_results_single_strategy(self):
        """
        Tests fusion with only one strategy.
        Should preserve original ranking with RRF scoring applied.
        """
        results = {
            "semantic": ["doc1", "doc2", "doc3"],
        }
        k = 60
        weights = {"semantic": 1.0}
        modified_times = {}

        fused = fuse_results(results, k, weights, modified_times)

        doc_ids = [doc_id for doc_id, _ in fused]

        # Original order should be preserved
        assert doc_ids == ["doc1", "doc2", "doc3"]

        # Scores should decrease with rank
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_fuse_results_score_ordering(self):
        """
        Tests that final results are correctly sorted by score.
        Verifies descending order regardless of input complexity.
        """
        results = {
            "semantic": ["doc1", "doc2", "doc3"],
            "keyword": ["doc3", "doc1", "doc4"],
            "graph": ["doc4", "doc5"],
        }
        k = 60
        weights = {"semantic": 1.0, "keyword": 1.0, "graph": 1.0}
        modified_times = {}

        fused = fuse_results(results, k, weights, modified_times)

        # Verify all scores are in descending order
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

        # Verify all unique documents are present
        doc_ids = [doc_id for doc_id, _ in fused]
        assert set(doc_ids) == {"doc1", "doc2", "doc3", "doc4", "doc5"}

    def test_fuse_results_default_weights(self):
        """
        Tests that missing strategies in weights dict default to 1.0.
        Ensures robust handling of partial weight specifications.
        """
        results = {
            "semantic": ["doc1"],
            "keyword": ["doc2"],
            "graph": ["doc3"],
        }
        k = 60
        weights = {"semantic": 2.0}  # Only semantic specified
        modified_times = {}

        fused = fuse_results(results, k, weights, modified_times)

        # doc1: 2.0 * 1/60 ≈ 0.0333
        # doc2: 1.0 * 1/60 ≈ 0.0167 (default weight)
        # doc3: 1.0 * 1/60 ≈ 0.0167 (default weight)

        doc_ids = [doc_id for doc_id, _ in fused]

        # doc1 should rank first due to explicit 2.0 weight
        assert doc_ids[0] == "doc1"
