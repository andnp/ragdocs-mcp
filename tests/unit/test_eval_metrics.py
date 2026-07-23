"""Unit tests for evaluation metrics."""

import math

from searchkernel.eval.metrics import average_precision, mrr, ndcg_at_k, recall_at_k


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_recall_at_k_perfect_match(self):
        """All relevant items in top-k."""
        ranked = ["a", "b", "c", "d"]
        relevant = {"a", "b", "c", "d"}
        assert recall_at_k(ranked, relevant, k=4) == 1.0

    def test_recall_at_k_partial_match(self):
        """Some relevant items in top-k.

        ranked: [a, b, c, d]
        relevant: {b, d}
        top-2: {a, b} -> hit on b -> 1/2 = 0.5
        """
        ranked = ["a", "b", "c", "d"]
        relevant = {"b", "d"}
        assert recall_at_k(ranked, relevant, k=2) == 0.5

    def test_recall_at_k_no_hits(self):
        """No relevant items in top-k."""
        ranked = ["a", "b", "c", "d"]
        relevant = {"x", "y"}
        assert recall_at_k(ranked, relevant, k=2) == 0.0

    def test_recall_at_k_empty_relevant(self):
        """Empty relevant set returns 0."""
        ranked = ["a", "b", "c"]
        relevant: set[str] = set()
        assert recall_at_k(ranked, relevant, k=2) == 0.0

    def test_recall_at_k_k_larger_than_ranked(self):
        """k larger than ranked list size."""
        ranked = ["a", "b"]
        relevant = {"a", "b", "c"}
        # Only 2 items in ranked, so can match at most a and b -> 2/3
        assert recall_at_k(ranked, relevant, k=10) == 2.0 / 3.0

    def test_recall_at_k_with_list_relevant(self):
        """relevant_ids can be a list."""
        ranked = ["a", "b", "c", "d"]
        relevant = ["b", "d"]
        assert recall_at_k(ranked, relevant, k=2) == 0.5

    def test_recall_at_k_all_relevant(self):
        """All items in top-k are relevant."""
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(ranked, relevant, k=3) == 1.0


class TestNdcgAtK:
    """Tests for ndcg_at_k function."""

    def test_ndcg_at_k_perfect_ranking(self):
        """Relevant items ranked first."""
        ranked = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        # DCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        # IDCG = 1/log2(2) + 1/log2(3) = 1.631
        # nDCG = 1.631 / 1.631 = 1.0
        assert ndcg_at_k(ranked, relevant, k=3) == 1.0

    def test_ndcg_at_k_binary_relevance(self):
        """Test with binary relevance (default).

        ranked: [a, b, c, d]
        relevant: {b, d}
        top-3: [a, b, c]

        DCG = 0 + 1/log2(3) + 0 = 0.631
        IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        nDCG = 0.631 / 1.631 = 0.387
        """
        ranked = ["a", "b", "c", "d"]
        relevant = {"b", "d"}
        ndcg = ndcg_at_k(ranked, relevant, k=3)
        assert abs(ndcg - 0.631 / 1.631) < 0.001

    def test_ndcg_at_k_no_hits(self):
        """No relevant items in top-k."""
        ranked = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert ndcg_at_k(ranked, relevant, k=2) == 0.0

    def test_ndcg_at_k_empty_relevant(self):
        """Empty relevant set."""
        ranked = ["a", "b", "c"]
        relevant: set[str] = set()
        assert ndcg_at_k(ranked, relevant, k=2) == 0.0

    def test_ndcg_at_k_with_graded_relevance(self):
        """nDCG with custom relevance gains.

        ranked: [a, b, c]
        relevant with gains: {a: 2.0, b: 1.0}
        top-3: [a, b, c]

        DCG = 2.0/log2(2) + 1.0/log2(3) + 0 = 2.0 + 0.631 = 2.631
        IDCG = 2.0/log2(2) + 1.0/log2(3) = 2.631
        nDCG = 1.0
        """
        ranked = ["a", "b", "c"]
        relevant = {"a", "b"}
        gains = {"a": 2.0, "b": 1.0}
        ndcg = ndcg_at_k(ranked, relevant, k=3, gains=gains)
        assert ndcg == 1.0

    def test_ndcg_at_k_at_end(self):
        """Relevant item at the very end of top-k."""
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"e"}
        # DCG = 1/log2(6) = 0.226
        # IDCG = 1/log2(2) = 1
        # nDCG = 0.226
        ndcg = ndcg_at_k(ranked, relevant, k=5)
        assert abs(ndcg - 1.0 / math.log2(6)) < 0.001


class TestMrr:
    """Tests for Mean Reciprocal Rank (MRR)."""

    def test_mrr_first_position(self):
        """Relevant item at rank 1."""
        ranked = ["a", "b", "c"]
        relevant = {"a"}
        assert mrr(ranked, relevant) == 1.0

    def test_mrr_second_position(self):
        """Relevant item at rank 2.

        MRR = 1/2 = 0.5
        """
        ranked = ["a", "b", "c"]
        relevant = {"b"}
        assert mrr(ranked, relevant) == 0.5

    def test_mrr_third_position(self):
        """Relevant item at rank 3.

        MRR = 1/3 = 0.333...
        """
        ranked = ["a", "b", "c"]
        relevant = {"c"}
        assert abs(mrr(ranked, relevant) - 1.0 / 3.0) < 0.001

    def test_mrr_no_relevant(self):
        """No relevant items found."""
        ranked = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert mrr(ranked, relevant) == 0.0

    def test_mrr_empty_relevant(self):
        """Empty relevant set."""
        ranked = ["a", "b", "c"]
        relevant: set[str] = set()
        assert mrr(ranked, relevant) == 0.0

    def test_mrr_multiple_relevant_first_counts(self):
        """Multiple relevant items; first one is used."""
        ranked = ["a", "b", "c", "d"]
        relevant = {"b", "d"}
        # First relevant is b at position 2
        assert mrr(ranked, relevant) == 0.5

    def test_mrr_empty_ranked(self):
        """Empty ranked list."""
        ranked: list[str] = []
        relevant = {"a"}
        assert mrr(ranked, relevant) == 0.0


class TestAveragePrecision:
    """Tests for Average Precision (AP)."""

    def test_ap_all_relevant_at_top(self):
        """All relevant items ranked at the top.

        ranked: [a, b, c, d]
        relevant: {a, b, c}

        Precision@1: 1/1 = 1.0 (hit on a)
        Precision@2: 2/2 = 1.0 (hit on b)
        Precision@3: 3/3 = 1.0 (hit on c)
        AP = (1 + 1 + 1) / 3 = 1.0
        """
        ranked = ["a", "b", "c", "d"]
        relevant = {"a", "b", "c"}
        assert average_precision(ranked, relevant) == 1.0

    def test_ap_scattered_relevant(self):
        """Relevant items scattered in ranking.

        ranked: [a, b, c, d, e]
        relevant: {b, d}

        Precision@2: 1/2 = 0.5 (hit on b)
        Precision@4: 2/4 = 0.5 (hit on d)
        AP = (0.5 + 0.5) / 2 = 0.5
        """
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"b", "d"}
        assert average_precision(ranked, relevant) == 0.5

    def test_ap_no_relevant(self):
        """No relevant items."""
        ranked = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert average_precision(ranked, relevant) == 0.0

    def test_ap_empty_relevant(self):
        """Empty relevant set."""
        ranked = ["a", "b", "c"]
        relevant: set[str] = set()
        assert average_precision(ranked, relevant) == 0.0

    def test_ap_one_relevant_at_position_3(self):
        """Single relevant item at position 3.

        ranked: [a, b, c, d]
        relevant: {c}

        Precision@3: 1/3 = 0.333...
        AP = 0.333... / 1 = 0.333...
        """
        ranked = ["a", "b", "c", "d"]
        relevant = {"c"}
        assert abs(average_precision(ranked, relevant) - 1.0 / 3.0) < 0.001

    def test_ap_with_list_relevant(self):
        """relevant_ids as list."""
        ranked = ["a", "b", "c"]
        relevant = ["a", "b"]
        assert average_precision(ranked, relevant) == 1.0
