"""Unit tests for evaluation runner."""

from searchkernel.eval.golden import GoldenEntry, GoldenSet
from searchkernel.eval.runner import ab_eval, run_eval


def test_run_eval_perfect_search():
    """Test eval with a perfect search function."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="query1", relevant_ids=["a", "b"]),
            GoldenEntry(query="query2", relevant_ids=["c"]),
        ]
    )

    def perfect_search(query: str) -> list[str]:
        """Returns all relevant IDs in order."""
        if query == "query1":
            return ["a", "b", "x", "y"]
        elif query == "query2":
            return ["c", "x", "y"]
        else:
            return []

    report = run_eval(golden_set, perfect_search, k=2)

    assert report.golden_set_size == 2
    assert report.k == 2
    assert len(report.metrics) == 2

    # Query 1: ["a", "b"], relevant: {a, b}
    # Recall@2 = 2/2 = 1.0
    # nDCG@2 = 1.0
    # MRR = 1.0 (first relevant at position 1)
    # AP = 1.0
    assert report.metrics[0].recall_at_k == 1.0
    assert report.metrics[0].ndcg_at_k == 1.0
    assert report.metrics[0].mrr == 1.0
    assert report.metrics[0].ap == 1.0

    # Query 2: ["c", "x", "y"], relevant: {c}
    # Recall@2 = 1/1 = 1.0
    # nDCG@2 = 1.0
    # MRR = 1.0
    # AP = 1.0
    assert report.metrics[1].recall_at_k == 1.0
    assert report.metrics[1].ndcg_at_k == 1.0
    assert report.metrics[1].mrr == 1.0
    assert report.metrics[1].ap == 1.0

    # Aggregates
    assert report.mean_recall_at_k == 1.0
    assert report.mean_ndcg_at_k == 1.0
    assert report.mean_mrr == 1.0
    assert report.mean_ap == 1.0


def test_run_eval_partial_search():
    """Test eval with partial matches."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a", "b", "c"]),
            GoldenEntry(query="q2", relevant_ids=["x", "y"]),
        ]
    )

    def partial_search(query: str) -> list[str]:
        """Returns some relevant items."""
        if query == "q1":
            return ["a", "z", "b"]  # Has a, b but missing c
        elif query == "q2":
            return ["z", "w"]  # No relevant items
        else:
            return []

    report = run_eval(golden_set, partial_search, k=3)

    # Query 1: ["a", "z", "b"], relevant: {a, b, c}
    # Recall@3 = 2/3 = 0.667
    assert abs(report.metrics[0].recall_at_k - 2.0 / 3.0) < 0.001

    # Query 2: ["z", "w"], relevant: {x, y}
    # Recall@3 = 0/2 = 0.0
    assert report.metrics[1].recall_at_k == 0.0

    # Aggregate recall = (2/3 + 0) / 2 = 0.333
    assert abs(report.mean_recall_at_k - 1.0 / 3.0) < 0.001


def test_run_eval_latency_percentiles():
    """Test that latency percentiles are computed."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
            GoldenEntry(query="q3", relevant_ids=["c"]),
        ]
    )

    def dummy_search(query: str) -> list[str]:
        return [query[1]]  # Just return something

    report = run_eval(golden_set, dummy_search, k=10)

    # Should have latency percentiles
    assert report.latency_p50_ms is not None
    assert report.latency_p95_ms is not None
    assert report.latency_p99_ms is not None

    # Percentiles should be ordered
    assert report.latency_p50_ms <= report.latency_p95_ms
    assert report.latency_p95_ms <= report.latency_p99_ms


def test_run_eval_empty_golden_set():
    """Test eval with empty golden set."""
    golden_set = GoldenSet(entries=[])

    def dummy_search(query: str) -> list[str]:
        return []

    report = run_eval(golden_set, dummy_search, k=10)

    assert report.golden_set_size == 0
    assert len(report.metrics) == 0
    assert report.mean_recall_at_k is None
    assert report.latency_p50_ms is None


def test_ab_eval():
    """Test A/B evaluation."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
        ]
    )

    def search_a(query: str) -> list[str]:
        """Baseline: always returns nothing."""
        return []

    def search_b(query: str) -> list[str]:
        """Candidate: returns correct answer."""
        if query == "q1":
            return ["a", "x"]
        elif query == "q2":
            return ["b", "y"]
        return []

    ab_report = ab_eval(golden_set, search_a, search_b, k=10)

    # Baseline should have 0 recall
    assert ab_report.report_a.mean_recall_at_k == 0.0
    # Candidate should have 1.0 recall
    assert ab_report.report_b.mean_recall_at_k == 1.0

    # Delta should be positive
    assert ab_report.recall_at_k_delta == 1.0
    assert ab_report.ndcg_at_k_delta == 1.0


def test_ab_eval_regression():
    """Test A/B eval detecting a regression."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a", "b"]),
        ]
    )

    def search_good(query: str) -> list[str]:
        return ["a", "b", "c"]

    def search_bad(query: str) -> list[str]:
        return ["c", "d", "e"]

    ab_report = ab_eval(golden_set, search_good, search_bad, k=5)

    # Good should have higher recall
    assert ab_report.report_a.mean_recall_at_k > ab_report.report_b.mean_recall_at_k
    # Delta should be negative
    assert ab_report.recall_at_k_delta < 0
