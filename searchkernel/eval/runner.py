"""Evaluation runner: execute retrieval evaluation on a golden set.

The runner is decoupled from SearchOrchestrator; it takes a plain search callable.
This enables testing without a live index.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from searchkernel.eval.golden import GoldenSet
from searchkernel.eval.metrics import average_precision, mrr, ndcg_at_k, recall_at_k
from searchkernel.runtime.trace import QueryTrace


@dataclass
class MetricSnapshot:
    """Per-query metric snapshot."""

    query: str
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    ap: float
    latency_ms: float | None


@dataclass
class EvalReport:
    """Evaluation report for a single search function."""

    golden_set_size: int
    """Number of queries in the golden set."""

    k: int
    """Cutoff rank for recall@k and nDCG@k."""

    metrics: list[MetricSnapshot] = field(default_factory=list)
    """Per-query metric snapshots."""

    mean_recall_at_k: float | None = None
    """Mean recall@k across all queries."""

    mean_ndcg_at_k: float | None = None
    """Mean nDCG@k across all queries."""

    mean_mrr: float | None = None
    """Mean MRR across all queries."""

    mean_ap: float | None = None
    """Mean average precision across all queries."""

    latency_p50_ms: float | None = None
    """50th percentile latency (median)."""

    latency_p95_ms: float | None = None
    """95th percentile latency."""

    latency_p99_ms: float | None = None
    """99th percentile latency."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "golden_set_size": self.golden_set_size,
            "k": self.k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "mean_mrr": self.mean_mrr,
            "mean_ap": self.mean_ap,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "per_query_metrics": [
                {
                    "query": m.query,
                    "recall_at_k": m.recall_at_k,
                    "ndcg_at_k": m.ndcg_at_k,
                    "mrr": m.mrr,
                    "ap": m.ap,
                    "latency_ms": m.latency_ms,
                }
                for m in self.metrics
            ],
        }


@dataclass
class AbReport:
    """A/B comparison report between two search functions."""

    report_a: EvalReport
    """Baseline report."""

    report_b: EvalReport
    """Candidate report."""

    recall_at_k_delta: float | None = None
    """Absolute difference in mean recall@k (B - A)."""

    ndcg_at_k_delta: float | None = None
    """Absolute difference in mean nDCG@k (B - A)."""

    mrr_delta: float | None = None
    """Absolute difference in mean MRR (B - A)."""

    ap_delta: float | None = None
    """Absolute difference in mean AP (B - A)."""

    latency_p50_delta_ms: float | None = None
    """Absolute difference in p50 latency (B - A)."""

    latency_p95_delta_ms: float | None = None
    """Absolute difference in p95 latency (B - A)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "report_a": self.report_a.to_dict(),
            "report_b": self.report_b.to_dict(),
            "deltas": {
                "recall_at_k": self.recall_at_k_delta,
                "ndcg_at_k": self.ndcg_at_k_delta,
                "mrr": self.mrr_delta,
                "ap": self.ap_delta,
                "latency_p50_ms": self.latency_p50_delta_ms,
                "latency_p95_ms": self.latency_p95_delta_ms,
            },
        }


def _percentile(values: list[float], p: int) -> float:
    """Compute the pth percentile of a list of values.

    Args:
        values: List of numeric values.
        p: Percentile (0-100).

    Returns:
        The percentile value.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((p / 100.0) * len(sorted_vals))
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_eval(
    golden_set: GoldenSet,
    search_fn: Callable[[str], list[str]],
    k: int = 10,
) -> EvalReport:
    """Run evaluation on a golden set using a search function.

    Args:
        golden_set: The GoldenSet with queries and relevant IDs.
        search_fn: A callable(query: str) -> list[str] of ranked result IDs.
                   Called once per query in the golden set.
        k: Cutoff rank for recall@k and nDCG@k.

    Returns:
        An EvalReport with per-query and aggregate metrics.
    """
    report = EvalReport(
        golden_set_size=len(golden_set),
        k=k,
        metrics=[],
    )

    latencies = []

    for entry in golden_set:
        # Time the search
        trace = QueryTrace(entry.query)
        with trace.span("search"):
            ranked_ids = search_fn(entry.query)
        trace.close()

        latency_ms = trace.total_duration_ms or 0.0
        latencies.append(latency_ms)

        # Compute metrics
        recall = recall_at_k(ranked_ids, entry.relevant_ids, k)
        ndcg = ndcg_at_k(ranked_ids, entry.relevant_ids, k)
        mrr_score = mrr(ranked_ids, entry.relevant_ids)
        ap_score = average_precision(ranked_ids, entry.relevant_ids)

        snapshot = MetricSnapshot(
            query=entry.query,
            recall_at_k=recall,
            ndcg_at_k=ndcg,
            mrr=mrr_score,
            ap=ap_score,
            latency_ms=latency_ms,
        )
        report.metrics.append(snapshot)

    # Compute aggregates
    if report.metrics:
        report.mean_recall_at_k = sum(m.recall_at_k for m in report.metrics) / len(report.metrics)
        report.mean_ndcg_at_k = sum(m.ndcg_at_k for m in report.metrics) / len(report.metrics)
        report.mean_mrr = sum(m.mrr for m in report.metrics) / len(report.metrics)
        report.mean_ap = sum(m.ap for m in report.metrics) / len(report.metrics)

    if latencies:
        report.latency_p50_ms = _percentile(latencies, 50)
        report.latency_p95_ms = _percentile(latencies, 95)
        report.latency_p99_ms = _percentile(latencies, 99)

    return report


def ab_eval(
    golden_set: GoldenSet,
    search_fn_a: Callable[[str], list[str]],
    search_fn_b: Callable[[str], list[str]],
    k: int = 10,
) -> AbReport:
    """Run A/B evaluation comparing two search functions.

    Args:
        golden_set: The GoldenSet with queries and relevant IDs.
        search_fn_a: Baseline search callable.
        search_fn_b: Candidate search callable.
        k: Cutoff rank for metrics.

    Returns:
        An AbReport with both reports and deltas.
    """
    report_a = run_eval(golden_set, search_fn_a, k)
    report_b = run_eval(golden_set, search_fn_b, k)

    ab_report = AbReport(
        report_a=report_a,
        report_b=report_b,
    )

    # Compute deltas
    if report_a.mean_recall_at_k is not None and report_b.mean_recall_at_k is not None:
        ab_report.recall_at_k_delta = report_b.mean_recall_at_k - report_a.mean_recall_at_k

    if report_a.mean_ndcg_at_k is not None and report_b.mean_ndcg_at_k is not None:
        ab_report.ndcg_at_k_delta = report_b.mean_ndcg_at_k - report_a.mean_ndcg_at_k

    if report_a.mean_mrr is not None and report_b.mean_mrr is not None:
        ab_report.mrr_delta = report_b.mean_mrr - report_a.mean_mrr

    if report_a.mean_ap is not None and report_b.mean_ap is not None:
        ab_report.ap_delta = report_b.mean_ap - report_a.mean_ap

    if report_a.latency_p50_ms is not None and report_b.latency_p50_ms is not None:
        ab_report.latency_p50_delta_ms = report_b.latency_p50_ms - report_a.latency_p50_ms

    if report_a.latency_p95_ms is not None and report_b.latency_p95_ms is not None:
        ab_report.latency_p95_delta_ms = report_b.latency_p95_ms - report_a.latency_p95_ms

    return ab_report
