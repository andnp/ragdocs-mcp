from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.search.evaluation_harness import build_search_evaluation_harness

pytestmark = pytest.mark.e2e


@dataclass(frozen=True)
class GoldenParityCase:
    label: str
    query: str
    relevant_paths: tuple[str, ...]
    expected_top_path: str
    expected_prefix: tuple[str, ...]
    project_filter: tuple[str, ...] = ()


GOLDEN_PARITY_CASES = (
    GoldenParityCase(
        label="exact-heading",
        query="Authentication Overview",
        relevant_paths=("alpha/docs/authentication-overview.md",),
        expected_top_path="alpha/docs/authentication-overview.md",
        expected_prefix=(
            "alpha/docs/authentication-overview.md",
            "alpha/docs/api-authentication.md",
        ),
    ),
    GoldenParityCase(
        label="refresh-token-fact",
        query="how many days until a refresh token expires",
        relevant_paths=("alpha/docs/token-lifecycle.md",),
        expected_top_path="alpha/docs/token-lifecycle.md",
        expected_prefix=(
            "alpha/docs/token-lifecycle.md",
            "alpha/docs/authentication-overview.md",
        ),
    ),
    GoldenParityCase(
        label="api-authentication-neighborhood",
        query="bearer tokens protected endpoints",
        relevant_paths=(
            "alpha/docs/api-authentication.md",
            "alpha/docs/authentication-overview.md",
        ),
        expected_top_path="alpha/docs/api-authentication.md",
        expected_prefix=(
            "alpha/docs/api-authentication.md",
            "alpha/docs/token-lifecycle.md",
        ),
    ),
    GoldenParityCase(
        label="file-shaped-query",
        query="mcp_server.py list_tools call_tool",
        relevant_paths=("alpha/docs/src-mcp_server-py.md",),
        expected_top_path="alpha/docs/src-mcp_server-py.md",
        expected_prefix=("alpha/docs/src-mcp_server-py.md",),
    ),
    GoldenParityCase(
        label="title-over-interior-section",
        query="testing strategy",
        relevant_paths=("alpha/docs/testing-strategy.md",),
        expected_top_path="alpha/docs/testing-strategy.md",
        expected_prefix=(
            "alpha/docs/testing-strategy.md",
            "alpha/docs/development.md",
        ),
    ),
    GoldenParityCase(
        label="conceptual-refresh-token",
        query="how long do refresh tokens last",
        relevant_paths=("alpha/docs/token-lifecycle.md",),
        expected_top_path="alpha/docs/token-lifecycle.md",
        expected_prefix=(
            "alpha/docs/token-lifecycle.md",
            "alpha/docs/authentication-overview.md",
        ),
    ),
    GoldenParityCase(
        label="project-scoped-alpha",
        query="Project Rollout Checklist",
        relevant_paths=("alpha/docs/project-rollout-checklist.md",),
        expected_top_path="alpha/docs/project-rollout-checklist.md",
        expected_prefix=("alpha/docs/project-rollout-checklist.md",),
        project_filter=("alpha",),
    ),
    GoldenParityCase(
        label="project-scoped-beta",
        query="Project Rollout Checklist",
        relevant_paths=("beta/docs/project-rollout-checklist.md",),
        expected_top_path="beta/docs/project-rollout-checklist.md",
        expected_prefix=("beta/docs/project-rollout-checklist.md",),
        project_filter=("beta",),
    ),
)


async def _run_case(harness, case: GoldenParityCase):
    results, _compression, _strategy = await harness.orchestrator.query(
        case.query,
        top_k=10,
        top_n=5,
        project_filter=list(case.project_filter),
    )
    paths = tuple(harness.doc_id_to_path[result.doc_id] for result in results)
    scores = tuple(result.score for result in results)
    relevant = {
        harness.path_to_doc_id[path]
        for path in case.relevant_paths
    }
    recall_at_5 = sum(result.doc_id in relevant for result in results) / len(relevant)
    return paths, scores, recall_at_5


@pytest.mark.asyncio
async def test_golden_cases_preserve_labeled_relevance_and_raw_rrf_order(
    tmp_path: Path,
) -> None:
    harness = build_search_evaluation_harness(tmp_path)
    recalls: list[float] = []

    for case in GOLDEN_PARITY_CASES:
        paths, scores, recall_at_5 = await _run_case(harness, case)
        assert paths[0] == case.expected_top_path, (
            f"{case.label}: expected {case.expected_top_path}, got {paths}"
        )
        assert paths[: len(case.expected_prefix)] == case.expected_prefix
        assert list(scores) == sorted(scores, reverse=True)
        assert scores[0] > 0.0
        assert scores[0] < 1.0
        recalls.append(recall_at_5)

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall == 1.0
    print(f"golden parity: cases={len(recalls)} recall@5={mean_recall:.3f}")


@pytest.mark.asyncio
@pytest.mark.performance
async def test_golden_cases_report_bounded_uncached_latency(
    tmp_path: Path,
) -> None:
    harness = build_search_evaluation_harness(tmp_path)
    latencies_ms: list[float] = []
    observed_counts: list[int] = []

    for case in GOLDEN_PARITY_CASES:
        started = time.perf_counter()
        paths, scores, _recall = await _run_case(harness, case)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        observed_counts.append(len(paths))
        assert paths[0] == case.expected_top_path
        assert scores == tuple(sorted(scores, reverse=True))

    ordered = sorted(latencies_ms)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    assert p95 < 5000.0
    print(
        "golden latency: "
        f"cases={len(latencies_ms)} p95_ms={p95:.2f} "
        f"results={sum(observed_counts)}"
    )
