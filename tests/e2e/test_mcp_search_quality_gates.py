from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mcp_markdown_ragdocs.context import IndexState
from mcp_markdown_ragdocs.config import SearchConfig
from mcp_markdown_ragdocs.lifecycle import LifecycleState
from mcp_markdown_ragdocs.mcp.handlers import HandlerContext, get_handler
from mcp_markdown_ragdocs.mcp.tools.document_tools import handle_query_documents
from tests.search.evaluation_harness import (
    SearchEvaluationCaseResult,
    SearchEvaluationAggregate,
    SearchEvaluationReport,
    build_search_evaluation_harness,
    compute_ranking_metrics,
)


class _ReadyCoordinator:
    state = LifecycleState.READY

    async def wait_ready(self, timeout: float = 60.0) -> None:
        return None


def _ready_context(harness: Any) -> HandlerContext:
    context = SimpleNamespace(
        config=SimpleNamespace(detected_project=None),
        documents_roots=[harness.corpus_root],
        git_indexing_enabled=False,
        orchestrator=harness.orchestrator,
        is_ready=lambda: True,
        get_index_state=lambda: IndexState(status="ready"),
    )
    return HandlerContext(cast(Any, lambda: context), _ReadyCoordinator())


async def _query(hctx: HandlerContext, arguments: dict[str, object]) -> dict[str, Any]:
    handler = get_handler("query_documents")
    assert handler is not None, "query_documents must remain registered"
    assert handler is handle_query_documents, "query_documents must remain registered"
    contents = await handler(hctx, arguments)
    assert len(contents) == 1
    return json.loads(contents[0].text)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_query_documents_retains_golden_quality_metrics(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)
    hctx = _ready_context(harness)
    case_results: list[SearchEvaluationCaseResult] = []

    for case in harness.cases:
        arguments: dict[str, object] = {"query": case.query, "top_n": case.top_n}
        if case.min_score is not None:
            arguments["min_score"] = case.min_score
        if case.project_context is not None:
            arguments.update(
                {
                    "scope_mode": "explicit_projects",
                    "scope_projects": [case.project_context],
                    "preferred_project": case.project_context,
                }
            )

        payload = await _query(hctx, arguments)
        assert payload["status"] == "ok"
        results = payload["results"]
        ranked_doc_ids = tuple(result["doc_id"] for result in results)
        ranked_paths = tuple(
            harness.doc_id_to_path.get(doc_id, f"<unknown:{doc_id}>")
            for doc_id in ranked_doc_ids
        )
        relevant_doc_ids = tuple(
            harness.path_to_doc_id[path] for path in case.relevant_paths
        )
        case_results.append(
            SearchEvaluationCaseResult(
                case=case,
                relevant_doc_ids=relevant_doc_ids,
                ranked_doc_ids=ranked_doc_ids,
                ranked_paths=ranked_paths,
                metrics=compute_ranking_metrics(ranked_doc_ids, relevant_doc_ids),
            )
        )

    report = SearchEvaluationReport(
        case_results=tuple(case_results),
        aggregate=_aggregate(case_results),
    )
    assert report.expectation_failures() == []
    print(report.format_summary())


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_query_documents_returns_unique_documents_without_debug_metadata(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)
    payload = await _query(
        _ready_context(harness),
        {
            "query": "Authentication Overview",
            "top_n": 5,
            "uniqueness_mode": "one_per_document",
        },
    )

    assert payload["status"] == "ok"
    results = payload["results"]
    doc_ids = [result["doc_id"] for result in results]
    assert len(doc_ids) == len(set(doc_ids))
    assert "meta" not in payload
    assert all("provenance" not in result for result in results)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_query_documents_handles_arbitrary_punctuation(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)
    payload = await _query(
        _ready_context(harness),
        {"query": "'''((((( !!! ??? :::", "top_n": 3},
    )

    assert payload["status"] == "ok"
    assert "meta" not in payload


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mcp_query_documents_does_not_leak_dotted_fts_failure(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)
    payload = await _query(
        _ready_context(harness),
        {"query": "mcp_server.py list_tools call_tool", "top_n": 3},
    )

    assert payload["status"] == "ok"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_artifact_query_returns_markdown_reference(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    results, _, strategy = await harness.orchestrator.query(
        "mcp_server.py list_tools call_tool",
        top_k=20,
        top_n=3,
    )

    assert results
    assert results[0].file_path.endswith("src-mcp_server-py.md")
    assert all(result.file_path.endswith(".md") for result in results)
    assert strategy.keyword_count is not None and strategy.keyword_count > 0
    assert results[0].provenance is not None
    assert "keyword" in results[0].provenance.strategies


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_graph_retrieval_reports_graph_strategy(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    results, _, strategy = await harness.orchestrator.query(
        "what links to token lifecycle",
        top_k=20,
        top_n=5,
    )

    assert strategy.graph_count is not None
    assert strategy.graph_count > 0
    assert results[0].doc_id.endswith("token-lifecycle")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_scoped_keyword_search_stays_in_requested_project(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    results, _, _ = await harness.orchestrator.query(
        "Project Rollout Checklist",
        top_k=10,
        top_n=5,
        project_filter=["beta"],
        source_filter=["note"],
    )

    assert results
    assert all(result.project_id == "beta" for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_typo_tolerance_keeps_authentication_result_visible(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    results, _, _ = await harness.orchestrator.query(
        "authentcation overview",
        top_k=10,
        top_n=3,
    )

    assert results
    assert results[0].file_path.endswith("authentication-overview.md")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_default_search_diversifies_documents(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    results, _, _ = await harness.orchestrator.query(
        "authentication",
        top_k=20,
        top_n=5,
    )

    doc_ids = [result.doc_id for result in results]
    assert len(doc_ids) == len(set(doc_ids))


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_no_answer_threshold_returns_empty(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    results, _, _ = await harness.orchestrator.query(
        "quantum teleportation safety protocol",
        top_k=20,
        top_n=5,
        min_score=0.02,
    )

    assert results == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_abstention_rejects_unrelated_and_keeps_hybrid_match(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    unrelated, _, _ = await harness.orchestrator.query(
        "quantum teleportation safety protocol",
        top_k=20,
        top_n=5,
    )
    relevant, _, strategy = await harness.orchestrator.query(
        "how frequently must access credentials be renewed",
        top_k=20,
        top_n=5,
        min_score=0.03,
    )

    assert unrelated == []
    assert relevant
    assert relevant[0].file_path.endswith("token-lifecycle.md")
    assert relevant[0].score >= 0.03
    provenance = relevant[0].provenance
    assert provenance is not None
    assert {"keyword", "vector"} <= set(provenance.strategies)
    assert strategy.keyword_count is not None and strategy.keyword_count > 0
    assert strategy.vector_count is not None and strategy.vector_count > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fixture_global_scores_are_monotonic(tmp_path) -> None:
    harness = build_search_evaluation_harness(tmp_path)

    results, _, _ = await harness.orchestrator.query(
        "authentication",
        top_k=20,
        top_n=5,
    )

    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_configured_abstention_filters_unrelated_fixture_results(tmp_path) -> None:
    harness = build_search_evaluation_harness(
        tmp_path,
        search_config=SearchConfig(abstention_threshold=0.02),
    )

    relevant, _, _ = await harness.orchestrator.query(
        "Authentication Overview",
        top_k=10,
        top_n=3,
    )
    unrelated, _, _ = await harness.orchestrator.query(
        "quantum teleportation safety protocol",
        top_k=10,
        top_n=3,
    )

    assert relevant
    assert relevant[0].file_path.endswith("authentication-overview.md")
    assert unrelated == []


def _aggregate(results: list[SearchEvaluationCaseResult]) -> SearchEvaluationAggregate:
    count = len(results)
    return SearchEvaluationAggregate(
        query_count=count,
        mrr=sum(r.metrics.reciprocal_rank for r in results) / count,
        recall_at_1=sum(r.metrics.recall_at_1 for r in results) / count,
        recall_at_3=sum(r.metrics.recall_at_3 for r in results) / count,
        recall_at_5=sum(r.metrics.recall_at_5 for r in results) / count,
        ndcg_at_3=sum(r.metrics.ndcg_at_3 for r in results) / count,
        ndcg_at_5=sum(r.metrics.ndcg_at_5 for r in results) / count,
    )
