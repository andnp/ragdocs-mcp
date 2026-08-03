from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mcp_markdown_ragdocs.context import IndexState
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
