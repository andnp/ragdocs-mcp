from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.context import IndexState
from src.lifecycle import LifecycleState
from src.mcp.handlers import HandlerContext
from src.mcp.tools.document_request import normalize_query_documents_request
from src.mcp.tools.document_tools import (
    handle_query_documents,
    handle_search_git_history,
)
from src.models import CompressionStats, SearchStrategyStats


class _FakeCoordinator:
    state = LifecycleState.INITIALIZING


class _ColdStartContext:
    def __init__(
        self,
        index_state: IndexState,
        *,
        ready: bool = False,
        commit_count: int = 0,
    ) -> None:
        self._index_state = index_state
        self._ready = ready
        self.documents_roots = [Path("/docs")]
        self.commit_indexer = SimpleNamespace(
            get_total_commits=lambda: commit_count,
        )

    def is_ready(self) -> bool:
        return self._ready

    def get_index_state(self) -> IndexState:
        return self._index_state


def _parse_query_documents_response(response_text: str) -> dict[str, object]:
    return json.loads(response_text)


@pytest.mark.asyncio
async def test_query_documents_returns_initializing_text_on_true_cold_start() -> None:
    hctx = HandlerContext(
        lambda: _ColdStartContext(
            IndexState(status="indexing", indexed_count=0, total_count=12)
        ),
        _FakeCoordinator(),
    )

    contents = await handle_query_documents(hctx, {"query": "daemon startup"})

    assert len(contents) == 1
    payload = _parse_query_documents_response(contents[0].text)
    assert payload["schema_version"] == "query_documents.response.v2"
    assert payload["status"] == "initializing"
    assert payload["results"] == []
    assert payload["meta"]["query"] == "daemon startup"
    assert payload["meta"]["index_state"] == {
        "status": "indexing",
        "indexed_count": 0,
        "total_count": 12,
        "last_error": None,
    }


@pytest.mark.asyncio
async def test_query_documents_preserves_validation_errors_during_cold_start() -> None:
    hctx = HandlerContext(
        lambda: _ColdStartContext(IndexState(status="indexing")),
        _FakeCoordinator(),
    )

    contents = await handle_query_documents(hctx, {"query": ""})

    assert len(contents) == 1
    payload = _parse_query_documents_response(contents[0].text)
    assert payload["status"] == "error"
    assert payload["meta"]["error"] == "validation_error"
    assert "cannot be empty" in payload["meta"]["message"]


@pytest.mark.asyncio
async def test_query_documents_runs_immediately_when_indices_are_queryable() -> None:
    captured: dict[str, object] = {}

    class _FakeOrchestrator:
        documents_path = Path("/docs")

        async def query(
            self,
            query: str,
            *,
            top_k: int,
            top_n: int,
            pipeline_config,
            excluded_files,
            project_filter,
            project_context,
        ):
            assert query == "daemon startup"
            assert top_n == 5
            captured["project_filter"] = project_filter
            captured["project_context"] = project_context
            return (
                [
                    SimpleNamespace(
                        chunk_id="plan_chunk_1",
                        doc_id="plan",
                        file_path="docs/plan.md",
                        header_path="Overview",
                        score=0.91,
                        content="Fast cold start contract.",
                        project_id="docs-project",
                        parent_chunk_id=None,
                    )
                ],
                CompressionStats(
                    original_count=1,
                    after_threshold=1,
                    after_content_dedup=1,
                    after_ngram_dedup=1,
                    after_dedup=1,
                    after_doc_limit=1,
                    clusters_merged=0,
                ),
                SearchStrategyStats(
                    vector_count=1,
                    keyword_count=1,
                    graph_count=0,
                    tag_expansion_count=0,
                ),
            )

    ready_ctx = _ColdStartContext(IndexState(status="ready"), ready=True)
    ready_ctx.orchestrator = _FakeOrchestrator()
    ready_ctx.config = SimpleNamespace(detected_project="ambient-project")

    hctx = HandlerContext(lambda: ready_ctx, _FakeCoordinator())

    contents = await handle_query_documents(hctx, {"query": "daemon startup"})

    assert len(contents) == 1
    payload = _parse_query_documents_response(contents[0].text)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == "query_documents.response.v2"
    assert payload["meta"]["results_count"] == 1
    assert payload["meta"]["observed_strategies"] == ["semantic", "keyword"]
    assert payload["meta"]["scope"] == {
        "mode": "global",
        "projects": [],
        "preferred_project": None,
        "applied_filter_projects": [],
        "applied_uplift_project": None,
    }
    assert payload["results"] == [
        {
            "rank": 1,
            "chunk_id": "plan_chunk_1",
            "doc_id": "plan",
            "file_path": "docs/plan.md",
            "header_path": "Overview",
            "score": 0.91,
            "content": "Fast cold start contract.",
            "project_id": "docs-project",
            "parent_chunk_id": None,
        }
    ]
    assert captured == {"project_filter": [], "project_context": None}


@pytest.mark.asyncio
async def test_query_documents_rejects_legacy_scope_aliases() -> None:
    hctx = HandlerContext(
        lambda: _ColdStartContext(IndexState(status="indexing")),
        _FakeCoordinator(),
    )

    contents = await handle_query_documents(
        hctx,
        {
            "query": "daemon startup",
            "project_filter": ["proj-a", "proj-b"],
            "project_context": "proj-b",
        },
    )

    payload = _parse_query_documents_response(contents[0].text)
    assert payload["status"] == "error"
    assert payload["meta"]["error"] == "validation_error"
    assert (
        payload["meta"]["message"]
        == "Unexpected parameter(s): project_context, project_filter. query_documents now accepts canonical scope fields only"
    )


@pytest.mark.asyncio
async def test_query_documents_returns_canonical_scope_and_meta() -> None:
    class _FakeOrchestrator:
        documents_path = Path("/docs")

        async def query(
            self,
            query: str,
            *,
            top_k: int,
            top_n: int,
            pipeline_config,
            excluded_files,
            project_filter,
            project_context,
        ):
            return (
                [
                    SimpleNamespace(
                        chunk_id="auth_chunk_2",
                        doc_id="auth-guide",
                        file_path="docs/auth.md",
                        header_path="Authentication > Tokens",
                        score=0.88,
                        content="Token exchange details.",
                        project_id="proj-a",
                        parent_chunk_id="auth_parent_1",
                    )
                ],
                CompressionStats(
                    original_count=5,
                    after_threshold=4,
                    after_content_dedup=4,
                    after_ngram_dedup=3,
                    after_dedup=3,
                    after_doc_limit=1,
                    clusters_merged=2,
                ),
                SearchStrategyStats(
                    vector_count=4,
                    keyword_count=2,
                    graph_count=1,
                    tag_expansion_count=1,
                ),
            )

    ready_ctx = _ColdStartContext(IndexState(status="ready"), ready=True)
    ready_ctx.orchestrator = _FakeOrchestrator()
    ready_ctx.config = SimpleNamespace(detected_project="ambient-project")

    hctx = HandlerContext(lambda: ready_ctx, _FakeCoordinator())

    contents = await handle_query_documents(
        hctx,
        {
            "query": "auth tokens",
            "scope_mode": "explicit_projects",
            "scope_projects": ["proj-a"],
            "preferred_project": "proj-a",
            "uniqueness_mode": "one_per_document",
        },
    )

    assert len(contents) == 1
    payload = _parse_query_documents_response(contents[0].text)
    assert payload["meta"]["query"] == "auth tokens"
    assert payload["meta"]["uniqueness_mode"] == "one_per_document"
    assert payload["meta"]["strategy_counts"] == {
        "semantic": 4,
        "keyword": 2,
        "graph": 1,
        "tag_expansion": 1,
    }
    assert payload["meta"]["observed_strategies"] == [
        "semantic",
        "keyword",
        "graph",
        "tag_expansion",
    ]
    assert payload["meta"]["scope"] == {
        "mode": "explicit_projects",
        "projects": ["proj-a"],
        "preferred_project": "proj-a",
        "applied_filter_projects": ["proj-a"],
        "applied_uplift_project": "proj-a",
    }
    assert payload["meta"]["compression"] == {
        "original_count": 5,
        "after_threshold": 4,
        "after_content_dedup": 4,
        "after_ngram_dedup": 3,
        "after_dedup": 3,
        "after_doc_limit": 1,
        "clusters_merged": 2,
    }


def test_normalize_query_documents_request_rejects_scope_projects_outside_explicit_mode() -> None:
    with pytest.raises(ValueError, match="scope_projects may only be provided"):
        normalize_query_documents_request(
            {
                "query": "daemon startup",
                "scope_mode": "global",
                "scope_projects": ["proj-c"],
            }
        )


def test_normalize_query_documents_request_rejects_preferred_project_in_global_mode() -> None:
    with pytest.raises(ValueError, match="preferred_project may only be provided"):
        normalize_query_documents_request(
            {
                "query": "daemon startup",
                "scope_mode": "global",
                "preferred_project": "proj-c",
            }
        )


def test_normalize_query_documents_request_requires_scope_projects_for_explicit_mode() -> None:
    with pytest.raises(ValueError, match="scope_projects must be provided"):
        normalize_query_documents_request(
            {
                "query": "daemon startup",
                "scope_mode": "explicit_projects",
            }
        )


def test_normalize_query_documents_request_accepts_active_project_scope() -> None:
    request = normalize_query_documents_request(
        {
            "query": "daemon startup",
            "scope_mode": "active_project",
            "preferred_project": "proj-c",
        }
    )

    assert request.scope_mode == "active_project"
    assert request.scope_projects == ()
    assert request.preferred_project == "proj-c"
    assert request.project_filter == []
    assert request.project_context == "proj-c"


def test_normalize_query_documents_request_accepts_canonical_explicit_scope() -> None:
    request = normalize_query_documents_request(
        {
            "query": "daemon startup",
            "scope_mode": "explicit_projects",
            "scope_projects": ["proj-a", "proj-b"],
            "preferred_project": "proj-b",
            "uniqueness_mode": "one_per_document",
        }
    )

    assert request.scope_mode == "explicit_projects"
    assert request.scope_projects == ("proj-a", "proj-b")
    assert request.preferred_project == "proj-b"
    assert request.project_filter == ["proj-a", "proj-b"]
    assert request.project_context == "proj-b"
    assert request.max_chunks_per_doc == 1


@pytest.mark.asyncio
async def test_query_documents_uses_detected_project_for_active_project_scope() -> None:
    captured: dict[str, object] = {}

    class _FakeOrchestrator:
        documents_path = Path("/docs")

        async def query(
            self,
            query: str,
            *,
            top_k: int,
            top_n: int,
            pipeline_config,
            excluded_files,
            project_filter,
            project_context,
        ):
            captured["project_filter"] = project_filter
            captured["project_context"] = project_context
            return (
                [],
                CompressionStats(
                    original_count=0,
                    after_threshold=0,
                    after_content_dedup=0,
                    after_ngram_dedup=0,
                    after_dedup=0,
                    after_doc_limit=0,
                    clusters_merged=0,
                ),
                SearchStrategyStats(
                    vector_count=0,
                    keyword_count=0,
                    graph_count=0,
                    tag_expansion_count=0,
                ),
            )

    ready_ctx = _ColdStartContext(IndexState(status="ready"), ready=True)
    ready_ctx.orchestrator = _FakeOrchestrator()
    ready_ctx.config = SimpleNamespace(detected_project="ambient-project")
    hctx = HandlerContext(lambda: ready_ctx, _FakeCoordinator())

    contents = await handle_query_documents(
        hctx,
        {
            "query": "daemon startup",
            "scope_mode": "active_project",
        },
    )

    payload = _parse_query_documents_response(contents[0].text)
    assert payload["meta"]["scope"] == {
        "mode": "active_project",
        "projects": [],
        "preferred_project": None,
        "applied_filter_projects": [],
        "applied_uplift_project": "ambient-project",
    }
    assert captured == {"project_filter": [], "project_context": "ambient-project"}


@pytest.mark.asyncio
async def test_search_git_history_returns_initializing_text_on_true_cold_start() -> None:
    hctx = HandlerContext(
        lambda: _ColdStartContext(
            IndexState(status="indexing", indexed_count=0, total_count=8),
            commit_count=7,
        ),
        _FakeCoordinator(),
    )

    contents = await handle_search_git_history(hctx, {"query": "daemon"})

    assert len(contents) == 1
    assert "# Git History Search Results" in contents[0].text
    assert "**Status:** initializing" in contents[0].text
    assert "**Total Commits Indexed:** 7" in contents[0].text
    assert "**Results Returned:** 0" in contents[0].text