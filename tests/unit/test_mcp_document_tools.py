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


def _extract_query_documents_metadata(response_text: str) -> dict[str, object]:
    marker = "<!-- query_documents_response_v1\n"
    start = response_text.index(marker) + len(marker)
    end = response_text.index("\n-->", start)
    return json.loads(response_text[start:end])


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
    assert "# Search Results" in contents[0].text
    assert "**Status:** initializing" in contents[0].text
    assert "**Query:** daemon startup" in contents[0].text
    assert "**Index State:** indexing (0/12)" in contents[0].text
    assert "**Results Returned:** 0" in contents[0].text


@pytest.mark.asyncio
async def test_query_documents_preserves_validation_errors_during_cold_start() -> None:
    hctx = HandlerContext(
        lambda: _ColdStartContext(IndexState(status="indexing")),
        _FakeCoordinator(),
    )

    contents = await handle_query_documents(hctx, {"query": ""})

    assert len(contents) == 1
    assert contents[0].text.startswith("Validation error:")


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
    assert "docs/plan.md" in contents[0].text
    assert "**Status:** initializing" not in contents[0].text
    metadata = _extract_query_documents_metadata(contents[0].text)
    assert metadata["schema_version"] == "query_documents.response.v1"
    assert metadata["results_count"] == 1
    assert metadata["observed_strategies"] == ["semantic", "keyword"]
    assert metadata["scope"] == {
        "mode": "global",
        "projects": [],
        "preferred_project": None,
        "applied_filter_projects": [],
        "applied_uplift_project": "ambient-project",
    }
    assert metadata["results"] == [
        {
            "rank": 1,
            "chunk_id": "plan_chunk_1",
            "doc_id": "plan",
            "file_path": "docs/plan.md",
            "header_path": "Overview",
            "score": 0.91,
            "project_id": "docs-project",
            "parent_chunk_id": None,
        }
    ]
    assert captured == {"project_filter": [], "project_context": None}


@pytest.mark.asyncio
async def test_query_documents_normalizes_legacy_scope_aliases() -> None:
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
            captured["query"] = query
            captured["top_k"] = top_k
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
    hctx = HandlerContext(lambda: ready_ctx, _FakeCoordinator())

    await handle_query_documents(
        hctx,
        {
            "query": "daemon startup",
            "top_n": 3,
            "project_filter": ["proj-a", "proj-b"],
            "project_context": "proj-b",
        },
    )

    assert captured == {
        "query": "daemon startup",
        "top_k": 30,
        "project_filter": ["proj-a", "proj-b"],
        "project_context": "proj-b",
    }


@pytest.mark.asyncio
async def test_query_documents_embeds_explicit_scope_and_stats_metadata() -> None:
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
            "show_stats": True,
            "scope_mode": "explicit_projects",
            "scope_projects": ["proj-a"],
            "preferred_project": "proj-a",
            "uniqueness_mode": "one_per_document",
        },
    )

    assert len(contents) == 1
    assert "# Compression Stats" in contents[0].text
    assert "- After document limit (1 per doc): 1" in contents[0].text

    metadata = _extract_query_documents_metadata(contents[0].text)
    assert metadata["query"] == "auth tokens"
    assert metadata["result_header"] == "Search Results (Unique Documents)"
    assert metadata["strategy_counts"] == {
        "semantic": 4,
        "keyword": 2,
        "graph": 1,
        "tag_expansion": 1,
    }
    assert metadata["observed_strategies"] == [
        "semantic",
        "keyword",
        "graph",
        "tag_expansion",
    ]
    assert metadata["scope"] == {
        "mode": "explicit_projects",
        "projects": ["proj-a"],
        "preferred_project": "proj-a",
        "applied_filter_projects": ["proj-a"],
        "applied_uplift_project": "proj-a",
    }
    assert metadata["compression_stats"] == {
        "original_count": 5,
        "after_threshold": 4,
        "after_content_dedup": 4,
        "after_ngram_dedup": 3,
        "after_dedup": 3,
        "after_doc_limit": 1,
        "clusters_merged": 2,
    }


def test_normalize_query_documents_request_maps_legacy_scope_aliases() -> None:
    request = normalize_query_documents_request(
        {
            "query": "daemon startup",
            "project_filter": ["proj-a", "proj-b"],
            "project_context": "proj-b",
        }
    )

    assert request.scope_mode == "explicit_projects"
    assert request.scope_projects == ("proj-a", "proj-b")
    assert request.preferred_project == "proj-b"
    assert request.project_filter == ["proj-a", "proj-b"]
    assert request.project_context == "proj-b"


def test_normalize_query_documents_request_prefers_canonical_scope_fields() -> None:
    request = normalize_query_documents_request(
        {
            "query": "daemon startup",
            "scope_mode": "global",
            "scope_projects": ["proj-c"],
            "preferred_project": "proj-c",
            "project_filter": ["proj-a"],
            "project_context": "proj-a",
        }
    )

    assert request.scope_mode == "global"
    assert request.scope_projects == ("proj-c",)
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
    assert request.result_header == "Search Results (Unique Documents)"


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