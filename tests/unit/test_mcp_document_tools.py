from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_markdown_ragdocs.context import IndexState
from mcp_markdown_ragdocs.lifecycle import LifecycleState
from mcp_markdown_ragdocs.mcp.handlers import HandlerContext
from mcp_markdown_ragdocs.mcp.tools.document_request import normalize_query_documents_request
from mcp_markdown_ragdocs.mcp.tools.document_tools import (
    handle_query_documents,
    handle_search_with_hypothesis,
    handle_search_git_history,
)
from mcp_markdown_ragdocs.models import (
    CompressionStats,
    SearchStrategyStats,
)


class _FakeCoordinator:
    state = LifecycleState.INITIALIZING

    async def wait_ready(self, timeout: float = 60.0) -> None:
        return None


class _ColdStartContext:
    orchestrator: Any
    config: Any
    git_indexing_enabled = False

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
        self._commit_count = commit_count

    def get_total_git_commits_indexed(self) -> int:
        return self._commit_count

    def is_ready(self) -> bool:
        return self._ready

    def get_index_state(self) -> IndexState:
        return self._index_state


def _parse_query_documents_response(response_text: str) -> dict[str, Any]:
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
    assert "schema_version" not in payload
    assert payload == {
        "status": "initializing",
        "message": "Search indices are still initializing. Retry shortly.",
        "results": [],
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
    assert payload["error"] == "validation_error"
    assert "cannot be empty" in payload["message"]


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
            min_score,
            similarity_threshold,
            max_chunks_per_doc,
        ):
            assert query == "daemon startup"
            assert top_n == 5
            captured["project_filter"] = project_filter
            captured["project_context"] = project_context
            captured["min_score"] = min_score
            captured["similarity_threshold"] = similarity_threshold
            captured["max_chunks_per_doc"] = max_chunks_per_doc
            return (
                [
                    SimpleNamespace(
                        chunk_id="plan_chunk_1",
                        record_id="plan",
                        score=0.91,
                        content="Fast cold start contract.",
                        parent_chunk_id=None,
                        parent_content=None,
                        provenance=None,
                        metadata={
                            "file_path": "docs/plan.md",
                            "header_path": "Overview",
                            "project_id": "docs-project",
                        },
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
    assert "schema_version" not in payload
    assert "meta" not in payload
    assert payload["results"] == [
        {
            "chunk_id": "plan_chunk_1",
            "doc_id": "plan",
            "file_path": "docs/plan.md",
            "header_path": "Overview",
            "score": 0.91,
            "content": "Fast cold start contract.",
            "project_id": "docs-project",
        }
    ]

    assert captured == {
        "project_filter": [],
        "project_context": None,
        "min_score": None,
        "similarity_threshold": 0.85,
        "max_chunks_per_doc": 1,
    }


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
    assert payload["error"] == "validation_error"
    assert (
        payload["message"]
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
            min_score,
            similarity_threshold,
            max_chunks_per_doc,
        ):
            return (
                [
                    SimpleNamespace(
                        chunk_id="auth_chunk_2",
                        record_id="auth-guide",
                        score=0.88,
                        content="Token exchange details.",
                        parent_chunk_id="auth_parent_1",
                        parent_content=None,
                        provenance=None,
                        metadata={
                            "file_path": "docs/auth.md",
                            "header_path": "Authentication > Tokens",
                            "project_id": "proj-a",
                        },
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
    assert "meta" not in payload
    assert payload["results"][0]["doc_id"] == "auth-guide"


def test_normalize_query_documents_request_rejects_scope_projects_outside_explicit_mode() -> None:
    with pytest.raises(ValueError, match="scope_projects may only be provided"):
        normalize_query_documents_request(
            {
                "query": "daemon startup",
                "scope_mode": "global",
                "scope_projects": ["proj-c"],
            }
        )


def test_normalize_query_documents_request_preserves_min_score_omission() -> None:
    omitted = normalize_query_documents_request({"query": "daemon startup"})
    explicit = normalize_query_documents_request(
        {"query": "daemon startup", "min_score": 0.0}
    )

    assert omitted.min_score is None
    assert explicit.min_score == 0.0


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
            min_score,
            similarity_threshold,
            max_chunks_per_doc,
        ):
            captured["project_filter"] = project_filter
            captured["project_context"] = project_context
            captured["min_score"] = min_score
            captured["similarity_threshold"] = similarity_threshold
            captured["max_chunks_per_doc"] = max_chunks_per_doc
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
    assert "meta" not in payload
    assert captured == {
        "project_filter": [],
        "project_context": "ambient-project",
        "min_score": None,
        "similarity_threshold": 0.85,
        "max_chunks_per_doc": 1,
    }


@pytest.mark.asyncio
async def test_hyde_handler_normalizes_exclusions_for_each_document_root() -> None:
    captured: dict[str, object] = {}

    class _FakeOrchestrator:
        documents_path = Path("/repo-one")

        async def query_with_hypothesis(
            self,
            hypothesis: str,
            *,
            top_k: int,
            top_n: int,
            excluded_files,
            project_filter,
            project_context,
        ):
            captured["hypothesis"] = hypothesis
            captured["excluded_files"] = excluded_files
            return (
                [],
                CompressionStats(0, 0, 0, 0, 0, 0, 0),
                SearchStrategyStats(0, 0, 0, 0),
            )

    ready_ctx = _ColdStartContext(IndexState(status="ready"), ready=True)
    ready_ctx.documents_roots = [Path("/repo-one"), Path("/repo-two")]
    ready_ctx.orchestrator = _FakeOrchestrator()
    ready_ctx.config = SimpleNamespace(detected_project=None)
    hctx = HandlerContext(lambda: ready_ctx, _FakeCoordinator())

    contents = await handle_search_with_hypothesis(
        hctx,
        {
            "hypothesis": "authentication setup",
            "excluded_files": ["/repo-two/private.md"],
        },
    )

    assert len(contents) == 1
    assert captured["hypothesis"] == "authentication setup"
    excluded_files = captured["excluded_files"]
    assert isinstance(excluded_files, set)
    assert "private" in excluded_files


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
    payload = json.loads(contents[0].text)
    assert payload == {
        "status": "initializing",
        "message": "Search indices are still initializing. Retry shortly.",
        "results": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"after_timestamp": "not-an-integer"},
            "after_timestamp must be an integer, got str",
        ),
        (
            {"after_timestamp": 100, "before_timestamp": 100},
            "after_timestamp must be less than before_timestamp",
        ),
    ],
)
async def test_search_git_history_validates_timestamp_filters(arguments, message) -> None:
    hctx = HandlerContext(
        lambda: _ColdStartContext(IndexState(status="indexing")),
        _FakeCoordinator(),
    )

    contents = await handle_search_git_history(
        hctx,
        {"query": "daemon", **arguments},
    )

    payload = json.loads(contents[0].text)
    assert payload["status"] == "error"
    assert payload["message"] == message


@pytest.mark.asyncio
async def test_search_git_history_compacts_results_and_opts_into_diff() -> None:
    class _FakeOrchestrator:
        async def query(self, *args, **kwargs):
            return (
                [
                    SimpleNamespace(
                        chunk_id="git-chunk",
                        record_id="git:abc123",
                        score=0.7,
                        content="@@ -1 +1 @@\n-old\n+new",
                        parent_chunk_id=None,
                        parent_content=None,
                        provenance=None,
                        metadata={
                            "title": "Update docs",
                            "author": "Andy",
                            "timestamp": 123,
                            "files_changed": ["README.md"],
                            "chunk_section": "diff",
                        },
                    )
                ],
                None,
                None,
            )

    ctx = _ColdStartContext(IndexState(status="ready"), ready=True, commit_count=1)
    ctx.git_indexing_enabled = True
    ctx.orchestrator = _FakeOrchestrator()
    hctx = HandlerContext(lambda: ctx, _FakeCoordinator())

    default_payload = json.loads(
        (await handle_search_git_history(hctx, {"query": "docs"}))[0].text
    )
    debug_payload = json.loads(
        (
            await handle_search_git_history(
                hctx, {"query": "docs", "include_diff": True}
            )
        )[0].text
    )

    assert "diff" not in default_payload["results"][0]
    assert debug_payload["results"][0]["diff"] == "@@ -1 +1 @@\n-old\n+new"
