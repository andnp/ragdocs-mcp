from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.context import IndexState
from src.lifecycle import LifecycleState
from src.mcp.handlers import HandlerContext
from src.mcp.tools.document_tools import (
    handle_query_documents,
    handle_search_git_history,
)


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
            return (
                [
                    SimpleNamespace(
                        file_path="docs/plan.md",
                        header_path="Overview",
                        score=0.91,
                        content="Fast cold start contract.",
                    )
                ],
                SimpleNamespace(
                    original_count=1,
                    after_threshold=1,
                    after_dedup=1,
                    after_doc_limit=1,
                    clusters_merged=0,
                ),
                None,
            )

    ready_ctx = _ColdStartContext(IndexState(status="ready"), ready=True)
    ready_ctx.orchestrator = _FakeOrchestrator()

    hctx = HandlerContext(lambda: ready_ctx, _FakeCoordinator())

    contents = await handle_query_documents(hctx, {"query": "daemon startup"})

    assert len(contents) == 1
    assert "docs/plan.md" in contents[0].text
    assert "**Status:** initializing" not in contents[0].text


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