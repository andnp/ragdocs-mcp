from __future__ import annotations

from pathlib import Path

import pytest

from mcp_markdown_ragdocs.context import IndexState
from mcp_markdown_ragdocs.lifecycle import LifecycleState
from mcp_markdown_ragdocs.mcp.handlers import (
    HandlerContext,
    format_search_status_text,
)


class _Coordinator:
    state = LifecycleState.INITIALIZING

    def __init__(self) -> None:
        self.wait_calls: list[float] = []

    async def wait_ready(self, timeout: float = 60.0) -> None:
        self.wait_calls.append(timeout)


class _Context:
    def __init__(self, state: IndexState, *, ready: bool = False) -> None:
        self._state = state
        self._ready = ready
        self.documents_roots = [Path("/docs")]
        self.orchestrator = None
        self.search_use_case = None
        self.git_indexing_enabled = False

    def is_ready(self) -> bool:
        return self._ready

    def get_index_state(self) -> IndexState:
        return self._state

    def get_total_git_commits_indexed(self) -> int:
        return 4


def test_handler_context_requires_initialized_context() -> None:
    context = HandlerContext(lambda: None, _Coordinator())

    with pytest.raises(RuntimeError, match="Server not initialized"):
        context.require_ctx()


@pytest.mark.asyncio
async def test_handler_context_waits_only_when_not_ready() -> None:
    coordinator = _Coordinator()
    context = HandlerContext(
        lambda: _Context(IndexState(status="indexing")),
        coordinator,
    )

    await context.wait_for_ready(timeout=3.0)
    assert coordinator.wait_calls == [3.0]

    ready_context = HandlerContext(
        lambda: _Context(IndexState(status="ready"), ready=True),
        coordinator,
    )
    await ready_context.wait_for_ready(timeout=4.0)
    assert coordinator.wait_calls == [3.0]


def test_handler_context_reports_failed_search_state_with_git_metadata() -> None:
    context = HandlerContext(
        lambda: _Context(
            IndexState(
                status="failed",
                indexed_count=1,
                total_count=3,
                last_error="index unavailable",
            )
        ),
        _Coordinator(),
    )

    payload = context.get_nonblocking_search_payload(
        query="daemon",
        include_git_metadata=True,
    )

    assert payload is not None
    assert payload["status"] == "error"
    assert payload["details"] == "index unavailable"
    assert payload["total_commits_indexed"] == 4


def test_format_search_status_text_renders_initializing_and_error_fields() -> None:
    text = format_search_status_text(
        "Document Search",
        {
            "status": "initializing",
            "message": "Still starting",
            "query": "daemon",
            "lifecycle": "initializing",
            "configured_root_count": 2,
            "index_state": {"status": "indexing", "indexed_count": 1, "total_count": 4},
            "results": [],
            "total_commits_indexed": 9,
        },
        include_git_metadata=True,
    )

    assert "**Status:** initializing" in text
    assert "**Query:** daemon" in text
    assert "**Index State:** indexing (1/4)" in text
    assert "**Total Commits Indexed:** 9" in text
    assert "_No results yet" in text
