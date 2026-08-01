from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import TextContent

from mcp_markdown_ragdocs.daemon.mcp_requests import (
    build_mcp_tools_payload,
    handle_mcp_tool_call,
)
from mcp_markdown_ragdocs.lifecycle import LifecycleState


class _FakeCoordinator:
    state = LifecycleState.READY

    async def wait_ready(self, timeout: float = 60.0) -> None:
        return None


class _FakeContext:
    documents_roots = [Path("/docs")]
    orchestrator: Any = None
    git_indexing_enabled = False

    def is_ready(self) -> bool:
        return True

    def get_index_state(self) -> Any:
        return None

    def get_total_git_commits_indexed(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_returns_contents(monkeypatch) -> None:
    ctx = _FakeContext()

    async def _fake_handler(hctx, arguments: dict[str, object]) -> list[TextContent]:
        assert arguments == {"query": "daemon"}
        assert hctx.require_ctx() is ctx
        return [TextContent(type="text", text="ok")]

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.mcp_requests.get_handler",
        lambda name: _fake_handler if name == "query_documents" else None,
    )

    payload = await handle_mcp_tool_call(
        ctx_getter=lambda: ctx,
        coordinator=_FakeCoordinator(),
        payload={"name": "query_documents", "arguments": {"query": "daemon"}},
    )

    assert payload == {
        "contents": [{"type": "text", "text": "ok"}],
    }


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_validates_arguments_object(monkeypatch) -> None:
    ctx = _FakeContext()
    monkeypatch.setattr("mcp_markdown_ragdocs.daemon.mcp_requests.get_handler", lambda name: None)

    payload = await handle_mcp_tool_call(
        ctx_getter=lambda: ctx,
        coordinator=_FakeCoordinator(),
        payload={"name": "query_documents", "arguments": "bad"},
    )

    assert payload == {
        "status": "error",
        "error": "tool_arguments_must_be_object",
    }


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_returns_unknown_tool_error(monkeypatch) -> None:
    ctx = _FakeContext()
    monkeypatch.setattr("mcp_markdown_ragdocs.daemon.mcp_requests.get_handler", lambda name: None)

    payload = await handle_mcp_tool_call(
        ctx_getter=lambda: ctx,
        coordinator=_FakeCoordinator(),
        payload={"name": "missing_tool", "arguments": {}},
    )

    assert payload == {
        "status": "error",
        "error": "unknown_tool:missing_tool",
    }


def test_build_mcp_tools_payload_uses_registered_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.daemon.mcp_requests.get_document_tools",
        lambda: [
            SimpleNamespace(
                name="query_documents",
                description="Search docs",
                inputSchema={"type": "object"},
            )
        ],
    )

    payload = build_mcp_tools_payload()

    assert payload == {
        "tools": [
            {
                "name": "query_documents",
                "description": "Search docs",
                "inputSchema": {"type": "object"},
            }
        ]
    }