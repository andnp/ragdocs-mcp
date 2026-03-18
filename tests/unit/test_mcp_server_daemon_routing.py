from __future__ import annotations

import pytest

from src.daemon.metadata import DaemonMetadata
from src.mcp.server import MCPServer


@pytest.mark.asyncio
async def test_mcp_server_prefers_daemon_tool_listing(monkeypatch):
    server = MCPServer(project_override="docs")

    monkeypatch.setattr(
        "src.mcp.server.start_daemon",
        lambda *, project_override, timeout_seconds=10.0, paths=None: DaemonMetadata(
            pid=1,
            started_at=1.0,
            status="ready",
            socket_path="/tmp/ragdocs.sock",
        ),
    )
    monkeypatch.setattr(
        "src.mcp.server.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds=30.0: {
            "tools": [
                {
                    "name": "query_documents",
                    "description": "remote",
                    "inputSchema": {"type": "object"},
                }
            ]
        },
    )

    tools = await server._get_remote_tools()

    assert tools is not None
    assert [tool.name for tool in tools] == ["query_documents"]


@pytest.mark.asyncio
async def test_mcp_server_prefers_daemon_tool_calls(monkeypatch):
    server = MCPServer(project_override="docs")

    monkeypatch.setattr(
        "src.mcp.server.start_daemon",
        lambda *, project_override, timeout_seconds=10.0, paths=None: DaemonMetadata(
            pid=1,
            started_at=1.0,
            status="ready",
            socket_path="/tmp/ragdocs.sock",
        ),
    )
    monkeypatch.setattr(
        "src.mcp.server.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds=30.0: {
            "contents": [{"type": "text", "text": "remote response"}]
        },
    )

    contents = await server._call_remote_tool("query_documents", {"query": "test"})

    assert contents is not None
    assert len(contents) == 1
    assert contents[0].text == "remote response"


@pytest.mark.asyncio
async def test_mcp_server_does_not_wait_for_ready_daemon_before_tool_call(monkeypatch):
    server = MCPServer(project_override="docs")

    monkeypatch.setattr(
        "src.mcp.server.start_daemon",
        lambda *, project_override, timeout_seconds=30.0, paths=None: DaemonMetadata(
            pid=1,
            started_at=1.0,
            status="initializing",
            socket_path="/tmp/ragdocs.sock",
        ),
    )

    monkeypatch.setattr(
        "src.mcp.server.wait_for_daemon_ready",
        lambda *, timeout_seconds=120.0, paths=None: (_ for _ in ()).throw(
            AssertionError("wait_for_daemon_ready should not be called")
        ),
    )
    monkeypatch.setattr(
        "src.mcp.server.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds=60.0: {
            "contents": [{"type": "text", "text": "remote response"}]
        },
    )

    contents = await server._call_remote_tool("query_documents", {"query": "test"})

    assert len(contents) == 1


@pytest.mark.asyncio
async def test_mcp_server_retries_tool_call_after_timeout(monkeypatch):
    server = MCPServer(project_override="docs")
    calls = {"count": 0}

    monkeypatch.setattr(
        "src.mcp.server.start_daemon",
        lambda *, project_override, timeout_seconds=30.0, paths=None: DaemonMetadata(
            pid=1,
            started_at=1.0,
            status="ready_primary",
            socket_path="/tmp/ragdocs.sock",
        ),
    )
    monkeypatch.setattr(
        "src.mcp.server.inspect_daemon",
        lambda paths=None: type(
            "Inspection",
            (),
            {"metadata": DaemonMetadata(pid=1, started_at=1.0, status="ready_primary", socket_path="/tmp/ragdocs.sock"), "running": True},
        )(),
    )
    monkeypatch.setattr(
        "src.mcp.server.wait_for_daemon_ready",
        lambda *, timeout_seconds=120.0, paths=None: DaemonMetadata(
            pid=1,
            started_at=1.0,
            status="ready_primary",
            socket_path="/tmp/ragdocs.sock",
        ),
    )

    def _fake_request(socket_path, path, payload, timeout_seconds=60.0):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "error", "error": "daemon_request_timed_out"}
        return {"contents": [{"type": "text", "text": "remote response"}]}

    monkeypatch.setattr("src.mcp.server.request_daemon_socket", _fake_request)

    contents = await server._call_remote_tool("query_documents", {"query": "test"})

    assert calls["count"] == 2
    assert len(contents) == 1
