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
        lambda socket_path, path, payload: {
            "tools": [
                {
                    "name": "query_documents",
                    "description": "remote",
                    "inputSchema": {"type": "object"},
                }
            ]
        },
    )

    tools = await server._maybe_get_remote_tools()

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
        lambda socket_path, path, payload: {
            "contents": [{"type": "text", "text": "remote response"}]
        },
    )

    contents = await server._maybe_call_remote_tool("query_documents", {"query": "test"})

    assert contents is not None
    assert len(contents) == 1
    assert contents[0].text == "remote response"