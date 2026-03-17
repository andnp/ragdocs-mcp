from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.daemon.metadata import DaemonMetadata
from src.daemon.transport import UnixSocketTransportClient, UnixSocketTransportServer


@pytest.mark.asyncio
async def test_unix_socket_transport_round_trip(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = UnixSocketTransportServer(
        socket_path=socket_path,
        metadata_provider=lambda: DaemonMetadata(pid=123, started_at=1.0, status="ready"),
        request_handler=lambda path, payload: asyncio.sleep(
            0,
            result={"path": path, "payload": payload},
        ),
    )

    await server.start()
    try:
        client = UnixSocketTransportClient()
        response = await asyncio.to_thread(
            client.send_request,
            socket_path,
            "/api/example",
            {"hello": "world"},
            timeout_seconds=1.0,
        )
    finally:
        await server.stop()

    assert response == {"path": "/api/example", "payload": {"hello": "world"}}


@pytest.mark.asyncio
async def test_unix_socket_transport_health_round_trip(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    metadata = DaemonMetadata(pid=321, started_at=1.0, status="ready")
    server = UnixSocketTransportServer(
        socket_path=socket_path,
        metadata_provider=lambda: metadata,
    )

    await server.start()
    try:
        client = UnixSocketTransportClient()
        response = await asyncio.to_thread(
            client.send_request,
            socket_path,
            "/internal/health",
            {},
            timeout_seconds=1.0,
        )
    finally:
        await server.stop()

    assert response["pid"] == metadata.pid
    assert response["status"] == metadata.status