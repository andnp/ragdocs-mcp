from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("zmq")

from src.daemon.metadata import DaemonMetadata
from src.daemon.transport import ZMQTransportClient, ZMQTransportServer


@pytest.mark.asyncio
async def test_zmq_transport_round_trip(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = ZMQTransportServer(
        socket_path=socket_path,
        metadata_provider=lambda: DaemonMetadata(pid=123, started_at=1.0, status="ready"),
        request_handler=lambda path, payload: asyncio.sleep(
            0,
            result={"path": path, "payload": payload},
        ),
    )

    await server.start()
    try:
        client = ZMQTransportClient()
        response = await asyncio.to_thread(
            client.send_request,
            socket_path,
            "/api/example",
            {"hello": "world"},
            timeout_seconds=1.0,
        )
    finally:
        await server.stop()

    assert response["path"] == "/api/example"
    assert response["payload"] == {"hello": "world"}
    assert isinstance(response.get("request_id"), str)


@pytest.mark.asyncio
async def test_zmq_transport_health_round_trip(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    metadata = DaemonMetadata(pid=321, started_at=1.0, status="ready")
    server = ZMQTransportServer(
        socket_path=socket_path,
        metadata_provider=lambda: metadata,
    )

    await server.start()
    try:
        client = ZMQTransportClient()
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
    assert isinstance(response.get("request_id"), str)


@pytest.mark.asyncio
async def test_zmq_transport_reports_explicit_timeout(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = ZMQTransportServer(
        socket_path=socket_path,
        metadata_provider=lambda: DaemonMetadata(pid=123, started_at=1.0, status="ready"),
        request_handler=lambda path, payload: asyncio.sleep(0.2, result={"path": path}),
    )

    await server.start()
    try:
        client = ZMQTransportClient()
        response = await asyncio.to_thread(
            client.send_request,
            socket_path,
            "/api/example",
            {},
            timeout_seconds=0.01,
        )
    finally:
        await server.stop()

    assert response == {"status": "error", "error": "daemon_request_timed_out"}