from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.daemon.health import DaemonHealthServer, probe_daemon_socket, request_daemon_socket
from src.daemon.metadata import DaemonMetadata


@pytest.mark.asyncio
async def test_health_server_responds_with_metadata(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    metadata = DaemonMetadata(pid=321, started_at=1.0, status="ready")
    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: metadata,
    )

    await server.start()
    try:
        probed = await asyncio.to_thread(probe_daemon_socket, socket_path)
    finally:
        await server.stop()

    assert probed == metadata


@pytest.mark.asyncio
async def test_health_server_removes_socket_on_stop(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: None,
    )

    await server.start()
    assert socket_path.exists()

    await server.stop()

    assert not socket_path.exists()


@pytest.mark.asyncio
async def test_health_server_dispatches_custom_request_handler(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: None,
        request_handler=lambda path, payload: asyncio.sleep(0, result={"path": path, "payload": payload}),
    )

    await server.start()
    try:
        response = await asyncio.to_thread(
            request_daemon_socket,
            socket_path,
            "/api/example",
            {"hello": "world"},
        )
    finally:
        await server.stop()

    assert response == {"path": "/api/example", "payload": {"hello": "world"}}


@pytest.mark.asyncio
async def test_request_daemon_socket_supports_slow_handler_with_custom_timeout(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: None,
        request_handler=lambda path, payload: asyncio.sleep(
            0.05,
            result={"path": path, "payload": payload},
        ),
    )

    await server.start()
    try:
        response = await asyncio.to_thread(
            request_daemon_socket,
            socket_path,
            "/api/slow",
            {"hello": "world"},
            timeout_seconds=0.2,
        )
    finally:
        await server.stop()

    assert response == {"path": "/api/slow", "payload": {"hello": "world"}}


@pytest.mark.asyncio
async def test_request_daemon_socket_reports_timeout_as_unavailable(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: None,
        request_handler=lambda path, payload: asyncio.sleep(
            0.05,
            result={"path": path, "payload": payload},
        ),
    )

    await server.start()
    try:
        response = await asyncio.to_thread(
            request_daemon_socket,
            socket_path,
            "/api/slow",
            {"hello": "world"},
            timeout_seconds=0.001,
        )
    finally:
        await server.stop()

    assert response == {"status": "error", "error": "daemon_socket_unavailable"}


@pytest.mark.asyncio
async def test_health_server_returns_error_payload_for_handler_exception(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "daemon.sock"

    async def _raise(path: str, payload: dict[str, object]) -> dict[str, object]:
        raise ValueError("boom")

    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: None,
        request_handler=_raise,
    )

    await server.start()
    try:
        response = await asyncio.to_thread(
            request_daemon_socket,
            socket_path,
            "/api/example",
            {"hello": "world"},
        )
    finally:
        await server.stop()

    assert response["status"] == "error"
    assert response["error"] == "handler_exception"
    assert response["details"] == "boom"


@pytest.mark.asyncio
async def test_health_server_survives_handler_exception_for_later_requests(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "daemon.sock"

    async def _handler(path: str, payload: dict[str, object]) -> dict[str, object]:
        if path == "/api/fail":
            raise ValueError("boom")
        return {"path": path, "payload": payload}

    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: None,
        request_handler=_handler,
    )

    await server.start()
    try:
        failed = await asyncio.to_thread(
            request_daemon_socket,
            socket_path,
            "/api/fail",
            {},
        )
        succeeded = await asyncio.to_thread(
            request_daemon_socket,
            socket_path,
            "/api/ok",
            {"hello": "world"},
        )
    finally:
        await server.stop()

    assert failed["error"] == "handler_exception"
    assert succeeded == {"path": "/api/ok", "payload": {"hello": "world"}}


@pytest.mark.asyncio
async def test_request_daemon_socket_reads_large_response(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    large_text = "x" * 5000
    server = DaemonHealthServer(
        socket_path=socket_path,
        metadata_provider=lambda: None,
        request_handler=lambda path, payload: asyncio.sleep(
            0,
            result={"path": path, "payload": payload, "content": large_text},
        ),
    )

    await server.start()
    try:
        response = await asyncio.to_thread(
            request_daemon_socket,
            socket_path,
            "/api/large",
            {"hello": "world"},
        )
    finally:
        await server.stop()

    assert response["path"] == "/api/large"
    assert response["content"] == large_text