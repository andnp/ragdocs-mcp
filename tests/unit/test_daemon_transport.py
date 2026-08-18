from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("zmq")

from mcp_markdown_ragdocs.daemon.metadata import DaemonMetadata
from mcp_markdown_ragdocs.daemon.transport import ZMQTransportClient, ZMQTransportServer


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


def test_zmq_transport_client_reuses_shared_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_markdown_ragdocs.daemon.transport as transport_module

    created_contexts: list[_FakeContext] = []

    class _FakeSocket:
        def __init__(self) -> None:
            self.closed = False
            self.linger = 0

        def connect(self, endpoint: str) -> None:
            return None

        def send(self, data: bytes) -> None:
            return None

        def recv_multipart(self) -> list[bytes]:
            return [b"", b'{"status": "ok"}']

        def close(self, linger: int = -1) -> None:
            self.closed = True

    class _FakePoller:
        def __init__(self) -> None:
            self._socket: _FakeSocket | None = None

        def register(self, socket: _FakeSocket, flags: int) -> None:
            self._socket = socket

        def poll(self, timeout: int) -> list[tuple[_FakeSocket, int]]:
            assert self._socket is not None
            return [(self._socket, 1)]

    class _FakeContext:
        def __init__(self) -> None:
            self.sockets: list[_FakeSocket] = []
            self.terminated = False
            created_contexts.append(self)

        def socket(self, socket_type: int) -> _FakeSocket:
            sock = _FakeSocket()
            self.sockets.append(sock)
            return sock

        def term(self) -> None:
            self.terminated = True

    class _FakeZMQ:
        DEALER = 1
        POLLIN = 1
        Poller = _FakePoller
        Context = _FakeContext

    monkeypatch.setattr(transport_module, "_require_zmq", lambda: (_FakeZMQ(), None))
    monkeypatch.setattr(transport_module, "_shared_client_context", None)

    client = ZMQTransportClient()
    for _ in range(2):
        response = client.send_request(
            Path("/tmp/unused.sock"),
            "/api/example",
            {},
            timeout_seconds=1.0,
        )
        assert response == {"status": "ok"}

    assert len(created_contexts) == 1
    context = created_contexts[0]
    assert context.terminated is False
    assert len(context.sockets) == 2
    assert all(sock.closed for sock in context.sockets)
