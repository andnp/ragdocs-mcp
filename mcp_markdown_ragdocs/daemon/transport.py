from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

from mcp_markdown_ragdocs.daemon.metadata import DaemonMetadata

logger = logging.getLogger(__name__)


class DaemonTransportServer(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class DaemonTransportClient(Protocol):
    def send_request(
        self,
        socket_path: Path,
        path: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]: ...


class _AsyncSocket(Protocol):
    linger: int

    def bind(self, endpoint: str) -> None: ...
    async def recv_multipart(self) -> list[bytes]: ...
    async def send_multipart(self, frames: list[bytes]) -> None: ...
    def close(self, linger: int = ...) -> None: ...


class _SyncSocket(Protocol):
    linger: int

    def connect(self, endpoint: str) -> None: ...
    def send(self, data: bytes) -> None: ...
    def recv_multipart(self) -> list[bytes]: ...
    def close(self, linger: int = ...) -> None: ...


class _AsyncContext(Protocol):
    def socket(self, socket_type: int) -> _AsyncSocket: ...
    def term(self) -> None: ...


class _SyncContext(Protocol):
    def socket(self, socket_type: int) -> _SyncSocket: ...
    def term(self) -> None: ...


class _Poller(Protocol):
    def register(self, socket: _SyncSocket, flags: int) -> None: ...
    def poll(self, timeout: int) -> list[tuple[_SyncSocket, int]]: ...


class _ZMQModule(Protocol):
    DEALER: int
    POLLIN: int
    ROUTER: int
    Context: type[_SyncContext]
    Poller: type[_Poller]


class _ZMQAsyncModule(Protocol):
    Context: type[_AsyncContext]


def _require_zmq() -> tuple[_ZMQModule, _ZMQAsyncModule]:
    try:
        zmq = importlib.import_module("zmq")
        zmq_asyncio = importlib.import_module("zmq.asyncio")
    except ImportError as exc:
        raise RuntimeError(
            "pyzmq is required for the Ragdocs daemon transport. Run 'uv sync' and retry."
        ) from exc
    return cast(_ZMQModule, zmq), cast(_ZMQAsyncModule, zmq_asyncio)


def transport_endpoint(socket_path: Path) -> str:
    return f"ipc://{socket_path}"


def remove_transport_socket(socket_path: Path) -> None:
    try:
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _attach_request_id(
    response: dict[str, object],
    request_id: str | None,
) -> dict[str, object]:
    if request_id is None or "request_id" in response:
        return response
    return {"request_id": request_id, **response}


class ZMQTransportServer:
    def __init__(
        self,
        *,
        socket_path: Path,
        metadata_provider: Callable[[], DaemonMetadata | None],
        request_handler: Callable[[str, dict[str, object]], Awaitable[dict[str, object]]] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._metadata_provider = metadata_provider
        self._request_handler = request_handler
        self._context: _AsyncContext | None = None
        self._socket: _AsyncSocket | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._request_tasks: set[asyncio.Task[None]] = set()
        self._send_lock = asyncio.Lock()

    async def start(self) -> None:
        zmq, zmq_asyncio = _require_zmq()
        remove_transport_socket(self._socket_path)
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        context = zmq_asyncio.Context()
        socket = context.socket(zmq.ROUTER)
        socket.linger = 0
        socket.bind(transport_endpoint(self._socket_path))
        self._context = context
        self._socket = socket
        self._serve_task = asyncio.create_task(self._serve())
        await asyncio.sleep(0)
        try:
            os.chmod(self._socket_path, 0o600)
        except OSError:
            pass

    async def stop(self) -> None:
        if self._serve_task is not None:
            self._serve_task.cancel()
            try:
                await self._serve_task
            except asyncio.CancelledError:
                pass
            self._serve_task = None
        if self._request_tasks:
            for task in list(self._request_tasks):
                task.cancel()
            await asyncio.gather(*self._request_tasks, return_exceptions=True)
            self._request_tasks.clear()
        socket = self._socket
        self._socket = None
        if socket is not None:
            socket.close(0)
        context = self._context
        self._context = None
        if context is not None:
            context.term()
        remove_transport_socket(self._socket_path)

    async def _serve(self) -> None:
        socket = self._socket
        assert socket is not None

        while True:
            try:
                frames = await socket.recv_multipart()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Daemon transport receive failed")
                continue

            if len(frames) < 2:
                logger.debug(
                    "Ignoring malformed ZMQ request with %d frame(s)", len(frames)
                )
                continue

            identity = frames[0]
            task = asyncio.create_task(
                self._handle_request(identity, frames[-1])
            )
            self._request_tasks.add(task)
            task.add_done_callback(self._request_tasks.discard)

    async def _handle_request(
        self,
        identity: bytes,
        request_line: bytes,
    ) -> None:
        payload = await self._dispatch_request(request_line)
        response = json.dumps(payload, sort_keys=True).encode("utf-8")

        socket = self._socket
        if socket is None:
            return

        try:
            async with self._send_lock:
                socket = self._socket
                if socket is None:
                    return
                await socket.send_multipart([identity, response])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Daemon transport send failed")

    async def _dispatch_request(self, request_line: bytes) -> dict[str, object]:
        request_id: str | None = None
        try:
            if not request_line:
                return {"status": "error", "error": "empty_request"}

            try:
                request = json.loads(request_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                request = {"path": "/internal/health", "payload": {}}

            if not isinstance(request, dict):
                return {"status": "error", "error": "invalid_request"}

            path = request.get("path", "/internal/health")
            payload = request.get("payload", {})
            raw_request_id = request.get("request_id")
            if isinstance(raw_request_id, str) and raw_request_id:
                request_id = raw_request_id
            if not isinstance(path, str) or not path:
                return _attach_request_id(
                    {"status": "error", "error": "request_path_required"},
                    request_id,
                )
            if not isinstance(payload, dict):
                return _attach_request_id(
                    {
                        "status": "error",
                        "error": "request_payload_must_be_object",
                    },
                    request_id,
                )

            if path == "/internal/health":
                metadata = self._metadata_provider()
                if metadata is None:
                    return _attach_request_id(
                        {
                            "status": "error",
                            "error": "daemon_metadata_unavailable",
                        },
                        request_id,
                    )
                return _attach_request_id(asdict(metadata), request_id)

            if self._request_handler is None:
                return _attach_request_id(
                    {"status": "error", "error": "unknown_request_path"},
                    request_id,
                )

            response = await self._request_handler(path, payload)
            return _attach_request_id(response, request_id)
        except Exception as exc:
            logger.exception("Daemon request handler failed")
            return _attach_request_id(
                {
                    "status": "error",
                    "error": "handler_exception",
                    "details": str(exc),
                },
                request_id,
            )


class ZMQTransportClient:
    def send_request(
        self,
        socket_path: Path,
        path: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        zmq, _ = _require_zmq()
        context: _SyncContext | None = None
        client: _SyncSocket | None = None
        try:
            context = zmq.Context()
            client = context.socket(zmq.DEALER)
            client.linger = 0
            client.connect(transport_endpoint(socket_path))
            request_id = str(uuid.uuid4())
            client.send(
                json.dumps(
                    {
                        "request_id": request_id,
                        "path": path,
                        "payload": payload,
                        "client": {
                            "kind": "unknown",
                            "pid": os.getpid(),
                        },
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )

            poller = zmq.Poller()
            poller.register(client, zmq.POLLIN)
            events = dict(poller.poll(int(timeout_seconds * 1000)))
            if client not in events:
                return {"status": "error", "error": "daemon_request_timed_out"}

            frames = client.recv_multipart()
            data = frames[-1] if frames else b""
        except Exception:  # noqa: BLE001 -- zmq/network transport errors vary
            return {"status": "error", "error": "daemon_socket_unavailable"}
        finally:
            if client is not None:
                client.close(0)
            if context is not None:
                context.term()

        if not data:
            return {"status": "error", "error": "empty_response"}

        try:
            response = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"status": "error", "error": "invalid_response"}

        if not isinstance(response, dict):
            return {"status": "error", "error": "invalid_response"}
        return response
