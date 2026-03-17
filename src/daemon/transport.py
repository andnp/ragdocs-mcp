from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import socket
from typing import Awaitable, Callable, Protocol

from src.daemon.metadata import DaemonMetadata


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


class UnixSocketTransportServer:
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
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        remove_unix_socket(self._socket_path)
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )
        try:
            os.chmod(self._socket_path, 0o600)
        except OSError:
            pass

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        remove_unix_socket(self._socket_path)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await reader.readline()
            payload = await self._dispatch_request(request_line)
            try:
                writer.write(
                    json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
                )
                await writer.drain()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("Daemon client disconnected before response drain")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("Daemon client connection already closed")

    async def _dispatch_request(self, request_line: bytes) -> dict[str, object]:
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
            if not isinstance(path, str) or not path:
                return {"status": "error", "error": "request_path_required"}
            if not isinstance(payload, dict):
                return {"status": "error", "error": "request_payload_must_be_object"}

            if path == "/internal/health":
                metadata = self._metadata_provider()
                if metadata is None:
                    return {
                        "status": "error",
                        "error": "daemon_metadata_unavailable",
                    }
                return asdict(metadata)

            if self._request_handler is None:
                return {"status": "error", "error": "unknown_request_path"}

            return await self._request_handler(path, payload)
        except Exception as exc:
            logger.error("Daemon request handler failed", exc_info=True)
            return {
                "status": "error",
                "error": "handler_exception",
                "details": str(exc),
            }


class UnixSocketTransportClient:
    def send_request(
        self,
        socket_path: Path,
        path: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(timeout_seconds)
                client.connect(str(socket_path))
                client.sendall(
                    json.dumps({"path": path, "payload": payload}, sort_keys=True).encode(
                        "utf-8"
                    )
                    + b"\n"
                )
                chunks = bytearray()
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    chunks.extend(chunk)
                    if b"\n" in chunk:
                        break
                data = bytes(chunks)
        except (FileNotFoundError, OSError, TimeoutError):
            return {"status": "error", "error": "daemon_socket_unavailable"}

        if not data:
            return {"status": "error", "error": "empty_response"}

        try:
            response = json.loads(data.splitlines()[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"status": "error", "error": "invalid_response"}

        if not isinstance(response, dict):
            return {"status": "error", "error": "invalid_response"}
        return response


def remove_unix_socket(socket_path: Path) -> None:
    try:
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return