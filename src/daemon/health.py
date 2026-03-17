from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
from typing import Callable

from src.daemon.metadata import DaemonMetadata, parse_daemon_metadata


class DaemonHealthServer:
    def __init__(
        self,
        *,
        socket_path: Path,
        metadata_provider: Callable[[], DaemonMetadata | None],
    ) -> None:
        self._socket_path = socket_path
        self._metadata_provider = metadata_provider
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        _remove_socket_file(self._socket_path)
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
        _remove_socket_file(self._socket_path)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await reader.readline()
            metadata = self._metadata_provider()
            if metadata is None:
                payload: dict[str, object] = {
                    "status": "error",
                    "error": "daemon_metadata_unavailable",
                }
            else:
                payload = asdict(metadata)

            writer.write(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def probe_daemon_socket(
    socket_path: Path,
    *,
    timeout_seconds: float = 0.2,
) -> DaemonMetadata | None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_seconds)
            client.connect(str(socket_path))
            client.sendall(b"health\n")
            data = client.recv(4096)
    except (FileNotFoundError, OSError, TimeoutError):
        return None

    if not data:
        return None

    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    return parse_daemon_metadata(payload)


def remove_daemon_socket(socket_path: Path) -> None:
    _remove_socket_file(socket_path)


def _remove_socket_file(socket_path: Path) -> None:
    try:
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return