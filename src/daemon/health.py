from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path


from src.daemon.transport import (
    UnixSocketTransportClient,
    UnixSocketTransportServer,
    remove_unix_socket,
)
DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_DAEMON_HEALTH_TIMEOUT_SECONDS = 0.2

logger = logging.getLogger(__name__)


class DaemonHealthServer(UnixSocketTransportServer):
    """Backward-compatible daemon transport server using Unix sockets."""
def probe_daemon_socket(
    socket_path: Path,
    *,
    timeout_seconds: float = DEFAULT_DAEMON_HEALTH_TIMEOUT_SECONDS,
) -> DaemonMetadata | None:
    payload = request_daemon_socket(
        socket_path,
        "/internal/health",
        {},
        timeout_seconds=timeout_seconds,
    )
    return parse_daemon_metadata(payload)


def request_daemon_socket(
    socket_path: Path,
    path: str,
    payload: dict[str, object],
    *,
    timeout_seconds: float = DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, object]:
    client = UnixSocketTransportClient()
    return client.send_request(
        socket_path,
        path,
        payload,
        timeout_seconds=timeout_seconds,
    )


def remove_daemon_socket(socket_path: Path) -> None:
    remove_unix_socket(socket_path)