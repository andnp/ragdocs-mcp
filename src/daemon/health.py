from __future__ import annotations

from pathlib import Path

from src.daemon.metadata import DaemonMetadata, parse_daemon_metadata
from src.daemon.transport import (
    ZMQTransportClient,
    ZMQTransportServer,
    remove_transport_socket,
)


DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_DAEMON_HEALTH_TIMEOUT_SECONDS = 0.2


class DaemonHealthServer(ZMQTransportServer):
    """Backward-compatible daemon transport server using ZMQ over IPC."""


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
    client = ZMQTransportClient()
    return client.send_request(
        socket_path,
        path,
        payload,
        timeout_seconds=timeout_seconds,
    )


def remove_daemon_socket(socket_path: Path) -> None:
    remove_transport_socket(socket_path)