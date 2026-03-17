from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.daemon.health import DaemonHealthServer, probe_daemon_socket
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