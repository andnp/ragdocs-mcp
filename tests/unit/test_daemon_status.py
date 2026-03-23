from __future__ import annotations

from pathlib import Path

from src.daemon import DaemonMetadata, RuntimePaths
from src.daemon.management import DaemonInspection
from src.daemon.status import (
    build_daemon_status_payload,
    format_daemon_startup_result,
    request_daemon_overview,
)


def _runtime_paths() -> RuntimePaths:
    return RuntimePaths(
        root=Path("/runtime"),
        index_db_path=Path("/runtime/index.db"),
        queue_db_path=Path("/runtime/queue.db"),
        metadata_path=Path("/runtime/daemon.json"),
        lock_path=Path("/runtime/daemon.lock"),
        socket_path=Path("/runtime/daemon.sock"),
    )


def _inspection(
    *,
    running: bool,
    ready: bool,
    metadata: DaemonMetadata | None,
) -> DaemonInspection:
    return DaemonInspection(
        running=running,
        ready=ready,
        stale=not running,
        metadata=metadata,
    )


def test_request_daemon_overview_returns_none_without_live_socket() -> None:
    inspection = _inspection(running=False, ready=False, metadata=None)

    payload = request_daemon_overview(inspection, runtime_paths=_runtime_paths())

    assert payload is None


def test_build_daemon_status_payload_reports_not_running() -> None:
    payload = build_daemon_status_payload(
        _inspection(running=False, ready=False, metadata=None),
        runtime_paths=_runtime_paths(),
    )

    assert payload == {
        "status": "not_running",
        "metadata_path": "/runtime/daemon.json",
        "lock_path": "/runtime/daemon.lock",
        "socket_path": "/runtime/daemon.sock",
    }


def test_build_daemon_status_payload_merges_overview_for_running_daemon() -> None:
    metadata = DaemonMetadata(
        pid=123,
        status="ready",
        started_at=1.0,
        socket_path="/runtime/live.sock",
        daemon_scope="global",
        index_db_path="/runtime/index.db",
        queue_db_path="/runtime/queue.db",
    )

    payload = build_daemon_status_payload(
        _inspection(running=True, ready=True, metadata=metadata),
        runtime_paths=_runtime_paths(),
        overview={
            "indexed_documents": 10,
            "pending_count": 2,
            "configured_root_count": 3,
        },
    )

    assert payload["status"] == "running"
    assert payload["pid"] == 123
    assert payload["lifecycle"] == "ready"
    assert payload["socket_path"] == "/runtime/live.sock"
    assert payload["indexed_documents"] == 10
    assert payload["pending_count"] == 2
    assert payload["configured_root_count"] == 3


def test_format_daemon_startup_result_marks_pending_socket_readiness() -> None:
    pending = DaemonMetadata(pid=1, status="starting", started_at=1.0)
    ready = DaemonMetadata(pid=2, status="ready", started_at=1.0)

    assert format_daemon_startup_result("started", pending) == (
        "Daemon started (pid=1, lifecycle=starting, socket readiness pending)"
    )
    assert format_daemon_startup_result("restarted", ready) == (
        "Daemon restarted (pid=2, lifecycle=ready)"
    )