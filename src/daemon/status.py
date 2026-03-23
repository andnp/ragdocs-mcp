from __future__ import annotations

import time
from pathlib import Path

from src.daemon import DaemonMetadata, RuntimePaths
from src.daemon.client import DAEMON_PENDING_READY_STATUSES
from src.daemon.health import request_daemon_socket
from src.daemon.management import DaemonInspection

DEFAULT_DAEMON_OVERVIEW_TIMEOUT_SECONDS = 1.0
READY_DAEMON_STATUSES = {"ready", "ready_primary", "ready_replica"}


def request_daemon_overview(
    inspection: DaemonInspection,
    *,
    runtime_paths: RuntimePaths,
    timeout_seconds: float = DEFAULT_DAEMON_OVERVIEW_TIMEOUT_SECONDS,
) -> dict[str, object] | None:
    metadata = inspection.metadata
    if metadata is None or not inspection.running or not metadata.socket_path:
        return None

    response = request_daemon_socket(
        Path(metadata.socket_path),
        "/api/admin/overview",
        {},
        timeout_seconds=timeout_seconds,
    )
    if response.get("status") == "error":
        return None
    return response


def build_daemon_status_payload(
    inspection: DaemonInspection,
    *,
    runtime_paths: RuntimePaths,
    overview: dict[str, object] | None = None,
) -> dict[str, object]:
    if inspection.metadata is None:
        return {
            "status": "not_running",
            "metadata_path": str(runtime_paths.metadata_path),
            "lock_path": str(runtime_paths.lock_path),
            "socket_path": str(runtime_paths.socket_path),
        }

    metadata_ready = inspection.metadata.status in READY_DAEMON_STATUSES
    state = (
        "running"
        if inspection.ready or (inspection.running and metadata_ready)
        else "starting" if inspection.running else "stale"
    )
    started_at = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(inspection.metadata.started_at)
    )

    payload = {
        "status": state,
        "pid": inspection.metadata.pid,
        "lifecycle": inspection.metadata.status,
        "daemon_scope": inspection.metadata.daemon_scope,
        "started_at": started_at,
        "metadata_path": str(runtime_paths.metadata_path),
        "lock_path": str(runtime_paths.lock_path),
        "socket_path": inspection.metadata.socket_path
        or str(runtime_paths.socket_path),
        "index_db_path": inspection.metadata.index_db_path,
        "queue_db_path": inspection.metadata.queue_db_path,
        "endpoint": inspection.metadata.transport_endpoint,
    }

    if overview is not None:
        payload.update(
            {
                key: overview[key]
                for key in (
                    "indexed_documents",
                    "indexed_chunks",
                    "git_commits",
                    "git_repositories",
                    "worker_health",
                    "worker_pid",
                    "pending_count",
                    "scheduled_count",
                    "running_count",
                    "failed_count",
                    "worker_running",
                    "configured_root_count",
                    "documents_roots",
                    "project_context_mode",
                )
                if key in overview
            }
        )

    return payload


def format_daemon_startup_result(action: str, metadata: DaemonMetadata) -> str:
    if metadata.status in DAEMON_PENDING_READY_STATUSES:
        return (
            f"Daemon {action} (pid={metadata.pid}, lifecycle={metadata.status}, "
            "socket readiness pending)"
        )

    return f"Daemon {action} (pid={metadata.pid}, lifecycle={metadata.status})"