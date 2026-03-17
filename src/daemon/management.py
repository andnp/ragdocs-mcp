from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.daemon.lock import FilesystemLock
from src.daemon.metadata import DaemonMetadata, read_daemon_metadata, remove_daemon_metadata
from src.daemon.paths import RuntimePaths


_READY_STATUSES = {"ready", "ready_primary", "ready_replica"}


class DaemonManagementError(RuntimeError):
    """Raised when daemon management operations fail."""


@dataclass(frozen=True)
class DaemonInspection:
    metadata: DaemonMetadata | None
    running: bool
    stale: bool
    ready: bool = False


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_daemon(paths: RuntimePaths | None = None) -> DaemonInspection:
    runtime_paths = paths or RuntimePaths.resolve()
    metadata = read_daemon_metadata(runtime_paths.metadata_path)
    if metadata is None:
        return DaemonInspection(metadata=None, running=False, stale=False, ready=False)

    running = is_process_running(metadata.pid)
    ready = running and metadata.status in _READY_STATUSES
    return DaemonInspection(
        metadata=metadata,
        running=running,
        stale=not running,
        ready=ready,
    )


def start_daemon(
    *,
    project_override: str | None = None,
    timeout_seconds: float = 10.0,
    paths: RuntimePaths | None = None,
) -> DaemonMetadata:
    runtime_paths = paths or RuntimePaths.resolve()
    runtime_paths.ensure_directories()
    deadline = time.monotonic() + timeout_seconds

    current = inspect_daemon(runtime_paths)
    if current.running and current.ready and current.metadata is not None:
        return current.metadata
    if current.running and current.metadata is not None:
        return _wait_for_ready_daemon(deadline=deadline, paths=runtime_paths)
    if current.stale:
        _cleanup_stale_runtime_state(runtime_paths)

    process = _spawn_daemon_process(project_override)
    return _wait_for_ready_daemon(
        deadline=deadline,
        paths=runtime_paths,
        spawned_process=process,
    )


def stop_daemon(
    *,
    timeout_seconds: float = 5.0,
    paths: RuntimePaths | None = None,
) -> DaemonMetadata | None:
    runtime_paths = paths or RuntimePaths.resolve()
    inspection = inspect_daemon(runtime_paths)
    metadata = inspection.metadata
    if metadata is None:
        return None

    if not inspection.running:
        _cleanup_stale_runtime_state(runtime_paths)
        return metadata

    _terminate_process(metadata.pid, force=False)
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if not is_process_running(metadata.pid):
            _cleanup_stale_runtime_state(runtime_paths)
            return metadata
        time.sleep(0.1)

    _terminate_process(metadata.pid, force=True)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not is_process_running(metadata.pid):
            break
        time.sleep(0.05)

    _cleanup_stale_runtime_state(runtime_paths)
    return metadata


def restart_daemon(
    *,
    project_override: str | None = None,
    start_timeout_seconds: float = 10.0,
    stop_timeout_seconds: float = 5.0,
    paths: RuntimePaths | None = None,
) -> DaemonMetadata:
    stop_daemon(timeout_seconds=stop_timeout_seconds, paths=paths)
    return start_daemon(
        project_override=project_override,
        timeout_seconds=start_timeout_seconds,
        paths=paths,
    )


def _spawn_daemon_process(project_override: str | None) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-m", "src.cli", "daemon-internal-run"]
    if project_override:
        command.extend(["--project", project_override])

    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path.cwd()),
        start_new_session=True,
    )


def _terminate_process(pid: int, *, force: bool) -> None:
    try:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        return


def _cleanup_stale_runtime_state(paths: RuntimePaths) -> None:
    remove_daemon_metadata(paths.metadata_path)
    try:
        paths.socket_path.unlink()
    except FileNotFoundError:
        pass


def acquire_boot_lock(
    *,
    paths: RuntimePaths | None = None,
    timeout_seconds: float = 10.0,
) -> FilesystemLock:
    runtime_paths = paths or RuntimePaths.resolve()
    runtime_paths.ensure_directories()
    lock = FilesystemLock(runtime_paths.lock_path)
    lock.acquire(timeout_seconds=timeout_seconds)
    return lock


def _wait_for_ready_daemon(
    *,
    deadline: float,
    paths: RuntimePaths,
    spawned_process: subprocess.Popen[bytes] | None = None,
) -> DaemonMetadata:
    exit_code: int | None = None

    while time.monotonic() < deadline:
        inspection = inspect_daemon(paths)
        if inspection.stale:
            _cleanup_stale_runtime_state(paths)
        elif inspection.running and inspection.ready and inspection.metadata is not None:
            return inspection.metadata

        if spawned_process is not None and exit_code is None:
            exit_code = spawned_process.poll()

        time.sleep(0.1)

    if spawned_process is not None:
        if exit_code is None:
            exit_code = spawned_process.poll()
        if exit_code is None:
            _terminate_process(spawned_process.pid, force=True)
            raise DaemonManagementError("Timed out waiting for daemon readiness")
        raise DaemonManagementError(
            f"Daemon exited before becoming ready (exit code {exit_code})"
        )

    raise DaemonManagementError("Timed out waiting for existing daemon readiness")