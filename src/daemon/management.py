from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from src.daemon.health import probe_daemon_socket, remove_daemon_socket, request_daemon_socket
from src.daemon.lock import FilesystemLock
from src.daemon.metadata import DaemonMetadata, read_daemon_metadata, remove_daemon_metadata
from src.daemon.paths import RuntimePaths


logger = logging.getLogger(__name__)


_READY_STATUSES = {"ready", "ready_primary", "ready_replica"}
_NONRESPONSIVE_METADATA_GRACE_SECONDS = 30.0
_INTERNAL_SHUTDOWN_TIMEOUT_SECONDS = 1.0
_RUNTIME_DAEMON_COMMAND = ("-m", "src.cli", "daemon-internal-run")
_RUNTIME_ROOT_OPTION = "--runtime-root"


class DaemonManagementError(RuntimeError):
    """Raised when daemon management operations fail."""


@dataclass(frozen=True)
class DaemonInspection:
    metadata: DaemonMetadata | None
    running: bool
    stale: bool
    responsive: bool = False
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
        return DaemonInspection(
            metadata=None,
            running=False,
            stale=False,
            responsive=False,
            ready=False,
        )

    running = is_process_running(metadata.pid)
    probed_metadata = (
        probe_daemon_socket(runtime_paths.socket_path) if running else None
    )
    responsive = (
        running and probed_metadata is not None and probed_metadata.pid == metadata.pid
    )
    ready = (
        responsive
        and metadata.status in _READY_STATUSES
    )
    return DaemonInspection(
        metadata=metadata,
        running=running,
        stale=not running,
        responsive=responsive,
        ready=ready,
    )


def start_daemon(
    *,
    timeout_seconds: float = 10.0,
    paths: RuntimePaths | None = None,
) -> DaemonMetadata:
    runtime_paths = paths or RuntimePaths.resolve()
    runtime_paths.ensure_directories()
    deadline = time.monotonic() + timeout_seconds

    current = inspect_daemon(runtime_paths)
    if current.running and current.ready and current.metadata is not None:
        _terminate_extra_runtime_daemon_processes(
            runtime_paths,
            keep_pid=current.metadata.pid,
        )
        return current.metadata
    if current.running and current.metadata is not None:
        _terminate_extra_runtime_daemon_processes(
            runtime_paths,
            keep_pid=current.metadata.pid,
        )
        if _metadata_has_exceeded_grace_period(current.metadata):
            _terminate_extra_runtime_daemon_processes(runtime_paths)
            _cleanup_stale_runtime_state(runtime_paths)
        else:
            return _wait_for_ready_daemon(deadline=deadline, paths=runtime_paths)
    if current.stale:
        _terminate_extra_runtime_daemon_processes(runtime_paths)
        _cleanup_stale_runtime_state(runtime_paths)
    elif current.metadata is None:
        _terminate_extra_runtime_daemon_processes(runtime_paths)

    process = _spawn_daemon_process(runtime_paths)
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
        terminated = _terminate_extra_runtime_daemon_processes(runtime_paths)
        if terminated:
            _cleanup_stale_runtime_state(runtime_paths)
        return None

    _terminate_extra_runtime_daemon_processes(runtime_paths, keep_pid=metadata.pid)

    if not inspection.running:
        _terminate_extra_runtime_daemon_processes(runtime_paths)
        _cleanup_stale_runtime_state(runtime_paths)
        return metadata

    shutdown_requested = _request_internal_shutdown(metadata)
    if not shutdown_requested:
        _terminate_process(metadata.pid, force=False)
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if not is_process_running(metadata.pid):
            _terminate_extra_runtime_daemon_processes(runtime_paths)
            _cleanup_stale_runtime_state(runtime_paths)
            return metadata
        time.sleep(0.1)

    _terminate_process(metadata.pid, force=True)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not is_process_running(metadata.pid):
            break
        time.sleep(0.05)

    _terminate_extra_runtime_daemon_processes(runtime_paths)
    _cleanup_stale_runtime_state(runtime_paths)
    return metadata


def _request_internal_shutdown(metadata: DaemonMetadata) -> bool:
    if not metadata.socket_path:
        return False

    try:
        response = request_daemon_socket(
            Path(metadata.socket_path),
            "/internal/shutdown",
            {},
            timeout_seconds=_INTERNAL_SHUTDOWN_TIMEOUT_SECONDS,
        )
    except Exception:
        return False

    return response.get("status") == "ok"


def restart_daemon(
    *,
    start_timeout_seconds: float = 10.0,
    stop_timeout_seconds: float = 5.0,
    paths: RuntimePaths | None = None,
) -> DaemonMetadata:
    stop_daemon(timeout_seconds=stop_timeout_seconds, paths=paths)
    return start_daemon(
        timeout_seconds=start_timeout_seconds,
        paths=paths,
    )


def wait_for_daemon_ready(
    *,
    timeout_seconds: float = 60.0,
    paths: RuntimePaths | None = None,
) -> DaemonMetadata:
    runtime_paths = paths or RuntimePaths.resolve()
    runtime_paths.ensure_directories()
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        inspection = inspect_daemon(runtime_paths)
        if inspection.stale:
            _cleanup_stale_runtime_state(runtime_paths)
        elif (
            inspection.metadata is not None
            and inspection.running
            and inspection.metadata.status in _READY_STATUSES
        ):
            return inspection.metadata

        time.sleep(0.1)

    raise DaemonManagementError("Timed out waiting for existing daemon readiness")


def _spawn_daemon_process(
    runtime_paths: RuntimePaths,
) -> subprocess.Popen[bytes]:
    command = [
        str(_resolve_daemon_python()),
        "-m",
        "src.cli",
        "daemon-internal-run",
        _RUNTIME_ROOT_OPTION,
        str(runtime_paths.root),
    ]

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )

    log_path = _daemon_log_path(runtime_paths)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_handle = log_path.open("wb")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
            cwd=str(Path.cwd()),
            env=env,
            start_new_session=True,
        )
    finally:
        stderr_handle.close()

    return process


def _resolve_daemon_python() -> Path:
    repo_root = Path(__file__).resolve().parents[2]

    candidates: list[Path] = []

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(_python_from_env_root(Path(virtual_env)))

    candidates.append(_python_from_env_root(repo_root / ".venv"))
    candidates.append(Path(sys.executable))

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    return Path(sys.executable)


def _python_from_env_root(env_root: Path) -> Path:
    if os.name == "nt":
        return env_root / "Scripts" / "python.exe"
    return env_root / "bin" / "python"


def _terminate_process(pid: int, *, force: bool) -> None:
    try:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        return


def _terminate_extra_runtime_daemon_processes(
    runtime_paths: RuntimePaths,
    *,
    keep_pid: int | None = None,
) -> list[int]:
    terminated: list[int] = []
    current_pid = os.getpid()
    for pid in _find_runtime_daemon_pids(runtime_paths):
        if pid == current_pid or pid == keep_pid:
            continue
        _terminate_pid_with_timeout(pid)
        terminated.append(pid)
    return terminated


def _terminate_pid_with_timeout(pid: int, timeout_seconds: float = 1.0) -> None:
    logger.warning("Terminating stale daemon process pid=%s", pid)

    _terminate_process(pid, force=False)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_process_running(pid):
            return
        time.sleep(0.05)
    _terminate_process(pid, force=True)


def _find_runtime_daemon_pids(runtime_paths: RuntimePaths) -> list[int]:
    runtime_root = runtime_paths.root.resolve()
    matching_pids: list[int] = []
    for pid in _iter_proc_pids():
        cmdline = _read_process_cmdline(pid)
        if cmdline is None or not _argv_contains_sequence(
            cmdline, _RUNTIME_DAEMON_COMMAND
        ):
            continue

        runtime_root_option = _read_option_path_value(cmdline, _RUNTIME_ROOT_OPTION)
        if runtime_root_option is not None:
            if runtime_root_option == runtime_root:
                matching_pids.append(pid)
            continue

        if _legacy_runtime_daemon_matches_root(pid, runtime_paths):
            matching_pids.append(pid)

    return matching_pids


def _legacy_runtime_daemon_matches_root(
    pid: int,
    runtime_paths: RuntimePaths,
) -> bool:
    expected_paths = {
        runtime_paths.socket_path.resolve(),
        runtime_paths.index_db_path.resolve(),
        runtime_paths.queue_db_path.resolve(),
        _daemon_log_path(runtime_paths).resolve(),
        _worker_log_path(runtime_paths).resolve(),
    }

    cwd_target = _read_proc_link(Path("/proc") / str(pid) / "cwd")
    if cwd_target is not None and _path_within_runtime_root(
        cwd_target, runtime_paths.root
    ):
        return True

    fd_root = Path("/proc") / str(pid) / "fd"
    try:
        entries = list(fd_root.iterdir())
    except OSError:
        return False

    for entry in entries:
        target = _read_proc_link(entry)
        if target is None:
            continue
        if target in expected_paths:
            return True

    return False


def _path_within_runtime_root(path: Path, runtime_root: Path) -> bool:
    try:
        path.resolve().relative_to(runtime_root.resolve())
    except ValueError:
        return False
    return True


def _iter_proc_pids() -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []

    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []

    return [int(entry.name) for entry in entries if entry.is_dir() and entry.name.isdigit()]


def _read_process_cmdline(pid: int) -> list[str] | None:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return []
    return [part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part]


def _argv_contains_sequence(argv: list[str], expected: tuple[str, ...]) -> bool:
    if not expected or len(argv) < len(expected):
        return False
    for index in range(len(argv) - len(expected) + 1):
        if tuple(argv[index : index + len(expected)]) == expected:
            return True
    return False


def _read_option_path_value(argv: list[str], option: str) -> Path | None:
    for index, token in enumerate(argv):
        if token != option:
            continue
        if index + 1 >= len(argv):
            return None
        return Path(argv[index + 1]).expanduser().resolve()
    return None


def _read_proc_link(path: Path) -> Path | None:
    try:
        raw_target = os.readlink(path)
    except OSError:
        return None

    cleaned_target = raw_target.removesuffix(" (deleted)")
    if not cleaned_target.startswith("/"):
        return None
    return Path(cleaned_target).resolve()


def _cleanup_stale_runtime_state(paths: RuntimePaths) -> None:
    remove_daemon_metadata(paths.metadata_path)
    remove_daemon_socket(paths.socket_path)


def _metadata_has_exceeded_grace_period(
    metadata: DaemonMetadata,
    *,
    now: float | None = None,
) -> bool:
    current_time = time.time() if now is None else now
    return (
        current_time - metadata.started_at
        >= _NONRESPONSIVE_METADATA_GRACE_SECONDS
    )


def _daemon_log_path(paths: RuntimePaths) -> Path:
    return paths.root / "daemon.log"


def _worker_log_path(paths: RuntimePaths) -> Path:
    return paths.root / "worker.log"


def _read_daemon_log_excerpt(paths: RuntimePaths, max_bytes: int = 4000) -> str | None:
    log_path = _daemon_log_path(paths)
    if not log_path.exists():
        return None
    try:
        data = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not data:
        return None
    excerpt = data[-max_bytes:].strip()
    return excerpt or None


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
    require_ready: bool = False,
) -> DaemonMetadata:
    exit_code: int | None = None

    while time.monotonic() < deadline:
        inspection = inspect_daemon(paths)
        if inspection.stale:
            _cleanup_stale_runtime_state(paths)
        elif inspection.metadata is not None:
            if require_ready and inspection.ready:
                return inspection.metadata
            if not require_ready and inspection.running and inspection.responsive:
                return inspection.metadata

        if require_ready and inspection.running and inspection.metadata is not None:
            time.sleep(0.1)
            continue

        if spawned_process is not None and exit_code is None:
            exit_code = spawned_process.poll()

        time.sleep(0.1)

    if spawned_process is not None:
        if exit_code is None:
            exit_code = spawned_process.poll()
        if exit_code is None:
            _terminate_process(spawned_process.pid, force=True)
            message = "Timed out waiting for daemon readiness"
            log_excerpt = _read_daemon_log_excerpt(paths)
            if log_excerpt is not None:
                message += f"\n\nDaemon log:\n{log_excerpt}"
            raise DaemonManagementError(message)
        message = f"Daemon exited before becoming ready (exit code {exit_code})"
        log_excerpt = _read_daemon_log_excerpt(paths)
        if log_excerpt is not None:
            message += f"\n\nDaemon log:\n{log_excerpt}"
        raise DaemonManagementError(message)

    raise DaemonManagementError("Timed out waiting for existing daemon readiness")
