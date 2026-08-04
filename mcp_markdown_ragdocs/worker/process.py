from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from mcp_markdown_ragdocs.daemon.management import _resolve_daemon_python
from mcp_markdown_ragdocs.daemon.paths import RuntimePaths
from mcp_markdown_ragdocs.daemon.producer import (
    ProducerMetadata,
    read_producer_metadata,
    read_process_start_time_ticks,
    write_producer_metadata,
)

logger = logging.getLogger(__name__)

WORKER_STARTUP_TIMEOUT_SECONDS = 60.0
WORKER_HEARTBEAT_TIMEOUT_SECONDS = 300.0

_DAEMON_PARENT_COMMAND = ("-m", "mcp_markdown_ragdocs.cli", "daemon-internal-run")
_WORKER_COMMAND = ("-m", "mcp_markdown_ragdocs.cli", "worker-run")


class HueyWorkerProcess:
    def __init__(
        self,
        *,
        runtime_paths: RuntimePaths,
    ) -> None:
        self._runtime_paths = runtime_paths
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        if self._process is None:
            return None
        return self._process.pid

    def is_healthy(self) -> bool:
        process = self._process
        if process is None or process.poll() is not None:
            return False

        status = _read_worker_status(self._runtime_paths)
        if status.get("status") != "ready":
            return False
        if status.get("pid") != process.pid:
            return False

        heartbeat = status.get("heartbeat")
        if not isinstance(heartbeat, (int, float)):
            return False
        return (time.time() - heartbeat) <= WORKER_HEARTBEAT_TIMEOUT_SECONDS

    def start(self) -> None:
        if self.is_running:
            return

        _terminate_runtime_worker_processes(self._runtime_paths)
        _remove_worker_status(self._runtime_paths)
        parent_start_time = current_process_start_time_ticks()

        command = [
            str(_resolve_daemon_python()),
            "-m",
            "mcp_markdown_ragdocs.cli",
            "worker-run",
            "--queue-db",
            str(self._runtime_paths.queue_db_path),
            "--index-root",
            str(self._runtime_paths.root),
            "--parent-pid",
            str(os.getpid()),
        ]
        if parent_start_time is not None:
            command.extend([
                "--parent-start-time",
                str(parent_start_time),
            ])

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            start_new_session=True,
        )

        producer_path = self._runtime_paths.producer_metadata_path
        if producer_path is not None:
            start_time_ticks = read_process_start_time_ticks(self._process.pid)
            if start_time_ticks is not None:
                previous = read_producer_metadata(producer_path)
                write_producer_metadata(
                    producer_path,
                    ProducerMetadata(
                        pid=self._process.pid,
                        start_time_ticks=start_time_ticks,
                        started_at=time.time(),
                        stop_reason=(
                            previous.stop_reason
                            if previous is not None
                            and previous.stop_reason == "restart"
                            else None
                        ),
                    ),
                )

        # Loading the embedding model can take longer than the daemon's normal
        # health interval. Keep the subprocess alive while it initializes so
        # the supervisor does not repeatedly restart a healthy-but-cold worker.
        deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.is_healthy():
                return
            if self._process is not None and self._process.poll() is not None:
                return
            time.sleep(0.1)

    def stop(self, timeout: float = 5.0, *, reason: str = "shutdown") -> None:
        process = self._process
        if process is None:
            return

        producer_path = self._runtime_paths.producer_metadata_path
        if producer_path is not None:
            metadata = ProducerMetadata(
                pid=process.pid,
                start_time_ticks=read_process_start_time_ticks(process.pid) or 0,
                started_at=time.time(),
                status="stopped",
                stop_reason=reason,
            )
            write_producer_metadata(producer_path, metadata)

        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

        self._process = None
        _remove_worker_status(self._runtime_paths)

    def restart(self, timeout: float = 5.0) -> None:
        self.stop(timeout=timeout, reason="restart")
        time.sleep(0.2)
        self.start()


def _worker_status_path(runtime_paths: RuntimePaths) -> Path:
    return runtime_paths.root / "worker.json"


def current_process_start_time_ticks() -> int | None:
    return _read_process_start_time_ticks(os.getpid())


def is_expected_daemon_parent(
    parent_pid: int,
    expected_start_time_ticks: int | None = None,
) -> bool:
    if not _process_exists(parent_pid):
        return False

    cmdline = _read_process_cmdline(parent_pid)
    if cmdline is None:
        return not _procfs_available()
    if not _argv_contains_sequence(cmdline, _DAEMON_PARENT_COMMAND):
        return False

    if expected_start_time_ticks is None:
        return True

    actual_start_time_ticks = _read_process_start_time_ticks(parent_pid)
    if actual_start_time_ticks is None:
        return not _procfs_available()
    return actual_start_time_ticks == expected_start_time_ticks


def _terminate_runtime_worker_processes(runtime_paths: RuntimePaths) -> None:
    keep_pid = os.getpid()
    for pid in _find_runtime_worker_pids(runtime_paths):
        if pid == keep_pid:
            continue
        logger.warning(
            "Terminating stale worker-run process for runtime root %s (pid=%s)",
            runtime_paths.root,
            pid,
        )
        _terminate_process(pid)


def _find_runtime_worker_pids(runtime_paths: RuntimePaths) -> list[int]:
    queue_db_path = str(runtime_paths.queue_db_path.resolve())
    index_root = str(runtime_paths.root.resolve())
    matching_pids: list[int] = []
    for pid in _iter_proc_pids():
        cmdline = _read_process_cmdline(pid)
        if cmdline is None or not _argv_contains_sequence(cmdline, _WORKER_COMMAND):
            continue
        if _read_option_value(cmdline, "--queue-db") != queue_db_path:
            continue
        if _read_option_value(cmdline, "--index-root") != index_root:
            continue
        matching_pids.append(pid)
    return matching_pids


def _iter_proc_pids() -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []

    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []

    return [int(entry.name) for entry in entries if entry.is_dir() and entry.name.isdigit()]


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_process_cmdline(pid: int) -> list[str] | None:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return []
    return [part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part]


def _read_process_start_time_ticks(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "stat"
    try:
        stat_line = path.read_text(encoding="utf-8")
    except OSError:
        return None

    closing_paren = stat_line.rfind(")")
    if closing_paren == -1:
        return None

    fields = stat_line[closing_paren + 2 :].split()
    if len(fields) <= 19:
        return None

    try:
        return int(fields[19])
    except ValueError:
        return None


def _argv_contains_sequence(argv: list[str], expected: tuple[str, ...]) -> bool:
    if not expected or len(argv) < len(expected):
        return False
    for index in range(len(argv) - len(expected) + 1):
        if tuple(argv[index : index + len(expected)]) == expected:
            return True
    return False


def _read_option_value(argv: list[str], option: str) -> str | None:
    for index, token in enumerate(argv):
        if token != option:
            continue
        if index + 1 >= len(argv):
            return None
        return str(Path(argv[index + 1]).resolve())
    return None


def _terminate_process(pid: int, timeout: float = 1.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def _procfs_available() -> bool:
    return Path("/proc").exists()


def _read_worker_status(runtime_paths: RuntimePaths) -> dict[str, object]:
    path = _worker_status_path(runtime_paths)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _remove_worker_status(runtime_paths: RuntimePaths) -> None:
    path = _worker_status_path(runtime_paths)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return
