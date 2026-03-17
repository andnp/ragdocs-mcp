from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from src.daemon.management import _resolve_daemon_python, _worker_log_path
from src.daemon.paths import RuntimePaths


class HueyWorkerProcess:
    def __init__(
        self,
        *,
        runtime_paths: RuntimePaths,
        project_override: str | None = None,
    ) -> None:
        self._runtime_paths = runtime_paths
        self._project_override = project_override
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
        return (time.time() - heartbeat) <= 10.0

    def start(self) -> None:
        if self.is_running:
            return

        _remove_worker_status(self._runtime_paths)

        command = [
            str(_resolve_daemon_python()),
            "-m",
            "src.cli",
            "worker-run",
            "--queue-db",
            str(self._runtime_paths.queue_db_path),
            "--index-root",
            str(self._runtime_paths.root),
            "--parent-pid",
            str(os.getpid()),
        ]
        if self._project_override:
            command.extend(["--project", self._project_override])

        log_path = _worker_log_path(self._runtime_paths)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = log_path.open("ab")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_handle,
                cwd=str(Path.cwd()),
                env=os.environ.copy(),
                start_new_session=True,
            )
        finally:
            stderr_handle.close()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self.is_healthy():
                return
            if self._process is not None and self._process.poll() is not None:
                return
            time.sleep(0.1)

    def stop(self, timeout: float = 5.0) -> None:
        process = self._process
        if process is None:
            return

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
        self.stop(timeout=timeout)
        time.sleep(0.2)
        self.start()


def _worker_status_path(runtime_paths: RuntimePaths) -> Path:
    return runtime_paths.root / "worker.json"


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