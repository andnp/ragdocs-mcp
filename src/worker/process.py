from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess

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

    def start(self) -> None:
        if self.is_running:
            return

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