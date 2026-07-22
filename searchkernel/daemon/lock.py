from __future__ import annotations

import fcntl
from dataclasses import dataclass
from pathlib import Path
import time
from typing import IO


class DaemonLockTimeoutError(TimeoutError):
    pass


@dataclass
class FilesystemLock:
    lock_path: Path
    _handle: IO[str] | None = None

    def acquire(self, timeout_seconds: float | None = 10.0) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds

        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return
            except BlockingIOError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    handle.close()
                    raise DaemonLockTimeoutError(
                        f"Timed out acquiring daemon lock: {self.lock_path}"
                    ) from exc
                time.sleep(0.05)
            except Exception:
                handle.close()
                raise

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> FilesystemLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()