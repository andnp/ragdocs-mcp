import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexLock:
    def __init__(self, index_data_path: Path, timeout_seconds: float = 5.0):
        self.lock_file_path = index_data_path / ".index.lock"
        self.timeout_seconds = timeout_seconds
        self._lock_fd: int | None = None
        self._lock_mode: str | None = None

    def acquire_exclusive(self) -> None:
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            self._acquire_exclusive_windows()
        else:
            self._acquire_exclusive_unix()

    def acquire_shared(self) -> None:
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            self._acquire_shared_windows()
        else:
            self._acquire_shared_unix()

    def _acquire_exclusive_unix(self) -> None:
        import fcntl

        fd = os.open(
            self.lock_file_path,
            os.O_CREAT | os.O_RDWR,
            0o644,
        )

        start_time = time.time()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_fd = fd
                self._lock_mode = "exclusive"
                logger.debug(f"Exclusive lock acquired: {self.lock_file_path}")
                return
            except (IOError, OSError):
                elapsed = time.time() - start_time
                if elapsed >= self.timeout_seconds:
                    os.close(fd)
                    raise TimeoutError(
                        f"Failed to acquire exclusive lock after {self.timeout_seconds}s: {self.lock_file_path}"
                    ) from None
                time.sleep(0.1)

    def _acquire_shared_unix(self) -> None:
        import fcntl

        fd = os.open(
            self.lock_file_path,
            os.O_CREAT | os.O_RDWR,
            0o644,
        )

        start_time = time.time()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                self._lock_fd = fd
                self._lock_mode = "shared"
                logger.debug(f"Shared lock acquired: {self.lock_file_path}")
                return
            except (IOError, OSError):
                elapsed = time.time() - start_time
                if elapsed >= self.timeout_seconds:
                    os.close(fd)
                    raise TimeoutError(
                        f"Failed to acquire shared lock after {self.timeout_seconds}s: {self.lock_file_path}"
                    ) from None
                time.sleep(0.1)

    def _acquire_exclusive_windows(self) -> None:
        import msvcrt

        fd = os.open(
            self.lock_file_path,
            os.O_CREAT | os.O_RDWR | os.O_BINARY,  # type: ignore[attr-defined]
            0o644,
        )

        start_time = time.time()
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                self._lock_fd = fd
                self._lock_mode = "exclusive"
                logger.debug(f"Exclusive lock acquired: {self.lock_file_path}")
                return
            except OSError:
                elapsed = time.time() - start_time
                if elapsed >= self.timeout_seconds:
                    os.close(fd)
                    raise TimeoutError(
                        f"Failed to acquire exclusive lock after {self.timeout_seconds}s: {self.lock_file_path}"
                    ) from None
                time.sleep(0.1)

    def _acquire_shared_windows(self) -> None:
        fd = os.open(
            self.lock_file_path,
            os.O_CREAT | os.O_RDONLY | os.O_BINARY,  # type: ignore[attr-defined]
            0o644,
        )

        self._lock_fd = fd
        self._lock_mode = "shared"
        logger.debug(f"Shared lock acquired (Windows simulation): {self.lock_file_path}")

    def release(self) -> None:
        if self._lock_fd is None:
            return

        try:
            if sys.platform == "win32" and self._lock_mode == "exclusive":
                import msvcrt
                try:
                    msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"Failed to unlock file descriptor: {e}")

            os.close(self._lock_fd)
            self._lock_fd = None
            logger.debug(f"Lock released ({self._lock_mode}): {self.lock_file_path}")
            self._lock_mode = None

        except Exception as e:
            logger.error(f"Failed to release lock: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
