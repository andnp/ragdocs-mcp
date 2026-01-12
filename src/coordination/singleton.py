import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SingletonGuard:
    def __init__(self, index_data_path: Path):
        self.lock_file_path = index_data_path / ".server.lock"
        self._lock_fd: int | None = None
        self._acquired = False

    def acquire(self) -> None:
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            self._acquire_windows()
        else:
            self._acquire_unix()

    def _acquire_unix(self) -> None:
        import fcntl

        fd: int | None = None
        try:
            fd = os.open(
                self.lock_file_path,
                os.O_CREAT | os.O_RDWR,
                0o644,
            )

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                self._check_stale_lock_unix(fd)
                os.close(fd)
                raise RuntimeError(
                    f"Another mcp-markdown-ragdocs instance is already running.\n"
                    f"Lock file: {self.lock_file_path}\n\n"
                    f"Troubleshooting:\n"
                    f"  1. Stop the other instance\n"
                    f"  2. If no instance is running, remove: {self.lock_file_path}\n"
                    f"  3. Use coordination_mode='file_lock' in config for multi-instance support"
                ) from None

            pid = os.getpid()
            timestamp = int(time.time())
            lock_content = f"{pid}\n{timestamp}\n"

            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, lock_content.encode())
            os.fsync(fd)

            self._lock_fd = fd
            self._acquired = True
            logger.info(f"Singleton lock acquired: {self.lock_file_path}")

        except RuntimeError:
            raise
        except Exception as e:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to acquire singleton lock: {e}") from e

    def _acquire_windows(self) -> None:
        import msvcrt

        fd: int | None = None
        try:
            fd = os.open(
                self.lock_file_path,
                os.O_CREAT | os.O_RDWR | os.O_BINARY,  # type: ignore[attr-defined]
                0o644,
            )

            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
            except OSError:
                self._check_stale_lock_windows(fd)
                os.close(fd)
                raise RuntimeError(
                    f"Another mcp-markdown-ragdocs instance is already running.\n"
                    f"Lock file: {self.lock_file_path}\n\n"
                    f"Troubleshooting:\n"
                    f"  1. Stop the other instance\n"
                    f"  2. If no instance is running, remove: {self.lock_file_path}\n"
                    f"  3. Use coordination_mode='file_lock' in config for multi-instance support"
                ) from None

            pid = os.getpid()
            timestamp = int(time.time())
            lock_content = f"{pid}\n{timestamp}\n"

            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, lock_content.encode())

            self._lock_fd = fd
            self._acquired = True
            logger.info(f"Singleton lock acquired: {self.lock_file_path}")

        except RuntimeError:
            raise
        except Exception as e:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to acquire singleton lock: {e}") from e

    def _check_stale_lock_unix(self, fd: int) -> None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            content = os.read(fd, 1024).decode().strip()
            lines = content.split('\n')

            if len(lines) >= 1:
                try:
                    pid = int(lines[0])
                    if not self._is_process_alive(pid):
                        logger.warning(
                            f"Stale lock detected (PID {pid} not running). "
                            f"Auto-recovery not implemented - please remove: {self.lock_file_path}"
                        )
                except ValueError:
                    logger.warning(f"Invalid lock file format: {self.lock_file_path}")

        except Exception as e:
            logger.debug(f"Failed to check stale lock: {e}")

    def _check_stale_lock_windows(self, fd: int) -> None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            content = os.read(fd, 1024).decode().strip()
            lines = content.split('\n')

            if len(lines) >= 1:
                try:
                    pid = int(lines[0])
                    if not self._is_process_alive(pid):
                        logger.warning(
                            f"Stale lock detected (PID {pid} not running). "
                            f"Auto-recovery not implemented - please remove: {self.lock_file_path}"
                        )
                except ValueError:
                    logger.warning(f"Invalid lock file format: {self.lock_file_path}")

        except Exception as e:
            logger.debug(f"Failed to check stale lock: {e}")

    def _is_process_alive(self, pid: int) -> bool:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_INFORMATION = 0x0400
            handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
            if handle == 0:
                return False
            kernel32.CloseHandle(handle)
            return True
        else:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

    def release(self) -> None:
        if not self._acquired or self._lock_fd is None:
            return

        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"Failed to unlock file descriptor: {e}")

            os.close(self._lock_fd)
            self._lock_fd = None

            try:
                self.lock_file_path.unlink()
            except FileNotFoundError:
                pass

            self._acquired = False
            logger.info(f"Singleton lock released: {self.lock_file_path}")

        except Exception as e:
            logger.error(f"Failed to release singleton lock: {e}")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
