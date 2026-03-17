from pathlib import Path

from src.daemon.lock import FilesystemLock


def test_filesystem_lock_acquire_and_release(tmp_path: Path) -> None:
    lock = FilesystemLock(tmp_path / "daemon.lock")

    lock.acquire(timeout_seconds=0.1)
    try:
        assert lock._handle is not None
    finally:
        lock.release()

    assert lock._handle is None