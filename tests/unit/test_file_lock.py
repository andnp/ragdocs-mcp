import sys
import tempfile
from pathlib import Path

import pytest

from src.coordination.file_lock import IndexLock


@pytest.fixture
def temp_index_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_exclusive_lock_success(temp_index_path: Path):
    lock = IndexLock(temp_index_path, timeout_seconds=1.0)

    lock.acquire_exclusive()

    assert lock._lock_fd is not None
    assert lock._lock_mode == "exclusive"
    assert lock.lock_file_path.exists()

    lock.release()
    assert lock._lock_fd is None
    assert lock._lock_mode is None


def test_shared_lock_success(temp_index_path: Path):
    lock = IndexLock(temp_index_path, timeout_seconds=1.0)

    lock.acquire_shared()

    assert lock._lock_fd is not None
    assert lock._lock_mode == "shared"
    assert lock.lock_file_path.exists()

    lock.release()
    assert lock._lock_fd is None


def test_exclusive_lock_blocks_exclusive_lock(temp_index_path: Path):
    if sys.platform == "win32":
        pytest.skip("Windows has limited shared/exclusive lock support")

    lock1 = IndexLock(temp_index_path, timeout_seconds=0.5)
    lock2 = IndexLock(temp_index_path, timeout_seconds=0.5)

    lock1.acquire_exclusive()

    with pytest.raises(TimeoutError, match="Failed to acquire exclusive lock"):
        lock2.acquire_exclusive()

    lock1.release()


def test_exclusive_lock_blocks_shared_lock(temp_index_path: Path):
    if sys.platform == "win32":
        pytest.skip("Windows has limited shared/exclusive lock support")

    lock1 = IndexLock(temp_index_path, timeout_seconds=0.5)
    lock2 = IndexLock(temp_index_path, timeout_seconds=0.5)

    lock1.acquire_exclusive()

    with pytest.raises(TimeoutError, match="Failed to acquire shared lock"):
        lock2.acquire_shared()

    lock1.release()


def test_shared_locks_are_concurrent(temp_index_path: Path):
    if sys.platform == "win32":
        pytest.skip("Windows has limited shared/exclusive lock support")

    lock1 = IndexLock(temp_index_path, timeout_seconds=1.0)
    lock2 = IndexLock(temp_index_path, timeout_seconds=1.0)

    lock1.acquire_shared()
    lock2.acquire_shared()

    assert lock1._lock_mode == "shared"
    assert lock2._lock_mode == "shared"

    lock1.release()
    lock2.release()


def test_shared_lock_blocks_exclusive_lock(temp_index_path: Path):
    if sys.platform == "win32":
        pytest.skip("Windows has limited shared/exclusive lock support")

    lock1 = IndexLock(temp_index_path, timeout_seconds=0.5)
    lock2 = IndexLock(temp_index_path, timeout_seconds=0.5)

    lock1.acquire_shared()

    with pytest.raises(TimeoutError, match="Failed to acquire exclusive lock"):
        lock2.acquire_exclusive()

    lock1.release()


def test_lock_timeout_behavior(temp_index_path: Path):
    if sys.platform == "win32":
        pytest.skip("Windows has limited shared/exclusive lock support")

    lock1 = IndexLock(temp_index_path, timeout_seconds=5.0)
    lock2 = IndexLock(temp_index_path, timeout_seconds=0.3)

    lock1.acquire_exclusive()

    import time
    start = time.time()

    with pytest.raises(TimeoutError):
        lock2.acquire_exclusive()

    elapsed = time.time() - start
    assert 0.2 < elapsed < 0.6

    lock1.release()


def test_lock_context_manager_exclusive(temp_index_path: Path):
    with IndexLock(temp_index_path, timeout_seconds=1.0) as lock:
        lock.acquire_exclusive()
        assert lock._lock_mode == "exclusive"

    assert lock._lock_fd is None


def test_lock_context_manager_shared(temp_index_path: Path):
    with IndexLock(temp_index_path, timeout_seconds=1.0) as lock:
        lock.acquire_shared()
        assert lock._lock_mode == "shared"

    assert lock._lock_fd is None


def test_lock_release_without_acquire(temp_index_path: Path):
    lock = IndexLock(temp_index_path, timeout_seconds=1.0)

    lock.release()


def test_lock_double_release(temp_index_path: Path):
    lock = IndexLock(temp_index_path, timeout_seconds=1.0)

    lock.acquire_exclusive()
    lock.release()
    lock.release()


def test_lock_reacquisition_after_release(temp_index_path: Path):
    lock1 = IndexLock(temp_index_path, timeout_seconds=1.0)
    lock2 = IndexLock(temp_index_path, timeout_seconds=1.0)

    lock1.acquire_exclusive()
    lock1.release()

    lock2.acquire_exclusive()
    assert lock2._lock_mode == "exclusive"

    lock2.release()
