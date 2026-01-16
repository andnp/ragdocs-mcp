import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from src.coordination.singleton import SingletonGuard


@pytest.fixture
def temp_index_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_singleton_acquire_success(temp_index_path: Path):
    """
    Verify successful lock acquisition and release.

    Tests that a guard can acquire a lock, writes correct metadata to the lock
    file, and properly cleans up on release.
    """
    guard = SingletonGuard(temp_index_path)

    guard.acquire()

    assert guard._acquired is True
    assert guard._lock_fd is not None
    assert guard.lock_file_path.exists()

    content = guard.lock_file_path.read_text()
    lines = content.strip().split('\n')
    assert len(lines) == 2
    assert lines[0] == str(os.getpid())

    guard.release()
    assert guard._acquired is False
    assert not guard.lock_file_path.exists()


def test_singleton_blocks_second_acquisition(temp_index_path: Path):
    """
    Verify that a second guard cannot acquire an already-held lock.

    This ensures the singleton pattern correctly prevents concurrent access.
    """
    guard1 = SingletonGuard(temp_index_path)
    guard2 = SingletonGuard(temp_index_path)

    guard1.acquire()

    with pytest.raises(RuntimeError, match="Another mcp-markdown-ragdocs instance is already running"):
        guard2.acquire()

    guard1.release()


def test_singleton_allows_reacquisition_after_release(temp_index_path: Path):
    """
    Verify that a new guard can acquire lock after previous guard releases.

    Tests proper cleanup allows subsequent acquisitions.
    """
    guard1 = SingletonGuard(temp_index_path)
    guard2 = SingletonGuard(temp_index_path)

    guard1.acquire()
    guard1.release()

    guard2.acquire()
    assert guard2._acquired is True

    guard2.release()


def test_singleton_stale_lock_detection_unix(temp_index_path: Path):
    """
    Verify that an active lock held by another file descriptor blocks acquisition.

    On Unix, file locks are advisory. This test confirms that when a lock file
    exists and is actively held, a second guard cannot acquire the lock.
    """
    if sys.platform == "win32":
        pytest.skip("Unix-specific test")

    import fcntl

    guard = SingletonGuard(temp_index_path)

    guard.lock_file_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(guard.lock_file_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    fake_pid = os.getpid()
    fake_timestamp = int(time.time())
    os.write(fd, f"{fake_pid}\n{fake_timestamp}\n".encode())

    try:
        with pytest.raises(RuntimeError, match="Another mcp-markdown-ragdocs instance is already running"):
            guard.acquire()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_singleton_context_manager(temp_index_path: Path):
    """
    Verify that SingletonGuard works correctly as a context manager.

    Lock should be acquired on entry and released on exit.
    """
    with SingletonGuard(temp_index_path) as guard:
        assert guard._acquired is True
        assert guard.lock_file_path.exists()

    assert not guard.lock_file_path.exists()


def test_singleton_is_process_alive_unix():
    """
    Verify _is_process_alive correctly detects running and dead processes.

    Current PID should be alive; a non-existent PID should be dead.
    """
    if sys.platform == "win32":
        pytest.skip("Unix-specific test")

    guard = SingletonGuard(Path("/tmp"))

    assert guard._is_process_alive(os.getpid()) is True

    assert guard._is_process_alive(999999) is False


def test_singleton_cross_platform_compatibility(temp_index_path: Path):
    """
    Verify basic acquire/release works on all platforms.

    This is a smoke test to ensure the guard functions on Windows/Unix.
    """
    guard = SingletonGuard(temp_index_path)

    guard.acquire()
    assert guard._acquired is True

    guard.release()
    assert guard._acquired is False


def test_singleton_release_without_acquire(temp_index_path: Path):
    """
    Verify that releasing an unacquired lock is a no-op.

    Should not raise an exception or cause issues.
    """
    guard = SingletonGuard(temp_index_path)

    guard.release()


def test_singleton_double_release(temp_index_path: Path):
    """
    Verify that releasing a lock twice is safe.

    The second release should be a no-op.
    """
    guard = SingletonGuard(temp_index_path)

    guard.acquire()
    guard.release()
    guard.release()


def test_singleton_error_message_includes_troubleshooting(temp_index_path: Path):
    """
    Verify that the lock error includes helpful troubleshooting information.

    Users should see guidance on how to resolve the lock conflict.
    """
    guard1 = SingletonGuard(temp_index_path)
    guard2 = SingletonGuard(temp_index_path)

    guard1.acquire()

    with pytest.raises(RuntimeError) as exc_info:
        guard2.acquire()

    error_msg = str(exc_info.value)
    assert "Troubleshooting:" in error_msg
    assert "Stop the other instance" in error_msg
    assert "coordination_mode='file_lock'" in error_msg

    guard1.release()
