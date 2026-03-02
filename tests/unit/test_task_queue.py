"""
Unit tests for Huey task queue setup and persistence.

Commit 3.1: Verifies SqliteHuey instance creation, task enqueue, and persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from huey import SqliteHuey

from src.coordination.queue import get_huey, reset_huey


@pytest.fixture(autouse=True)
def _clean_huey():
    """Reset the module-level Huey instance between tests."""
    reset_huey()
    yield
    reset_huey()


class TestHueySetup:
    def test_get_huey_returns_sqlite_instance(self, tmp_path: Path) -> None:
        """get_huey() returns a SqliteHuey instance."""
        huey = get_huey(tmp_path / "queue.db")
        assert isinstance(huey, SqliteHuey)

    def test_get_huey_is_singleton(self, tmp_path: Path) -> None:
        """Repeated calls return the same instance."""
        first = get_huey(tmp_path / "queue.db")
        second = get_huey()
        assert first is second

    def test_get_huey_requires_path_on_first_call(self) -> None:
        """get_huey() raises RuntimeError if no path on first call."""
        with pytest.raises(RuntimeError, match="db_path required"):
            get_huey()

    def test_huey_creates_parent_dirs(self, tmp_path: Path) -> None:
        """get_huey() creates parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "queue.db"
        huey = get_huey(deep_path)
        assert deep_path.parent.exists()
        assert isinstance(huey, SqliteHuey)


class TestTaskPersistence:
    def test_task_persistence_across_instances(self, tmp_path: Path) -> None:
        """Tasks survive across Huey instance restarts.

        This verifies that SqliteHuey persists tasks to disk — the core
        guarantee we need for crash recovery.
        """
        db_path = tmp_path / "queue.db"

        # Create a Huey instance and define a task
        huey1 = SqliteHuey(name="test", filename=str(db_path), immediate=False)

        @huey1.task()
        def sample_task(x: int) -> int:
            return x * 2

        # Enqueue tasks
        for i in range(5):
            sample_task(i)

        # Verify tasks are in the queue
        pending1 = huey1.pending_count()
        assert pending1 == 5

        # "Restart" — create a new instance pointing at the same DB
        huey2 = SqliteHuey(name="test", filename=str(db_path), immediate=False)

        # Tasks persist
        pending2 = huey2.pending_count()
        assert pending2 == 5

    def test_enqueue_and_dequeue(self, tmp_path: Path) -> None:
        """Tasks can be enqueued and dequeued from the queue."""
        db_path = tmp_path / "queue.db"
        huey = SqliteHuey(name="test", filename=str(db_path), immediate=False)

        @huey.task()
        def add(a: int, b: int) -> int:
            return a + b

        # Enqueue
        add(3, 4)
        assert huey.pending_count() == 1

        # Execute the task manually
        task = huey.dequeue()
        huey.execute(task)

        # Queue should be empty now
        assert huey.pending_count() == 0

    def test_reset_huey_allows_reinit(self, tmp_path: Path) -> None:
        """After reset_huey(), a new instance can be created."""
        path1 = tmp_path / "q1.db"
        path2 = tmp_path / "q2.db"

        h1 = get_huey(path1)
        reset_huey()
        h2 = get_huey(path2)

        assert h1 is not h2
