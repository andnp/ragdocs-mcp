"""
Unit tests for Huey task queue setup and persistence.

Commit 3.1: Verifies SqliteHuey instance creation, task enqueue, and persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from huey import SqliteHuey

from mcp_markdown_ragdocs.coordination.queue import get_huey, reset_huey
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore


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


class TestTaskLeases:
    def test_claim_heartbeat_and_reclaim(self, tmp_path: Path) -> None:
        store = TaskLeaseStore(tmp_path / "queue.db", timeout_seconds=10)

        assert store.claim(
            "task-1",
            task_name="index_document",
            owner_token="owner-1",
            payload=b"serialized-task",
            now=100.0,
        )
        assert store.active_count(now=105.0) == 1
        assert store.heartbeat("task-1", owner_token="owner-1", now=109.0)
        assert store.active_count(now=118.0) == 1
        assert not store.heartbeat("task-1", owner_token="owner-2", now=110.0)

        reclaimed = store.reclaim_expired(now=130.0)

        assert len(reclaimed) == 1
        assert reclaimed[0].task_id == "task-1"
        assert reclaimed[0].state == "reclaimed"
        assert reclaimed[0].payload == b"serialized-task"
        assert store.active_count(now=130.0) == 0

        assert store.claim(
            "task-1",
            task_name="index_document",
            owner_token="owner-2",
            payload=b"serialized-task",
            now=131.0,
        )
        assert store.complete("task-1", owner_token="owner-2", now=132.0)
        assert store.active_count(now=132.0) == 0
        lease = store.get("task-1")
        assert lease is not None
        assert lease.state == "completed"

    def test_writer_lease_is_exclusive_and_expires(self, tmp_path: Path) -> None:
        store = TaskLeaseStore(tmp_path / "queue.db", timeout_seconds=10)
        resource = "rebuild-writer"

        assert store.acquire_writer(resource, "rebuild-1", now=100.0)
        assert not store.acquire_writer(resource, "rebuild-2", now=105.0)
        assert store.writer_owner(resource, now=105.0) == "rebuild-1"
        assert store.writer_owner(resource, now=111.0) is None
        assert store.acquire_writer(resource, "rebuild-2", now=112.0)
        assert store.release_writer(resource, "rebuild-2")
        assert store.writer_owner(resource, now=112.0) is None
