"""
Unit tests for Huey task queue setup and persistence.

Commit 3.1: Verifies SqliteHuey instance creation, task enqueue, and persistence.
"""

from __future__ import annotations

from pathlib import Path

from huey import SqliteHuey

from mcp_markdown_ragdocs.coordination.queue import QueueRuntime, build_queue_runtime
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore


class TestHueySetup:
    def test_queue_runtime_validates_storage_identity(self, tmp_path: Path) -> None:
        """A runtime rejects a database path that does not match its queue."""
        huey = SqliteHuey(
            name="test",
            filename=str(tmp_path / "queue.db"),
            immediate=False,
        )

        try:
            QueueRuntime(huey=huey, db_path=tmp_path / "other.db")
        except ValueError as error:
            assert str(error) == "QueueRuntime path must match the Huey storage path"
        else:
            raise AssertionError("mismatched queue identity should be rejected")

    def test_queue_runtime_instances_are_isolated(self, tmp_path: Path) -> None:
        """Separate runtime construction produces separate queues and paths."""
        first = build_queue_runtime(tmp_path / "first" / "queue.db")
        second = build_queue_runtime(tmp_path / "second" / "queue.db")

        assert first.huey is not second.huey
        assert first.db_path != second.db_path
        assert first.huey.name == "ragdocs"
        assert second.huey.name == "ragdocs"
        assert first.huey.immediate is False
        assert second.huey.immediate is False

    def test_queue_runtime_rejects_directory_path(self, tmp_path: Path) -> None:
        """A runtime requires a database filename rather than a directory."""
        try:
            build_queue_runtime(tmp_path)
        except ValueError as error:
            assert str(error) == "QueueRuntime requires a database file path"
        else:
            raise AssertionError("directory paths should be rejected")


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

        assert store.acquire_writer("rebuild-1", now=100.0)
        assert not store.acquire_writer("rebuild-2", now=105.0)
        assert store.writer_owner(now=105.0) == "rebuild-1"
        assert store.writer_owner(now=111.0) is None
        assert store.acquire_writer("rebuild-2", now=112.0)
        assert store.release_writer("rebuild-2")
        assert store.writer_owner(now=112.0) is None
