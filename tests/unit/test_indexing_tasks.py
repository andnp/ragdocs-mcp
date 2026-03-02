"""
Unit tests for Huey-based indexing tasks.

Commit 3.3: Verifies indexing operations work as Huey tasks.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from huey import SqliteHuey

import src.indexing.tasks as tasks_mod
from src.indexing.tasks import (
    enqueue_index,
    enqueue_remove,
    register_tasks,
)


class FakeIndexManager:
    """Lightweight stub that records calls."""

    def __init__(self) -> None:
        self.indexed: list[tuple[str, bool]] = []
        self.removed: list[str] = []

    def index_document(self, file_path: str, force: bool = False) -> None:
        self.indexed.append((file_path, force))

    def remove_document(self, doc_id: str) -> None:
        self.removed.append(doc_id)


@pytest.fixture()
def huey_instance(tmp_path: Path) -> SqliteHuey:
    return SqliteHuey(
        name="test-tasks", filename=str(tmp_path / "tasks.db"), immediate=False
    )


@pytest.fixture()
def fake_manager() -> FakeIndexManager:
    return FakeIndexManager()


@pytest.fixture(autouse=True)
def _reset_tasks():
    """Reset module-level state between tests."""
    tasks_mod._huey = None
    tasks_mod._index_manager = None
    tasks_mod.index_document_task = None
    tasks_mod.remove_document_task = None
    yield
    tasks_mod._huey = None
    tasks_mod._index_manager = None
    tasks_mod.index_document_task = None
    tasks_mod.remove_document_task = None


class TestTaskRegistration:
    def test_register_tasks_creates_task_functions(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        """register_tasks() creates index and remove tasks."""
        register_tasks(huey_instance, fake_manager)
        assert tasks_mod.index_document_task is not None
        assert tasks_mod.remove_document_task is not None

    def test_enqueue_without_registration_returns_false(self) -> None:
        """enqueue_index/remove return False when tasks aren't registered."""
        assert enqueue_index("/some/file.md") is False
        assert enqueue_remove("some-doc") is False

    def test_enqueue_with_registration_returns_true(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        """enqueue_index/remove return True when tasks are registered."""
        register_tasks(huey_instance, fake_manager)
        assert enqueue_index("/some/file.md") is True
        assert enqueue_remove("some-doc") is True


class TestTaskExecution:
    def test_index_task_calls_manager(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        """Dequeued index task calls index_manager.index_document()."""
        register_tasks(huey_instance, fake_manager)

        # Enqueue
        enqueue_index("/docs/test.md", force=True)
        assert huey_instance.pending_count() == 1

        # Execute
        task = huey_instance.dequeue()
        huey_instance.execute(task)

        assert huey_instance.pending_count() == 0
        assert len(fake_manager.indexed) == 1
        assert fake_manager.indexed[0] == ("/docs/test.md", True)

    def test_remove_task_calls_manager(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        """Dequeued remove task calls index_manager.remove_document()."""
        register_tasks(huey_instance, fake_manager)

        enqueue_remove("docs/readme")
        assert huey_instance.pending_count() == 1

        task = huey_instance.dequeue()
        huey_instance.execute(task)

        assert huey_instance.pending_count() == 0
        assert fake_manager.removed == ["docs/readme"]

    def test_end_to_end_with_worker(
        self, tmp_path: Path, fake_manager: FakeIndexManager
    ) -> None:
        """Full flow: enqueue -> worker processes -> manager called."""
        from src.worker.consumer import HueyWorker

        huey = SqliteHuey(
            name="test-e2e", filename=str(tmp_path / "e2e.db"), immediate=False
        )
        register_tasks(huey, fake_manager)

        # Enqueue a task
        enqueue_index("/docs/guide.md")
        assert huey.pending_count() == 1

        # Start worker
        worker = HueyWorker(huey)
        worker.start()

        # Wait for processing
        deadline = time.monotonic() + 5.0
        while huey.pending_count() > 0 and time.monotonic() < deadline:
            time.sleep(0.1)

        worker.stop()

        assert huey.pending_count() == 0
        assert len(fake_manager.indexed) == 1
        assert fake_manager.indexed[0] == ("/docs/guide.md", False)

    def test_task_failure_does_not_crash_worker(self, tmp_path: Path) -> None:
        """A failing task doesn't crash the worker."""
        from src.worker.consumer import HueyWorker

        class FailingManager:
            def index_document(self, file_path: str, force: bool = False) -> None:
                raise RuntimeError("Simulated failure")

            def remove_document(self, doc_id: str) -> None:
                raise RuntimeError("Simulated failure")

        huey = SqliteHuey(
            name="test-fail", filename=str(tmp_path / "fail.db"), immediate=False
        )
        register_tasks(huey, FailingManager())

        enqueue_index("/bad/file.md")

        worker = HueyWorker(huey)
        worker.start()

        deadline = time.monotonic() + 5.0
        while huey.pending_count() > 0 and time.monotonic() < deadline:
            time.sleep(0.1)

        # Worker should still be running
        assert worker.is_running
        worker.stop()
