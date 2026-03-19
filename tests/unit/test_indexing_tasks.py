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
from src.daemon.queue_status import get_queue_stats
from src.indexing.tasks import (
    enqueue_index,
    enqueue_index_batch,
    enqueue_refresh_git,
    enqueue_refresh_git_batch,
    enqueue_remove,
    register_tasks,
)


class FakeIndexManager:
    """Lightweight stub that records calls."""

    def __init__(self) -> None:
        self.indexed: list[tuple[str, bool]] = []
        self.removed: list[str] = []
        self.persist_calls = 0

    def index_document(self, file_path: str, force: bool = False) -> None:
        self.indexed.append((file_path, force))

    def remove_document(self, doc_id: str) -> None:
        self.removed.append(doc_id)

    def persist(self) -> None:
        self.persist_calls += 1


class FakeCommitIndexer:
    def __init__(self) -> None:
        self.last_indexed_requests: list[str] = []

    def get_last_indexed_timestamp(self, repo_path: str) -> int | None:
        self.last_indexed_requests.append(repo_path)
        return 123


@pytest.fixture()
def huey_instance(tmp_path: Path) -> SqliteHuey:
    return SqliteHuey(
        name="test-tasks", filename=str(tmp_path / "tasks.db"), immediate=False
    )


@pytest.fixture()
def fake_manager() -> FakeIndexManager:
    return FakeIndexManager()


@pytest.fixture()
def fake_commit_indexer() -> FakeCommitIndexer:
    return FakeCommitIndexer()


@pytest.fixture(autouse=True)
def _reset_tasks():
    """Reset module-level state between tests."""
    tasks_mod._huey = None
    tasks_mod._index_manager = None
    tasks_mod._commit_indexer = None
    tasks_mod._task_backpressure_limit = 100
    tasks_mod.index_document_task = None
    tasks_mod.remove_document_task = None
    tasks_mod.refresh_git_repository_task = None
    yield
    tasks_mod._huey = None
    tasks_mod._index_manager = None
    tasks_mod._commit_indexer = None
    tasks_mod._task_backpressure_limit = 100
    tasks_mod.index_document_task = None
    tasks_mod.remove_document_task = None
    tasks_mod.refresh_git_repository_task = None


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

    def test_enqueue_respects_backpressure_limit(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager, task_backpressure_limit=1)

        assert enqueue_index("/some/file.md") is True
        assert enqueue_index("/some/other.md") is False
        assert enqueue_remove("some-doc") is False

    def test_startup_batch_enqueue_bypasses_backpressure_limit(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        fake_commit_indexer: FakeCommitIndexer,
    ) -> None:
        register_tasks(
            huey_instance,
            fake_manager,
            fake_commit_indexer,
            task_backpressure_limit=1,
        )

        indexed = enqueue_index_batch(["/some/file.md", "/some/other.md"])
        refreshed = enqueue_refresh_git_batch(["/repo-a/.git", "/repo-b/.git"])

        assert indexed == 2
        assert refreshed == 2
        assert huey_instance.pending_count() == 4

    def test_queue_stats_include_backpressure_utilization(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)
        enqueue_index("/some/file.md")

        stats = get_queue_stats(huey_instance, backpressure_limit=4)

        assert stats.backpressure_limit == 4
        assert stats.backpressure_utilization == 0.25

    def test_enqueue_refresh_git_returns_false_without_registration(self) -> None:
        assert enqueue_refresh_git("/repo/.git") is False

    def test_register_tasks_creates_git_task_when_commit_indexer_provided(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        fake_commit_indexer: FakeCommitIndexer,
    ) -> None:
        register_tasks(huey_instance, fake_manager, fake_commit_indexer)
        assert tasks_mod.refresh_git_repository_task is not None
        assert enqueue_refresh_git("/repo/.git") is True


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
        assert fake_manager.persist_calls == 1

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
        assert fake_manager.persist_calls == 1

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
        assert fake_manager.persist_calls == 1

    def test_task_failure_does_not_crash_worker(self, tmp_path: Path) -> None:
        """A failing task doesn't crash the worker."""
        from src.worker.consumer import HueyWorker

        class FailingManager:
            def index_document(self, file_path: str, force: bool = False) -> None:
                raise RuntimeError("Simulated failure")

            def remove_document(self, doc_id: str) -> None:
                raise RuntimeError("Simulated failure")

            def persist(self) -> None:
                raise AssertionError("persist should not be called after failed task")

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

    def test_refresh_git_task_calls_parallel_indexing(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        fake_commit_indexer: FakeCommitIndexer,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        register_tasks(huey_instance, fake_manager, fake_commit_indexer)

        observed: dict[str, object] = {}

        def _fake_get_commits_after_timestamp(git_dir: Path, last_timestamp: int | None):
            observed["git_dir"] = git_dir
            observed["last_timestamp"] = last_timestamp
            return ["abc123"]

        def _fake_index_commits_parallel_sync(
            commit_hashes,
            git_dir,
            commit_indexer,
            config,
            max_delta_lines,
        ):
            observed["commit_hashes"] = commit_hashes
            observed["index_git_dir"] = git_dir
            observed["commit_indexer"] = commit_indexer
            observed["max_delta_lines"] = max_delta_lines
            return 1

        monkeypatch.setattr(
            "src.git.repository.get_commits_after_timestamp",
            _fake_get_commits_after_timestamp,
        )
        monkeypatch.setattr(
            "src.git.parallel_indexer.index_commits_parallel_sync",
            _fake_index_commits_parallel_sync,
        )

        git_dir = tmp_path / "repo" / ".git"
        git_dir.parent.mkdir(parents=True)
        git_dir.mkdir()

        enqueue_refresh_git(str(git_dir))
        task = huey_instance.dequeue()
        huey_instance.execute(task)

        assert fake_commit_indexer.last_indexed_requests == [str(git_dir)]
        assert observed["git_dir"] == git_dir
        assert observed["last_timestamp"] == 123
        assert observed["commit_hashes"] == ["abc123"]
        assert observed["index_git_dir"] == git_dir
        assert observed["commit_indexer"] is fake_commit_indexer
        assert observed["max_delta_lines"] == 200
