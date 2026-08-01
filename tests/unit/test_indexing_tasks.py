"""
Unit tests for Huey-based indexing tasks.

Commit 3.3: Verifies indexing operations work as Huey tasks.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from huey import SqliteHuey
from searchkernel.domain import Record
from searchkernel.indexing.bootstrap_checkpoint import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    load_bootstrap_checkpoint,
    save_bootstrap_checkpoint,
)

import mcp_markdown_ragdocs.indexing.tasks as tasks_mod
from mcp_markdown_ragdocs.daemon.queue_status import get_queue_stats
from mcp_markdown_ragdocs.indexing.git_refresh_state import get_cursor, save_cursor, save_head
from mcp_markdown_ragdocs.indexing.tasks import (
    GIT_REFRESH_TASK_PRIORITY,
    RECORD_BATCH_TASK_PRIORITY,
    enqueue_index,
    enqueue_index_batch,
    enqueue_refresh_git,
    enqueue_refresh_git_batch,
    enqueue_remove,
    get_pending_index_document_count,
    register_tasks,
    submit_index_batch,
    submit_index_request_batch,
    submit_record_batch,
    submit_refresh_git_request,
    submit_remove_request_batch,
)


class FakeIndexManager:
    """Lightweight stub that records calls."""

    def __init__(self) -> None:
        self.indexed: list[tuple[str, bool]] = []
        self.removed: list[str] = []
        self.indexed_records: list[Record] = []
        self.persist_calls = 0

    def index_document(self, file_path: str, force: bool = False) -> None:
        self.indexed.append((file_path, force))

    def index_documents(
        self,
        file_paths: list[str],
        force: bool = False,
        persist: bool = False,
    ) -> None:
        for file_path in file_paths:
            self.indexed.append((file_path, force))
        if persist:
            self.persist_calls += 1

    def remove_document(self, doc_id: str) -> None:
        self.removed.append(doc_id)

    def remove_documents(self, doc_ids: list[str], persist: bool = False) -> None:
        self.removed.extend(doc_ids)
        if persist:
            self.persist_calls += 1

    def persist(self) -> None:
        self.persist_calls += 1

    def index_record(self, record: Record) -> None:
        self.indexed_records.append(record)


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
    tasks_mod._task_backpressure_limit = 100
    tasks_mod._bootstrap_index_path = None
    tasks_mod._bootstrap_documents_roots = []
    tasks_mod._git_refresh_in_flight.clear()
    tasks_mod.index_document_task = None
    tasks_mod.index_documents_batch_task = None
    tasks_mod.index_records_batch_task = None
    tasks_mod.remove_document_task = None
    tasks_mod.remove_documents_batch_task = None
    tasks_mod.refresh_git_repository_task = None
    yield
    tasks_mod._huey = None
    tasks_mod._index_manager = None
    tasks_mod._task_backpressure_limit = 100
    tasks_mod._bootstrap_index_path = None
    tasks_mod._bootstrap_documents_roots = []
    tasks_mod._git_refresh_in_flight.clear()
    tasks_mod.index_document_task = None
    tasks_mod.index_documents_batch_task = None
    tasks_mod.index_records_batch_task = None
    tasks_mod.remove_document_task = None
    tasks_mod.remove_documents_batch_task = None
    tasks_mod.refresh_git_repository_task = None


class TestTaskRegistration:
    def test_register_tasks_creates_task_functions(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        """register_tasks() creates index and remove tasks."""
        register_tasks(huey_instance, fake_manager)
        assert tasks_mod.index_document_task is not None
        assert tasks_mod.index_documents_batch_task is not None
        assert tasks_mod.index_records_batch_task is not None
        assert tasks_mod.remove_document_task is not None
        assert tasks_mod.remove_documents_batch_task is not None

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

    def test_record_batch_task_indexes_and_persists_once(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)
        payload = Record(
            source_kind="note",
            source_id="note:1",
            title="A note",
            body="Body",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        ).to_dict()

        result = submit_record_batch([payload])
        assert result is not None
        task = huey_instance.dequeue()
        assert task is not None
        assert task.priority == RECORD_BATCH_TASK_PRIORITY
        assert huey_instance.execute(task) == {"status": "ok", "indexed_count": 1}
        assert [record.source_id for record in fake_manager.indexed_records] == ["note:1"]
        assert fake_manager.persist_calls == 1

    def test_enqueue_respects_backpressure_limit(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager, task_backpressure_limit=1)

        assert enqueue_index("/some/file.md") is True
        assert enqueue_index("/some/other.md") is False
        assert enqueue_remove("some-doc") is False

    def test_startup_git_batch_respects_backpressure_limit(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
    ) -> None:
        register_tasks(
            huey_instance,
            fake_manager,
            task_backpressure_limit=1,
        )

        refreshed = enqueue_refresh_git_batch(["/repo-a/.git", "/repo-b/.git"])

        assert refreshed == 1
        assert huey_instance.pending_count() == 1

    def test_git_refresh_has_lower_priority_than_record_ingestion(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_refresh_git("/repo/.git") is True
        payload = Record(
            source_kind="note",
            source_id="note:priority",
            title="A note",
            body="Body",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        ).to_dict()
        assert submit_record_batch([payload]) is not None

        first_task = huey_instance.dequeue()
        assert first_task is not None
        assert first_task.priority == RECORD_BATCH_TASK_PRIORITY
        assert first_task.priority > GIT_REFRESH_TASK_PRIORITY

    def test_startup_batch_skips_files_already_pending_in_queue(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_index("/some/file.md") is True

        indexed = enqueue_index_batch([
            "/some/file.md",
            "/some/other.md",
        ])

        assert indexed == 1
        assert huey_instance.pending_count() == 2

        first_task = huey_instance.dequeue()
        second_task = huey_instance.dequeue()
        assert first_task is not None
        assert second_task is not None

        assert first_task.args == ("/some/file.md",)
        assert second_task.args == (["/some/other.md"],)

    def test_get_pending_index_document_count_counts_matching_pending_paths(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_index("/some/file.md") is True
        assert enqueue_index("/some/other.md") is True

        pending = get_pending_index_document_count(
            [
                "/some/file.md",
                "/some/file.md",
                "/some/missing.md",
            ]
        )

        assert pending == 1

    def test_startup_batch_deduplicates_duplicate_paths_within_batch(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        indexed = enqueue_index_batch([
            "/some/file.md",
            "/some/file.md",
            "/some/other.md",
        ])

        assert indexed == 2
        assert huey_instance.pending_count() == 1

    def test_startup_batch_preserves_force_reindex_behavior(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_index("/some/file.md") is True

        indexed = enqueue_index_batch(["/some/file.md"], force=True)

        assert indexed == 1
        assert huey_instance.pending_count() == 2

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

    def test_register_tasks_creates_git_refresh_task(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
    ) -> None:
        register_tasks(huey_instance, fake_manager)
        assert tasks_mod.refresh_git_repository_task is not None
        assert enqueue_refresh_git("/repo/.git") is True

    def test_enqueue_refresh_git_skips_repo_already_pending_in_queue(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_refresh_git("/repo/.git") is True
        assert enqueue_refresh_git("/repo/.git") is False

        assert huey_instance.pending_count() == 1
        task = huey_instance.dequeue()
        assert task is not None
        assert task.args == ("/repo/.git",)

    def test_submit_refresh_git_request_reports_already_pending_status(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_refresh_git("/repo/.git") is True

        submission = submit_refresh_git_request("/repo/.git")

        assert submission.status == "already_pending"
        assert submission.accepted_by_queue is True
        assert submission.enqueued is False

    def test_submit_refresh_git_request_reports_running_as_already_pending(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
    ) -> None:
        register_tasks(huey_instance, fake_manager)
        git_dir = "/repo/.git"
        tasks_mod._git_refresh_in_flight.add(str(Path(git_dir).resolve()))

        submission = submit_refresh_git_request(git_dir)

        assert submission.status == "already_pending"
        assert huey_instance.pending_count() == 0

    def test_startup_git_batch_skips_repos_already_pending_in_queue(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_refresh_git("/repo-a/.git") is True

        refreshed = enqueue_refresh_git_batch(
            [
                "/repo-a/.git",
                "/repo-b/.git",
            ]
        )

        assert refreshed == 1
        assert huey_instance.pending_count() == 2

        first_task = huey_instance.dequeue()
        second_task = huey_instance.dequeue()
        assert first_task is not None
        assert second_task is not None

        assert first_task.args == ("/repo-a/.git",)
        assert second_task.args == ("/repo-b/.git",)

    def test_startup_git_batch_deduplicates_duplicate_paths_within_batch(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        refreshed = enqueue_refresh_git_batch(
            [
                "/repo-a/.git",
                "/repo-a/.git",
                "/repo-b/.git",
            ]
        )

        assert refreshed == 2
        assert huey_instance.pending_count() == 2

    def test_submit_index_batch_reports_pending_items_as_already_represented(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_index("/some/file.md") is True

        submission = submit_index_batch([
            "/some/file.md",
            "/some/other.md",
        ])

        assert submission.queue_available is True
        assert submission.requested_unique_count == 2
        assert submission.enqueued_count == 1
        assert submission.already_pending_count == 1
        assert submission.all_represented is True

    def test_pending_index_count_includes_batch_tasks(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_index_batch(["/some/file.md", "/some/other.md"]) == 2

        pending = get_pending_index_document_count(
            ["/some/file.md", "/some/other.md", "/some/missing.md"]
        )

        assert pending == 2

    def test_submit_index_request_batch_deduplicates_against_pending_single_and_batch_tasks(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_index("/some/file.md") is True
        assert enqueue_index_batch(["/some/batch.md"]) == 1

        submission = submit_index_request_batch(
            ["/some/file.md", "/some/batch.md", "/some/new.md"]
        )

        assert submission.queue_available is True
        assert submission.enqueued_count == 1
        assert submission.already_pending_count == 2
        assert submission.backpressured_items == ()

        assert huey_instance.pending_count() == 3
        huey_instance.dequeue()
        huey_instance.dequeue()
        third_task = huey_instance.dequeue()
        assert third_task is not None
        assert third_task.args == (["/some/new.md"],)

    def test_submit_index_request_batch_reports_only_unrepresented_items_as_backpressured(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager, task_backpressure_limit=1)

        assert enqueue_index("/some/file.md") is True

        submission = submit_index_request_batch(
            ["/some/file.md", "/some/new.md", "/some/other.md"]
        )

        assert submission.queue_available is True
        assert submission.already_pending_count == 1
        assert submission.enqueued_count == 0
        assert submission.backpressured_items == (
            "/some/new.md",
            "/some/other.md",
        )

    def test_submit_remove_request_batch_deduplicates_against_pending_single_and_batch_tasks(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        assert enqueue_remove("docs/existing") is True
        first_batch = submit_remove_request_batch(["docs/batched"])
        assert first_batch.enqueued_count == 1

        submission = submit_remove_request_batch(
            ["docs/existing", "docs/batched", "docs/new"]
        )

        assert submission.queue_available is True
        assert submission.enqueued_count == 1
        assert submission.already_pending_count == 2
        assert submission.backpressured_items == ()

        assert huey_instance.pending_count() == 3
        huey_instance.dequeue()
        huey_instance.dequeue()
        third_task = huey_instance.dequeue()
        assert third_task is not None
        assert third_task.args == (["docs/new"],)

    def test_submit_remove_request_batch_reports_only_unrepresented_items_as_backpressured(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager, task_backpressure_limit=1)

        assert enqueue_remove("docs/existing") is True

        submission = submit_remove_request_batch(
            ["docs/existing", "docs/new", "docs/other"]
        )

        assert submission.queue_available is True
        assert submission.already_pending_count == 1
        assert submission.enqueued_count == 0
        assert submission.backpressured_items == (
            "docs/new",
            "docs/other",
        )


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

    def test_index_task_marks_bootstrap_checkpoint_after_persist(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        tmp_path: Path,
    ) -> None:
        docs_root = tmp_path / "docs"
        docs_root.mkdir()
        file_path = docs_root / "guide.md"
        file_path.write_text("# Guide")

        save_bootstrap_checkpoint(
            tmp_path,
            BootstrapCheckpoint(
                schema_version="1.0.0",
                generation="current",
                complete=False,
                targets={
                    "guide.md": BootstrapFileStamp(
                        "guide.md",
                        mtime_ns=0,
                        size=0,
                    )
                },
                completed={},
            ),
        )

        register_tasks(
            huey_instance,
            fake_manager,
            bootstrap_index_path=tmp_path,
            bootstrap_documents_roots=[docs_root],
        )

        enqueue_index(str(file_path))
        task = huey_instance.dequeue()
        huey_instance.execute(task)

        checkpoint = load_bootstrap_checkpoint(tmp_path)

        assert checkpoint is not None
        assert checkpoint.complete is True
        assert set(checkpoint.completed) == {"guide.md"}

    def test_index_batch_task_marks_all_bootstrap_files_after_single_persist(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        tmp_path: Path,
    ) -> None:
        docs_root = tmp_path / "docs"
        docs_root.mkdir()
        first = docs_root / "guide.md"
        second = docs_root / "api.md"
        first.write_text("# Guide")
        second.write_text("# API")

        save_bootstrap_checkpoint(
            tmp_path,
            BootstrapCheckpoint(
                schema_version="1.0.0",
                generation="current",
                complete=False,
                targets={
                    "guide.md": BootstrapFileStamp("guide.md", mtime_ns=0, size=0),
                    "api.md": BootstrapFileStamp("api.md", mtime_ns=0, size=0),
                },
                completed={},
            ),
        )

        register_tasks(
            huey_instance,
            fake_manager,
            bootstrap_index_path=tmp_path,
            bootstrap_documents_roots=[docs_root],
        )

        assert enqueue_index_batch([str(first), str(second)]) == 2

        task = huey_instance.dequeue()
        huey_instance.execute(task)

        checkpoint = load_bootstrap_checkpoint(tmp_path)

        assert checkpoint is not None
        assert checkpoint.complete is True
        assert set(checkpoint.completed) == {"guide.md", "api.md"}
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

    def test_remove_batch_task_calls_manager_once_for_batch(
        self, huey_instance: SqliteHuey, fake_manager: FakeIndexManager
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        submission = submit_remove_request_batch(["docs/a", "docs/b", "docs/a"])

        assert submission.enqueued_count == 2

        task = huey_instance.dequeue()
        huey_instance.execute(task)

        assert fake_manager.removed == ["docs/a", "docs/b"]
        assert fake_manager.persist_calls == 1

    def test_end_to_end_with_worker(
        self, tmp_path: Path, fake_manager: FakeIndexManager
    ) -> None:
        """Full flow: enqueue -> worker processes -> manager called."""
        from mcp_markdown_ragdocs.worker.consumer import HueyWorker

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
        from mcp_markdown_ragdocs.worker.consumer import HueyWorker

        class FailingManager:
            def index_document(self, file_path: str, force: bool = False) -> None:
                raise RuntimeError("Simulated failure")

            def index_documents(
                self,
                file_paths: list[str],
                force: bool = False,
                persist: bool = False,
            ) -> None:
                raise RuntimeError("Simulated failure")

            def remove_document(self, doc_id: str) -> None:
                raise RuntimeError("Simulated failure")

            def remove_documents(
                self,
                doc_ids: list[str],
                persist: bool = False,
            ) -> None:
                raise RuntimeError("Simulated failure")

            def persist(self) -> None:
                raise AssertionError("persist should not be called after failed task")

            def index_record(self, record: Record) -> None:
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

    def test_refresh_git_task_ingests_via_content_source(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        register_tasks(huey_instance, fake_manager)

        observed: dict[str, object] = {}

        class _FakeIngestor:
            def __init__(self, index_manager):
                observed["index_manager"] = index_manager

            async def index_records(self, records, *, checkpoint=None):
                observed["since"] = checkpoint
                return type("Receipt", (), {"records": (object(),)})()

        monkeypatch.setattr(
            "mcp_markdown_ragdocs.indexing.tasks.AsyncIndexIngestor",
            _FakeIngestor,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.adapters.sources.git.GitContentSource.iter_records",
            lambda source, since: observed.update(repo_path=source.repo_path) or [],
        )

        git_dir = tmp_path / "repo" / ".git"
        git_dir.parent.mkdir(parents=True)
        git_dir.mkdir()

        enqueue_refresh_git(str(git_dir))
        task = huey_instance.dequeue()
        huey_instance.execute(task)

        assert observed["index_manager"] is fake_manager
        assert observed["repo_path"] == git_dir.parent
        assert observed["since"] is None
        assert fake_manager.persist_calls == 1

    def test_refresh_git_task_uses_cursor_and_persists_cursor(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        state_root = tmp_path / "index"
        git_dir = tmp_path / "repo" / ".git"
        git_dir.parent.mkdir(parents=True)
        git_dir.mkdir()
        save_cursor(state_root, git_dir, 123)
        observed: dict[str, object] = {}
        record = Record(
            source_kind="git_commit",
            source_id="git:abc",
            title="Commit",
            body="Body",
            created_at=datetime.fromtimestamp(124, tz=UTC),
            updated_at=datetime.fromtimestamp(124, tz=UTC),
        )

        class _FakeIngestor:
            def __init__(self, _index_manager):
                pass

            async def index_records(self, records, *, checkpoint=None):
                observed["since"] = checkpoint
                return type("Receipt", (), {"records": (record,)})()

        monkeypatch.setattr(
            "mcp_markdown_ragdocs.indexing.tasks.AsyncIndexIngestor",
            _FakeIngestor,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.adapters.sources.git.GitContentSource.iter_records",
            lambda _source, since: [record],
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.indexing.tasks.get_git_ref_signature",
            lambda _git_dir: "head-1",
        )

        register_tasks(
            huey_instance,
            fake_manager,
            bootstrap_index_path=state_root,
        )
        enqueue_refresh_git(str(git_dir))
        task = huey_instance.dequeue()
        assert task is not None
        assert huey_instance.execute(task) is True

        assert observed["since"] == "122"
        assert fake_manager.persist_calls == 1
        assert get_cursor(state_root, git_dir) == 124

    def test_refresh_git_task_skips_completed_head(
        self,
        huey_instance: SqliteHuey,
        fake_manager: FakeIndexManager,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        state_root = tmp_path / "index"
        git_dir = tmp_path / "repo" / ".git"
        git_dir.parent.mkdir(parents=True)
        git_dir.mkdir()
        save_head(state_root, git_dir, "head-1")
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.indexing.tasks.get_git_ref_signature",
            lambda _git_dir: "head-1",
        )

        def _unexpected_ingest(*args, **kwargs):
            raise AssertionError("unchanged repository should not be ingested")

        monkeypatch.setattr(
            "mcp_markdown_ragdocs.indexing.tasks.AsyncIndexIngestor",
            _unexpected_ingest,
        )
        register_tasks(
            huey_instance,
            fake_manager,
            bootstrap_index_path=state_root,
        )
        enqueue_refresh_git(str(git_dir))
        task = huey_instance.dequeue()
        assert task is not None

        assert huey_instance.execute(task) is True
        assert fake_manager.persist_calls == 0
