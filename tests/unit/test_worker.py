"""
Unit tests for HueyWorker consumer management.

Commit 3.2: Verifies the Huey consumer thread lifecycle.
"""

from __future__ import annotations

import errno
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from huey import SqliteHuey

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.indexing import tasks as indexing_tasks
from mcp_markdown_ragdocs.worker.consumer import HueyWorker, _promote_due_tasks


@pytest.fixture()
def huey_instance(tmp_path: Path) -> SqliteHuey:
    return SqliteHuey(name="test", filename=str(tmp_path / "queue.db"), immediate=False)


def _register_tasks(huey: SqliteHuey, index_manager: object) -> Any:
    queue_path = Path(cast(Any, huey.storage).filename)
    return indexing_tasks.register_tasks(
        huey,
        cast(Any, index_manager),
        TaskLeaseStore(queue_path),
        WorkIntentStore(queue_path),
        config=Config(),
    )


class TestHueyWorker:
    def test_promotes_overdue_scheduled_task(self, huey_instance: SqliteHuey) -> None:
        """
        An overdue scheduled task becomes available to the worker queue.
        """
        results: list[int] = []

        @huey_instance.task()
        def append_value(value: int) -> None:
            results.append(value)

        huey_instance.add_schedule(append_value.s(1, eta=datetime.now(timezone.utc)))
        _promote_due_tasks(huey_instance)
        _promote_due_tasks(huey_instance)

        assert huey_instance.pending_count() == 1
        task = huey_instance.dequeue()
        assert task is not None
        huey_instance.execute(task)
        assert results == [1]

    def test_promotion_transaction_rolls_back_on_queue_failure(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        """
        A failed queue insert keeps the due schedule row for recovery.
        """
        @huey_instance.task()
        def scheduled_task() -> None:
            return

        huey_instance.add_schedule(
            scheduled_task.s(eta=datetime.now(timezone.utc))
        )
        with huey_instance.storage.db(commit=True) as cursor:
            cursor.execute(
                """
                CREATE TRIGGER fail_task_insert
                BEFORE INSERT ON task
                BEGIN
                    SELECT RAISE(ABORT, 'queue unavailable');
                END
                """
            )

        with pytest.raises(sqlite3.IntegrityError, match="queue unavailable"):
            _promote_due_tasks(huey_instance)

        assert huey_instance.storage.schedule_size() == 1
        assert huey_instance.pending_count() == 0

    def test_malformed_scheduled_task_remains_recoverable(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        """
        An undecodable due row is retained instead of being discarded.
        """
        with huey_instance.storage.db(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO schedule (queue, data, timestamp) VALUES (?, ?, ?)",
                (huey_instance.storage.name, b"malformed", 0),
            )

        _promote_due_tasks(huey_instance)

        assert huey_instance.storage.schedule_size() == 1

    def test_promotion_limits_one_batch(
        self,
        huey_instance: SqliteHuey,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Promotion leaves later due rows for a subsequent worker iteration.
        """
        @huey_instance.task()
        def scheduled_task(value: int) -> None:
            del value

        for value in (1, 2):
            huey_instance.add_schedule(
                scheduled_task.s(value, eta=datetime.now(timezone.utc))
            )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.SCHEDULE_PROMOTION_BATCH_SIZE",
            1,
        )

        _promote_due_tasks(huey_instance)

        assert huey_instance.pending_count() == 1
        assert huey_instance.storage.schedule_size() == 1

    def test_start_creates_consumer_thread(self, huey_instance: SqliteHuey) -> None:
        """start() creates and starts a consumer thread."""
        worker = HueyWorker(huey_instance)
        worker.start()
        assert worker.is_running
        worker.stop()

    def test_stop_terminates_consumer(self, huey_instance: SqliteHuey) -> None:
        """stop() terminates the consumer thread."""
        worker = HueyWorker(huey_instance)
        worker.start()
        assert worker.is_running
        worker.stop(timeout=2.0)
        assert not worker.is_running

    def test_consumer_processes_tasks(self, huey_instance: SqliteHuey) -> None:
        """Consumer processes enqueued tasks."""
        results: list[int] = []

        @huey_instance.task()
        def append_value(x: int) -> None:
            results.append(x)

        # Enqueue a task
        append_value(42)
        assert huey_instance.pending_count() == 1

        # Start worker to process it
        worker = HueyWorker(huey_instance)
        worker.start()

        # Wait for task to be processed
        deadline = time.monotonic() + 5.0
        while huey_instance.pending_count() > 0 and time.monotonic() < deadline:
            time.sleep(0.1)

        worker.stop()
        assert huey_instance.pending_count() == 0
        assert 42 in results

    def test_consumer_reports_active_lease_during_execution(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        @huey_instance.task()
        def blocking_task() -> None:
            started.set()
            release.wait(timeout=5.0)

        blocking_task()
        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            assert started.wait(timeout=5.0)
            lease_store = TaskLeaseStore(
                Path(str(getattr(huey_instance.storage, "filename")))
            )
            assert lease_store.active_count() == 1
            release.set()
            deadline = time.monotonic() + 5.0
            while lease_store.active_count() > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert lease_store.active_count() == 0
        finally:
            release.set()
            worker.stop()

    def test_consumer_marks_huey_errors_failed(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        @huey_instance.task()
        def failing_task() -> None:
            raise RuntimeError("expected failure")

        result = failing_task()
        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while huey_instance.pending_count() > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            worker.stop()

        lease = TaskLeaseStore(
            Path(str(getattr(huey_instance.storage, "filename")))
        ).get(result.id)
        assert lease is not None
        assert lease.state == "failed"
        assert lease.error is not None
        assert "expected failure" in lease.error

    def test_start_requeues_expired_lease(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        results: list[int] = []

        @huey_instance.task()
        def append_value(x: int) -> None:
            results.append(x)

        append_value(7)
        task = huey_instance.dequeue()
        assert task is not None
        lease_store = TaskLeaseStore(
            Path(str(getattr(huey_instance.storage, "filename"))),
            timeout_seconds=1.0,
        )
        assert lease_store.claim(
            task.id,
            task_name=task.name,
            owner_token="stale-owner",
            payload=huey_instance.serialize_task(task),
            now=100.0,
        )

        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while not results and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            worker.stop()

        assert results == [7]

    def test_expired_drive_scope_lease_does_not_stop_worker(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        """
        Expired Drive scope ownership must not be treated as a queued task.
        The worker stays alive so the next Drive task can reclaim the scope.
        """
        lease_store = TaskLeaseStore(
            Path(str(getattr(huey_instance.storage, "filename"))),
        )
        assert lease_store.claim(
            "gdrive-scope:workspace:shared-drive:drive-1",
            task_name="gdrive_scope_sync",
            owner_token="stale-owner",
            payload=b'{"scope_identity":"shared-drive:drive-1"}',
            now=time.time() - 60.0,
        )

        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            time.sleep(0.1)
            assert worker.is_running
        finally:
            worker.stop()

    def test_requeues_lease_that_expires_after_start(
        self,
        huey_instance: SqliteHuey,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        results: list[int] = []

        @huey_instance.task()
        def append_value(x: int) -> None:
            results.append(x)

        append_value(11)
        task = huey_instance.dequeue()
        assert task is not None
        lease_store = TaskLeaseStore(
            Path(str(cast(Any, huey_instance.storage).filename)),
            timeout_seconds=0.1,
        )
        assert lease_store.claim(
            task.id,
            task_name=task.name,
            owner_token="stale-owner",
            payload=huey_instance.serialize_task(task),
            now=time.time() + 0.2,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_TIMEOUT_SECONDS",
            0.1,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_RECLAIM_INTERVAL_SECONDS",
            0.01,
        )

        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while not results and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            worker.stop()

        assert results == [11]
        lease = lease_store.get(task.id)
        assert lease is not None
        assert lease.state == "completed"
        assert lease.attempt == 2

    def test_requeues_delayed_intent_with_a_fresh_claim_token(
        self,
        huey_instance: SqliteHuey,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        class _Manager:
            ingestor = object()

            def index_document(self, file_path: str, force: bool = False) -> bool:
                del file_path, force
                return True

            def persist(self) -> None:
                return

            def index_documents(
                self,
                file_paths: list[str],
                force: bool = False,
                persist: bool = False,
            ) -> None:
                del file_paths, force, persist

            def remove_document(self, doc_id: str) -> None:
                del doc_id

            def remove_documents(self, doc_ids: list[str], persist: bool = False) -> None:
                del doc_ids, persist

            def index_record(self, record: Any) -> None:
                del record

        runtime = _register_tasks(huey_instance, cast(Any, _Manager()))
        queue_path = Path(str(cast(Any, huey_instance.storage).filename))
        runtime.work_intent_store = WorkIntentStore(
            queue_path,
            claim_timeout_seconds=0.1,
        )
        assert runtime.submission.submit_index_request("delayed.md").enqueued
        task = huey_instance.dequeue()
        assert task is not None
        lease_store = TaskLeaseStore(queue_path, timeout_seconds=0.1)
        assert lease_store.claim(
            task.id,
            task_name=task.name,
            owner_token="stale-owner",
            payload=huey_instance.serialize_task(task),
            now=time.time() - 1.0,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_TIMEOUT_SECONDS",
            0.1,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_RECLAIM_INTERVAL_SECONDS",
            0.01,
        )

        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while huey_instance.pending_count() > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            worker.stop()

        intent = runtime.work_intent_store.find(
            "index_document", "delayed.md"
        )
        assert intent is not None
        assert intent.state == "succeeded"

    def test_delayed_queued_intent_survives_claim_timeout(
        self,
        huey_instance: SqliteHuey,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _Manager:
            ingestor = object()

            def index_document(self, file_path: str, force: bool = False) -> bool:
                del file_path, force
                return True

            def persist(self) -> None:
                return

        runtime = _register_tasks(huey_instance, cast(Any, _Manager()))
        queue_path = Path(str(cast(Any, huey_instance.storage).filename))
        runtime.work_intent_store = WorkIntentStore(
            queue_path,
            claim_timeout_seconds=0.1,
        )
        assert runtime.submission.submit_index_request("queued.md").enqueued
        with sqlite3.connect(queue_path) as connection:
            connection.execute(
                "UPDATE work_intents SET claim_observed_at = ?",
                (time.time() - 1.0,),
            )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_TIMEOUT_SECONDS",
            0.1,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_RECLAIM_INTERVAL_SECONDS",
            0.01,
        )

        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while huey_instance.pending_count() > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            worker.stop()

        intent = runtime.work_intent_store.find("index_document", "queued.md")
        assert intent is not None
        assert intent.state == "succeeded"

    def test_long_running_intent_claim_is_not_recovered(
        self,
        huey_instance: SqliteHuey,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        class _Manager:
            ingestor = object()

            def index_document(self, file_path: str, force: bool = False) -> bool:
                del file_path, force
                started.set()
                release.wait(timeout=5.0)
                return True

            def persist(self) -> None:
                return

        runtime = _register_tasks(huey_instance, cast(Any, _Manager()))
        queue_path = Path(str(cast(Any, huey_instance.storage).filename))
        runtime.work_intent_store = WorkIntentStore(
            queue_path,
            claim_timeout_seconds=0.1,
        )
        assert runtime.submission.submit_index_request("long-running.md").enqueued
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_TIMEOUT_SECONDS",
            0.1,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_RECLAIM_INTERVAL_SECONDS",
            0.01,
        )

        worker = HueyWorker(huey_instance)
        worker.start()
        try:
            assert started.wait(timeout=5.0)
            time.sleep(0.25)
            intent = runtime.work_intent_store.find(
                "index_document", "long-running.md"
            )
            assert intent is not None
            assert intent.state == "running"
            release.set()
            deadline = time.monotonic() + 5.0
            while huey_instance.pending_count() > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            release.set()
            worker.stop()

        intent = runtime.work_intent_store.find(
            "index_document", "long-running.md"
        )
        assert intent is not None
        assert intent.state == "succeeded"

    def test_pool_processes_tasks_concurrently(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        """Multiple worker threads dequeue and run tasks at the same time."""
        active = 0
        max_active = 0
        lock = threading.Lock()
        release = threading.Event()

        @huey_instance.task()
        def blocking_task(_: int) -> None:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            release.wait(timeout=5.0)
            with lock:
                active -= 1

        blocking_task(1)
        blocking_task(2)

        worker = HueyWorker(huey_instance, workers=2)
        worker.start()
        try:
            deadline = time.monotonic() + 5.0
            while max_active < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert max_active == 2
        finally:
            release.set()
            worker.stop()

    def test_only_one_pool_thread_reclaims_expired_leases(
        self,
        huey_instance: SqliteHuey,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only the designated reclaimer thread runs periodic reclaim."""
        thread_idents: set[int] = set()
        lock = threading.Lock()

        def fake_requeue(huey: object, lease_store: object) -> None:
            del huey, lease_store
            with lock:
                thread_idents.add(threading.get_ident())

        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer._requeue_expired_leases",
            fake_requeue,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.worker.consumer.LEASE_RECLAIM_INTERVAL_SECONDS",
            0.01,
        )
        main_ident = threading.get_ident()

        worker = HueyWorker(huey_instance, workers=3)
        worker.start()
        try:
            time.sleep(0.3)
        finally:
            worker.stop()

        non_main_idents = thread_idents - {main_ident}
        assert len(non_main_idents) == 1

    def test_stop_joins_pool_and_clears_running_state(
        self,
        huey_instance: SqliteHuey,
    ) -> None:
        """stop() joins every spawned thread and leaves is_running False."""
        worker = HueyWorker(huey_instance, workers=3)
        worker.start()
        assert worker.is_running
        worker.stop(timeout=2.0)
        assert not worker.is_running

    def test_double_start_is_safe(self, huey_instance: SqliteHuey) -> None:
        """Starting twice doesn't create duplicate threads."""
        worker = HueyWorker(huey_instance)
        worker.start()
        worker.start()  # Should log warning, not error
        assert worker.is_running
        worker.stop()

    def test_stop_without_start_is_safe(self, huey_instance: SqliteHuey) -> None:
        """Stopping without starting is a no-op."""
        worker = HueyWorker(huey_instance)
        worker.stop()  # Should not error
        assert not worker.is_running


class TestWorkerWithLifecycle:
    @pytest.mark.asyncio
    async def test_worker_spawns_on_leader(self, tmp_path: Path) -> None:
        """Huey worker only runs when lifecycle state is READY_PRIMARY."""
        from dataclasses import dataclass, field
        from typing import Any, cast

        from searchkernel.api import DatabaseManager

        from mcp_markdown_ragdocs.lifecycle import LifecycleCoordinator, LifecycleState

        @dataclass
        class FakeGitConfig:
            enabled: bool = False
            watch_enabled: bool = False

        @dataclass
        class FakeIndexingConfig:
            documents_path: str = "/tmp"
            exclude: list[str] = field(default_factory=list)
            exclude_hidden_dirs: bool = True

        @dataclass
        class FakeConfig:
            git_indexing: FakeGitConfig = field(default_factory=FakeGitConfig)
            indexing: FakeIndexingConfig = field(default_factory=FakeIndexingConfig)

        @dataclass
        class FakeContext:
            config: FakeConfig = field(default_factory=FakeConfig)
            git_indexing_enabled: bool = False

            async def start(self, background_index: bool = False) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def ensure_ready(self, timeout: float = 60.0) -> None:
                pass

        db = DatabaseManager(tmp_path / "test.db")
        huey = SqliteHuey(
            name="test", filename=str(tmp_path / "queue.db"), immediate=False
        )
        worker = HueyWorker(huey)

        # Start as primary with worker
        coord = LifecycleCoordinator()
        await coord.start(cast(Any, FakeContext()), db_manager=db, huey_worker=worker)

        assert coord.state == LifecycleState.READY_PRIMARY
        assert worker.is_running

        # Shutdown should stop worker
        await coord.shutdown()
        assert not worker.is_running

    @pytest.mark.asyncio
    async def test_worker_not_spawned_on_replica(self, tmp_path: Path) -> None:
        """Huey worker does NOT start when lifecycle state is READY_REPLICA."""
        from dataclasses import dataclass, field
        from typing import Any, cast

        from searchkernel.api import DatabaseManager

        from mcp_markdown_ragdocs.lifecycle import LifecycleCoordinator, LifecycleState

        @dataclass
        class FakeGitConfig:
            enabled: bool = False
            watch_enabled: bool = False

        @dataclass
        class FakeIndexingConfig:
            documents_path: str = "/tmp"
            exclude: list[str] = field(default_factory=list)
            exclude_hidden_dirs: bool = True

        @dataclass
        class FakeConfig:
            git_indexing: FakeGitConfig = field(default_factory=FakeGitConfig)
            indexing: FakeIndexingConfig = field(default_factory=FakeIndexingConfig)

        @dataclass
        class FakeContext:
            config: FakeConfig = field(default_factory=FakeConfig)
            git_indexing_enabled: bool = False

            async def start(self, background_index: bool = False) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def ensure_ready(self, timeout: float = 60.0) -> None:
                pass

        db = DatabaseManager(tmp_path / "test.db")
        huey = SqliteHuey(
            name="test", filename=str(tmp_path / "queue.db"), immediate=False
        )
        worker = HueyWorker(huey)

        # First coordinator takes primary
        coord1 = LifecycleCoordinator()
        await coord1.start(cast(Any, FakeContext()), db_manager=db)
        assert coord1.state == LifecycleState.READY_PRIMARY

        # Second coordinator becomes replica — worker should NOT start
        coord2 = LifecycleCoordinator()
        await coord2.start(cast(Any, FakeContext()), db_manager=db, huey_worker=worker)
        assert coord2.state == LifecycleState.READY_REPLICA
        assert not worker.is_running

        await coord1.shutdown()
        await coord2.shutdown()


class TestWorkerRuntimeStartup:
    @pytest.mark.asyncio
    async def test_worker_runtime_continues_when_file_watcher_hits_emfile(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        from mcp_markdown_ragdocs.cli import _run_worker_forever_async
        from mcp_markdown_ragdocs.daemon.paths import RuntimePaths

        class _FakeIndexManager:
            def load(self) -> None:
                return None

        class _FakeWatcher:
            def __init__(self) -> None:
                self.stop_calls = 0

            def start(self) -> None:
                raise OSError(errno.EMFILE, "Too many open files")

            async def stop(self) -> None:
                self.stop_calls += 1

        class _FakeGitConfig:
            watch_enabled = False

        class _FakeIndexingConfig:
            task_backpressure_limit = 100
            embedding_cache_prune_cooldown_seconds = 86400

        class _FakeConfig:
            git_indexing = _FakeGitConfig()
            indexing = _FakeIndexingConfig()

        class _FakeContext:
            def __init__(self) -> None:
                self.index_manager = _FakeIndexManager()
                self.watcher = _FakeWatcher()
                self.git_indexing_enabled = False
                self.config = _FakeConfig()
                self.index_path = tmp_path / "index_data"
                self.documents_roots: list[Path] = []

            def attach_task_runtime(self, task_runtime: object) -> None:
                self.task_runtime = task_runtime

        class _FakeHueyWorker:
            def __init__(self, _huey: object) -> None:
                self.is_running = True
                self.start_calls = 0
                self.stop_calls = 0

            def start(self) -> None:
                self.start_calls += 1

            def stop(self, timeout: float = 5.0) -> None:
                self.stop_calls += 1
                self.is_running = False

        fake_ctx = _FakeContext()
        fake_worker = _FakeHueyWorker(object())
        runtime_paths = RuntimePaths(
            root=tmp_path,
            index_db_path=tmp_path / "index.db",
            queue_db_path=tmp_path / "queue.db",
            metadata_path=tmp_path / "daemon.json",
            lock_path=tmp_path / "daemon.lock",
            socket_path=tmp_path / "daemon.sock",
        )
        fake_queue_runtime = SimpleNamespace(
            huey=object(),
            db_path=runtime_paths.queue_db_path,
        )

        monkeypatch.setattr("mcp_markdown_ragdocs.context.ApplicationContext.create", lambda **kwargs: fake_ctx)
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.coordination.queue.build_queue_runtime",
            lambda path: fake_queue_runtime,
        )
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.indexing.tasks.register_tasks",
            lambda *args, **kwargs: SimpleNamespace(
                queue_runtime=fake_queue_runtime,
                submission=SimpleNamespace(submit_refresh_git_request=lambda git_dir: None),
            ),
        )
        monkeypatch.setattr("mcp_markdown_ragdocs.worker.consumer.HueyWorker", lambda _huey: fake_worker)
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.cli._parent_process_alive",
            lambda _pid, _parent_start_time=None: False,
        )
        resolve_calls = 0

        def _unexpected_resolve(cls):
            nonlocal resolve_calls
            resolve_calls += 1
            return runtime_paths

        monkeypatch.setattr(RuntimePaths, "resolve", classmethod(_unexpected_resolve))

        await _run_worker_forever_async(None, runtime_paths.queue_db_path, runtime_paths.root, 123)

        assert fake_ctx.watcher.stop_calls >= 1
        assert fake_worker.start_calls == 1
        assert fake_worker.stop_calls >= 1
        assert resolve_calls == 0
