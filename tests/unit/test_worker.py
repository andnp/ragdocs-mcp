"""
Unit tests for HueyWorker consumer management.

Commit 3.2: Verifies the Huey consumer thread lifecycle.
"""

from __future__ import annotations

import errno
import time
from pathlib import Path

import pytest
from huey import SqliteHuey

from src.worker.consumer import HueyWorker


@pytest.fixture()
def huey_instance(tmp_path: Path) -> SqliteHuey:
    return SqliteHuey(name="test", filename=str(tmp_path / "queue.db"), immediate=False)


class TestHueyWorker:
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

        from src.lifecycle import LifecycleCoordinator, LifecycleState
        from src.storage.db import DatabaseManager

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
            commit_indexer: None = None

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

        from src.lifecycle import LifecycleCoordinator, LifecycleState
        from src.storage.db import DatabaseManager

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
            commit_indexer: None = None

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
        from src.cli import _run_worker_forever_async
        from src.daemon.paths import RuntimePaths

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

        class _FakeConfig:
            git_indexing = _FakeGitConfig()
            indexing = _FakeIndexingConfig()

        class _FakeContext:
            def __init__(self) -> None:
                self.index_manager = _FakeIndexManager()
                self.watcher = _FakeWatcher()
                self.commit_indexer = None
                self.config = _FakeConfig()

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

        monkeypatch.setattr("src.cli.ApplicationContext.create", lambda **kwargs: fake_ctx)
        monkeypatch.setattr("src.cli.get_huey", lambda _path: object())
        monkeypatch.setattr("src.cli.register_tasks", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.cli.HueyWorker", lambda _huey: fake_worker)
        monkeypatch.setattr("src.cli._parent_process_alive", lambda _pid: False)
        monkeypatch.setattr(
            RuntimePaths,
            "resolve",
            classmethod(lambda cls: runtime_paths),
        )

        await _run_worker_forever_async(None, runtime_paths.queue_db_path, runtime_paths.root, 123)

        assert fake_ctx.watcher.stop_calls >= 1
        assert fake_worker.start_calls == 1
        assert fake_worker.stop_calls >= 1
