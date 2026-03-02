"""
Unit tests for HueyWorker consumer management.

Commit 3.2: Verifies the Huey consumer thread lifecycle.
"""

from __future__ import annotations

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
        huey = SqliteHuey(name="test", filename=str(tmp_path / "queue.db"), immediate=False)
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
        huey = SqliteHuey(name="test", filename=str(tmp_path / "queue.db"), immediate=False)
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
