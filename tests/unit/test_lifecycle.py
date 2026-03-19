"""
Unit tests for LifecycleCoordinator state machine, emergency timer, and signal handling.

Commit 2.1: Verifies LifecycleCoordinator is the source of truth for process state.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.lifecycle import LifecycleCoordinator, LifecycleState
from src.storage.db import DatabaseManager


# ---------------------------------------------------------------------------
# Lightweight stubs (no mock library — real objects with minimal behavior)
# ---------------------------------------------------------------------------


@dataclass
class FakeGitConfig:
    enabled: bool = False
    watch_enabled: bool = False
    poll_interval_seconds: float = 30.0


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
    """Minimal stub to satisfy LifecycleCoordinator.start()."""

    config: FakeConfig = field(default_factory=FakeConfig)
    commit_indexer: None = None

    async def start(self, background_index: bool = False) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        pass

    def discover_git_repositories(self) -> list[str]:
        return []


def _make_coordinator(**overrides: Any) -> LifecycleCoordinator:
    return LifecycleCoordinator(**overrides)


def _fake_ctx() -> Any:
    """Return a FakeContext cast to Any so it satisfies the ApplicationContext param."""
    return cast(Any, FakeContext())


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------


class TestLifecycleStateMachine:
    def test_initial_state_is_uninitialized(self) -> None:
        """Fresh coordinator starts in UNINITIALIZED."""
        coord = _make_coordinator()
        assert coord.state == LifecycleState.UNINITIALIZED

    @pytest.mark.asyncio
    async def test_start_transitions_to_ready(self) -> None:
        """start() transitions: UNINITIALIZED -> STARTING -> READY."""
        coord = _make_coordinator()
        await coord.start(_fake_ctx())
        assert coord.state == LifecycleState.READY

    @pytest.mark.asyncio
    async def test_start_with_background_index_stays_initializing(self) -> None:
        """start(background_index=True) transitions to INITIALIZING, not READY."""
        coord = _make_coordinator()
        await coord.start(_fake_ctx(), background_index=True)
        assert coord.state == LifecycleState.INITIALIZING

    @pytest.mark.asyncio
    async def test_start_runs_huey_worker_in_thread(self, monkeypatch) -> None:
        """Huey worker startup is offloaded so daemon startup stays responsive."""
        coord = _make_coordinator()

        class _FakeLeaderElection:
            def __init__(self, _db: object) -> None:
                self.is_leader = False

            def try_acquire(self) -> bool:
                self.is_leader = True
                return True

        class _FakeWorker:
            def __init__(self) -> None:
                self.start_calls = 0

            def start(self) -> None:
                self.start_calls += 1

            def is_healthy(self) -> bool:
                return True

            def stop(self, timeout: float = 5.0) -> None:
                return None

        to_thread_calls: list[str] = []

        async def _fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append(getattr(func, "__name__", repr(func)))
            return func(*args, **kwargs)

        monkeypatch.setattr("src.lifecycle.LeaderElection", _FakeLeaderElection)
        monkeypatch.setattr("src.lifecycle.asyncio.to_thread", _fake_to_thread)

        worker = _FakeWorker()
        await coord.start(_fake_ctx(), db_manager=object(), huey_worker=worker)

        assert worker.start_calls == 1
        assert "start" in to_thread_calls
        assert coord.state == LifecycleState.READY_PRIMARY

    @pytest.mark.asyncio
    async def test_start_uses_context_git_discovery_for_git_watcher(
        self,
        monkeypatch,
    ) -> None:
        coord = _make_coordinator()
        observed: dict[str, object] = {"discover_calls": 0}

        @dataclass
        class _GitContext(FakeContext):
            config: FakeConfig = field(
                default_factory=lambda: FakeConfig(
                    git_indexing=FakeGitConfig(enabled=True, watch_enabled=True),
                )
            )
            commit_indexer: object | None = field(default_factory=object)

            def discover_git_repositories(self) -> list[str]:
                observed["discover_calls"] = observed["discover_calls"] + 1
                return ["/tmp/project-a/.git", "/tmp/project-b/.git"]

        class _FakeGitWatcher:
            def __init__(
                self,
                *,
                git_repos,
                commit_indexer,
                config,
                poll_interval,
                use_tasks,
            ) -> None:
                observed["git_repos"] = git_repos
                observed["commit_indexer"] = commit_indexer
                observed["use_tasks"] = use_tasks

            def start(self) -> None:
                observed["started"] = True

            async def stop(self) -> None:
                return None

        async def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("src.lifecycle.GitWatcher", _FakeGitWatcher)
        monkeypatch.setattr("src.lifecycle.asyncio.to_thread", _fake_to_thread)

        ctx = cast(Any, _GitContext())
        await coord.start(ctx)

        assert observed["discover_calls"] == 1
        assert observed["git_repos"] == [
            "/tmp/project-a/.git",
            "/tmp/project-b/.git",
        ]
        assert observed["started"] is True

    @pytest.mark.asyncio
    async def test_cannot_start_twice(self) -> None:
        """start() raises RuntimeError if already started."""
        coord = _make_coordinator()
        await coord.start(_fake_ctx())
        with pytest.raises(RuntimeError, match="Cannot start from state"):
            await coord.start(_fake_ctx())

    @pytest.mark.asyncio
    async def test_shutdown_transitions_to_terminated(self) -> None:
        """shutdown() transitions to TERMINATED."""
        coord = _make_coordinator()
        await coord.start(_fake_ctx())
        assert coord.state == LifecycleState.READY
        await coord.shutdown()
        assert coord.state == LifecycleState.TERMINATED

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self) -> None:
        """Multiple shutdown() calls don't error."""
        coord = _make_coordinator()
        await coord.start(_fake_ctx())
        await coord.shutdown()
        assert coord.state == LifecycleState.TERMINATED
        # Second call should be a no-op
        await coord.shutdown()
        assert coord.state == LifecycleState.TERMINATED

    def test_request_shutdown_sets_shutting_down(self) -> None:
        """request_shutdown() sets state to SHUTTING_DOWN."""
        coord = _make_coordinator()
        # Manually move to READY so request_shutdown has something to do
        coord._state = LifecycleState.READY
        with patch.object(coord, "_close_stdin"):
            coord.request_shutdown()
        assert coord.state == LifecycleState.SHUTTING_DOWN
        coord._cancel_emergency_timer()


# ---------------------------------------------------------------------------
# Emergency timer tests
# ---------------------------------------------------------------------------


class TestEmergencyTimer:
    def test_emergency_timer_starts_on_shutdown_request(self) -> None:
        """Emergency timer is set when shutdown is requested."""
        coord = _make_coordinator()
        coord._state = LifecycleState.READY
        assert coord._emergency_timer is None

        with patch.object(coord, "_close_stdin"):
            coord.request_shutdown()

        assert coord._emergency_timer is not None
        assert coord._emergency_timer.is_alive()
        # Clean up to avoid the timer firing during test
        coord._cancel_emergency_timer()

    @pytest.mark.asyncio
    async def test_emergency_timer_cancelled_on_clean_shutdown(self) -> None:
        """Timer is cancelled when shutdown completes normally."""
        coord = _make_coordinator()
        await coord.start(_fake_ctx())

        # Patch _close_stdin to prevent it from closing fd 0 in the test process
        with patch.object(coord, "_close_stdin"):
            coord.request_shutdown()
            assert coord._emergency_timer is not None

        # Full shutdown should cancel it
        await coord.shutdown()
        assert coord._emergency_timer is None


# ---------------------------------------------------------------------------
# Double-signal / force-exit tests
# ---------------------------------------------------------------------------


class TestDoubleSignal:
    def test_second_signal_triggers_force_exit(self) -> None:
        """Second shutdown request calls _force_exit."""
        coord = _make_coordinator()
        coord._state = LifecycleState.READY

        # First signal → SHUTTING_DOWN
        with patch.object(coord, "_close_stdin"):
            coord.request_shutdown()
        assert coord.state == LifecycleState.SHUTTING_DOWN

        # Second signal → force exit
        with patch.object(coord, "_force_exit") as mock_exit:
            coord.request_shutdown()
            mock_exit.assert_called_once()

        # Clean up
        coord._cancel_emergency_timer()


class TestWorkerSupervision:
    @pytest.mark.asyncio
    async def test_supervision_restarts_unhealthy_worker(self) -> None:
        coord = _make_coordinator()
        coord._state = LifecycleState.READY_PRIMARY

        class _FakeLeader:
            is_leader = True

        class _FakeWorker:
            def __init__(self) -> None:
                self.restart_calls = 0

            def is_healthy(self) -> bool:
                return self.restart_calls > 0

            def restart(self, timeout: float = 5.0) -> None:
                self.restart_calls += 1

        worker = _FakeWorker()
        coord._leader_election = _FakeLeader()
        coord._huey_worker = worker

        sleep_calls = 0

        async def _fake_sleep(_seconds: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError

        with patch("src.lifecycle.asyncio.sleep", _fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await coord._supervise_worker_health()

        assert worker.restart_calls == 1


class TestLeaderFailover:
    @pytest.mark.asyncio
    async def test_replica_promotes_to_primary_after_timeout_and_starts_worker(
        self,
        db: DatabaseManager,
    ) -> None:
        primary = _make_coordinator()
        primary._leader_monitor_interval = 0.01
        await primary.start(_fake_ctx(), db_manager=db)
        assert primary.state == LifecycleState.READY_PRIMARY

        replica = _make_coordinator()
        replica._leader_monitor_interval = 0.01

        class _FakeWorker:
            def __init__(self) -> None:
                self.start_calls = 0

            def start(self) -> None:
                self.start_calls += 1

            def is_healthy(self) -> bool:
                return True

            def stop(self, timeout: float = 5.0) -> None:
                return None

        worker = _FakeWorker()
        await replica.start(_fake_ctx(), db_manager=db, huey_worker=worker)
        assert replica.state == LifecycleState.READY_REPLICA
        assert replica._leader_monitor_task is not None

        if primary._leader_heartbeat_task is not None:
            primary._leader_heartbeat_task.cancel()
            await asyncio.gather(
                primary._leader_heartbeat_task,
                return_exceptions=True,
            )
            primary._leader_heartbeat_task = None

        conn = db.get_connection()
        stale_data = json.dumps(
            {
                "instance_id": primary._leader_election.instance_id,
                "heartbeat": time.time() - 30.0,
                "acquired_at": time.time() - 30.0,
            }
        )
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            ("leader_id", stale_data),
        )
        conn.commit()

        deadline = time.monotonic() + 1.0
        while (
            replica.state != LifecycleState.READY_PRIMARY
            or worker.start_calls != 1
            or replica._worker_supervision_task is None
        ):
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "Replica did not promote to primary and start worker supervision"
                )
            await asyncio.sleep(0.01)

        assert worker.start_calls == 1
        assert replica._worker_supervision_task is not None

        await replica.shutdown()
        await primary.shutdown()

    @pytest.mark.asyncio
    async def test_initializing_replica_acquires_leadership_and_stays_initializing(
        self,
        db: DatabaseManager,
    ) -> None:
        primary = _make_coordinator()
        primary._leader_monitor_interval = 0.01
        await primary.start(_fake_ctx(), db_manager=db)
        assert primary.state == LifecycleState.READY_PRIMARY

        class _BlockingReadyContext(FakeContext):
            def __init__(self) -> None:
                super().__init__()
                self.ready_event = asyncio.Event()

            async def ensure_ready(self, timeout: float = 60.0) -> None:
                if timeout < 60.0:
                    await asyncio.wait_for(self.ready_event.wait(), timeout=timeout)
                    return
                await self.ready_event.wait()

        replica = _make_coordinator()
        replica._leader_monitor_interval = 0.01

        class _FakeWorker:
            def __init__(self) -> None:
                self.start_calls = 0

            def start(self) -> None:
                self.start_calls += 1

            def is_healthy(self) -> bool:
                return True

            def stop(self, timeout: float = 5.0) -> None:
                return None

        ctx = cast(Any, _BlockingReadyContext())
        worker = _FakeWorker()
        await replica.start(
            ctx,
            background_index=True,
            db_manager=db,
            huey_worker=worker,
        )
        assert replica.state == LifecycleState.INITIALIZING
        assert replica._leader_monitor_task is not None

        if primary._leader_heartbeat_task is not None:
            primary._leader_heartbeat_task.cancel()
            await asyncio.gather(
                primary._leader_heartbeat_task,
                return_exceptions=True,
            )
            primary._leader_heartbeat_task = None

        conn = db.get_connection()
        stale_data = json.dumps(
            {
                "instance_id": primary._leader_election.instance_id,
                "heartbeat": time.time() - 30.0,
                "acquired_at": time.time() - 30.0,
            }
        )
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            ("leader_id", stale_data),
        )
        conn.commit()

        deadline = time.monotonic() + 1.0
        while (
            worker.start_calls != 1
            or not replica._leader_election.is_leader
            or replica._leader_heartbeat_task is None
            or replica._worker_supervision_task is None
        ):
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "Initializing replica did not acquire leadership and start worker"
                )
            await asyncio.sleep(0.01)

        assert replica.state == LifecycleState.INITIALIZING

        ctx.ready_event.set()
        deadline = time.monotonic() + 1.0
        while replica.state != LifecycleState.READY_PRIMARY:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "Initializing leader did not transition to READY_PRIMARY after readiness"
                )
            await asyncio.sleep(0.01)

        await replica.shutdown()
        await primary.shutdown()
