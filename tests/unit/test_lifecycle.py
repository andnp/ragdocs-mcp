"""
Unit tests for LifecycleCoordinator state machine, emergency timer, and signal handling.

Commit 2.1: Verifies LifecycleCoordinator is the source of truth for process state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.lifecycle import LifecycleCoordinator, LifecycleState


# ---------------------------------------------------------------------------
# Lightweight stubs (no mock library — real objects with minimal behavior)
# ---------------------------------------------------------------------------


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
    """Minimal stub to satisfy LifecycleCoordinator.start()."""

    config: FakeConfig = field(default_factory=FakeConfig)
    commit_indexer: None = None

    async def start(self, background_index: bool = False) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        pass


def _make_coordinator(**overrides: Any) -> LifecycleCoordinator:
    return LifecycleCoordinator(**overrides)


def _fake_ctx() -> Any:
    """Return a FakeContext cast to Any so it satisfies the ApplicationContext param."""
    return cast(Any, FakeContext())


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
