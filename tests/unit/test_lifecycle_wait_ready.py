"""
Unit tests for LifecycleCoordinator.wait_ready() state handling.

Regression tests for the bug where queries during STARTING state would
raise "Cannot wait for ready from state starting" instead of waiting.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from src.lifecycle import LifecycleCoordinator, LifecycleState


@dataclass
class MockApplicationContext:
    """Mock context for testing lifecycle transitions."""

    _ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    call_count: int = field(default=0, repr=False)
    ensure_ready_delay: float = field(default=0.0, repr=False)

    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    async def ensure_ready(self, timeout: float = 60.0):
        self.call_count += 1
        if self.ensure_ready_delay > 0:
            await asyncio.sleep(self.ensure_ready_delay)
        self._ready_event.set()

    async def start(self, background_index: bool = False):
        if not background_index:
            self._ready_event.set()

    async def stop(self):
        pass


def _set_ctx(coordinator: LifecycleCoordinator, mock_ctx: MockApplicationContext):
    """Helper to set mock context, bypassing type checking."""
    object.__setattr__(coordinator, "_ctx", cast(Any, mock_ctx))


class TestWaitReadyFromStartingState:
    """Tests for wait_ready() when called during STARTING state."""

    @pytest.mark.asyncio
    async def test_wait_ready_during_starting_state_waits_for_transition(self):
        """
        Verify wait_ready() waits for STARTING to transition instead of raising.

        This is the regression test for the bug where queries during startup
        would fail with "Cannot wait for ready from state starting".
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockApplicationContext()

        # Manually set coordinator to STARTING state (simulating early query)
        coordinator._state = LifecycleState.STARTING
        _set_ctx(coordinator, mock_ctx)

        async def transition_after_delay():
            await asyncio.sleep(0.1)
            coordinator._state = LifecycleState.INITIALIZING

        transition_task = asyncio.create_task(transition_after_delay())

        # This should NOT raise, it should wait for STARTING → INITIALIZING
        await coordinator.wait_ready(timeout=5.0)

        await transition_task

        assert coordinator._state == LifecycleState.READY

    @pytest.mark.asyncio
    async def test_wait_ready_starting_state_transitions_to_ready_directly(self):
        """
        Verify wait_ready() handles direct STARTING → READY transition.
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockApplicationContext()

        coordinator._state = LifecycleState.STARTING
        _set_ctx(coordinator, mock_ctx)

        async def transition_to_ready():
            await asyncio.sleep(0.1)
            coordinator._state = LifecycleState.READY

        transition_task = asyncio.create_task(transition_to_ready())

        await coordinator.wait_ready(timeout=5.0)

        await transition_task

        assert coordinator._state == LifecycleState.READY

    @pytest.mark.asyncio
    async def test_wait_ready_starting_state_timeout(self):
        """
        Verify wait_ready() times out if stuck in STARTING state.
        """
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.STARTING

        with pytest.raises(RuntimeError, match="stuck in STARTING"):
            await coordinator.wait_ready(timeout=0.2)

    @pytest.mark.asyncio
    async def test_wait_ready_starting_uses_remaining_timeout(self):
        """
        Verify timeout is properly tracked across STARTING wait and ensure_ready.

        After waiting for STARTING → INITIALIZING, ensure_ready should get
        the remaining timeout, not the full original timeout.
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockApplicationContext(ensure_ready_delay=0.1)

        coordinator._state = LifecycleState.STARTING
        _set_ctx(coordinator, mock_ctx)

        async def slow_transition():
            await asyncio.sleep(0.15)
            coordinator._state = LifecycleState.INITIALIZING

        transition_task = asyncio.create_task(slow_transition())

        # Total timeout 0.3s: 0.15s waiting for STARTING, 0.1s in ensure_ready
        # Should complete with ~0.05s to spare
        await coordinator.wait_ready(timeout=0.3)

        await transition_task

        assert coordinator._state == LifecycleState.READY


class TestWaitReadyStateValidation:
    """Tests for wait_ready() state validation."""

    @pytest.mark.asyncio
    async def test_wait_ready_from_ready_returns_immediately(self):
        """
        Verify wait_ready() returns immediately when already READY.
        """
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.READY

        # Should complete instantly
        await asyncio.wait_for(coordinator.wait_ready(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_ready_from_initializing_state(self):
        """
        Verify wait_ready() works from INITIALIZING state.
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockApplicationContext()

        coordinator._state = LifecycleState.INITIALIZING
        _set_ctx(coordinator, mock_ctx)

        await coordinator.wait_ready(timeout=5.0)

        assert coordinator._state == LifecycleState.READY
        assert mock_ctx.call_count == 1

    @pytest.mark.asyncio
    async def test_wait_ready_from_degraded_state(self):
        """
        Verify wait_ready() works from DEGRADED state.
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockApplicationContext()

        coordinator._state = LifecycleState.DEGRADED
        _set_ctx(coordinator, mock_ctx)

        await coordinator.wait_ready(timeout=5.0)

        assert coordinator._state == LifecycleState.READY

    @pytest.mark.asyncio
    async def test_wait_ready_from_uninitialized_raises(self):
        """
        Verify wait_ready() raises from UNINITIALIZED state.
        """
        coordinator = LifecycleCoordinator()
        assert coordinator._state == LifecycleState.UNINITIALIZED

        with pytest.raises(RuntimeError, match="Cannot wait for ready from state"):
            await coordinator.wait_ready()

    @pytest.mark.asyncio
    async def test_wait_ready_from_shutting_down_raises(self):
        """
        Verify wait_ready() raises from SHUTTING_DOWN state.
        """
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.SHUTTING_DOWN

        with pytest.raises(RuntimeError, match="Cannot wait for ready from state"):
            await coordinator.wait_ready()

    @pytest.mark.asyncio
    async def test_wait_ready_from_terminated_raises(self):
        """
        Verify wait_ready() raises from TERMINATED state.
        """
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.TERMINATED

        with pytest.raises(RuntimeError, match="Cannot wait for ready from state"):
            await coordinator.wait_ready()


class TestWaitReadyTimeoutBehavior:
    """Tests for wait_ready() timeout handling."""

    @pytest.mark.asyncio
    async def test_wait_ready_timeout_during_ensure_ready(self):
        """
        Verify timeout is propagated to ensure_ready correctly.
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockApplicationContext(ensure_ready_delay=10.0)

        coordinator._state = LifecycleState.INITIALIZING
        _set_ctx(coordinator, mock_ctx)

        with pytest.raises(asyncio.TimeoutError):
            await coordinator.wait_ready(timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_ready_timeout_exhausted_before_ensure_ready(self):
        """
        Verify timeout exhausted in STARTING wait raises before ensure_ready.
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockApplicationContext()

        coordinator._state = LifecycleState.STARTING
        _set_ctx(coordinator, mock_ctx)

        async def slow_transition():
            await asyncio.sleep(0.5)
            coordinator._state = LifecycleState.INITIALIZING

        transition_task = asyncio.create_task(slow_transition())

        with pytest.raises(RuntimeError, match="stuck in STARTING"):
            await coordinator.wait_ready(timeout=0.1)

        transition_task.cancel()
        try:
            await transition_task
        except asyncio.CancelledError:
            pass
