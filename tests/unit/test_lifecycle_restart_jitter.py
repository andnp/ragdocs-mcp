"""
Unit tests for LifecycleCoordinator restart jitter.

Tests the exponential backoff with jitter to prevent thundering herd
when multiple instances restart simultaneously.
"""

import random
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from src.config import WorkerConfig
from src.lifecycle import LifecycleCoordinator, LifecycleState


@dataclass
class MockReadOnlyContext:
    """Mock readonly context for testing restart behavior."""

    @dataclass
    class MockConfig:
        worker: WorkerConfig = field(default_factory=WorkerConfig)

    config: MockConfig = field(default_factory=MockConfig)

    async def start_sync_watcher(self):
        pass

    def is_ready(self) -> bool:
        return True

    async def stop(self):
        pass


def _set_readonly_ctx(coordinator: LifecycleCoordinator, ctx: MockReadOnlyContext):
    """Helper to set mock readonly context."""
    object.__setattr__(coordinator, "_readonly_ctx", cast(Any, ctx))


class TestRestartJitterCalculation:
    """Tests for jitter calculation in _restart_worker."""

    @pytest.mark.asyncio
    async def test_jitter_applied_within_expected_range(self):
        """Verify jitter stays within ±jitter_factor of base delay."""
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(
            restart_backoff_base=2.0,
            restart_jitter_factor=0.25,
            restart_max_delay=60.0,
            max_restart_attempts=5,
        )
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED

        # Use seeded random for deterministic testing
        random.seed(42)

        delays: list[float] = []

        async def capture_delay(delay: float):
            delays.append(delay)

        with (
            patch("asyncio.sleep", side_effect=capture_delay),
            patch.object(coordinator, "_stop_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_start_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_wait_for_init", return_value=True),
        ):
            await coordinator._restart_worker()

        assert len(delays) == 1
        delay = delays[0]

        # First attempt: base_delay = 2.0 * (2^0) = 2.0
        base_delay = 2.0
        jitter_factor = 0.25
        min_expected = max(1.0, base_delay * (1 - jitter_factor))  # 1.5
        max_expected = base_delay * (1 + jitter_factor)  # 2.5

        assert min_expected <= delay <= max_expected, (
            f"Delay {delay} not in expected range [{min_expected}, {max_expected}]"
        )

    @pytest.mark.asyncio
    async def test_delay_capped_at_max_delay(self):
        """Verify delay never exceeds restart_max_delay."""
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(
            restart_backoff_base=10.0,
            restart_jitter_factor=0.25,
            restart_max_delay=30.0,
            max_restart_attempts=10,
        )
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED
        coordinator._restart_attempts = 5  # High attempt count for large exponent

        # Seed for deterministic max jitter
        random.seed(0)

        delays: list[float] = []

        async def capture_delay(delay: float):
            delays.append(delay)

        with (
            patch("asyncio.sleep", side_effect=capture_delay),
            patch.object(coordinator, "_stop_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_start_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_wait_for_init", return_value=True),
        ):
            await coordinator._restart_worker()

        assert len(delays) == 1
        delay = delays[0]

        # Even with jitter, delay should be capped
        max_allowed = 30.0 * 1.25  # max_delay + max jitter
        assert delay <= max_allowed, f"Delay {delay} exceeds max allowed {max_allowed}"

    @pytest.mark.asyncio
    async def test_minimum_delay_enforced(self):
        """Verify delay is at least 1 second even with negative jitter."""
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(
            restart_backoff_base=0.5,  # Very small base
            restart_jitter_factor=0.5,  # Large jitter that could go below 1.0
            restart_max_delay=60.0,
            max_restart_attempts=5,
        )
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED

        delays: list[float] = []

        async def capture_delay(delay: float):
            delays.append(delay)

        # Run multiple times to test with different random values
        for seed in range(10):
            random.seed(seed)
            coordinator._restart_attempts = 0
            delays.clear()

            with (
                patch("asyncio.sleep", side_effect=capture_delay),
                patch.object(coordinator, "_stop_worker", new_callable=AsyncMock),
                patch.object(coordinator, "_start_worker", new_callable=AsyncMock),
                patch.object(coordinator, "_wait_for_init", return_value=True),
            ):
                await coordinator._restart_worker()

            if delays:
                assert delays[0] >= 1.0, f"Delay {delays[0]} below minimum 1.0 (seed={seed})"

    @pytest.mark.asyncio
    async def test_exponential_backoff_with_jitter_progression(self):
        """Verify delays increase exponentially across restart attempts."""
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(
            restart_backoff_base=1.0,
            restart_jitter_factor=0.0,  # No jitter for predictable progression
            restart_max_delay=60.0,
            max_restart_attempts=5,
        )
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED

        all_delays: list[float] = []

        async def capture_delay(delay: float):
            all_delays.append(delay)

        with (
            patch("asyncio.sleep", side_effect=capture_delay),
            patch.object(coordinator, "_stop_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_start_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_wait_for_init", return_value=True),
        ):
            for _ in range(4):
                await coordinator._restart_worker()

        # Expected: 1.0, 2.0, 4.0, 8.0 (exponential, but clamped to min 1.0)
        expected = [1.0, 2.0, 4.0, 8.0]
        assert all_delays == expected, f"Expected {expected}, got {all_delays}"


class TestRestartJitterConfiguration:
    """Tests for jitter configuration in WorkerConfig."""

    def test_default_jitter_factor(self):
        """Verify default jitter factor is 0.25 (±25%)."""
        config = WorkerConfig()
        assert config.restart_jitter_factor == 0.25

    def test_default_max_delay(self):
        """Verify default max delay is 60 seconds."""
        config = WorkerConfig()
        assert config.restart_max_delay == 60.0

    def test_custom_jitter_factor(self):
        """Verify custom jitter factor can be set."""
        config = WorkerConfig(restart_jitter_factor=0.5)
        assert config.restart_jitter_factor == 0.5

    def test_custom_max_delay(self):
        """Verify custom max delay can be set."""
        config = WorkerConfig(restart_max_delay=120.0)
        assert config.restart_max_delay == 120.0


class TestRestartJitterDistribution:
    """Tests for jitter randomness distribution."""

    @pytest.mark.asyncio
    async def test_jitter_distribution_covers_range(self):
        """
        Verify jitter covers the expected range over many samples.

        This is a statistical test: with enough samples and a non-zero
        jitter factor, we should see delays both above and below base.
        """
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(
            restart_backoff_base=10.0,
            restart_jitter_factor=0.25,
            restart_max_delay=60.0,
            max_restart_attempts=100,  # Allow many restarts for testing
        )
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED

        delays: list[float] = []

        async def capture_delay(delay: float):
            delays.append(delay)

        # Collect many samples
        for seed in range(50):
            random.seed(seed)
            coordinator._restart_attempts = 0

            with (
                patch("asyncio.sleep", side_effect=capture_delay),
                patch.object(coordinator, "_stop_worker", new_callable=AsyncMock),
                patch.object(coordinator, "_start_worker", new_callable=AsyncMock),
                patch.object(coordinator, "_wait_for_init", return_value=True),
            ):
                await coordinator._restart_worker()

        # First attempt: base = 10.0, expect delays in [7.5, 12.5]
        base_delay = 10.0
        min_expected = base_delay * 0.75
        max_expected = base_delay * 1.25

        # Should have values both above and below base (statistical check)
        above_base = sum(1 for d in delays if d > base_delay)
        below_base = sum(1 for d in delays if d < base_delay)

        assert above_base > 0, "Expected some delays above base"
        assert below_base > 0, "Expected some delays below base"
        assert all(min_expected <= d <= max_expected for d in delays), (
            f"Some delays outside range [{min_expected}, {max_expected}]"
        )

    @pytest.mark.asyncio
    async def test_zero_jitter_gives_deterministic_delays(self):
        """Verify zero jitter factor gives deterministic results."""
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(
            restart_backoff_base=5.0,
            restart_jitter_factor=0.0,  # No jitter
            restart_max_delay=60.0,
            max_restart_attempts=100,
        )
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED

        delays: list[float] = []

        async def capture_delay(delay: float):
            delays.append(delay)

        for _ in range(5):
            coordinator._restart_attempts = 0

            with (
                patch("asyncio.sleep", side_effect=capture_delay),
                patch.object(coordinator, "_stop_worker", new_callable=AsyncMock),
                patch.object(coordinator, "_start_worker", new_callable=AsyncMock),
                patch.object(coordinator, "_wait_for_init", return_value=True),
            ):
                await coordinator._restart_worker()

        # All delays should be exactly base_delay (5.0)
        assert all(d == 5.0 for d in delays), f"Expected all 5.0, got {delays}"


class TestRestartAttemptTracking:
    """Tests for restart attempt tracking with jitter."""

    @pytest.mark.asyncio
    async def test_restart_attempts_increment_correctly(self):
        """Verify restart attempts increment on each call."""
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(max_restart_attempts=5)
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED

        assert coordinator._restart_attempts == 0

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch.object(coordinator, "_stop_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_start_worker", new_callable=AsyncMock),
            patch.object(coordinator, "_wait_for_init", return_value=True),
        ):
            await coordinator._restart_worker()
            assert coordinator._restart_attempts == 1

            await coordinator._restart_worker()
            assert coordinator._restart_attempts == 2

            await coordinator._restart_worker()
            assert coordinator._restart_attempts == 3

    @pytest.mark.asyncio
    async def test_max_attempts_stops_restarts(self):
        """Verify no restart after max attempts reached."""
        coordinator = LifecycleCoordinator()
        mock_ctx = MockReadOnlyContext()
        mock_ctx.config.worker = WorkerConfig(max_restart_attempts=2)
        _set_readonly_ctx(coordinator, mock_ctx)
        coordinator._state = LifecycleState.DEGRADED
        coordinator._restart_attempts = 2  # Already at max

        sleep_called = False

        async def track_sleep(delay: float):
            nonlocal sleep_called
            sleep_called = True

        with patch("asyncio.sleep", side_effect=track_sleep):
            await coordinator._restart_worker()

        assert not sleep_called, "Sleep should not be called when max attempts reached"
        assert coordinator._state == LifecycleState.DEGRADED
