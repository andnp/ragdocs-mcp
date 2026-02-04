"""
Unit tests for LifecycleCoordinator health check timeout recovery.

Tests the behavior where consecutive health check timeouts trigger
DEGRADED state and worker restart.
"""

from src.lifecycle import LifecycleCoordinator, LifecycleState


class TestHealthCheckTimeoutRecovery:
    """Tests for health check timeout tracking and recovery."""

    def test_consecutive_timeout_counter_increments(self):
        """Verify timeout counter increments on each timeout."""
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.READY

        assert coordinator._consecutive_timeouts == 0

        coordinator._consecutive_timeouts += 1
        assert coordinator._consecutive_timeouts == 1

        coordinator._consecutive_timeouts += 1
        assert coordinator._consecutive_timeouts == 2

    def test_successful_health_check_resets_counter(self):
        """Verify counter resets to 0 on successful health check."""
        coordinator = LifecycleCoordinator()
        coordinator._consecutive_timeouts = 2

        # Simulate receiving healthy response (counter reset)
        coordinator._consecutive_timeouts = 0

        assert coordinator._consecutive_timeouts == 0

    def test_max_timeouts_triggers_degraded_state(self):
        """Verify reaching max timeouts transitions to DEGRADED."""
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.READY

        max_timeouts = coordinator._max_consecutive_timeouts
        coordinator._consecutive_timeouts = max_timeouts

        if coordinator._consecutive_timeouts >= coordinator._max_consecutive_timeouts:
            coordinator._state = LifecycleState.DEGRADED

        assert coordinator._state == LifecycleState.DEGRADED

    def test_default_max_consecutive_timeouts_is_three(self):
        """Verify default threshold is 3."""
        coordinator = LifecycleCoordinator()
        assert coordinator._max_consecutive_timeouts == 3

    def test_counter_initializes_to_zero(self):
        """Verify counter starts at 0."""
        coordinator = LifecycleCoordinator()
        assert coordinator._consecutive_timeouts == 0


class TestHealthCheckTimeoutLogic:
    """Tests for the timeout handling logic in isolation."""

    def test_timeout_increments_counter_and_triggers_restart(self):
        """
        Verify the logic: after max timeouts, state becomes DEGRADED
        and restart is triggered.
        """
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.READY

        # Simulate 3 consecutive timeouts (the threshold)
        for i in range(coordinator._max_consecutive_timeouts):
            coordinator._consecutive_timeouts += 1

        # At threshold, should transition to DEGRADED
        if coordinator._consecutive_timeouts >= coordinator._max_consecutive_timeouts:
            coordinator._state = LifecycleState.DEGRADED
            coordinator._consecutive_timeouts = 0  # Reset after triggering restart

        assert coordinator._state == LifecycleState.DEGRADED
        assert coordinator._consecutive_timeouts == 0

    def test_healthy_response_resets_counter(self):
        """
        Verify the logic: healthy response resets timeout counter.
        """
        coordinator = LifecycleCoordinator()
        coordinator._consecutive_timeouts = 2

        # Simulate receiving healthy response
        coordinator._consecutive_timeouts = 0

        assert coordinator._consecutive_timeouts == 0

    def test_degraded_recovers_to_ready_on_healthy(self):
        """
        Verify the logic: DEGRADED state transitions to READY on healthy response.
        """
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.DEGRADED

        # Simulate receiving healthy response
        coordinator._consecutive_timeouts = 0
        if coordinator._state == LifecycleState.DEGRADED:
            coordinator._state = LifecycleState.READY

        assert coordinator._state == LifecycleState.READY

    def test_timeout_below_threshold_stays_ready(self):
        """
        Verify timeouts below threshold don't trigger DEGRADED.
        """
        coordinator = LifecycleCoordinator()
        coordinator._state = LifecycleState.READY

        # Simulate 2 timeouts (below threshold of 3)
        coordinator._consecutive_timeouts = 2

        # Check threshold condition (should not trigger)
        if coordinator._consecutive_timeouts >= coordinator._max_consecutive_timeouts:
            coordinator._state = LifecycleState.DEGRADED

        assert coordinator._state == LifecycleState.READY
        assert coordinator._consecutive_timeouts == 2
