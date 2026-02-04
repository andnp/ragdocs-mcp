"""Tests for VectorIndex circuit breaker functionality."""
import time

import pytest

from src.indices.vector import VectorIndex
from src.utils.circuit_breaker import CircuitState


class TestCircuitBreakerState:
    """Test circuit_state property and get_circuit_breaker_status()."""

    def test_initial_state_is_closed(self, shared_embedding_model):
        """Circuit breaker starts in CLOSED state."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        assert vector_index.circuit_state == CircuitState.CLOSED

    def test_circuit_state_property_returns_enum(self, shared_embedding_model):
        """circuit_state property returns CircuitState enum."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        state = vector_index.circuit_state
        assert isinstance(state, CircuitState)
        assert state.value == "closed"

    def test_get_circuit_breaker_status_structure(self, shared_embedding_model):
        """get_circuit_breaker_status() returns expected dict structure."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        status = vector_index.get_circuit_breaker_status()

        assert "state" in status
        assert "recent_failures" in status
        assert "open_until" in status
        assert "failure_threshold" in status
        assert "recovery_timeout" in status

        assert status["state"] == "closed"
        assert status["recent_failures"] == 0
        assert status["open_until"] is None
        assert status["failure_threshold"] == 5
        assert status["recovery_timeout"] == 60.0

    def test_status_reflects_open_state(self, shared_embedding_model):
        """Status dict reflects OPEN state with open_until timestamp."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Manually force circuit to OPEN state
        breaker._state = CircuitState.OPEN
        breaker._open_time = time.time()

        status = vector_index.get_circuit_breaker_status()

        assert status["state"] == "open"
        assert status["open_until"] is not None
        assert status["open_until"] > time.time()  # Should be in the future


class TestCircuitBreakerReset:
    """Test reset_circuit_breaker() method."""

    def test_reset_clears_failure_count(self, shared_embedding_model):
        """Manual reset clears failure timestamps."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Simulate some failures
        breaker._failure_timestamps = [time.time() - 10, time.time() - 5, time.time()]
        assert len(breaker._failure_timestamps) == 3

        vector_index.reset_circuit_breaker()

        assert len(breaker._failure_timestamps) == 0

    def test_reset_transitions_from_open_to_closed(self, shared_embedding_model):
        """Manual reset transitions from OPEN to CLOSED state."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Force OPEN state
        breaker._state = CircuitState.OPEN
        breaker._open_time = time.time()
        assert vector_index.circuit_state == CircuitState.OPEN

        vector_index.reset_circuit_breaker()

        assert vector_index.circuit_state == CircuitState.CLOSED

    def test_reset_transitions_from_half_open_to_closed(self, shared_embedding_model):
        """Manual reset transitions from HALF_OPEN to CLOSED state."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Force HALF_OPEN state
        breaker._state = CircuitState.HALF_OPEN
        assert vector_index.circuit_state == CircuitState.HALF_OPEN

        vector_index.reset_circuit_breaker()

        assert vector_index.circuit_state == CircuitState.CLOSED

    def test_reset_allows_subsequent_operations(self, shared_embedding_model):
        """After reset, embedding operations should be allowed."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Force OPEN state
        breaker._state = CircuitState.OPEN
        breaker._open_time = time.time()

        # Reset and verify operations work
        vector_index.reset_circuit_breaker()

        # Should be able to get embeddings (model already loaded via fixture)
        embedding = vector_index.get_text_embedding("test query")
        assert embedding is not None
        assert len(embedding) > 0


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions (closed -> open -> half_open -> closed)."""

    def test_multiple_failures_opens_circuit(self, tmp_path):
        """Circuit opens after failure_threshold failures in window."""
        # Create index without pre-loaded model to test failure path
        vector_index = VectorIndex(
            embedding_model_name="nonexistent/model",
            embedding_model=None,
        )
        breaker = vector_index._embedding_circuit_breaker

        # Configure for faster testing
        breaker.config.failure_threshold = 3
        breaker.config.window_duration = 60.0

        # Simulate failures by calling circuit breaker directly
        for i in range(3):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("simulated failure")))
            except RuntimeError:
                pass

        assert vector_index.circuit_state == CircuitState.OPEN

    def test_half_open_after_recovery_timeout(self, shared_embedding_model):
        """Circuit transitions to HALF_OPEN after recovery_timeout."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Set short recovery timeout
        breaker.config.recovery_timeout = 0.1

        # Force OPEN state with past open_time
        breaker._state = CircuitState.OPEN
        breaker._open_time = time.time() - 1.0  # 1 second ago

        # Next call should transition to HALF_OPEN and execute
        result = breaker.call(lambda: "success")

        assert result == "success"
        # After success in half-open, may transition to closed depending on success_threshold

    def test_success_in_half_open_closes_circuit(self, shared_embedding_model):
        """Success in HALF_OPEN state transitions back to CLOSED."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Configure single success to close
        breaker.config.success_threshold = 1

        # Force HALF_OPEN state
        breaker._state = CircuitState.HALF_OPEN
        breaker._success_count = 0

        # Successful call should close circuit
        breaker.call(lambda: "success")

        assert vector_index.circuit_state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self, shared_embedding_model):
        """Failure in HALF_OPEN state reopens circuit."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Force HALF_OPEN state
        breaker._state = CircuitState.HALF_OPEN

        # Failure should reopen
        try:
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("test failure")))
        except RuntimeError:
            pass

        assert vector_index.circuit_state == CircuitState.OPEN


class TestCircuitBreakerStatusReporting:
    """Test status reporting accurately reflects breaker state."""

    def test_status_tracks_failure_count(self, shared_embedding_model):
        """Status shows accurate recent failure count."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Add some failures
        current_time = time.time()
        breaker._failure_timestamps = [current_time - 30, current_time - 20, current_time - 10]

        status = vector_index.get_circuit_breaker_status()
        assert status["recent_failures"] == 3

    def test_status_excludes_old_failures(self, shared_embedding_model):
        """Status only counts failures within window_duration."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Add old failures outside window
        old_time = time.time() - breaker.config.window_duration - 100
        current_time = time.time()
        breaker._failure_timestamps = [old_time, old_time - 10, current_time - 5]

        # Trigger timestamp cleanup via a call
        breaker.call(lambda: "cleanup")

        status = vector_index.get_circuit_breaker_status()
        # Old failures should have been cleaned up
        assert status["recent_failures"] <= 1

    def test_open_until_calculation(self, shared_embedding_model):
        """open_until correctly calculates when circuit will test recovery."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Force OPEN with known open_time
        open_time = time.time()
        breaker._state = CircuitState.OPEN
        breaker._open_time = open_time

        status = vector_index.get_circuit_breaker_status()

        expected_open_until = open_time + breaker.config.recovery_timeout
        assert status["open_until"] == pytest.approx(expected_open_until, abs=0.01)

    def test_open_until_none_when_closed(self, shared_embedding_model):
        """open_until is None when circuit is CLOSED."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)

        status = vector_index.get_circuit_breaker_status()
        assert status["open_until"] is None


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with actual embedding operations."""

    def test_circuit_breaker_opens_after_failures(self, shared_embedding_model):
        """Circuit breaker opens after configured number of failures."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Configure for faster testing
        breaker.config.failure_threshold = 3
        breaker.config.recovery_timeout = 60.0

        # Simulate failures by calling circuit breaker directly
        for _ in range(3):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("simulated failure")))
            except RuntimeError:
                pass

        # Circuit should now be OPEN
        assert vector_index.circuit_state == CircuitState.OPEN

        # Subsequent call should fail fast
        from src.utils.circuit_breaker import CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(lambda: "should not execute")

    def test_manual_reset_after_failure_allows_retry(self, shared_embedding_model):
        """After manual reset, embedding operations can be retried."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)
        breaker = vector_index._embedding_circuit_breaker

        # Simulate OPEN state
        breaker._state = CircuitState.OPEN
        breaker._open_time = time.time()

        # Reset to allow retry
        vector_index.reset_circuit_breaker()

        # Now should be able to use embeddings
        embedding = vector_index.get_text_embedding("test query")
        assert len(embedding) > 0

    def test_status_in_health_check_reflects_breaker_state(self, shared_embedding_model):
        """Health check status accurately reports circuit breaker state."""
        vector_index = VectorIndex(embedding_model=shared_embedding_model)

        # Initial state
        status1 = vector_index.get_circuit_breaker_status()
        assert status1["state"] == "closed"

        # Force open
        vector_index._embedding_circuit_breaker._state = CircuitState.OPEN
        vector_index._embedding_circuit_breaker._open_time = time.time()

        status2 = vector_index.get_circuit_breaker_status()
        assert status2["state"] == "open"

        # Reset
        vector_index.reset_circuit_breaker()

        status3 = vector_index.get_circuit_breaker_status()
        assert status3["state"] == "closed"
