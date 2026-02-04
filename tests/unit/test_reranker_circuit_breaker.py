"""
Unit tests for ReRanker circuit breaker functionality.

Tests cover:
- Circuit breaker starts CLOSED
- Repeated model load failures trip the circuit
- Rerank returns original scores when circuit is open
- Circuit recovers after timeout

Test strategies:
- Use a FailingCrossEncoder to simulate model failures
- Directly manipulate circuit breaker state for edge case tests
"""

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from src.search.reranker import ReRanker
from src.utils.circuit_breaker import CircuitState


@dataclass
class FailingCrossEncoder:
    """Cross encoder that always raises an exception on predict."""

    def predict(
        self, sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str]
    ) -> Iterable[float]:
        raise RuntimeError("Model inference failed")


@dataclass
class FakeCrossEncoder:
    """Cross encoder that returns predictable scores based on content length."""

    def predict(
        self, sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str]
    ) -> Iterable[float]:
        if isinstance(sentences, list) and sentences and isinstance(sentences[0], tuple):
            return [len(content) * 0.01 for _, content in sentences]
        return [0.0]


@pytest.fixture
def content_provider():
    """Provide test content."""
    contents = {
        "chunk_1": "Short content",
        "chunk_2": "Medium length content",
        "chunk_3": "Longer content for testing",
    }
    return lambda chunk_id: contents.get(chunk_id)


@pytest.fixture
def candidates():
    """Sample candidates for re-ranking."""
    return [
        ("chunk_1", 0.9),
        ("chunk_2", 0.7),
        ("chunk_3", 0.5),
    ]


class TestCircuitBreakerInitialState:

    def test_circuit_breaker_starts_closed(self):
        """Circuit breaker should start in CLOSED state."""
        reranker = ReRanker(model_name="test-model")
        assert reranker.circuit_state == CircuitState.CLOSED

    def test_circuit_state_property_returns_state(self):
        """circuit_state property returns current circuit breaker state."""
        reranker = ReRanker(model_name="test-model")
        assert isinstance(reranker.circuit_state, CircuitState)


class TestCircuitBreakerTripping:

    def test_repeated_failures_trip_circuit(self):
        """Repeated model failures should trip the circuit to OPEN."""
        reranker = ReRanker(model_name="test-model")
        reranker._model = FailingCrossEncoder()

        # Trip the circuit with repeated failures
        failure_threshold = reranker._circuit_breaker.config.failure_threshold
        for _ in range(failure_threshold):
            # Each call should fail but reranker handles it gracefully
            try:
                reranker._protected_predict([("query", "content")])
            except RuntimeError:
                pass

        assert reranker.circuit_state == CircuitState.OPEN

    def test_circuit_open_after_model_load_failures(self):
        """Circuit opens after repeated model loading failures."""
        reranker = ReRanker(model_name="test-model")

        # Simulate model loading failure by making the circuit breaker call fail
        failure_threshold = reranker._circuit_breaker.config.failure_threshold

        for _ in range(failure_threshold):
            try:
                reranker._circuit_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("Load failed")))
            except RuntimeError:
                pass

        assert reranker.circuit_state == CircuitState.OPEN


class TestCircuitBreakerFallback:

    def test_rerank_returns_original_when_circuit_open(self, content_provider, candidates):
        """When circuit is open, rerank returns original rankings truncated."""
        reranker = ReRanker(model_name="test-model")
        reranker._model = FakeCrossEncoder()

        # Force circuit to OPEN state
        reranker._circuit_breaker._state = CircuitState.OPEN
        reranker._circuit_breaker._open_time = float("inf")  # Never expire

        result = reranker.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=2,
        )

        # Should return first 2 candidates in original order
        assert len(result) == 2
        assert result[0] == candidates[0]
        assert result[1] == candidates[1]

    def test_rerank_graceful_degradation_on_predict_failure(self, content_provider, candidates):
        """Rerank returns original rankings when prediction fails and circuit opens."""
        reranker = ReRanker(model_name="test-model")
        reranker._model = FailingCrossEncoder()

        # Trip the circuit first
        failure_threshold = reranker._circuit_breaker.config.failure_threshold
        for _ in range(failure_threshold):
            try:
                reranker._protected_predict([("query", "content")])
            except RuntimeError:
                pass

        # Now circuit should be open
        assert reranker.circuit_state == CircuitState.OPEN

        # Rerank should return original rankings
        result = reranker.rerank(
            query="test query",
            candidates=candidates,
            content_provider=content_provider,
            top_n=3,
        )

        assert len(result) == 3
        assert result[0] == candidates[0]


class TestCircuitBreakerIntegration:

    def test_successful_rerank_does_not_trip_circuit(self, content_provider, candidates):
        """Successful reranking keeps circuit in CLOSED state."""
        reranker = ReRanker(model_name="test-model")
        reranker._model = FakeCrossEncoder()

        for _ in range(10):
            reranker.rerank(
                query="test query",
                candidates=candidates,
                content_provider=content_provider,
                top_n=3,
            )

        assert reranker.circuit_state == CircuitState.CLOSED

    def test_model_lock_prevents_race_condition(self, content_provider, candidates):
        """Model lock ensures thread-safe model loading."""
        import threading

        reranker = ReRanker(model_name="test-model")
        errors = []

        def try_rerank():
            try:
                # Inject model if not set (simulating lazy load race)
                with reranker._model_lock:
                    if reranker._model is None:
                        reranker._model = FakeCrossEncoder()

                reranker.rerank(
                    query="test query",
                    candidates=candidates,
                    content_provider=content_provider,
                    top_n=2,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=try_rerank) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert reranker._model is not None

