"""Circuit breaker pattern implementation for resilient service calls.

Implements a three-state circuit breaker (CLOSED -> OPEN -> HALF_OPEN) to prevent
cascading failures when a service becomes unreliable.

States:
- CLOSED: Normal operation, all calls pass through
- OPEN: Service is unavailable, calls fail fast without attempting operation
- HALF_OPEN: Testing recovery, limited calls allowed to verify service health
"""
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure state, rejecting calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before attempting recovery
    success_threshold: int = 2  # Successes in half-open to close circuit
    window_duration: float = 60.0  # Time window for counting failures


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open (service unavailable)."""
    pass


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures.

    Usage:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )

        try:
            result = breaker.call(lambda: risky_operation())
        except CircuitBreakerOpen:
            # Fallback logic
            result = safe_fallback()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        window_duration: float = 60.0,
    ):
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            window_duration=window_duration,
        )
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._open_time = 0.0
        self._lock = threading.Lock()
        self._failure_timestamps: list[float] = []

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Callable to execute

        Returns:
            Result from func()

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception raised by func()
        """
        with self._lock:
            current_time = time.time()

            # Clean old failure timestamps outside window
            self._failure_timestamps = [
                ts for ts in self._failure_timestamps
                if current_time - ts < self.config.window_duration
            ]

            # State transition: OPEN -> HALF_OPEN
            if self._state == CircuitState.OPEN:
                if current_time - self._open_time >= self.config.recovery_timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN (testing recovery)")
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker is OPEN. Retry after "
                        f"{self.config.recovery_timeout - (current_time - self._open_time):.1f}s"
                    )

        # Execute function (outside lock to avoid holding during slow operation)
        try:
            result = func()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            current_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker HALF_OPEN: {self._success_count}/"
                    f"{self.config.success_threshold} successes"
                )

                # State transition: HALF_OPEN -> CLOSED
                if self._success_count >= self.config.success_threshold:
                    logger.info("Circuit breaker transitioning to CLOSED (service recovered)")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._failure_timestamps.clear()

            elif self._state == CircuitState.CLOSED:
                # Clear old failures on success
                self._failure_timestamps = [
                    ts for ts in self._failure_timestamps
                    if current_time - ts < self.config.window_duration
                ]

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            current_time = time.time()
            self._last_failure_time = current_time
            self._failure_timestamps.append(current_time)

            # Clean old timestamps
            self._failure_timestamps = [
                ts for ts in self._failure_timestamps
                if current_time - ts < self.config.window_duration
            ]

            failure_count = len(self._failure_timestamps)

            if self._state == CircuitState.HALF_OPEN:
                # State transition: HALF_OPEN -> OPEN (recovery failed)
                logger.warning("Circuit breaker transitioning to OPEN (recovery failed)")
                self._state = CircuitState.OPEN
                self._open_time = current_time

            elif self._state == CircuitState.CLOSED:
                # State transition: CLOSED -> OPEN (threshold exceeded)
                if failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit breaker transitioning to OPEN "
                        f"({failure_count} failures in {self.config.window_duration}s)"
                    )
                    self._state = CircuitState.OPEN
                    self._open_time = current_time

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            logger.info("Circuit breaker manually reset to CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._failure_timestamps.clear()
