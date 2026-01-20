"""
Circuit breaker pattern implementation for external service calls.

Prevents cascading failures when external services (like Ollama) are unavailable
by failing fast after a threshold of failures.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar, Generic

from .exceptions import CircuitOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """States for the circuit breaker."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "success_rate": round(self.successful_calls / max(self.total_calls, 1), 3),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening circuit
    success_threshold: int = 2          # Successes to close circuit
    timeout_seconds: float = 30.0       # Time before trying again (half-open)
    exception_types: tuple = (Exception,)  # Exceptions that count as failures


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation for protecting external service calls.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service considered failed, calls rejected immediately
    - HALF_OPEN: Testing if service recovered with limited calls

    Usage:
        breaker = CircuitBreaker("ollama", config=CircuitBreakerConfig(failure_threshold=3))

        try:
            result = breaker.call(lambda: make_api_call())
        except CircuitOpenError:
            # Circuit is open, use fallback
            result = fallback_value

    Or as a decorator:
        @breaker
        def make_api_call():
            ...
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        fallback: Callable[[], T] | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Service name for logging/identification
            config: Circuit breaker configuration
            fallback: Optional fallback function when circuit is open
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_failure_time: float = 0.0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing calls)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self.state == CircuitState.OPEN

    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN:
            time_since_failure = time.monotonic() - self._last_failure_time
            if time_since_failure >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._stats.state_changes += 1
            logger.info(
                f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value}"
            )

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._stats.consecutive_successes = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.monotonic()
            self._last_failure_time = time.monotonic()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure #{self._stats.consecutive_failures}: {error}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.rejected_calls += 1

    def call(self, fn: Callable[[], T]) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            fn: Function to execute

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open and no fallback
            Exception: Any exception from the function
        """
        with self._lock:
            self._check_state_transition()
            current_state = self._state

        if current_state == CircuitState.OPEN:
            self._record_rejection()
            retry_after = self.config.timeout_seconds - (
                time.monotonic() - self._last_failure_time
            )

            if self.fallback:
                logger.debug(
                    f"Circuit breaker '{self.name}' open, using fallback"
                )
                return self.fallback()

            raise CircuitOpenError(self.name, max(0, retry_after))

        try:
            result = fn()
            self._record_success()
            return result
        except self.config.exception_types as e:
            self._record_failure(e)
            raise

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator interface for the circuit breaker."""
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(lambda: fn(*args, **kwargs))
        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def force_open(self) -> None:
        """Manually open the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._last_failure_time = time.monotonic()
            logger.info(f"Circuit breaker '{self.name}' manually opened")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to circuit breakers across the application.
    """

    _instance: "CircuitBreakerRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "CircuitBreakerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers = {}
            return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_breakers"):
            self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        fallback: Callable | None = None,
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create a new one.

        Args:
            name: Circuit breaker name
            config: Configuration (only used for new breakers)
            fallback: Fallback function (only used for new breakers)

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config, fallback)
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {name: cb.stats.to_dict() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._breakers.values():
            cb.reset()


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    fallback: Callable | None = None,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker from the global registry.

    Args:
        name: Circuit breaker name
        config: Optional configuration
        fallback: Optional fallback function

    Returns:
        CircuitBreaker instance
    """
    return circuit_registry.get_or_create(name, config, fallback)
