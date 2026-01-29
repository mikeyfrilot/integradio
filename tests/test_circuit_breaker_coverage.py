"""
Batch 2 continued: Circuit Breaker Coverage Tests (12 tests)

Tests for integradio/circuit_breaker.py - HIGH priority
"""

import pytest
from unittest.mock import MagicMock, patch
import time
import threading


class TestCircuitBreakerOpensOnFailures:
    """Tests for circuit breaker opening on failures."""

    def test_circuit_breaker_opens_on_failures(self):
        """Verify circuit opens after threshold failures."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        assert breaker.state == CircuitState.CLOSED

        # Cause failures
        for i in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_stays_closed_below_threshold(self):
        """Verify circuit stays closed below failure threshold."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=5,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Cause fewer failures than threshold
        for i in range(4):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        # Still closed
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_counts_consecutive_failures(self):
        """Verify only consecutive failures count."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=3,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Fail twice
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        # Success resets counter
        result = breaker.call(lambda: "success")
        assert result == "success"

        # Fail twice more (not enough consecutive)
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerHalfOpenRecovery:
    """Tests for half-open state recovery."""

    def test_circuit_breaker_half_open_recovery(self):
        """Verify circuit transitions to half-open after timeout."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,  # Very short for testing
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_half_open_success_closes(self):
        """Verify success in half-open closes circuit."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout_seconds=0.1,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Success should close
        breaker.call(lambda: "success")

        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_half_open_failure_reopens(self):
        """Verify failure in half-open reopens circuit."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure should reopen
        with pytest.raises(ValueError):
            breaker.call(lambda: exec('raise ValueError("fail")'))

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerResetsOnSuccess:
    """Tests for circuit breaker reset on success."""

    def test_circuit_breaker_resets_on_success(self):
        """Verify consecutive failure count resets on success."""
        from integradio.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=3,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Cause 2 failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        assert breaker.stats.consecutive_failures == 2

        # Success resets
        breaker.call(lambda: "success")

        assert breaker.stats.consecutive_failures == 0

    def test_circuit_breaker_manual_reset(self):
        """Verify manual reset works."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 0


class TestCircuitBreakerTimeoutBehavior:
    """Tests for circuit breaker timeout behavior."""

    def test_circuit_breaker_timeout_behavior(self):
        """Verify circuit respects timeout before allowing retry."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )
        from integradio.exceptions import CircuitOpenError

        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.5,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        # Should reject immediately
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "test")

        # Wait less than timeout
        time.sleep(0.1)

        # Still open
        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_fallback_on_open(self):
        """Verify fallback is called when circuit is open."""
        from integradio.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        fallback_value = "fallback result"
        config = CircuitBreakerConfig(
            failure_threshold=2,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker(
            "test",
            config=config,
            fallback=lambda: fallback_value,
        )

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError("fail")'))

        # Should use fallback
        result = breaker.call(lambda: "normal")

        assert result == fallback_value


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_circuit_registry_get_or_create(self):
        """Verify registry creates and reuses circuit breakers."""
        from integradio.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        cb1 = registry.get_or_create("service1")
        cb2 = registry.get_or_create("service1")
        cb3 = registry.get_or_create("service2")

        assert cb1 is cb2  # Same name, same instance
        assert cb1 is not cb3  # Different name, different instance

    def test_circuit_registry_all_stats(self):
        """Verify registry returns all stats."""
        from integradio.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        registry.get_or_create("service1")
        registry.get_or_create("service2")

        stats = registry.all_stats()

        assert "service1" in stats
        assert "service2" in stats
        assert "total_calls" in stats["service1"]

    def test_circuit_registry_reset_all(self):
        """Verify registry can reset all breakers."""
        from integradio.circuit_breaker import (
            CircuitBreakerRegistry,
            CircuitBreakerConfig,
            CircuitState,
        )

        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        config = CircuitBreakerConfig(failure_threshold=1)
        cb1 = registry.get_or_create("service1", config=config)

        # Open circuit
        with pytest.raises(Exception):
            cb1.call(lambda: exec('raise Exception()'))

        assert cb1.state == CircuitState.OPEN

        # Reset all
        registry.reset_all()

        assert cb1.state == CircuitState.CLOSED
