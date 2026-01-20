"""Tests for circuit breaker pattern implementation."""

import time
import threading
import pytest
from unittest.mock import Mock, patch

from integradio.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState,
    CircuitBreakerRegistry,
    get_circuit_breaker,
    circuit_registry,
)
from integradio.exceptions import CircuitOpenError


class TestCircuitBreakerConfig:
    """Tests for circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30.0
        assert config.exception_types == (Exception,)

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=10.0,
            exception_types=(ValueError, RuntimeError),
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout_seconds == 10.0
        assert config.exception_types == (ValueError, RuntimeError)


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = CircuitBreakerStats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.rejected_calls == 0
        assert stats.state_changes == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0

    def test_to_dict(self):
        """Test statistics dictionary conversion."""
        stats = CircuitBreakerStats(
            total_calls=100,
            successful_calls=90,
            failed_calls=10,
            rejected_calls=5,
        )
        result = stats.to_dict()
        assert result["total_calls"] == 100
        assert result["successful_calls"] == 90
        assert result["failed_calls"] == 10
        assert result["rejected_calls"] == 5
        assert result["success_rate"] == 0.9

    def test_success_rate_zero_calls(self):
        """Test success rate calculation with zero calls."""
        stats = CircuitBreakerStats()
        result = stats.to_dict()
        assert result["success_rate"] == 0.0


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_closed(self):
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_transitions_to_open_after_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config=config)

        # Simulate 3 failures
        for i in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_open_circuit_rejects_calls(self):
        """Test open circuit rejects calls immediately."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=60.0)
        breaker = CircuitBreaker("test", config=config)

        # Trigger open state
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Next call should be rejected
        with pytest.raises(CircuitOpenError) as exc_info:
            breaker.call(lambda: "success")

        assert exc_info.value.details["service_name"] == "test"
        assert breaker.stats.rejected_calls == 1

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        breaker = CircuitBreaker("test", config=config)

        # Trigger open state
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # State should transition to half-open on next check
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Test successful call in half-open state closes circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout_seconds=0.1,
        )
        breaker = CircuitBreaker("test", config=config)

        # Trigger open state
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for half-open
        time.sleep(0.15)

        # Successful call should close circuit
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        """Test failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        breaker = CircuitBreaker("test", config=config)

        # Trigger open state
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure should reopen circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerCalls:
    """Tests for circuit breaker call handling."""

    def test_successful_call_returns_result(self):
        """Test successful calls return function result."""
        breaker = CircuitBreaker("test")
        result = breaker.call(lambda: 42)
        assert result == 42
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.total_calls == 1

    def test_failed_call_raises_exception(self):
        """Test failed calls raise original exception."""
        breaker = CircuitBreaker("test")

        with pytest.raises(RuntimeError) as exc_info:
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("test error")))

        assert "test error" in str(exc_info.value)
        assert breaker.stats.failed_calls == 1

    def test_non_configured_exception_not_counted(self):
        """Test exceptions not in config don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            exception_types=(ValueError,),
        )
        breaker = CircuitBreaker("test", config=config)

        # RuntimeError should not be counted as failure
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("not counted")))

        # Circuit should still be closed
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 0


class TestCircuitBreakerFallback:
    """Tests for circuit breaker fallback functionality."""

    def test_fallback_on_open_circuit(self):
        """Test fallback is called when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=60.0)
        breaker = CircuitBreaker("test", config=config, fallback=lambda: "fallback_value")

        # Trigger open state
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Next call should use fallback
        result = breaker.call(lambda: "normal_value")
        assert result == "fallback_value"

    def test_no_fallback_raises_circuit_open_error(self):
        """Test CircuitOpenError when no fallback and circuit open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=60.0)
        breaker = CircuitBreaker("test", config=config)

        # Trigger open state
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "value")


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator interface."""

    def test_decorator_basic(self):
        """Test using circuit breaker as decorator."""
        breaker = CircuitBreaker("test")

        @breaker
        def my_function(x, y):
            return x + y

        result = my_function(1, 2)
        assert result == 3
        assert breaker.stats.successful_calls == 1

    def test_decorator_with_failure(self):
        """Test decorator with failing function."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config=config)

        @breaker
        def failing_function():
            raise ValueError("intentional failure")

        with pytest.raises(ValueError):
            failing_function()

        assert breaker.stats.failed_calls == 1


class TestCircuitBreakerManualControl:
    """Tests for manual circuit breaker control."""

    def test_manual_reset(self):
        """Test manually resetting circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config=config)

        # Trigger open state
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 0

    def test_force_open(self):
        """Test manually forcing circuit open."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED

        breaker.force_open()
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_get_or_create_new(self):
        """Test creating new circuit breaker via registry."""
        # Use a unique name to avoid conflicts with global state
        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        breaker = registry.get_or_create("unique_test_service")
        assert breaker.name == "unique_test_service"
        assert isinstance(breaker, CircuitBreaker)

    def test_get_or_create_existing(self):
        """Test getting existing circuit breaker."""
        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        breaker1 = registry.get_or_create("service_a")
        breaker2 = registry.get_or_create("service_a")
        assert breaker1 is breaker2

    def test_get_nonexistent(self):
        """Test getting nonexistent circuit breaker."""
        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        result = registry.get("nonexistent")
        assert result is None

    def test_all_stats(self):
        """Test getting stats for all breakers."""
        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        registry.get_or_create("service_1")
        registry.get_or_create("service_2")

        stats = registry.all_stats()
        assert "service_1" in stats
        assert "service_2" in stats
        assert "total_calls" in stats["service_1"]

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry.__new__(CircuitBreakerRegistry)
        registry._breakers = {}

        config = CircuitBreakerConfig(failure_threshold=1)
        b1 = registry.get_or_create("s1", config=config)
        b2 = registry.get_or_create("s2", config=config)

        # Open both circuits
        for breaker in [b1, b2]:
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert b1.is_open
        assert b2.is_open

        # Reset all
        registry.reset_all()
        assert b1.is_closed
        assert b2.is_closed


class TestGetCircuitBreaker:
    """Tests for get_circuit_breaker helper function."""

    def test_get_circuit_breaker_creates_new(self):
        """Test helper function creates circuit breaker."""
        # This uses the global registry
        breaker = get_circuit_breaker("helper_test_service")
        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "helper_test_service"

    def test_get_circuit_breaker_with_config(self):
        """Test helper function with custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = get_circuit_breaker("configured_service", config=config)
        assert breaker.config.failure_threshold == 10

    def test_get_circuit_breaker_with_fallback(self):
        """Test helper function with fallback."""
        fallback = lambda: "default"
        breaker = get_circuit_breaker("fallback_service", fallback=fallback)
        assert breaker.fallback is fallback


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety of circuit breaker."""

    def test_concurrent_calls(self):
        """Test circuit breaker handles concurrent calls."""
        breaker = CircuitBreaker("concurrent_test")
        results = []
        errors = []

        def make_call():
            try:
                result = breaker.call(lambda: threading.current_thread().name)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0
        assert breaker.stats.successful_calls == 10

    def test_concurrent_failures(self):
        """Test concurrent failures trigger state transition correctly."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("concurrent_failure_test", config=config)
        errors = []

        def make_failing_call():
            try:
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except (ValueError, CircuitOpenError) as e:
                errors.append(e)

        threads = [threading.Thread(target=make_failing_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All calls should have resulted in an error
        assert len(errors) == 10
        # Circuit should be open
        assert breaker.is_open
