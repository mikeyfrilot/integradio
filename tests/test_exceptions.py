"""Tests for custom exception hierarchy."""

import pytest
from integradio.exceptions import (
    IntegradioError,
    EmbedderError,
    EmbedderUnavailableError,
    EmbedderTimeoutError,
    EmbedderResponseError,
    CacheError,
    RegistryError,
    RegistryDatabaseError,
    ComponentNotFoundError,
    ComponentRegistrationError,
    ComponentError,
    InvalidComponentError,
    ComponentIdError,
    VisualizationError,
    GraphSerializationError,
    APIError,
    ValidationError,
    CircuitBreakerError,
    CircuitOpenError,
)


class TestIntegradioError:
    """Tests for base exception class."""

    def test_basic_instantiation(self):
        """Test basic exception creation."""
        error = IntegradioError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_with_details(self):
        """Test exception with details dict."""
        error = IntegradioError("Error occurred", details={"key": "value", "count": 42})
        assert error.message == "Error occurred"
        assert error.details == {"key": "value", "count": 42}

    def test_is_exception(self):
        """Test that it's a proper Exception subclass."""
        error = IntegradioError("Test")
        assert isinstance(error, Exception)

        with pytest.raises(IntegradioError):
            raise error

    def test_str_with_details(self):
        """Test string representation includes details."""
        error = IntegradioError("Test error", details={"x": 1})
        str_repr = str(error)
        assert "Test error" in str_repr
        assert "Details" in str_repr or "x" in str_repr


class TestEmbedderExceptions:
    """Tests for embedder-related exceptions."""

    def test_embedder_error_hierarchy(self):
        """Test EmbedderError inherits from base."""
        error = EmbedderError("Embedder failed")
        assert isinstance(error, IntegradioError)
        assert isinstance(error, Exception)

    def test_embedder_unavailable_error(self):
        """Test EmbedderUnavailableError with service URL."""
        error = EmbedderUnavailableError("http://localhost:11434")
        assert "localhost:11434" in str(error)
        assert error.details["service_url"] == "http://localhost:11434"
        assert isinstance(error, EmbedderError)

    def test_embedder_unavailable_error_with_cause(self):
        """Test EmbedderUnavailableError with cause."""
        error = EmbedderUnavailableError("http://localhost:11434", cause="Connection refused")
        assert error.details["cause"] == "Connection refused"

    def test_embedder_timeout_error(self):
        """Test EmbedderTimeoutError with duration."""
        error = EmbedderTimeoutError(30.0, "sample text preview")
        assert "30" in str(error)
        assert error.details["timeout_seconds"] == 30.0
        assert isinstance(error, EmbedderError)

    def test_embedder_timeout_error_truncates_preview(self):
        """Test EmbedderTimeoutError truncates long text preview."""
        long_text = "x" * 100
        error = EmbedderTimeoutError(5.0, long_text)
        # Preview should be truncated
        assert len(error.details.get("text_preview", "")) <= 53  # 50 + "..."

    def test_embedder_response_error(self):
        """Test EmbedderResponseError."""
        error = EmbedderResponseError("Invalid response format", status_code=500)
        assert "Invalid response format" in str(error)
        assert error.details["status_code"] == 500
        assert error.status_code == 500

    def test_embedder_response_error_without_status(self):
        """Test EmbedderResponseError without status code."""
        error = EmbedderResponseError("Parse error")
        assert error.status_code is None

    def test_cache_error(self):
        """Test CacheError."""
        error = CacheError("save", "/path/to/cache", "Disk full")
        assert "save" in str(error)
        assert "/path/to/cache" in str(error)
        assert error.details["operation"] == "save"
        assert error.details["path"] == "/path/to/cache"
        assert error.details["cause"] == "Disk full"
        assert isinstance(error, EmbedderError)


class TestRegistryExceptions:
    """Tests for registry-related exceptions."""

    def test_registry_error_hierarchy(self):
        """Test RegistryError inherits from base."""
        error = RegistryError("Registry failed")
        assert isinstance(error, IntegradioError)

    def test_registry_database_error(self):
        """Test RegistryDatabaseError."""
        error = RegistryDatabaseError("insert", "UNIQUE constraint failed")
        assert "insert" in str(error)
        assert error.details["operation"] == "insert"
        assert error.details["cause"] == "UNIQUE constraint failed"
        assert isinstance(error, RegistryError)

    def test_component_not_found_error(self):
        """Test ComponentNotFoundError with component ID."""
        error = ComponentNotFoundError(123)
        assert "123" in str(error)
        assert error.details["component_id"] == 123
        assert error.component_id == 123
        assert isinstance(error, RegistryError)

    def test_component_registration_error(self):
        """Test ComponentRegistrationError."""
        error = ComponentRegistrationError(456, "Duplicate ID")
        assert "456" in str(error)
        assert error.details["component_id"] == 456
        assert error.details["cause"] == "Duplicate ID"
        assert error.component_id == 456
        assert isinstance(error, RegistryError)


class TestComponentExceptions:
    """Tests for component-related exceptions."""

    def test_component_error_hierarchy(self):
        """Test ComponentError inherits from base."""
        error = ComponentError("Component failed")
        assert isinstance(error, IntegradioError)

    def test_invalid_component_error(self):
        """Test InvalidComponentError."""
        error = InvalidComponentError("Missing label", component_type="Textbox")
        assert "Missing label" in str(error)
        assert error.details["component_type"] == "Textbox"
        assert isinstance(error, ComponentError)

    def test_invalid_component_error_without_type(self):
        """Test InvalidComponentError without component type."""
        error = InvalidComponentError("Generic error")
        assert error.details == {}

    def test_component_id_error(self):
        """Test ComponentIdError."""
        error = ComponentIdError("<Button object>")
        assert "_id" in str(error)
        assert error.details["component_repr"] == "<Button object>"
        assert isinstance(error, ComponentError)


class TestVisualizationExceptions:
    """Tests for visualization exceptions."""

    def test_visualization_error(self):
        """Test VisualizationError."""
        error = VisualizationError("Graph generation failed")
        assert isinstance(error, IntegradioError)
        assert "Graph generation failed" in str(error)

    def test_graph_serialization_error(self):
        """Test GraphSerializationError."""
        error = GraphSerializationError("JSON", "Circular reference")
        assert "JSON" in str(error)
        assert error.details["format"] == "JSON"
        assert error.details["cause"] == "Circular reference"
        assert isinstance(error, VisualizationError)


class TestAPIExceptions:
    """Tests for API exceptions."""

    def test_api_error(self):
        """Test APIError with status code."""
        error = APIError("Bad request", status_code=400)
        assert error.status_code == 400
        assert isinstance(error, IntegradioError)

    def test_api_error_default_status(self):
        """Test APIError default status code."""
        error = APIError("Server error")
        assert error.status_code == 500

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("email", "Invalid email format")
        assert "email" in str(error)
        assert error.field == "email"
        assert error.details["field"] == "email"
        assert error.status_code == 400
        assert isinstance(error, APIError)


class TestCircuitBreakerExceptions:
    """Tests for circuit breaker exceptions."""

    def test_circuit_breaker_error_hierarchy(self):
        """Test CircuitBreakerError inherits from base."""
        error = CircuitBreakerError("Circuit breaker failed")
        assert isinstance(error, IntegradioError)

    def test_circuit_open_error(self):
        """Test CircuitOpenError with retry info."""
        error = CircuitOpenError("ollama", 15.5)
        assert "ollama" in str(error)
        assert error.details["service_name"] == "ollama"
        assert error.details["retry_after_seconds"] == 15.5
        assert isinstance(error, CircuitBreakerError)

    def test_circuit_open_error_attributes(self):
        """Test CircuitOpenError attributes."""
        error = CircuitOpenError("service", 30.0)
        assert error.service_name == "service"
        assert error.retry_after_seconds == 30.0


class TestExceptionChaining:
    """Tests for exception chaining and handling."""

    def test_catch_by_base_class(self):
        """Test catching specific exceptions by base class."""
        with pytest.raises(IntegradioError):
            raise EmbedderUnavailableError("http://localhost:11434")

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        with pytest.raises(EmbedderUnavailableError) as exc_info:
            raise EmbedderUnavailableError("http://localhost:11434")

        assert exc_info.value.details["service_url"] == "http://localhost:11434"

    def test_exception_inheritance_chain(self):
        """Test full inheritance chain."""
        error = CircuitOpenError("test", 10.0)
        assert isinstance(error, CircuitOpenError)
        assert isinstance(error, CircuitBreakerError)
        assert isinstance(error, IntegradioError)
        assert isinstance(error, Exception)
        assert isinstance(error, BaseException)

    def test_try_except_pattern(self):
        """Test typical try/except usage pattern."""
        def failing_function():
            raise EmbedderTimeoutError(30.0)

        try:
            failing_function()
        except EmbedderError as e:
            assert "timeout" in str(e).lower() or "30" in str(e)
            assert e.details["timeout_seconds"] == 30.0
        except IntegradioError:
            pytest.fail("Should have been caught by EmbedderError")

    def test_registry_hierarchy(self):
        """Test registry exception hierarchy."""
        errors = [
            ComponentNotFoundError(1),
            ComponentRegistrationError(2, "cause"),
            RegistryDatabaseError("op", "cause"),
        ]
        for error in errors:
            assert isinstance(error, RegistryError)
            assert isinstance(error, IntegradioError)

    def test_component_hierarchy(self):
        """Test component exception hierarchy."""
        errors = [
            InvalidComponentError("msg"),
            ComponentIdError(),
        ]
        for error in errors:
            assert isinstance(error, ComponentError)
            assert isinstance(error, IntegradioError)
