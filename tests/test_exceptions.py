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
    FileUploadError,
    FileValidationError,
    FileSanitizationError,
    FileSizeError,
    BlockedExtensionError,
    WebSocketError,
    WebSocketConnectionError,
    WebSocketAuthenticationError,
    WebSocketTimeoutError,
    WebSocketDisconnectedError,
    EventSignatureError,
    EventExpiredError,
    RateLimitExceededError,
    ReplayAttackError,
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


class TestFileUploadExceptions:
    """Tests for file upload exceptions."""

    def test_file_upload_error_hierarchy(self):
        """Test FileUploadError inherits from base."""
        error = FileUploadError("Upload failed")
        assert isinstance(error, IntegradioError)
        assert "Upload failed" in str(error)

    def test_file_validation_error(self):
        """Test FileValidationError with filename and reason."""
        error = FileValidationError("test.exe", "Executable files not allowed")
        assert "test.exe" in str(error)
        assert "Executable files not allowed" in str(error)
        assert error.details["filename"] == "test.exe"
        assert error.details["reason"] == "Executable files not allowed"
        assert error.filename == "test.exe"
        assert error.reason == "Executable files not allowed"
        assert isinstance(error, FileUploadError)

    def test_file_sanitization_error(self):
        """Test FileSanitizationError with original filename."""
        error = FileSanitizationError("../../../etc/passwd", "Path traversal detected")
        assert "../../../etc/passwd" in str(error)
        assert "Path traversal detected" in str(error)
        assert error.details["original_filename"] == "../../../etc/passwd"
        assert error.details["reason"] == "Path traversal detected"
        assert error.original_filename == "../../../etc/passwd"
        assert isinstance(error, FileUploadError)

    def test_file_size_error(self):
        """Test FileSizeError with size calculations."""
        # 15MB file, 10MB limit
        size_bytes = 15 * 1024 * 1024
        max_bytes = 10 * 1024 * 1024
        error = FileSizeError("large_file.zip", size_bytes, max_bytes)

        # Check message includes MB conversion
        assert "large_file.zip" in str(error)
        assert "15.0MB" in str(error)
        assert "10.0MB" in str(error)

        # Check details
        assert error.details["filename"] == "large_file.zip"
        assert error.details["size_bytes"] == size_bytes
        assert error.details["max_bytes"] == max_bytes

        # Check attributes
        assert error.filename == "large_file.zip"
        assert error.size_bytes == size_bytes
        assert error.max_bytes == max_bytes
        assert isinstance(error, FileUploadError)

    def test_blocked_extension_error(self):
        """Test BlockedExtensionError with extension."""
        error = BlockedExtensionError("malware.exe", ".exe")
        assert ".exe" in str(error)
        assert "not allowed" in str(error)
        assert error.details["filename"] == "malware.exe"
        assert error.details["extension"] == ".exe"
        assert error.filename == "malware.exe"
        assert error.extension == ".exe"
        assert isinstance(error, FileUploadError)

    def test_file_upload_hierarchy(self):
        """Test all file upload exceptions inherit correctly."""
        errors = [
            FileValidationError("f.txt", "invalid"),
            FileSanitizationError("f.txt", "bad"),
            FileSizeError("f.txt", 100, 50),
            BlockedExtensionError("f.exe", ".exe"),
        ]
        for error in errors:
            assert isinstance(error, FileUploadError)
            assert isinstance(error, IntegradioError)


class TestWebSocketExceptions:
    """Tests for WebSocket and event exceptions."""

    def test_websocket_error_hierarchy(self):
        """Test WebSocketError inherits from base."""
        error = WebSocketError("WebSocket failed")
        assert isinstance(error, IntegradioError)
        assert "WebSocket failed" in str(error)

    def test_websocket_connection_error(self):
        """Test WebSocketConnectionError with reason."""
        error = WebSocketConnectionError("Connection refused")
        assert "Connection refused" in str(error)
        assert error.details["reason"] == "Connection refused"
        assert error.client_ip is None
        assert isinstance(error, WebSocketError)

    def test_websocket_connection_error_with_client_ip(self):
        """Test WebSocketConnectionError with client IP."""
        error = WebSocketConnectionError("Timeout", client_ip="192.168.1.100")
        assert "Timeout" in str(error)
        assert error.details["client_ip"] == "192.168.1.100"
        assert error.client_ip == "192.168.1.100"

    def test_websocket_authentication_error(self):
        """Test WebSocketAuthenticationError with reason."""
        error = WebSocketAuthenticationError("Invalid token")
        assert "Invalid token" in str(error)
        assert "authentication" in str(error).lower()
        assert error.details["reason"] == "Invalid token"
        assert error.client_ip is None
        assert isinstance(error, WebSocketError)

    def test_websocket_authentication_error_with_client_ip(self):
        """Test WebSocketAuthenticationError with client IP."""
        error = WebSocketAuthenticationError("Expired token", client_ip="10.0.0.1")
        assert error.details["client_ip"] == "10.0.0.1"
        assert error.client_ip == "10.0.0.1"

    def test_websocket_timeout_error(self):
        """Test WebSocketTimeoutError with operation and duration."""
        error = WebSocketTimeoutError("send", 30.0)
        assert "send" in str(error)
        assert "30" in str(error)
        assert error.details["operation"] == "send"
        assert error.details["timeout_seconds"] == 30.0
        assert error.operation == "send"
        assert error.timeout_seconds == 30.0
        assert isinstance(error, WebSocketError)

    def test_websocket_disconnected_error(self):
        """Test WebSocketDisconnectedError with defaults."""
        error = WebSocketDisconnectedError()
        assert "Client disconnected" in str(error)
        assert error.details["reason"] == "Client disconnected"
        assert error.client_id is None
        assert isinstance(error, WebSocketError)

    def test_websocket_disconnected_error_with_client_id(self):
        """Test WebSocketDisconnectedError with client ID and custom reason."""
        error = WebSocketDisconnectedError(client_id="user-123", reason="Network error")
        assert "Network error" in str(error)
        assert error.details["client_id"] == "user-123"
        assert error.details["reason"] == "Network error"
        assert error.client_id == "user-123"

    def test_event_signature_error(self):
        """Test EventSignatureError with defaults."""
        error = EventSignatureError()
        assert "signature" in str(error).lower()
        assert error.details["reason"] == "Invalid signature"
        assert error.event_id is None
        assert isinstance(error, WebSocketError)

    def test_event_signature_error_with_event_id(self):
        """Test EventSignatureError with event ID and custom reason."""
        error = EventSignatureError(event_id="evt-456", reason="Tampered payload")
        assert "Tampered payload" in str(error)
        assert error.details["event_id"] == "evt-456"
        assert error.details["reason"] == "Tampered payload"
        assert error.event_id == "evt-456"

    def test_event_expired_error(self):
        """Test EventExpiredError with timing details."""
        error = EventExpiredError("evt-789", age_seconds=120.5, max_age_seconds=60.0)
        assert "evt-789" in str(error)
        assert "expired" in str(error).lower()
        assert "120.5" in str(error)
        assert "60" in str(error)
        assert error.details["event_id"] == "evt-789"
        assert error.details["age_seconds"] == 120.5
        assert error.details["max_age_seconds"] == 60.0
        assert error.event_id == "evt-789"
        assert isinstance(error, WebSocketError)

    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError with limit info."""
        error = RateLimitExceededError("client-abc", limit=100, window_seconds=60.0)
        assert "client-abc" in str(error)
        assert "100" in str(error)
        assert "60" in str(error)
        assert error.details["client_id"] == "client-abc"
        assert error.details["limit"] == 100
        assert error.details["window_seconds"] == 60.0
        assert error.client_id == "client-abc"
        assert isinstance(error, WebSocketError)

    def test_replay_attack_error(self):
        """Test ReplayAttackError with nonce."""
        error = ReplayAttackError("nonce-12345")
        assert "nonce-12345" in str(error)
        assert "replay" in str(error).lower()
        assert error.details["nonce"] == "nonce-12345"
        assert error.nonce == "nonce-12345"
        assert isinstance(error, WebSocketError)

    def test_replay_attack_error_with_client_id(self):
        """Test ReplayAttackError with client ID."""
        error = ReplayAttackError("nonce-xyz", client_id="attacker-ip")
        assert error.details["client_id"] == "attacker-ip"
        assert error.nonce == "nonce-xyz"

    def test_websocket_hierarchy(self):
        """Test all WebSocket exceptions inherit correctly."""
        errors = [
            WebSocketConnectionError("reason"),
            WebSocketAuthenticationError("reason"),
            WebSocketTimeoutError("op", 10.0),
            WebSocketDisconnectedError(),
            EventSignatureError(),
            EventExpiredError("id", 100.0, 60.0),
            RateLimitExceededError("client", 100, 60.0),
            ReplayAttackError("nonce"),
        ]
        for error in errors:
            assert isinstance(error, WebSocketError)
            assert isinstance(error, IntegradioError)
