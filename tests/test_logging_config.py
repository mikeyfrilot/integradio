"""Tests for structured logging configuration."""

import json
import logging
import io
import time
import pytest

from integradio.logging_config import (
    configure_logging,
    get_logger,
    StructuredFormatter,
    HumanReadableFormatter,
    LogContext,
    OperationContext,
    request_id_var,
    component_id_var,
)


class TestStructuredFormatter:
    """Tests for JSON structured formatter."""

    def test_basic_format(self):
        """Test basic log message formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Test message"
        assert data["logger"] == "test.logger"
        assert data["level"] == "INFO"
        assert data["line"] == 42

    def test_format_with_timestamp(self):
        """Test formatter includes timestamp."""
        formatter = StructuredFormatter(include_timestamp=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" in data
        # Should be ISO format
        assert "T" in data["timestamp"]

    def test_format_without_timestamp(self):
        """Test formatter excludes timestamp when disabled."""
        formatter = StructuredFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data

    def test_format_with_context_vars(self):
        """Test formatter includes context variables."""
        formatter = StructuredFormatter()

        # Set context
        old_req = request_id_var.get()
        old_comp = component_id_var.get()
        request_id_var.set("test-request-123")
        component_id_var.set(456)

        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test with context",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            data = json.loads(result)

            assert data.get("request_id") == "test-request-123"
            assert data.get("component_id") == 456
        finally:
            request_id_var.set(old_req)
            component_id_var.set(old_comp)

    def test_format_with_exception(self):
        """Test formatter includes exception info."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "Test error" in data["exception"]["message"]

    def test_valid_json_output(self):
        """Test output is always valid JSON."""
        formatter = StructuredFormatter()
        messages = [
            "Simple message",
            "Message with 'quotes'",
            'Message with "double quotes"',
            "Message with\nnewline",
            "Message with unicode: \u00e9\u00e8\u00ea",
        ]

        for msg in messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None,
            )
            result = formatter.format(record)
            # Should not raise
            data = json.loads(result)
            assert data["message"] == msg


class TestHumanReadableFormatter:
    """Tests for human-readable formatter."""

    def test_basic_format(self):
        """Test basic human-readable format."""
        formatter = HumanReadableFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "test.logger" in result
        assert "Test message" in result

    def test_format_with_context(self):
        """Test formatter includes context vars."""
        formatter = HumanReadableFormatter()

        old_req = request_id_var.get()
        request_id_var.set("req-abc-12345678")

        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            # Should include truncated request ID
            assert "req=" in result
        finally:
            request_id_var.set(old_req)


class TestLogContext:
    """Tests for LogContext dataclass."""

    def test_default_values(self):
        """Test LogContext default values."""
        ctx = LogContext()
        assert ctx.request_id == ""
        assert ctx.component_id == 0
        assert ctx.operation == ""
        assert ctx.duration_ms == 0.0
        assert ctx.extra == {}

    def test_custom_values(self):
        """Test LogContext with custom values."""
        ctx = LogContext(
            request_id="req-123",
            component_id=456,
            operation="search",
            duration_ms=15.5,
            extra={"key": "value"},
        )
        assert ctx.request_id == "req-123"
        assert ctx.component_id == 456
        assert ctx.operation == "search"
        assert ctx.duration_ms == 15.5
        assert ctx.extra == {"key": "value"}


class TestOperationContext:
    """Tests for operation timing context manager."""

    def test_operation_timing(self):
        """Test OperationContext measures time."""
        logger = logging.getLogger("test_operation")
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with OperationContext(logger, "test_op", None, {}) as ctx:
            time.sleep(0.05)

        # Duration should be recorded
        assert ctx._context.duration_ms >= 50

    def test_operation_with_component_id(self):
        """Test OperationContext sets component context."""
        logger = logging.getLogger("test_comp")
        logger.setLevel(logging.DEBUG)

        old_comp = component_id_var.get()

        with OperationContext(logger, "test_op", 123, {}):
            # Inside context, component ID should be set
            assert component_id_var.get() == 123

        # After context, should be reset
        assert component_id_var.get() == old_comp

    def test_operation_add_extra(self):
        """Test adding extra context during operation."""
        logger = logging.getLogger("test_extra")
        logger.setLevel(logging.DEBUG)

        with OperationContext(logger, "test_op", None, {}) as ctx:
            ctx.add("count", 10)
            ctx.add("status", "success")

        assert ctx.extra["count"] == 10
        assert ctx.extra["status"] == "success"

    def test_operation_failure_logging(self):
        """Test OperationContext logs failures."""
        stream = io.StringIO()
        logger = logging.getLogger("test_failure")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)

        with pytest.raises(ValueError):
            with OperationContext(logger, "failing_op", None, {}):
                raise ValueError("test error")

        output = stream.getvalue()
        # Should have logged the failure
        assert "failing_op" in output or "error" in output.lower()


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_defaults(self):
        """Test configure_logging with default options."""
        # Should not raise
        configure_logging()

    def test_configure_json_format(self):
        """Test configure_logging with JSON format."""
        configure_logging(format_type="json")

        logger = get_logger("test_json")
        # Should not raise
        logger.info("Test JSON log")

    def test_configure_human_format(self):
        """Test configure_logging with human-readable format."""
        configure_logging(format_type="human")

        logger = get_logger("test_human")
        logger.info("Test human log")

    def test_configure_log_level(self):
        """Test configure_logging sets log level."""
        configure_logging(level=logging.WARNING)

        # Root integradio logger should have the level
        root_logger = logging.getLogger("integradio")
        assert root_logger.level == logging.WARNING

    def test_configure_custom_stream(self):
        """Test configure_logging with custom stream."""
        stream = io.StringIO()
        configure_logging(stream=stream, format_type="human")

        logger = get_logger("test_stream")
        logger.warning("Test stream message")

        output = stream.getvalue()
        assert "Test stream message" in output


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a Logger instance."""
        logger = get_logger("my_module")
        assert isinstance(logger, logging.Logger)
        assert "integradio.my_module" in logger.name or logger.name == "my_module"

    def test_get_logger_same_name_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("shared_module")
        logger2 = get_logger("shared_module")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test get_logger returns different instances for different names."""
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not logger2

    def test_get_logger_already_prefixed(self):
        """Test get_logger with already prefixed name."""
        logger = get_logger("integradio.submodule")
        assert "integradio" in logger.name


class TestContextVariables:
    """Tests for context variable functions."""

    def test_request_id_var(self):
        """Test request_id context variable."""
        old = request_id_var.get()
        request_id_var.set("test-request")
        assert request_id_var.get() == "test-request"
        request_id_var.set(old)

    def test_component_id_var(self):
        """Test component_id context variable."""
        old = component_id_var.get()
        component_id_var.set(999)
        assert component_id_var.get() == 999
        component_id_var.set(old)


class TestLoggerIntegration:
    """Integration tests for logging system."""

    def test_full_logging_flow(self):
        """Test complete logging flow with all features."""
        # Configure
        stream = io.StringIO()
        configure_logging(format_type="json", level=logging.DEBUG, stream=stream)

        # Get logger
        logger = get_logger("integration_test")

        # Set context
        old_req = request_id_var.get()
        old_comp = component_id_var.get()
        request_id_var.set("req-full-test")
        component_id_var.set(100)

        try:
            # Log with operation timing
            with OperationContext(logger, "full_test_operation", 100, {"extra_data": "test"}):
                logger.info("Inside operation")
                time.sleep(0.01)

            output = stream.getvalue()
            # Should have some output
            assert len(output) > 0
        finally:
            request_id_var.set(old_req)
            component_id_var.set(old_comp)

    def test_structured_log_includes_all_fields(self):
        """Test structured log includes all expected fields."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger = logging.getLogger("field_test")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        logger.info("Test message")

        output = stream.getvalue()
        data = json.loads(output)

        # Check all expected fields
        assert "message" in data
        assert "logger" in data
        assert "level" in data
        assert "timestamp" in data
        assert "module" in data
        assert "function" in data
        assert "line" in data
