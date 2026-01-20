"""
Structured logging configuration for integradio.

Provides consistent JSON logging with context fields for debugging
and monitoring across all modules.
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
component_id_var: ContextVar[int] = ContextVar("component_id", default=0)


@dataclass
class LogContext:
    """Structured context for log entries."""
    request_id: str = ""
    component_id: int = 0
    operation: str = ""
    duration_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs in a consistent JSON format suitable for log aggregation
    systems like ELK, Splunk, or CloudWatch.
    """

    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_level:
            log_data["level"] = record.levelname

        # Add context variables
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        component_id = component_id_var.get()
        if component_id:
            log_data["component_id"] = component_id

        # Add extra fields from record
        if hasattr(record, "context") and record.context:
            log_data["context"] = asdict(record.context) if isinstance(record.context, LogContext) else record.context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development/debugging.

    Provides colored, easy-to-read output for terminal usage.
    """

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Build context string
        context_parts = []
        request_id = request_id_var.get()
        if request_id:
            context_parts.append(f"req={request_id[:8]}")

        component_id = component_id_var.get()
        if component_id:
            context_parts.append(f"comp={component_id}")

        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format message
        msg = f"{timestamp} {color}{record.levelname:8}{self.RESET} {record.name}:{record.funcName}{context_str} - {record.getMessage()}"

        # Add extra context if present
        if hasattr(record, "context") and record.context:
            ctx = asdict(record.context) if isinstance(record.context, LogContext) else record.context
            if ctx:
                msg += f"\n  Context: {ctx}"

        return msg


class SemanticLogger(logging.Logger):
    """
    Extended logger with integradio specific helpers.
    """

    def operation(
        self,
        name: str,
        component_id: int | None = None,
        **extra: Any,
    ) -> "OperationContext":
        """
        Context manager for logging operations with timing.

        Usage:
            with logger.operation("search", component_id=123) as ctx:
                # do work
                ctx.add("results_count", 10)
        """
        return OperationContext(self, name, component_id, extra)

    def with_context(self, **context: Any) -> logging.LoggerAdapter:
        """Return a logger adapter with additional context."""
        return logging.LoggerAdapter(self, {"context": context})


class OperationContext:
    """Context manager for operation timing and logging."""

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        component_id: int | None,
        extra: dict[str, Any],
    ):
        self.logger = logger
        self.operation = operation
        self.component_id = component_id
        self.extra = extra
        self.start_time: float = 0
        self._context = LogContext(operation=operation)

    def __enter__(self) -> "OperationContext":
        self.start_time = time.perf_counter()
        if self.component_id:
            component_id_var.set(self.component_id)
            self._context.component_id = self.component_id

        self.logger.debug(
            f"Starting {self.operation}",
            extra={"context": self._context},
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self._context.duration_ms = round(duration_ms, 2)
        self._context.extra = self.extra

        if exc_type:
            self.logger.error(
                f"Failed {self.operation} after {duration_ms:.2f}ms: {exc_val}",
                extra={"context": self._context},
                exc_info=(exc_type, exc_val, exc_tb),
            )
        else:
            self.logger.debug(
                f"Completed {self.operation} in {duration_ms:.2f}ms",
                extra={"context": self._context},
            )

        # Reset context
        if self.component_id:
            component_id_var.set(0)

    def add(self, key: str, value: Any) -> None:
        """Add extra context to the operation."""
        self.extra[key] = value


def configure_logging(
    level: str | int = logging.INFO,
    format_type: str = "human",  # "human" or "json"
    stream: Any = None,
) -> None:
    """
    Configure logging for integradio.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: "human" for readable output, "json" for structured logs
        stream: Output stream (defaults to sys.stderr)
    """
    # Set custom logger class
    logging.setLoggerClass(SemanticLogger)

    # Get root integradio logger
    logger = logging.getLogger("integradio")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    # Set formatter
    if format_type == "json":
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False


def get_logger(name: str) -> SemanticLogger:
    """
    Get a integradio logger.

    Args:
        name: Logger name (will be prefixed with 'integradio.')

    Returns:
        SemanticLogger instance
    """
    full_name = f"integradio.{name}" if not name.startswith("integradio") else name
    return logging.getLogger(full_name)  # type: ignore


# Auto-configure with default settings on import
configure_logging()
