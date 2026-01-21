"""
Custom exception hierarchy for integradio.

Provides structured exception types for different error categories,
enabling better error handling and debugging.
"""


class IntegradioError(Exception):
    """Base exception for all integradio errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Embedder Exceptions
# =============================================================================


class EmbedderError(IntegradioError):
    """Base exception for embedder-related errors."""
    pass


class EmbedderUnavailableError(EmbedderError):
    """Raised when the embedding service is unavailable."""

    def __init__(self, service_url: str, cause: str | None = None):
        details = {"service_url": service_url}
        if cause:
            details["cause"] = cause
        super().__init__(
            f"Embedding service unavailable at {service_url}",
            details=details,
        )
        self.service_url = service_url


class EmbedderTimeoutError(EmbedderError):
    """Raised when embedding request times out."""

    def __init__(self, timeout_seconds: float, text_preview: str = ""):
        details = {"timeout_seconds": timeout_seconds}
        if text_preview:
            details["text_preview"] = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        super().__init__(
            f"Embedding request timed out after {timeout_seconds}s",
            details=details,
        )


class EmbedderResponseError(EmbedderError):
    """Raised when embedding API returns invalid response."""

    def __init__(self, message: str, status_code: int | None = None):
        details = {}
        if status_code is not None:
            details["status_code"] = status_code
        super().__init__(message, details=details)
        self.status_code = status_code


class CacheError(EmbedderError):
    """Raised when cache operations fail."""

    def __init__(self, operation: str, path: str, cause: str):
        super().__init__(
            f"Cache {operation} failed for {path}: {cause}",
            details={"operation": operation, "path": path, "cause": cause},
        )


# =============================================================================
# Registry Exceptions
# =============================================================================


class RegistryError(IntegradioError):
    """Base exception for registry-related errors."""
    pass


class RegistryDatabaseError(RegistryError):
    """Raised when database operations fail."""

    def __init__(self, operation: str, cause: str):
        super().__init__(
            f"Registry database error during {operation}: {cause}",
            details={"operation": operation, "cause": cause},
        )


class ComponentNotFoundError(RegistryError):
    """Raised when a component is not found in the registry."""

    def __init__(self, component_id: int):
        super().__init__(
            f"Component with ID {component_id} not found",
            details={"component_id": component_id},
        )
        self.component_id = component_id


class ComponentRegistrationError(RegistryError):
    """Raised when component registration fails."""

    def __init__(self, component_id: int, cause: str):
        super().__init__(
            f"Failed to register component {component_id}: {cause}",
            details={"component_id": component_id, "cause": cause},
        )
        self.component_id = component_id


# =============================================================================
# Component Exceptions
# =============================================================================


class ComponentError(IntegradioError):
    """Base exception for component-related errors."""
    pass


class InvalidComponentError(ComponentError):
    """Raised when a component is invalid or missing required attributes."""

    def __init__(self, message: str, component_type: str | None = None):
        details = {}
        if component_type:
            details["component_type"] = component_type
        super().__init__(message, details=details)


class ComponentIdError(ComponentError):
    """Raised when component ID is missing or invalid."""

    def __init__(self, component_repr: str = "unknown"):
        super().__init__(
            f"Component must have _id attribute: {component_repr}",
            details={"component_repr": component_repr},
        )


# =============================================================================
# Visualization Exceptions
# =============================================================================


class VisualizationError(IntegradioError):
    """Base exception for visualization-related errors."""
    pass


class GraphSerializationError(VisualizationError):
    """Raised when graph serialization fails."""

    def __init__(self, format_type: str, cause: str):
        super().__init__(
            f"Failed to serialize graph to {format_type}: {cause}",
            details={"format": format_type, "cause": cause},
        )


# =============================================================================
# API Exceptions
# =============================================================================


class APIError(IntegradioError):
    """Base exception for API-related errors."""

    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        super().__init__(message, details=details)
        self.status_code = status_code


class ValidationError(APIError):
    """Raised when API input validation fails."""

    def __init__(self, field: str, message: str):
        super().__init__(
            f"Validation error for '{field}': {message}",
            status_code=400,
            details={"field": field},
        )
        self.field = field


# =============================================================================
# Circuit Breaker Exceptions
# =============================================================================


class CircuitBreakerError(IntegradioError):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open and calls are being rejected."""

    def __init__(self, service_name: str, retry_after_seconds: float):
        super().__init__(
            f"Circuit breaker open for {service_name}. Retry after {retry_after_seconds:.1f}s",
            details={"service_name": service_name, "retry_after_seconds": retry_after_seconds},
        )
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds


# =============================================================================
# File Upload Exceptions
# =============================================================================


class FileUploadError(IntegradioError):
    """Base exception for file upload errors."""
    pass


class FileValidationError(FileUploadError):
    """Raised when uploaded file fails validation."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"File validation failed for '{filename}': {reason}",
            details={"filename": filename, "reason": reason},
        )
        self.filename = filename
        self.reason = reason


class FileSanitizationError(FileUploadError):
    """Raised when filename sanitization fails."""

    def __init__(self, original_filename: str, reason: str):
        super().__init__(
            f"Cannot sanitize filename '{original_filename}': {reason}",
            details={"original_filename": original_filename, "reason": reason},
        )
        self.original_filename = original_filename


class FileSizeError(FileUploadError):
    """Raised when file exceeds size limits."""

    def __init__(self, filename: str, size_bytes: int, max_bytes: int):
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(
            f"File '{filename}' ({size_mb:.1f}MB) exceeds maximum size ({max_mb:.1f}MB)",
            details={
                "filename": filename,
                "size_bytes": size_bytes,
                "max_bytes": max_bytes,
            },
        )
        self.filename = filename
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class BlockedExtensionError(FileUploadError):
    """Raised when file has a blocked extension."""

    def __init__(self, filename: str, extension: str):
        super().__init__(
            f"File type '{extension}' is not allowed for security reasons",
            details={"filename": filename, "extension": extension},
        )
        self.filename = filename
        self.extension = extension


# =============================================================================
# WebSocket/Event Exceptions
# =============================================================================


class WebSocketError(IntegradioError):
    """Base exception for WebSocket-related errors."""
    pass


class WebSocketConnectionError(WebSocketError):
    """Raised when WebSocket connection fails."""

    def __init__(self, reason: str, client_ip: str | None = None):
        details = {"reason": reason}
        if client_ip:
            details["client_ip"] = client_ip
        super().__init__(f"WebSocket connection failed: {reason}", details=details)
        self.client_ip = client_ip


class WebSocketAuthenticationError(WebSocketError):
    """Raised when WebSocket authentication fails."""

    def __init__(self, reason: str, client_ip: str | None = None):
        details = {"reason": reason}
        if client_ip:
            details["client_ip"] = client_ip
        super().__init__(f"WebSocket authentication failed: {reason}", details=details)
        self.client_ip = client_ip


class WebSocketTimeoutError(WebSocketError):
    """Raised when WebSocket operation times out."""

    def __init__(self, operation: str, timeout_seconds: float):
        super().__init__(
            f"WebSocket {operation} timed out after {timeout_seconds}s",
            details={"operation": operation, "timeout_seconds": timeout_seconds},
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class WebSocketDisconnectedError(WebSocketError):
    """Raised when client disconnects unexpectedly."""

    def __init__(self, client_id: str | None = None, reason: str = "Client disconnected"):
        details = {"reason": reason}
        if client_id:
            details["client_id"] = client_id
        super().__init__(reason, details=details)
        self.client_id = client_id


class EventSignatureError(WebSocketError):
    """Raised when event signature verification fails."""

    def __init__(self, event_id: str | None = None, reason: str = "Invalid signature"):
        details = {"reason": reason}
        if event_id:
            details["event_id"] = event_id
        super().__init__(f"Event signature verification failed: {reason}", details=details)
        self.event_id = event_id


class EventExpiredError(WebSocketError):
    """Raised when event has expired."""

    def __init__(self, event_id: str, age_seconds: float, max_age_seconds: float):
        super().__init__(
            f"Event '{event_id}' has expired (age: {age_seconds:.1f}s, max: {max_age_seconds}s)",
            details={
                "event_id": event_id,
                "age_seconds": age_seconds,
                "max_age_seconds": max_age_seconds,
            },
        )
        self.event_id = event_id


class RateLimitExceededError(WebSocketError):
    """Raised when rate limit is exceeded."""

    def __init__(self, client_id: str, limit: int, window_seconds: float):
        super().__init__(
            f"Rate limit exceeded for client '{client_id}' ({limit} requests per {window_seconds}s)",
            details={
                "client_id": client_id,
                "limit": limit,
                "window_seconds": window_seconds,
            },
        )
        self.client_id = client_id


class ReplayAttackError(WebSocketError):
    """Raised when replay attack is detected (duplicate nonce)."""

    def __init__(self, nonce: str, client_id: str | None = None):
        details = {"nonce": nonce}
        if client_id:
            details["client_id"] = client_id
        super().__init__(
            f"Replay attack detected: nonce '{nonce}' already used",
            details=details,
        )
        self.nonce = nonce
