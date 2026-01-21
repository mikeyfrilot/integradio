"""
Semantic Events - Secure WebSocket event mesh for real-time UI updates.

Features:
- CloudEvents-compliant message format
- HMAC-SHA256 message signing for integrity
- AsyncIO pub/sub with pattern matching
- Rate limiting and connection management
- Automatic reconnection with exponential backoff

Security (2026 Best Practices):
- WSS (TLS) required in production
- Token-based authentication on handshake
- Per-message authorization checks
- HMAC signing for message integrity
- Rate limiting to prevent DDoS
- Origin validation
- Input validation with size limits

Usage:
    from integradio.events import EventMesh, SemanticEvent

    mesh = EventMesh(secret_key="your-secret")

    @mesh.on("ui.component.*")
    async def handle_component_event(event: SemanticEvent):
        print(f"Component updated: {event.data}")

    await mesh.emit("ui.component.click", {"id": 123})
"""

from .event import SemanticEvent, EventType
from .mesh import EventMesh
from .websocket import WebSocketServer, WebSocketClient
from .security import (
    EventSigner,
    RateLimiter,
    ConnectionManager,
    validate_origin,
    # Security headers (2026 additions)
    SecurityHeadersConfig,
    create_security_headers_middleware,
    get_secure_gradio_config,
    validate_gradio_version,
    DEFAULT_SECURITY_HEADERS,
    DEFAULT_CSP,
)
from .handlers import EventHandler, on_event

# Import exceptions for convenience
from ..exceptions import (
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

# Backwards compatibility alias
MutantEvent = SemanticEvent

__all__ = [
    # Core
    "SemanticEvent",
    "MutantEvent",  # Backwards compatibility
    "EventType",
    "EventMesh",
    # WebSocket
    "WebSocketServer",
    "WebSocketClient",
    # Security
    "EventSigner",
    "RateLimiter",
    "ConnectionManager",
    "validate_origin",
    # Security Headers (2026 additions)
    "SecurityHeadersConfig",
    "create_security_headers_middleware",
    "get_secure_gradio_config",
    "validate_gradio_version",
    "DEFAULT_SECURITY_HEADERS",
    "DEFAULT_CSP",
    # Handlers
    "EventHandler",
    "on_event",
    # Exceptions (for consumers)
    "WebSocketError",
    "WebSocketConnectionError",
    "WebSocketAuthenticationError",
    "WebSocketTimeoutError",
    "WebSocketDisconnectedError",
    "EventSignatureError",
    "EventExpiredError",
    "RateLimitExceededError",
    "ReplayAttackError",
]
