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
)
from .handlers import EventHandler, on_event

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
    # Handlers
    "EventHandler",
    "on_event",
]
