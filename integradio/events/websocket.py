"""
WebSocket server and client for real-time event streaming.

Security features (2026 best practices):
- Token-based authentication on handshake
- Origin validation
- Per-connection rate limiting
- Message size limits
- Automatic reconnection with backoff
- Connection timeout management
"""

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Set
from urllib.parse import parse_qs, urlparse

from .event import SemanticEvent
from .mesh import EventMesh
from .security import (
    ConnectionManager,
    ConnectionInfo,
    RateLimiter,
    validate_origin,
    validate_message_size,
)

logger = logging.getLogger(__name__)

# Message size limit (64KB default)
MAX_MESSAGE_SIZE = 65536


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket server/client."""

    # Security
    allowed_origins: list[str] = None  # None = allow all (dev only!)
    require_auth: bool = True
    auth_timeout: float = 10.0
    max_message_size: int = MAX_MESSAGE_SIZE

    # Rate limiting
    rate_limit: float = 100.0  # messages/second per client
    rate_burst: int = 200

    # Connection management
    max_connections: int = 10000
    max_per_ip: int = 100
    idle_timeout: float = 300.0
    heartbeat_interval: float = 30.0

    # Reconnection (client)
    reconnect: bool = True
    reconnect_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    reconnect_max_attempts: int = 10

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]
            logger.warning(
                "WebSocket accepting all origins - not secure for production!"
            )


class WebSocketServer:
    """
    Secure WebSocket server integrated with EventMesh.

    Features:
    - FastAPI/Starlette compatible
    - Token authentication
    - Automatic event broadcasting
    - Connection management

    Usage:
        mesh = EventMesh()
        server = WebSocketServer(mesh)

        # In FastAPI
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await server.handle(websocket)
    """

    def __init__(
        self,
        mesh: EventMesh,
        config: Optional[WebSocketConfig] = None,
        auth_handler: Optional[Callable[[str], Optional[str]]] = None,
    ):
        """
        Initialize WebSocket server.

        Args:
            mesh: EventMesh to bridge
            config: Server configuration
            auth_handler: Function to validate auth tokens, returns user_id or None
        """
        self.mesh = mesh
        self.config = config or WebSocketConfig()
        self.auth_handler = auth_handler

        # Connection tracking
        self._connections: dict[str, Any] = {}  # client_id -> websocket
        self._conn_manager = ConnectionManager(
            max_connections=self.config.max_connections,
            max_per_ip=self.config.max_per_ip,
            idle_timeout=self.config.idle_timeout,
        )

        # Per-client rate limiters
        self._rate_limiters: dict[str, RateLimiter] = {}

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("WebSocket server started")

    async def stop(self) -> None:
        """Stop background tasks and close connections."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Close all connections
        for ws in list(self._connections.values()):
            try:
                await ws.close(1001, "Server shutting down")
            except Exception:
                pass

        logger.info("WebSocket server stopped")

    async def handle(self, websocket: Any) -> None:
        """
        Handle a WebSocket connection.

        Compatible with FastAPI/Starlette WebSocket.

        Args:
            websocket: WebSocket connection object
        """
        client_id = secrets.token_hex(16)
        client_ip = self._get_client_ip(websocket)

        try:
            # Check if connection allowed
            allowed, reason = await self._conn_manager.can_connect(client_ip)
            if not allowed:
                await websocket.close(1008, reason)  # Policy violation
                return

            # Validate origin
            origin = self._get_origin(websocket)
            if not validate_origin(
                origin, self.config.allowed_origins, allow_none=True
            ):
                logger.warning(f"Rejected origin: {origin}")
                await websocket.close(1008, "Origin not allowed")
                return

            # Accept connection
            await websocket.accept()

            # Authenticate if required
            user_id = None
            if self.config.require_auth:
                user_id = await self._authenticate(websocket, client_id)
                if user_id is None:
                    await websocket.close(1008, "Authentication failed")
                    return

            # Register connection
            conn_info = await self._conn_manager.add(
                client_id, client_ip, origin, user_id
            )
            self._connections[client_id] = websocket
            self._rate_limiters[client_id] = RateLimiter(
                rate=self.config.rate_limit,
                burst=self.config.rate_burst,
            )

            # Subscribe to mesh events for this client
            sub_id = await self._setup_client_subscription(client_id, websocket)

            logger.info(f"Client connected: {client_id} from {client_ip}")

            # Emit connection event
            await self.mesh.emit(
                "system.connection.open",
                {"client_id": client_id, "user_id": user_id},
                source="websocket-server",
            )

            # Handle messages
            await self._message_loop(websocket, client_id, conn_info)

        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")

        finally:
            # Cleanup
            await self._disconnect(client_id)

    async def _authenticate(
        self, websocket: Any, client_id: str
    ) -> Optional[str]:
        """
        Authenticate client via token.

        Expects first message to be: {"type": "auth", "token": "..."}
        """
        try:
            # Wait for auth message with timeout
            data = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self.config.auth_timeout,
            )

            msg = json.loads(data)
            if msg.get("type") != "auth" or "token" not in msg:
                return None

            token = msg["token"]

            # Validate token
            if self.auth_handler:
                user_id = self.auth_handler(token)
                if user_id:
                    # Send auth success
                    await websocket.send_text(
                        json.dumps({"type": "auth_success", "client_id": client_id})
                    )
                    return user_id
            else:
                # No auth handler = accept any token (dev mode)
                await websocket.send_text(
                    json.dumps({"type": "auth_success", "client_id": client_id})
                )
                return token  # Use token as user_id

            return None

        except asyncio.TimeoutError:
            logger.warning(f"Auth timeout for {client_id}")
            return None
        except Exception as e:
            logger.error(f"Auth error for {client_id}: {e}")
            return None

    async def _message_loop(
        self,
        websocket: Any,
        client_id: str,
        conn_info: ConnectionInfo,
    ) -> None:
        """Main message receive loop."""
        while True:
            try:
                data = await websocket.receive_text()

                # Validate message size
                if not validate_message_size(data, self.config.max_message_size):
                    await self._send_error(
                        websocket, "Message too large", client_id
                    )
                    continue

                # Rate limiting
                limiter = self._rate_limiters.get(client_id)
                if limiter:
                    result = await limiter.check(client_id)
                    if not result.allowed:
                        await self._send_error(
                            websocket,
                            f"Rate limited. Retry after {result.retry_after:.1f}s",
                            client_id,
                        )
                        continue

                # Update activity
                await self._conn_manager.update_activity(client_id)

                # Parse and handle message
                await self._handle_message(websocket, client_id, data)

            except Exception as e:
                # Connection closed or error
                if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                    break
                logger.error(f"Message error for {client_id}: {e}")
                break

    async def _handle_message(
        self, websocket: Any, client_id: str, data: str
    ) -> None:
        """Handle incoming message."""
        try:
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "subscribe":
                # Subscribe to patterns
                patterns = msg.get("patterns", [])
                await self._handle_subscribe(client_id, patterns)

            elif msg_type == "unsubscribe":
                # Unsubscribe from patterns
                patterns = msg.get("patterns", [])
                await self._handle_unsubscribe(client_id, patterns)

            elif msg_type == "event":
                # Publish event to mesh
                event_data = msg.get("event", {})
                event = SemanticEvent.from_dict(event_data)
                await self.mesh.publish(event, client_id=client_id)

            elif msg_type == "ping":
                # Respond to ping
                await websocket.send_text(
                    json.dumps({"type": "pong", "time": time.time()})
                )

            else:
                logger.warning(f"Unknown message type from {client_id}: {msg_type}")

        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON", client_id)
        except Exception as e:
            logger.error(f"Handle message error: {e}")
            await self._send_error(websocket, str(e), client_id)

    async def _handle_subscribe(
        self, client_id: str, patterns: list[str]
    ) -> None:
        """Handle subscription request."""
        conn_info = await self._conn_manager.get(client_id)
        if conn_info:
            conn_info.subscriptions.update(patterns)
            logger.debug(f"Client {client_id} subscribed to: {patterns}")

    async def _handle_unsubscribe(
        self, client_id: str, patterns: list[str]
    ) -> None:
        """Handle unsubscription request."""
        conn_info = await self._conn_manager.get(client_id)
        if conn_info:
            conn_info.subscriptions.difference_update(patterns)
            logger.debug(f"Client {client_id} unsubscribed from: {patterns}")

    async def _setup_client_subscription(
        self, client_id: str, websocket: Any
    ) -> str:
        """Set up mesh subscription for a client."""

        async def forward_to_client(event: SemanticEvent) -> None:
            """Forward matching events to WebSocket client."""
            conn_info = await self._conn_manager.get(client_id)
            if not conn_info:
                return

            # Check if event matches any subscription
            for pattern in conn_info.subscriptions:
                if event.matches_pattern(pattern):
                    try:
                        ws = self._connections.get(client_id)
                        if ws:
                            await ws.send_text(
                                json.dumps({
                                    "type": "event",
                                    "event": event.to_dict(),
                                })
                            )
                    except Exception as e:
                        logger.error(f"Forward error to {client_id}: {e}")
                    break

        # Subscribe to all events and filter in handler
        return self.mesh.subscribe(["**"], forward_to_client, client_id)

    async def _disconnect(self, client_id: str) -> None:
        """Clean up disconnected client."""
        # Remove from connections
        self._connections.pop(client_id, None)
        self._rate_limiters.pop(client_id, None)

        # Remove from manager
        conn_info = await self._conn_manager.remove(client_id)

        if conn_info:
            logger.info(f"Client disconnected: {client_id}")

            # Emit disconnection event
            await self.mesh.emit(
                "system.connection.close",
                {"client_id": client_id, "user_id": conn_info.user_id},
                source="websocket-server",
            )

    async def _send_error(
        self, websocket: Any, message: str, client_id: str
    ) -> None:
        """Send error message to client."""
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": message})
            )
        except Exception:
            pass

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all clients."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Send heartbeat to all connections
                event = {
                    "type": "heartbeat",
                    "time": time.time(),
                    "connections": self._conn_manager.connection_count,
                }
                message = json.dumps(event)

                for client_id, ws in list(self._connections.items()):
                    try:
                        await ws.send_text(message)
                    except Exception:
                        # Connection dead, will be cleaned up
                        pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up idle connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                idle = await self._conn_manager.get_idle_connections()
                for client_id in idle:
                    ws = self._connections.get(client_id)
                    if ws:
                        try:
                            await ws.close(1000, "Idle timeout")
                        except Exception:
                            pass
                    await self._disconnect(client_id)

                if idle:
                    logger.info(f"Cleaned up {len(idle)} idle connections")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def _get_client_ip(self, websocket: Any) -> str:
        """Extract client IP from websocket."""
        try:
            # Try common attributes
            if hasattr(websocket, "client"):
                return websocket.client.host
            if hasattr(websocket, "scope"):
                client = websocket.scope.get("client")
                if client:
                    return client[0]
        except Exception:
            pass
        return "unknown"

    def _get_origin(self, websocket: Any) -> Optional[str]:
        """Extract origin header from websocket."""
        try:
            if hasattr(websocket, "headers"):
                return websocket.headers.get("origin")
            if hasattr(websocket, "scope"):
                headers = dict(websocket.scope.get("headers", []))
                return headers.get(b"origin", b"").decode()
        except Exception:
            pass
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        return {
            "connections": self._conn_manager.connection_count,
            "mesh_stats": self.mesh.get_stats(),
        }


class WebSocketClient:
    """
    Secure WebSocket client with automatic reconnection.

    Features:
    - Token authentication
    - Automatic reconnection with exponential backoff
    - Pattern subscriptions
    - Event callbacks

    Usage:
        client = WebSocketClient("ws://localhost:8000/ws", token="my-token")

        @client.on("ui.component.*")
        async def handle_component(event):
            print(event)

        await client.connect()
    """

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        config: Optional[WebSocketConfig] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            url: WebSocket server URL
            token: Authentication token
            config: Client configuration
        """
        self.url = url
        self.token = token
        self.config = config or WebSocketConfig()

        self._websocket: Any = None
        self._connected = False
        self._reconnect_attempt = 0
        self._subscriptions: Set[str] = set()
        self._handlers: dict[str, list[Callable]] = {}

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to WebSocket server.

        Returns:
            True if connected successfully
        """
        try:
            # Import websockets library
            import websockets

            self._websocket = await websockets.connect(
                self.url,
                max_size=self.config.max_message_size,
            )

            # Authenticate
            if self.token:
                await self._websocket.send(
                    json.dumps({"type": "auth", "token": self.token})
                )

                response = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=self.config.auth_timeout,
                )
                msg = json.loads(response)
                if msg.get("type") != "auth_success":
                    logger.error("Authentication failed")
                    await self._websocket.close()
                    return False

            self._connected = True
            self._reconnect_attempt = 0

            # Re-subscribe to patterns
            if self._subscriptions:
                await self._websocket.send(
                    json.dumps({
                        "type": "subscribe",
                        "patterns": list(self._subscriptions),
                    })
                )

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info(f"Connected to {self.url}")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass

        logger.info("Disconnected")

    async def subscribe(self, *patterns: str) -> None:
        """
        Subscribe to event patterns.

        Args:
            patterns: Patterns to subscribe to
        """
        self._subscriptions.update(patterns)

        if self._connected and self._websocket:
            await self._websocket.send(
                json.dumps({"type": "subscribe", "patterns": list(patterns)})
            )

    async def unsubscribe(self, *patterns: str) -> None:
        """
        Unsubscribe from event patterns.

        Args:
            patterns: Patterns to unsubscribe from
        """
        self._subscriptions.difference_update(patterns)

        if self._connected and self._websocket:
            await self._websocket.send(
                json.dumps({"type": "unsubscribe", "patterns": list(patterns)})
            )

    async def emit(
        self,
        event_type: str,
        data: Any = None,
        source: str = "client",
    ) -> None:
        """
        Emit an event to the server.

        Args:
            event_type: Event type
            data: Event payload
            source: Event source
        """
        if not self._connected or not self._websocket:
            raise RuntimeError("Not connected")

        event = SemanticEvent(type=event_type, source=source, data=data)
        await self._websocket.send(
            json.dumps({"type": "event", "event": event.to_dict()})
        )

    def on(self, *patterns: str) -> Callable:
        """
        Decorator to register event handler.

        Usage:
            @client.on("ui.component.*")
            async def handler(event):
                print(event)
        """
        def decorator(fn: Callable) -> Callable:
            for pattern in patterns:
                if pattern not in self._handlers:
                    self._handlers[pattern] = []
                self._handlers[pattern].append(fn)
                self._subscriptions.add(pattern)
            return fn
        return decorator

    async def _receive_loop(self) -> None:
        """Receive and process messages."""
        while self._connected:
            try:
                data = await self._websocket.recv()
                msg = json.loads(data)

                msg_type = msg.get("type")

                if msg_type == "event":
                    event = SemanticEvent.from_dict(msg["event"])
                    await self._dispatch(event)

                elif msg_type == "heartbeat":
                    # Respond with pong
                    pass

                elif msg_type == "error":
                    logger.warning(f"Server error: {msg.get('message')}")

            except Exception as e:
                if self._connected:
                    logger.error(f"Receive error: {e}")
                    self._connected = False
                    await self._schedule_reconnect()
                break

    async def _dispatch(self, event: SemanticEvent) -> None:
        """Dispatch event to registered handlers."""
        for pattern, handlers in self._handlers.items():
            if event.matches_pattern(pattern):
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")

    async def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt."""
        if not self.config.reconnect:
            return

        if self._reconnect_attempt >= self.config.reconnect_max_attempts:
            logger.error("Max reconnection attempts reached")
            return

        # Exponential backoff
        delay = min(
            self.config.reconnect_delay * (2 ** self._reconnect_attempt),
            self.config.reconnect_max_delay,
        )
        self._reconnect_attempt += 1

        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempt})")
        await asyncio.sleep(delay)

        if await self.connect():
            logger.info("Reconnected successfully")
        else:
            await self._schedule_reconnect()
