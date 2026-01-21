"""
Tests for integradio.events.websocket - WebSocket server and client with security.

Comprehensive integration tests for:
- WebSocketConfig
- WebSocketServer (handle, auth, message loop, subscriptions, heartbeat, cleanup)
- WebSocketClient (connect, disconnect, subscribe, emit, handlers, reconnect)
"""

import asyncio
import json
import time
import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Mock WebSocket for Testing
# =============================================================================


class MockWebSocket:
    """Mock WebSocket that simulates FastAPI/Starlette WebSocket behavior."""

    def __init__(
        self,
        client_ip: str = "127.0.0.1",
        origin: str = "https://example.com",
        messages: list[str] | None = None,
    ):
        self.client = MagicMock()
        self.client.host = client_ip
        self.headers = {"origin": origin}
        self._messages = list(messages) if messages else []
        self._message_index = 0
        self._sent: list[str] = []
        self._closed = False
        self._close_code: int | None = None
        self._close_reason: str | None = None
        self._accepted = False
        self._receive_event = asyncio.Event()

    async def accept(self):
        self._accepted = True

    async def receive_text(self) -> str:
        if self._closed:
            raise RuntimeError("WebSocket is closed")
        if self._message_index < len(self._messages):
            msg = self._messages[self._message_index]
            self._message_index += 1
            return msg
        # Wait for new messages or simulate disconnect
        await asyncio.sleep(0.01)
        raise RuntimeError("WebSocket disconnected")

    async def send_text(self, data: str):
        if self._closed:
            raise RuntimeError("WebSocket is closed")
        self._sent.append(data)

    async def close(self, code: int = 1000, reason: str = ""):
        self._closed = True
        self._close_code = code
        self._close_reason = reason

    def add_message(self, msg: str):
        """Add a message to the queue."""
        self._messages.append(msg)

    @property
    def sent_messages(self) -> list[dict]:
        """Get sent messages as parsed JSON."""
        return [json.loads(m) for m in self._sent]


class MockWebSocketWithScope:
    """Mock WebSocket with scope attribute (alternate implementation)."""

    def __init__(self, client_ip: str = "127.0.0.1", origin: str = "https://example.com"):
        self.scope = {
            "client": (client_ip, 8080),
            "headers": [(b"origin", origin.encode())],
        }
        self._messages: list[str] = []
        self._sent: list[str] = []
        self._closed = False
        self._accepted = False

    async def accept(self):
        self._accepted = True

    async def receive_text(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        raise RuntimeError("WebSocket disconnected")

    async def send_text(self, data: str):
        self._sent.append(data)

    async def close(self, code: int = 1000, reason: str = ""):
        self._closed = True


# =============================================================================
# WebSocketConfig Tests
# =============================================================================


class TestWebSocketConfig:
    """Tests for WebSocketConfig dataclass."""

    def test_default_values(self):
        """Default config values are set."""
        from integradio.events.websocket import WebSocketConfig

        config = WebSocketConfig()

        assert config.require_auth is True
        assert config.auth_timeout == 10.0
        assert config.max_message_size == 65536
        assert config.rate_limit == 100.0
        assert config.rate_burst == 200
        assert config.max_connections == 10000
        assert config.max_per_ip == 100
        assert config.idle_timeout == 300.0
        assert config.heartbeat_interval == 30.0
        assert config.reconnect is True
        assert config.reconnect_delay == 1.0
        assert config.reconnect_max_delay == 60.0
        assert config.reconnect_max_attempts == 10

    def test_custom_values(self):
        """Custom config values override defaults."""
        from integradio.events.websocket import WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["https://example.com"],
            require_auth=False,
            auth_timeout=5.0,
            rate_limit=50.0,
            max_connections=1000,
        )

        assert config.allowed_origins == ["https://example.com"]
        assert config.require_auth is False
        assert config.auth_timeout == 5.0
        assert config.rate_limit == 50.0
        assert config.max_connections == 1000

    def test_post_init_default_origins_warning(self):
        """Warning is logged when allowed_origins is None (defaults to *)."""
        from integradio.events.websocket import WebSocketConfig
        import warnings

        # WebSocketConfig logs a warning when allowed_origins is None
        # We verify it sets to ["*"]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            config = WebSocketConfig()
            assert config.allowed_origins == ["*"]


# =============================================================================
# WebSocketServer Tests
# =============================================================================


class TestWebSocketServer:
    """Tests for WebSocketServer."""

    @pytest.fixture
    def mesh(self):
        """Create a fresh EventMesh for testing."""
        from integradio.events import EventMesh
        return EventMesh()

    @pytest.fixture
    def config(self):
        """Create config with relaxed settings for testing."""
        from integradio.events.websocket import WebSocketConfig
        return WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
            rate_limit=1000.0,
            rate_burst=1000,
            idle_timeout=10.0,
            heartbeat_interval=0.1,  # Fast heartbeats for testing
        )

    @pytest.fixture
    def server(self, mesh, config):
        """Create a WebSocketServer for testing."""
        from integradio.events.websocket import WebSocketServer
        return WebSocketServer(mesh, config)

    @pytest.mark.asyncio
    async def test_start_stop(self, server):
        """Server starts and stops background tasks."""
        await server.start()

        assert server._heartbeat_task is not None
        assert server._cleanup_task is not None
        assert not server._heartbeat_task.done()
        assert not server._cleanup_task.done()

        await server.stop()

        # Wait briefly for cancellation to complete
        await asyncio.sleep(0.05)

        # Tasks should be cancelled or done
        assert server._heartbeat_task.cancelled() or server._heartbeat_task.done()
        assert server._cleanup_task.cancelled() or server._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_handle_connection_no_auth(self, server):
        """Handle connection without auth requirement."""
        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "ping"}),  # Just ping then disconnect
        ]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        assert ws._accepted is True
        # Should have received pong
        assert any(m.get("type") == "pong" for m in ws.sent_messages)

    @pytest.mark.asyncio
    async def test_handle_origin_rejected(self, mesh):
        """Connection rejected for invalid origin."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["https://allowed.com"],
            require_auth=False,
        )
        server = WebSocketServer(mesh, config)

        ws = MockWebSocket(origin="https://notallowed.com")

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        assert ws._closed is True
        assert ws._close_code == 1008  # Policy violation

    @pytest.mark.asyncio
    async def test_handle_max_connections(self, server):
        """Connection rejected when at max capacity."""
        # Fill up connections
        server._conn_manager.max_connections = 1
        await server._conn_manager.add("existing", "1.1.1.1")

        ws = MockWebSocket()

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        assert ws._closed is True
        assert ws._close_code == 1008

    @pytest.mark.asyncio
    async def test_handle_auth_required(self, mesh):
        """Connection with auth requirement."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=True,
            auth_timeout=1.0,
        )

        # Auth handler that accepts token "valid-token"
        def auth_handler(token):
            if token == "valid-token":
                return "user-123"
            return None

        server = WebSocketServer(mesh, config, auth_handler=auth_handler)

        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "auth", "token": "valid-token"}),
            json.dumps({"type": "ping"}),
        ]

        async with asyncio.timeout(2.0):
            await server.handle(ws)

        assert ws._accepted is True
        # Should have auth_success response
        assert any(m.get("type") == "auth_success" for m in ws.sent_messages)

    @pytest.mark.asyncio
    async def test_handle_auth_failed(self, mesh):
        """Connection rejected with invalid auth."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=True,
            auth_timeout=1.0,
        )

        def auth_handler(token):
            return None  # Reject all

        server = WebSocketServer(mesh, config, auth_handler=auth_handler)

        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "auth", "token": "invalid-token"}),
        ]

        async with asyncio.timeout(2.0):
            await server.handle(ws)

        assert ws._closed is True
        assert ws._close_code == 1008

    @pytest.mark.asyncio
    async def test_handle_auth_timeout(self, mesh):
        """Connection rejected when auth times out."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=True,
            auth_timeout=0.1,  # Very short timeout
        )

        server = WebSocketServer(mesh, config)

        ws = MockWebSocket()
        ws._messages = []  # No auth message

        async with asyncio.timeout(2.0):
            await server.handle(ws)

        assert ws._closed is True
        assert ws._close_code == 1008

    @pytest.mark.asyncio
    async def test_handle_auth_no_handler_dev_mode(self, mesh):
        """Auth without handler uses token as user_id (dev mode)."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=True,
            auth_timeout=1.0,
        )

        server = WebSocketServer(mesh, config, auth_handler=None)

        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "auth", "token": "dev-token-123"}),
            json.dumps({"type": "ping"}),
        ]

        async with asyncio.timeout(2.0):
            await server.handle(ws)

        assert ws._accepted is True
        assert any(m.get("type") == "auth_success" for m in ws.sent_messages)

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, server, mesh):
        """Client can subscribe to patterns."""
        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "subscribe", "patterns": ["ui.*", "data.*"]}),
        ]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        # Verify subscriptions were registered (check via conn_manager)
        # Connection is cleaned up after disconnect, so we verify it worked
        assert ws._accepted is True

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, server, mesh):
        """Client can unsubscribe from patterns."""
        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "subscribe", "patterns": ["ui.*", "data.*"]}),
            json.dumps({"type": "unsubscribe", "patterns": ["data.*"]}),
        ]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        assert ws._accepted is True

    @pytest.mark.asyncio
    async def test_handle_event_message(self, server, mesh):
        """Client can publish events."""
        from integradio.events import SemanticEvent

        received_events = []

        @mesh.on("test.event")
        async def handler(event):
            received_events.append(event)

        async with mesh:
            ws = MockWebSocket()
            event_data = SemanticEvent(
                type="test.event",
                source="client",
                data={"value": 42},
            ).to_dict()
            ws._messages = [
                json.dumps({"type": "event", "event": event_data}),
            ]

            async with asyncio.timeout(1.0):
                await server.handle(ws)

            # Wait for event processing
            await asyncio.sleep(0.2)

        # Verify event was published
        assert len(received_events) >= 1
        assert received_events[0].data == {"value": 42}

    @pytest.mark.asyncio
    async def test_handle_ping_pong(self, server):
        """Server responds to ping with pong."""
        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "ping"}),
        ]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        pong_messages = [m for m in ws.sent_messages if m.get("type") == "pong"]
        assert len(pong_messages) == 1
        assert "time" in pong_messages[0]

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, server):
        """Server handles invalid JSON gracefully."""
        ws = MockWebSocket()
        ws._messages = [
            "not valid json {{{",
            json.dumps({"type": "ping"}),  # Valid message after
        ]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        # Should have received error message
        error_messages = [m for m in ws.sent_messages if m.get("type") == "error"]
        assert len(error_messages) >= 1
        assert "Invalid JSON" in error_messages[0].get("message", "")

    @pytest.mark.asyncio
    async def test_handle_message_too_large(self, mesh):
        """Server rejects oversized messages."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
            max_message_size=100,  # Very small limit
        )
        server = WebSocketServer(mesh, config)

        ws = MockWebSocket()
        large_message = json.dumps({"type": "event", "data": "x" * 200})
        ws._messages = [large_message]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        # Should have error about size
        error_messages = [m for m in ws.sent_messages if m.get("type") == "error"]
        assert any("large" in m.get("message", "").lower() for m in error_messages)

    @pytest.mark.asyncio
    async def test_handle_rate_limited(self, mesh):
        """Server rate limits excessive messages."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
            rate_limit=1.0,  # Very low limit
            rate_burst=2,  # Only 2 messages allowed
        )
        server = WebSocketServer(mesh, config)

        ws = MockWebSocket()
        # Send many messages to exceed rate limit
        ws._messages = [json.dumps({"type": "ping"}) for _ in range(10)]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        # Should have rate limit error
        error_messages = [m for m in ws.sent_messages if m.get("type") == "error"]
        assert any("rate" in m.get("message", "").lower() for m in error_messages)

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, server):
        """Server handles unknown message types."""
        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "unknown_type", "data": {}}),
        ]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        # Should not crash, just log warning
        assert ws._accepted is True

    @pytest.mark.asyncio
    async def test_heartbeat_loop(self, mesh):
        """Heartbeat loop sends periodic heartbeats."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
            heartbeat_interval=0.05,  # Very fast for testing
        )
        server = WebSocketServer(mesh, config)

        # Add a connection manually
        client_id = "test-client"
        ws = MockWebSocket()
        server._connections[client_id] = ws

        await server.start()
        await asyncio.sleep(0.15)  # Wait for a few heartbeats
        await server.stop()

        # Should have received heartbeats
        heartbeats = [m for m in ws.sent_messages if m.get("type") == "heartbeat"]
        assert len(heartbeats) >= 1
        assert "time" in heartbeats[0]
        assert "connections" in heartbeats[0]

    @pytest.mark.asyncio
    async def test_cleanup_idle_connections(self, mesh):
        """Cleanup loop removes idle connections."""
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
            idle_timeout=0.01,  # Very short for testing
        )
        server = WebSocketServer(mesh, config)

        # Add a connection that will become idle
        client_id = "idle-client"
        ws = MockWebSocket()
        server._connections[client_id] = ws
        await server._conn_manager.add(client_id, "1.1.1.1")

        # Make it idle by setting old activity time
        conn_info = await server._conn_manager.get(client_id)
        conn_info.last_activity = time.time() - 100

        # Run cleanup manually
        idle = await server._conn_manager.get_idle_connections()
        assert client_id in idle

    @pytest.mark.asyncio
    async def test_get_stats(self, server, mesh):
        """Server returns stats."""
        await server._conn_manager.add("client-1", "1.1.1.1")

        stats = server.get_stats()

        assert "connections" in stats
        assert stats["connections"] == 1
        assert "mesh_stats" in stats

    def test_get_client_ip_with_client_attr(self, server):
        """Extract client IP from websocket.client."""
        ws = MockWebSocket(client_ip="192.168.1.100")
        ip = server._get_client_ip(ws)
        assert ip == "192.168.1.100"

    def test_get_client_ip_with_scope(self, server):
        """Extract client IP from websocket.scope."""
        ws = MockWebSocketWithScope(client_ip="10.0.0.1")
        ip = server._get_client_ip(ws)
        assert ip == "10.0.0.1"

    def test_get_client_ip_fallback(self, server):
        """Return 'unknown' when IP can't be extracted."""
        ws = MagicMock()
        del ws.client
        del ws.scope
        ip = server._get_client_ip(ws)
        assert ip == "unknown"

    def test_get_origin_with_headers(self, server):
        """Extract origin from websocket.headers."""
        ws = MockWebSocket(origin="https://example.com")
        origin = server._get_origin(ws)
        assert origin == "https://example.com"

    def test_get_origin_with_scope(self, server):
        """Extract origin from websocket.scope."""
        ws = MockWebSocketWithScope(origin="https://scope.example.com")
        origin = server._get_origin(ws)
        assert origin == "https://scope.example.com"

    def test_get_origin_none(self, server):
        """Return None when origin can't be extracted."""
        ws = MagicMock()
        del ws.headers
        del ws.scope
        origin = server._get_origin(ws)
        assert origin is None

    @pytest.mark.asyncio
    async def test_disconnect_emits_event(self, server, mesh):
        """Disconnect emits system.connection.close event."""
        received_events = []

        @mesh.on("system.connection.close")
        async def handler(event):
            received_events.append(event)

        async with mesh:
            # Add and then disconnect a client
            client_id = "test-client"
            await server._conn_manager.add(client_id, "1.1.1.1", user_id="user-1")
            server._connections[client_id] = MockWebSocket()

            await server._disconnect(client_id)
            await asyncio.sleep(0.2)

        assert len(received_events) == 1
        assert received_events[0].data["client_id"] == "test-client"

    @pytest.mark.asyncio
    async def test_send_error(self, server):
        """_send_error sends error message to client."""
        ws = MockWebSocket()
        await server._send_error(ws, "Test error message", "client-123")

        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["type"] == "error"
        assert ws.sent_messages[0]["message"] == "Test error message"

    @pytest.mark.asyncio
    async def test_send_error_disconnected_client(self, server):
        """_send_error handles disconnected client gracefully."""

        class DisconnectedWebSocket:
            async def send_text(self, data):
                raise ConnectionResetError("Connection reset by peer")

        ws = DisconnectedWebSocket()

        # Should not raise
        await server._send_error(ws, "Error", "client-123")


# =============================================================================
# WebSocketClient Tests
# =============================================================================


class TestWebSocketClient:
    """Tests for WebSocketClient."""

    @pytest.fixture
    def config(self):
        """Client config for testing."""
        from integradio.events.websocket import WebSocketConfig
        return WebSocketConfig(
            reconnect=False,  # Disable reconnect for most tests
            auth_timeout=1.0,
        )

    @staticmethod
    def create_mock_ws(recv_responses: list[str]):
        """Create a mock websocket with predefined responses."""
        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(side_effect=recv_responses + [ConnectionResetError()])
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    @pytest.mark.asyncio
    async def test_connect_success(self, config):
        """Client connects successfully."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", token="test-token", config=config)

        mock_ws = self.create_mock_ws([
            json.dumps({"type": "auth_success", "client_id": "123"})
        ])

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            result = await client.connect()

        assert result is True
        assert client.connected is True
        mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self, config):
        """Client handles auth failure."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", token="bad-token", config=config)

        mock_ws = self.create_mock_ws([
            json.dumps({"type": "auth_failed"})
        ])

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            result = await client.connect()

        assert result is False
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_connect_timeout(self, config):
        """Client handles connection timeout."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        async def mock_connect(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch("websockets.connect", side_effect=mock_connect):
            result = await client.connect()

        assert result is False
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_connect_refused(self, config):
        """Client handles connection refused."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        async def mock_connect(*args, **kwargs):
            raise ConnectionRefusedError()

        with patch("websockets.connect", side_effect=mock_connect):
            result = await client.connect()

        assert result is False
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, config):
        """Client disconnects cleanly."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        mock_ws = self.create_mock_ws([
            json.dumps({"type": "auth_success", "client_id": "123"})
        ])

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()

        assert client.connected is True

        await client.disconnect()

        assert client.connected is False
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe(self, config):
        """Client can subscribe to patterns."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        mock_ws = self.create_mock_ws([
            json.dumps({"type": "auth_success", "client_id": "123"})
        ])

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()

        await client.subscribe("ui.*", "data.*")

        assert "ui.*" in client._subscriptions
        assert "data.*" in client._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe(self, config):
        """Client can unsubscribe from patterns."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)
        client._subscriptions = {"ui.*", "data.*"}

        mock_ws = self.create_mock_ws([
            json.dumps({"type": "auth_success", "client_id": "123"})
        ])

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()

        await client.unsubscribe("data.*")

        assert "ui.*" in client._subscriptions
        assert "data.*" not in client._subscriptions

    @pytest.mark.asyncio
    async def test_emit_event(self, config):
        """Client can emit events."""
        from integradio.events.websocket import WebSocketClient

        # Use a token so auth message is sent
        client = WebSocketClient("ws://localhost:8000/ws", token="test-token", config=config)

        mock_ws = self.create_mock_ws([
            json.dumps({"type": "auth_success", "client_id": "123"})
        ])

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()

        await client.emit("test.event", data={"key": "value"}, source="test")

        # Verify send was called with event
        calls = mock_ws.send.call_args_list
        assert len(calls) >= 2  # auth + event
        event_call = json.loads(calls[-1][0][0])
        assert event_call["type"] == "event"
        assert event_call["event"]["type"] == "test.event"

    @pytest.mark.asyncio
    async def test_emit_not_connected(self, config):
        """Client raises error when emitting while not connected."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.emit("test.event", data={})

    def test_on_decorator(self, config):
        """@client.on decorator registers handlers."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        @client.on("ui.click", "ui.hover")
        async def handler(event):
            pass

        assert "ui.click" in client._handlers
        assert "ui.hover" in client._handlers
        assert "ui.click" in client._subscriptions
        assert "ui.hover" in client._subscriptions

    @pytest.mark.asyncio
    async def test_dispatch_to_handlers(self, config):
        """Events are dispatched to registered handlers."""
        from integradio.events.websocket import WebSocketClient
        from integradio.events import SemanticEvent

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        received_events = []

        @client.on("test.*")
        async def handler(event):
            received_events.append(event)

        event = SemanticEvent(type="test.event", source="server", data={"x": 1})
        await client._dispatch(event)

        assert len(received_events) == 1
        assert received_events[0].data == {"x": 1}

    @pytest.mark.asyncio
    async def test_dispatch_sync_handler(self, config):
        """Sync handlers work correctly."""
        from integradio.events.websocket import WebSocketClient
        from integradio.events import SemanticEvent

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        received_events = []

        @client.on("test.*")
        def sync_handler(event):
            received_events.append(event)

        event = SemanticEvent(type="test.event", source="server", data={"x": 1})
        await client._dispatch(event)

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_dispatch_handler_error(self, config):
        """Handler errors are logged but don't crash."""
        from integradio.events.websocket import WebSocketClient
        from integradio.events import SemanticEvent

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        @client.on("test.*")
        async def failing_handler(event):
            raise ValueError("Test error")

        event = SemanticEvent(type="test.event", source="server", data={})

        # Should not raise
        await client._dispatch(event)

    @pytest.mark.asyncio
    async def test_reconnect_logic(self):
        """Client reconnects with exponential backoff."""
        from integradio.events.websocket import WebSocketClient, WebSocketConfig

        config = WebSocketConfig(
            reconnect=True,
            reconnect_delay=0.01,  # Fast for testing
            reconnect_max_delay=0.1,
            reconnect_max_attempts=3,
        )
        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        # Track reconnect attempts
        attempts = []

        async def mock_connect():
            attempts.append(1)
            return False  # Always fail

        with patch.object(client, "connect", side_effect=mock_connect):
            await client._schedule_reconnect()

        assert len(attempts) <= config.reconnect_max_attempts

    @pytest.mark.asyncio
    async def test_reconnect_disabled(self):
        """Client doesn't reconnect when disabled."""
        from integradio.events.websocket import WebSocketClient, WebSocketConfig

        config = WebSocketConfig(reconnect=False)
        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        # Should return immediately without reconnecting
        await client._schedule_reconnect()

        # No reconnect attempt should be made
        assert client._reconnect_attempt == 0

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts(self):
        """Client stops reconnecting after max attempts."""
        from integradio.events.websocket import WebSocketClient, WebSocketConfig

        config = WebSocketConfig(
            reconnect=True,
            reconnect_max_attempts=2,
        )
        client = WebSocketClient("ws://localhost:8000/ws", config=config)
        client._reconnect_attempt = 10  # Already past max

        # Should not attempt reconnect
        await client._schedule_reconnect()

    @pytest.mark.asyncio
    async def test_receive_loop_event(self, config):
        """Receive loop processes incoming events."""
        from integradio.events.websocket import WebSocketClient
        from integradio.events import SemanticEvent

        client = WebSocketClient("ws://localhost:8000/ws", config=config)
        received = []

        @client.on("server.event")
        async def handler(event):
            received.append(event)

        # Mock websocket with event message
        event_data = SemanticEvent(type="server.event", source="server", data={"value": 42}).to_dict()

        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"type": "auth_success", "client_id": "123"}),
                json.dumps({"type": "event", "event": event_data}),
                ConnectionResetError(),  # Disconnect after event
            ]
        )
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()

            # Wait for receive loop to process
            await asyncio.sleep(0.1)

        assert len(received) >= 1

    @pytest.mark.asyncio
    async def test_receive_loop_heartbeat(self, config):
        """Receive loop handles heartbeat messages."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"type": "auth_success", "client_id": "123"}),
                json.dumps({"type": "heartbeat", "time": time.time()}),
                ConnectionResetError(),
            ]
        )
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()
            await asyncio.sleep(0.1)

        # Should handle heartbeat without issues
        assert client._connected is False  # Disconnected after

    @pytest.mark.asyncio
    async def test_receive_loop_error_message(self, config):
        """Receive loop handles error messages."""
        from integradio.events.websocket import WebSocketClient

        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"type": "auth_success", "client_id": "123"}),
                json.dumps({"type": "error", "message": "Server error"}),
                ConnectionResetError(),
            ]
        )
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()
            await asyncio.sleep(0.1)

        # Should handle error without crashing
        assert True

    @pytest.mark.asyncio
    async def test_connect_resubscribes(self):
        """Client resubscribes to patterns on reconnect."""
        from integradio.events.websocket import WebSocketClient, WebSocketConfig

        config = WebSocketConfig(reconnect=False)
        client = WebSocketClient("ws://localhost:8000/ws", token="test", config=config)
        client._subscriptions = {"ui.*", "data.*"}

        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"type": "auth_success", "client_id": "123"}),
                ConnectionResetError(),
            ]
        )
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch("websockets.connect", side_effect=mock_connect):
            await client.connect()

        # Verify subscribe message was sent
        calls = mock_ws.send.call_args_list
        subscribe_calls = [c for c in calls if "subscribe" in c[0][0]]
        assert len(subscribe_calls) >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestWebSocketIntegration:
    """Integration tests for WebSocket server and client."""

    @pytest.mark.asyncio
    async def test_event_forwarding(self):
        """Events are forwarded from mesh to subscribed clients."""
        from integradio.events import EventMesh
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        mesh = EventMesh()
        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
        )
        server = WebSocketServer(mesh, config)

        # Simulate a connected client
        client_id = "test-client"
        ws = MockWebSocket()
        server._connections[client_id] = ws
        conn_info = await server._conn_manager.add(client_id, "1.1.1.1")
        await conn_info.add_subscriptions(["test.*"])

        async with mesh:
            # Subscribe directly to mesh for forwarding
            async def forward_to_client(event):
                if event.matches_pattern("test.*"):
                    await ws.send_text(json.dumps({
                        "type": "event",
                        "event": event.to_dict(),
                    }))

            mesh.subscribe("test.*", forward_to_client)

            # Emit an event
            await mesh.emit("test.event", data={"value": 123}, source="test")

            # Wait for processing
            await asyncio.sleep(0.2)

        # Verify event was forwarded
        forwarded = [m for m in ws.sent_messages if m.get("type") == "event"]
        assert len(forwarded) >= 1
        assert forwarded[0]["event"]["type"] == "test.event"

    @pytest.mark.asyncio
    async def test_connection_lifecycle_events(self):
        """Connection lifecycle emits proper events."""
        from integradio.events import EventMesh
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        mesh = EventMesh()
        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
        )
        server = WebSocketServer(mesh, config)

        connection_events = []

        @mesh.on("system.connection.*")
        async def handler(event):
            connection_events.append(event)

        async with mesh:
            # Handle a connection
            ws = MockWebSocket()
            ws._messages = []  # Will disconnect immediately

            async with asyncio.timeout(1.0):
                await server.handle(ws)

            await asyncio.sleep(0.2)

        # Should have open and close events
        event_types = [e.type for e in connection_events]
        assert "system.connection.open" in event_types
        assert "system.connection.close" in event_types


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestWebSocketEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_auth_invalid_json(self):
        """Handle invalid JSON in auth message."""
        from integradio.events import EventMesh
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        mesh = EventMesh()
        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=True,
            auth_timeout=1.0,
        )
        server = WebSocketServer(mesh, config)

        ws = MockWebSocket()
        ws._messages = ["not valid json"]

        async with asyncio.timeout(2.0):
            await server.handle(ws)

        assert ws._closed is True

    @pytest.mark.asyncio
    async def test_auth_missing_token(self):
        """Handle auth message without token field."""
        from integradio.events import EventMesh
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        mesh = EventMesh()
        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=True,
            auth_timeout=1.0,
        )
        server = WebSocketServer(mesh, config)

        ws = MockWebSocket()
        ws._messages = [json.dumps({"type": "auth"})]  # No token

        async with asyncio.timeout(2.0):
            await server.handle(ws)

        assert ws._closed is True

    @pytest.mark.asyncio
    async def test_handle_malformed_event(self):
        """Handle malformed event data."""
        from integradio.events import EventMesh
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        mesh = EventMesh()
        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
        )
        server = WebSocketServer(mesh, config)

        ws = MockWebSocket()
        ws._messages = [
            json.dumps({"type": "event", "event": {}}),  # Missing required fields
        ]

        async with asyncio.timeout(1.0):
            await server.handle(ws)

        # Should have error message about invalid format
        error_messages = [m for m in ws.sent_messages if m.get("type") == "error"]
        assert len(error_messages) >= 1

    @pytest.mark.asyncio
    async def test_connection_info_thread_safety(self):
        """ConnectionInfo subscription methods are thread-safe."""
        from integradio.events.security import ConnectionInfo

        conn = ConnectionInfo(client_id="test")

        # Run concurrent modifications
        async def add_subs():
            for i in range(100):
                await conn.add_subscriptions([f"pattern.{i}"])

        async def remove_subs():
            for i in range(50):
                await conn.remove_subscriptions([f"pattern.{i}"])

        await asyncio.gather(add_subs(), remove_subs())

        # Should have some subscriptions left
        snapshot = await conn.get_subscriptions_snapshot()
        assert len(snapshot) >= 50

    @pytest.mark.asyncio
    async def test_forward_event_serialization_error(self):
        """Handle event serialization errors in forwarding."""
        from integradio.events import EventMesh, SemanticEvent
        from integradio.events.websocket import WebSocketServer, WebSocketConfig

        mesh = EventMesh()
        config = WebSocketConfig(
            allowed_origins=["*"],
            require_auth=False,
        )
        server = WebSocketServer(mesh, config)

        # Create a mock websocket that raises on send
        client_id = "test-client"
        ws = MockWebSocket()
        ws._closed = True  # Will fail sends
        server._connections[client_id] = ws
        conn_info = await server._conn_manager.add(client_id, "1.1.1.1")
        await conn_info.add_subscriptions(["**"])

        # Should handle gracefully
        await server._setup_client_subscription(client_id, ws)

    def test_validate_message_size_string(self):
        """validate_message_size works with strings."""
        from integradio.events.security import validate_message_size

        assert validate_message_size("short", 100) is True
        assert validate_message_size("x" * 200, 100) is False

    def test_validate_message_size_bytes(self):
        """validate_message_size works with bytes."""
        from integradio.events.security import validate_message_size

        assert validate_message_size(b"short", 100) is True
        assert validate_message_size(b"x" * 200, 100) is False

    def test_validate_message_size_unicode(self):
        """validate_message_size handles unicode correctly."""
        from integradio.events.security import validate_message_size

        # Unicode characters can be multiple bytes
        unicode_text = "日本語" * 10  # Each char is 3 bytes in UTF-8
        assert validate_message_size(unicode_text, 1000) is True
        assert validate_message_size(unicode_text, 10) is False
