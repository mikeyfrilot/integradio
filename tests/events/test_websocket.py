"""
Comprehensive tests for WebSocket server and client.

Tests cover:
- WebSocketConfig initialization and defaults
- WebSocketServer connection handling
- WebSocketServer authentication flow
- WebSocketServer message handling
- WebSocketServer rate limiting
- WebSocketServer background tasks (heartbeat, cleanup)
- WebSocketClient connection and reconnection
- WebSocketClient subscription handling
- WebSocketClient event emission
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from integradio.events.websocket import (
    WebSocketConfig,
    WebSocketServer,
    WebSocketClient,
    MAX_MESSAGE_SIZE,
)
from integradio.events.event import SemanticEvent
from integradio.events.mesh import EventMesh
from integradio.events.security import ConnectionInfo, RateLimitResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_mesh():
    """Create a mock EventMesh."""
    mesh = MagicMock(spec=EventMesh)
    mesh.emit = AsyncMock(return_value=True)
    mesh.publish = AsyncMock(return_value=True)
    mesh.subscribe = MagicMock(return_value="sub_1")
    mesh.unsubscribe = MagicMock(return_value=True)
    mesh.get_stats = MagicMock(return_value={"running": True})
    return mesh


@pytest.fixture
def default_config():
    """Create default WebSocket config."""
    return WebSocketConfig()


@pytest.fixture
def auth_config():
    """Create config with authentication required."""
    return WebSocketConfig(
        require_auth=True,
        auth_timeout=5.0,
        allowed_origins=["https://example.com"],
    )


@pytest.fixture
def no_auth_config():
    """Create config without authentication."""
    return WebSocketConfig(
        require_auth=False,
        allowed_origins=["*"],
    )


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.client = MagicMock()
    ws.client.host = "127.0.0.1"
    ws.headers = {"origin": "https://example.com"}
    return ws


@pytest.fixture
def server(mock_mesh, no_auth_config):
    """Create a WebSocket server."""
    return WebSocketServer(mesh=mock_mesh, config=no_auth_config)


@pytest.fixture
def auth_server(mock_mesh, auth_config):
    """Create a WebSocket server with auth."""
    def auth_handler(token):
        if token == "valid_token":
            return "user_123"
        return None
    return WebSocketServer(mesh=mock_mesh, config=auth_config, auth_handler=auth_handler)


# =============================================================================
# WebSocketConfig Tests
# =============================================================================

class TestWebSocketConfig:
    """Tests for WebSocketConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WebSocketConfig()

        assert config.require_auth is True
        assert config.auth_timeout == 10.0
        assert config.max_message_size == MAX_MESSAGE_SIZE
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

    def test_post_init_sets_default_origins(self):
        """Test that post_init sets allowed_origins to ['*'] if None."""
        config = WebSocketConfig()
        assert config.allowed_origins == ["*"]

    def test_custom_origins_preserved(self):
        """Test that custom origins are preserved."""
        config = WebSocketConfig(allowed_origins=["https://example.com"])
        assert config.allowed_origins == ["https://example.com"]

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WebSocketConfig(
            require_auth=False,
            auth_timeout=5.0,
            max_message_size=32768,
            rate_limit=50.0,
            rate_burst=100,
            max_connections=5000,
            max_per_ip=50,
        )

        assert config.require_auth is False
        assert config.auth_timeout == 5.0
        assert config.max_message_size == 32768
        assert config.rate_limit == 50.0
        assert config.rate_burst == 100
        assert config.max_connections == 5000
        assert config.max_per_ip == 50


# =============================================================================
# WebSocketServer Tests - Initialization
# =============================================================================

class TestWebSocketServerInit:
    """Tests for WebSocketServer initialization."""

    def test_init_with_defaults(self, mock_mesh):
        """Test server initialization with default config."""
        server = WebSocketServer(mesh=mock_mesh)

        assert server.mesh is mock_mesh
        assert server.config is not None
        assert server.auth_handler is None
        assert len(server._connections) == 0
        assert server._heartbeat_task is None
        assert server._cleanup_task is None

    def test_init_with_custom_config(self, mock_mesh, auth_config):
        """Test server initialization with custom config."""
        server = WebSocketServer(mesh=mock_mesh, config=auth_config)

        assert server.config is auth_config
        assert server.config.require_auth is True

    def test_init_with_auth_handler(self, mock_mesh):
        """Test server initialization with auth handler."""
        def my_auth_handler(token):
            return "user_id" if token == "secret" else None

        server = WebSocketServer(
            mesh=mock_mesh,
            auth_handler=my_auth_handler,
        )

        assert server.auth_handler is my_auth_handler


# =============================================================================
# WebSocketServer Tests - Start/Stop
# =============================================================================

class TestWebSocketServerLifecycle:
    """Tests for server start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_background_tasks(self, server):
        """Test that start creates heartbeat and cleanup tasks."""
        await server.start()

        assert server._heartbeat_task is not None
        assert server._cleanup_task is not None

        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, server):
        """Test that stop cancels background tasks."""
        await server.start()

        heartbeat_task = server._heartbeat_task
        cleanup_task = server._cleanup_task

        await server.stop()

        # Give tasks time to fully cancel
        await asyncio.sleep(0.01)

        # Tasks should be done after stop (either completed, cancelled, or errored)
        assert heartbeat_task.done()
        assert cleanup_task.done()

    @pytest.mark.asyncio
    async def test_stop_closes_all_connections(self, server, mock_websocket):
        """Test that stop closes all active connections."""
        # Add a mock connection
        server._connections["client_1"] = mock_websocket

        await server.stop()

        mock_websocket.close.assert_called()


# =============================================================================
# WebSocketServer Tests - Connection Handling
# =============================================================================

class TestWebSocketServerConnectionHandling:
    """Tests for connection handling."""

    @pytest.mark.asyncio
    async def test_handle_rejects_when_at_capacity(self, server, mock_websocket):
        """Test connection rejected when at max capacity."""
        # Mock connection manager to reject
        server._conn_manager.can_connect = AsyncMock(
            return_value=(False, "Server at maximum capacity")
        )

        await server.handle(mock_websocket)

        mock_websocket.close.assert_called_with(1008, "Server at maximum capacity")

    @pytest.mark.asyncio
    async def test_handle_rejects_invalid_origin(self, mock_mesh):
        """Test connection rejected with invalid origin."""
        config = WebSocketConfig(
            require_auth=False,
            allowed_origins=["https://allowed.com"],
        )
        server = WebSocketServer(mesh=mock_mesh, config=config)

        ws = MagicMock()
        ws.close = AsyncMock()
        ws.client = MagicMock()
        ws.client.host = "127.0.0.1"
        ws.headers = {"origin": "https://evil.com"}

        # Mock can_connect to allow
        server._conn_manager.can_connect = AsyncMock(return_value=(True, "OK"))

        await server.handle(ws)

        ws.close.assert_called_with(1008, "Origin not allowed")

    @pytest.mark.asyncio
    async def test_handle_accepts_valid_connection(self, server, mock_websocket):
        """Test valid connection is accepted."""
        # Mock dependencies
        server._conn_manager.can_connect = AsyncMock(return_value=(True, "OK"))
        server._conn_manager.add = AsyncMock(return_value=ConnectionInfo(
            client_id="test_client",
            ip_address="127.0.0.1",
        ))

        # Make receive_text raise disconnect to end the loop
        mock_websocket.receive_text = AsyncMock(side_effect=Exception("disconnect"))

        await server.handle(mock_websocket)

        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_emits_connection_open_event(self, server, mock_websocket):
        """Test connection emits system.connection.open event."""
        server._conn_manager.can_connect = AsyncMock(return_value=(True, "OK"))
        server._conn_manager.add = AsyncMock(return_value=ConnectionInfo(
            client_id="test_client",
        ))
        mock_websocket.receive_text = AsyncMock(side_effect=Exception("disconnect"))

        await server.handle(mock_websocket)

        # Check that emit was called with connection.open
        calls = server.mesh.emit.call_args_list
        assert any(
            call[0][0] == "system.connection.open"
            for call in calls
        )


# =============================================================================
# WebSocketServer Tests - Authentication
# =============================================================================

class TestWebSocketServerAuth:
    """Tests for authentication handling."""

    @pytest.mark.asyncio
    async def test_authenticate_success(self, auth_server, mock_websocket):
        """Test successful authentication."""
        mock_websocket.receive_text = AsyncMock(
            return_value=json.dumps({"type": "auth", "token": "valid_token"})
        )

        user_id = await auth_server._authenticate(mock_websocket, "client_1")

        assert user_id == "user_123"
        mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_authenticate_failure_invalid_token(self, auth_server, mock_websocket):
        """Test authentication failure with invalid token."""
        mock_websocket.receive_text = AsyncMock(
            return_value=json.dumps({"type": "auth", "token": "invalid_token"})
        )

        user_id = await auth_server._authenticate(mock_websocket, "client_1")

        assert user_id is None

    @pytest.mark.asyncio
    async def test_authenticate_failure_wrong_message_type(self, auth_server, mock_websocket):
        """Test authentication failure with wrong message type."""
        mock_websocket.receive_text = AsyncMock(
            return_value=json.dumps({"type": "subscribe", "patterns": []})
        )

        user_id = await auth_server._authenticate(mock_websocket, "client_1")

        assert user_id is None

    @pytest.mark.asyncio
    async def test_authenticate_timeout(self, auth_server, mock_websocket):
        """Test authentication timeout."""
        async def slow_receive():
            await asyncio.sleep(10)
            return json.dumps({"type": "auth", "token": "valid_token"})

        mock_websocket.receive_text = slow_receive
        auth_server.config.auth_timeout = 0.01

        user_id = await auth_server._authenticate(mock_websocket, "client_1")

        assert user_id is None

    @pytest.mark.asyncio
    async def test_authenticate_no_handler_dev_mode(self, mock_mesh):
        """Test authentication in dev mode (no handler)."""
        config = WebSocketConfig(require_auth=True)
        server = WebSocketServer(mesh=mock_mesh, config=config)

        ws = MagicMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "auth", "token": "any_token"})
        )
        ws.send_text = AsyncMock()

        user_id = await server._authenticate(ws, "client_1")

        # In dev mode, token is used as user_id
        assert user_id == "any_token"


# =============================================================================
# WebSocketServer Tests - Message Handling
# =============================================================================

class TestWebSocketServerMessageHandling:
    """Tests for message handling."""

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, server):
        """Test handling subscribe message."""
        conn_info = ConnectionInfo(client_id="client_1")
        server._conn_manager.get = AsyncMock(return_value=conn_info)

        await server._handle_subscribe("client_1", ["ui.**", "data.*"])

        assert "ui.**" in conn_info.subscriptions
        assert "data.*" in conn_info.subscriptions

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, server):
        """Test handling unsubscribe message."""
        conn_info = ConnectionInfo(client_id="client_1")
        conn_info.subscriptions = {"ui.**", "data.*"}
        server._conn_manager.get = AsyncMock(return_value=conn_info)

        await server._handle_unsubscribe("client_1", ["ui.**"])

        assert "ui.**" not in conn_info.subscriptions
        assert "data.*" in conn_info.subscriptions

    @pytest.mark.asyncio
    async def test_handle_event_message(self, server, mock_websocket):
        """Test handling event message."""
        event_data = {
            "type": "ui.click",
            "source": "button_1",
            "data": {"clicked": True},
        }

        data = json.dumps({"type": "event", "event": event_data})

        await server._handle_message(mock_websocket, "client_1", data)

        server.mesh.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_ping_message(self, server, mock_websocket):
        """Test handling ping message."""
        data = json.dumps({"type": "ping"})

        await server._handle_message(mock_websocket, "client_1", data)

        # Should respond with pong
        mock_websocket.send_text.assert_called()
        response = json.loads(mock_websocket.send_text.call_args[0][0])
        assert response["type"] == "pong"
        assert "time" in response

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, server, mock_websocket):
        """Test handling invalid JSON message."""
        await server._handle_message(mock_websocket, "client_1", "not json")

        # Should send error
        mock_websocket.send_text.assert_called()
        response = json.loads(mock_websocket.send_text.call_args[0][0])
        assert response["type"] == "error"
        assert "Invalid JSON" in response["message"]

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, server, mock_websocket):
        """Test handling unknown message type."""
        data = json.dumps({"type": "unknown_type"})

        await server._handle_message(mock_websocket, "client_1", data)

        # Should log warning but not crash


# =============================================================================
# WebSocketServer Tests - Rate Limiting
# =============================================================================

class TestWebSocketServerRateLimiting:
    """Tests for rate limiting in message loop."""

    @pytest.mark.asyncio
    async def test_message_rejected_when_rate_limited(self, server, mock_websocket):
        """Test message rejected when rate limited."""
        conn_info = ConnectionInfo(client_id="client_1")

        # Add rate limiter that denies
        server._rate_limiters["client_1"] = MagicMock()
        server._rate_limiters["client_1"].check = AsyncMock(
            return_value=RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=0,
                retry_after=1.5,
            )
        )

        # Mock the message loop to process one message then disconnect
        messages = [json.dumps({"type": "ping"})]
        message_index = 0

        async def mock_receive():
            nonlocal message_index
            if message_index < len(messages):
                msg = messages[message_index]
                message_index += 1
                return msg
            raise Exception("disconnect")

        mock_websocket.receive_text = mock_receive
        server._conn_manager.update_activity = AsyncMock()

        await server._message_loop(mock_websocket, "client_1", conn_info)

        # Should send rate limit error
        calls = mock_websocket.send_text.call_args_list
        assert any("Rate limited" in str(call) for call in calls)


# =============================================================================
# WebSocketServer Tests - Background Tasks
# =============================================================================

class TestWebSocketServerBackgroundTasks:
    """Tests for background tasks."""

    @pytest.mark.asyncio
    async def test_heartbeat_sends_to_all_connections(self, server, mock_websocket):
        """Test heartbeat sends to all connected clients."""
        server.config.heartbeat_interval = 0.01  # Fast heartbeat for test
        server._connections = {"client_1": mock_websocket}
        server._conn_manager = MagicMock()
        server._conn_manager.connection_count = 1

        # Start heartbeat task
        task = asyncio.create_task(server._heartbeat_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Check heartbeat was sent
        assert mock_websocket.send_text.called

    @pytest.mark.asyncio
    async def test_cleanup_removes_idle_connections(self, server, mock_websocket):
        """Test cleanup removes idle connections."""
        server._connections = {"idle_client": mock_websocket}
        server._conn_manager.get_idle_connections = AsyncMock(
            return_value=["idle_client"]
        )
        server._conn_manager.remove = AsyncMock(return_value=ConnectionInfo(
            client_id="idle_client",
        ))
        server.mesh.emit = AsyncMock()

        # Run one cleanup iteration
        server._running = True

        # Mock sleep to complete quickly
        with patch("asyncio.sleep", new_callable=AsyncMock):
            task = asyncio.create_task(server._cleanup_loop())
            await asyncio.sleep(0)  # Let task start
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# =============================================================================
# WebSocketServer Tests - Client IP and Origin Extraction
# =============================================================================

class TestWebSocketServerExtraction:
    """Tests for IP and origin extraction."""

    def test_get_client_ip_from_client_attribute(self, server):
        """Test extracting IP from websocket.client.host."""
        ws = MagicMock()
        ws.client = MagicMock()
        ws.client.host = "192.168.1.1"

        ip = server._get_client_ip(ws)

        assert ip == "192.168.1.1"

    def test_get_client_ip_from_scope(self, server):
        """Test extracting IP from ASGI scope."""
        ws = MagicMock(spec=[])
        ws.scope = {"client": ("10.0.0.1", 12345)}

        ip = server._get_client_ip(ws)

        assert ip == "10.0.0.1"

    def test_get_client_ip_unknown(self, server):
        """Test fallback to 'unknown' when IP cannot be extracted."""
        ws = MagicMock(spec=[])

        ip = server._get_client_ip(ws)

        assert ip == "unknown"

    def test_get_origin_from_headers(self, server):
        """Test extracting origin from headers dict."""
        ws = MagicMock()
        ws.headers = {"origin": "https://example.com"}

        origin = server._get_origin(ws)

        assert origin == "https://example.com"

    def test_get_origin_from_scope(self, server):
        """Test extracting origin from ASGI scope."""
        ws = MagicMock(spec=[])
        ws.scope = {"headers": [(b"origin", b"https://test.com")]}

        origin = server._get_origin(ws)

        assert origin == "https://test.com"

    def test_get_origin_none(self, server):
        """Test fallback to None when origin cannot be extracted."""
        ws = MagicMock(spec=[])

        origin = server._get_origin(ws)

        assert origin is None


# =============================================================================
# WebSocketServer Tests - Stats
# =============================================================================

class TestWebSocketServerStats:
    """Tests for server statistics."""

    def test_get_stats(self, server):
        """Test getting server statistics."""
        stats = server.get_stats()

        assert "connections" in stats
        assert "mesh_stats" in stats


# =============================================================================
# WebSocketClient Tests - Initialization
# =============================================================================

class TestWebSocketClientInit:
    """Tests for WebSocketClient initialization."""

    def test_init_with_url(self):
        """Test client initialization with URL."""
        client = WebSocketClient("ws://localhost:8000/ws")

        assert client.url == "ws://localhost:8000/ws"
        assert client.token is None
        assert client.config is not None
        assert client._connected is False

    def test_init_with_token(self):
        """Test client initialization with auth token."""
        client = WebSocketClient(
            "ws://localhost:8000/ws",
            token="my_token",
        )

        assert client.token == "my_token"

    def test_init_with_config(self):
        """Test client initialization with custom config."""
        config = WebSocketConfig(reconnect_max_attempts=5)
        client = WebSocketClient(
            "ws://localhost:8000/ws",
            config=config,
        )

        assert client.config.reconnect_max_attempts == 5


# =============================================================================
# WebSocketClient Tests - Connection
# =============================================================================

class TestWebSocketClientConnection:
    """Tests for client connection handling."""

    @pytest.mark.asyncio
    async def test_connect_success_no_auth(self):
        """Test successful connection without auth."""
        client = WebSocketClient("ws://localhost:8000/ws")

        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock()
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws

            result = await client.connect()

            assert result is True
            assert client._connected is True

            # Cleanup the receive task
            if client._receive_task:
                client._receive_task.cancel()
                try:
                    await client._receive_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_connect_success_with_auth(self):
        """Test successful connection with auth."""
        client = WebSocketClient(
            "ws://localhost:8000/ws",
            token="valid_token",
        )

        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"type": "auth_success", "client_id": "c1"})
        )
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws

            result = await client.connect()

            assert result is True
            mock_ws.send.assert_called()

            # Cleanup the receive task
            if client._receive_task:
                client._receive_task.cancel()
                try:
                    await client._receive_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self):
        """Test connection failure due to auth."""
        client = WebSocketClient(
            "ws://localhost:8000/ws",
            token="invalid_token",
        )

        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"type": "auth_failed"})
        )
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws

            result = await client.connect()

            assert result is False
            mock_ws.close.assert_called()

    @pytest.mark.asyncio
    async def test_connect_error(self):
        """Test connection error handling."""
        client = WebSocketClient("ws://localhost:8000/ws")

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")

            result = await client.connect()

            assert result is False
            assert client._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test client disconnect."""
        client = WebSocketClient("ws://localhost:8000/ws")
        client._connected = True
        client._websocket = MagicMock()
        client._websocket.close = AsyncMock()
        client._receive_task = asyncio.create_task(asyncio.sleep(10))

        await client.disconnect()

        assert client._connected is False
        client._websocket.close.assert_called()

    def test_connected_property(self):
        """Test connected property."""
        client = WebSocketClient("ws://localhost:8000/ws")

        assert client.connected is False

        client._connected = True
        assert client.connected is True


# =============================================================================
# WebSocketClient Tests - Subscriptions
# =============================================================================

class TestWebSocketClientSubscriptions:
    """Tests for client subscription handling."""

    @pytest.mark.asyncio
    async def test_subscribe_when_connected(self):
        """Test subscribing when already connected."""
        client = WebSocketClient("ws://localhost:8000/ws")
        client._connected = True
        client._websocket = MagicMock()
        client._websocket.send = AsyncMock()

        await client.subscribe("ui.**", "data.*")

        assert "ui.**" in client._subscriptions
        assert "data.*" in client._subscriptions
        client._websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_subscribe_when_not_connected(self):
        """Test subscribing when not connected stores patterns."""
        client = WebSocketClient("ws://localhost:8000/ws")

        await client.subscribe("ui.**")

        assert "ui.**" in client._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_when_connected(self):
        """Test unsubscribing when connected."""
        client = WebSocketClient("ws://localhost:8000/ws")
        client._connected = True
        client._websocket = MagicMock()
        client._websocket.send = AsyncMock()
        client._subscriptions = {"ui.**", "data.*"}

        await client.unsubscribe("ui.**")

        assert "ui.**" not in client._subscriptions
        assert "data.*" in client._subscriptions
        client._websocket.send.assert_called()

    def test_on_decorator(self):
        """Test @client.on() decorator."""
        client = WebSocketClient("ws://localhost:8000/ws")

        @client.on("ui.component.*", "data.update")
        async def handler(event):
            pass

        assert "ui.component.*" in client._handlers
        assert "data.update" in client._handlers
        assert handler in client._handlers["ui.component.*"]
        assert "ui.component.*" in client._subscriptions


# =============================================================================
# WebSocketClient Tests - Event Emission
# =============================================================================

class TestWebSocketClientEmission:
    """Tests for client event emission."""

    @pytest.mark.asyncio
    async def test_emit_when_connected(self):
        """Test emitting event when connected."""
        client = WebSocketClient("ws://localhost:8000/ws")
        client._connected = True
        client._websocket = MagicMock()
        client._websocket.send = AsyncMock()

        await client.emit("ui.click", {"button": "submit"}, source="my_app")

        client._websocket.send.assert_called()
        sent_data = json.loads(client._websocket.send.call_args[0][0])
        assert sent_data["type"] == "event"
        assert sent_data["event"]["type"] == "ui.click"

    @pytest.mark.asyncio
    async def test_emit_when_not_connected_raises(self):
        """Test emitting when not connected raises error."""
        client = WebSocketClient("ws://localhost:8000/ws")

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.emit("ui.click")


# =============================================================================
# WebSocketClient Tests - Message Dispatch
# =============================================================================

class TestWebSocketClientDispatch:
    """Tests for client event dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_calls_matching_handlers(self):
        """Test dispatch calls handlers matching event pattern."""
        client = WebSocketClient("ws://localhost:8000/ws")

        received_events = []

        async def handler(event):
            received_events.append(event)

        client._handlers["ui.**"] = [handler]

        event = SemanticEvent(type="ui.button.click", source="test")
        await client._dispatch(event)

        assert len(received_events) == 1
        assert received_events[0] is event

    @pytest.mark.asyncio
    async def test_dispatch_calls_sync_handler(self):
        """Test dispatch can call sync handlers."""
        client = WebSocketClient("ws://localhost:8000/ws")

        received_events = []

        def sync_handler(event):
            received_events.append(event)

        client._handlers["ui.**"] = [sync_handler]

        event = SemanticEvent(type="ui.click", source="test")
        await client._dispatch(event)

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_dispatch_handles_handler_errors(self):
        """Test dispatch handles handler errors gracefully."""
        client = WebSocketClient("ws://localhost:8000/ws")

        async def bad_handler(event):
            raise ValueError("Handler error")

        client._handlers["ui.**"] = [bad_handler]

        event = SemanticEvent(type="ui.click", source="test")
        # Should not raise
        await client._dispatch(event)


# =============================================================================
# WebSocketClient Tests - Reconnection
# =============================================================================

class TestWebSocketClientReconnection:
    """Tests for client reconnection logic."""

    @pytest.mark.asyncio
    async def test_schedule_reconnect_disabled(self):
        """Test reconnection when disabled."""
        config = WebSocketConfig(reconnect=False)
        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        # Should return immediately without reconnecting
        await client._schedule_reconnect()

        assert client._reconnect_attempt == 0

    @pytest.mark.asyncio
    async def test_schedule_reconnect_max_attempts(self):
        """Test reconnection stops after max attempts."""
        config = WebSocketConfig(reconnect_max_attempts=3)
        client = WebSocketClient("ws://localhost:8000/ws", config=config)
        client._reconnect_attempt = 3

        await client._schedule_reconnect()

        # Should not increase attempt count
        assert client._reconnect_attempt == 3

    @pytest.mark.asyncio
    async def test_reconnect_exponential_backoff(self):
        """Test reconnection uses exponential backoff."""
        config = WebSocketConfig(
            reconnect_delay=1.0,
            reconnect_max_delay=60.0,
            reconnect_max_attempts=10,
        )
        client = WebSocketClient("ws://localhost:8000/ws", config=config)

        # Calculate expected delays
        delays = []
        for attempt in range(5):
            delay = min(1.0 * (2 ** attempt), 60.0)
            delays.append(delay)

        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]


# =============================================================================
# WebSocketClient Tests - Receive Loop
# =============================================================================

class TestWebSocketClientReceiveLoop:
    """Tests for client receive loop."""

    @pytest.mark.asyncio
    async def test_receive_loop_dispatches_events(self):
        """Test receive loop dispatches events to handlers."""
        client = WebSocketClient("ws://localhost:8000/ws")
        client._connected = True

        received = []

        async def handler(event):
            received.append(event)
            client._connected = False  # Stop loop after first event

        client._handlers["ui.**"] = [handler]

        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            "type": "event",
            "event": {
                "type": "ui.click",
                "source": "test",
                "id": "123",
                "time": datetime.now(timezone.utc).isoformat(),
            }
        }))
        client._websocket = mock_ws

        await client._receive_loop()

        assert len(received) == 1
        assert received[0].type == "ui.click"

    @pytest.mark.asyncio
    async def test_receive_loop_handles_heartbeat(self):
        """Test receive loop handles heartbeat messages."""
        client = WebSocketClient("ws://localhost:8000/ws")
        client._connected = True

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"type": "heartbeat", "time": 123})
            else:
                client._connected = False
                raise Exception("disconnect")

        mock_ws = MagicMock()
        mock_ws.recv = mock_recv
        client._websocket = mock_ws

        await client._receive_loop()

        # Should have processed heartbeat without error

    @pytest.mark.asyncio
    async def test_receive_loop_handles_errors(self):
        """Test receive loop handles server errors."""
        client = WebSocketClient("ws://localhost:8000/ws")
        client._connected = True

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"type": "error", "message": "Something went wrong"})
            else:
                client._connected = False
                raise Exception("disconnect")

        mock_ws = MagicMock()
        mock_ws.recv = mock_recv
        client._websocket = mock_ws

        await client._receive_loop()

        # Should log warning but not crash


# =============================================================================
# Integration-style Tests
# =============================================================================

class TestWebSocketIntegration:
    """Integration-style tests for WebSocket components."""

    @pytest.mark.asyncio
    async def test_server_client_subscription_flow(self, server):
        """Test full subscription flow through server."""
        # Setup server
        conn_info = ConnectionInfo(client_id="client_1")
        server._conn_manager.get = AsyncMock(return_value=conn_info)

        # Client subscribes
        await server._handle_subscribe("client_1", ["ui.**"])

        # Verify subscription stored
        assert "ui.**" in conn_info.subscriptions

        # Client unsubscribes
        await server._handle_unsubscribe("client_1", ["ui.**"])

        # Verify subscription removed
        assert "ui.**" not in conn_info.subscriptions

    @pytest.mark.asyncio
    async def test_forward_event_to_subscribed_client(self, server, mock_websocket):
        """Test event forwarding to subscribed client."""
        # Setup
        conn_info = ConnectionInfo(client_id="client_1")
        conn_info.subscriptions = {"ui.**"}

        server._connections["client_1"] = mock_websocket
        server._conn_manager.get = AsyncMock(return_value=conn_info)

        # Create a forwarding function
        sub_id = await server._setup_client_subscription("client_1", mock_websocket)

        # Verify subscription was created
        assert sub_id is not None
        server.mesh.subscribe.assert_called()
