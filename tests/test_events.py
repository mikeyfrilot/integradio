"""
Tests for integradio.events - CloudEvents-compliant event mesh with security features.

Tests cover:
- SemanticEvent and EventType (event.py)
- EventSigner, RateLimiter, NonceTracker, ConnectionManager, validate_origin (security.py)
- EventMesh pub/sub (mesh.py)
- EventHandler, HandlerRegistry, on_event decorator (handlers.py)
- WebSocketConfig (websocket.py)
"""

import asyncio
import json
import time
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch


# =============================================================================
# SemanticEvent Tests (event.py)
# =============================================================================


class TestSemanticEvent:
    """Tests for SemanticEvent dataclass."""

    def test_create_event_basic(self):
        """Create event with required fields."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="ui.click", source="button-1")

        assert event.type == "ui.click"
        assert event.source == "button-1"
        assert event.specversion == "1.0"
        assert event.datacontenttype == "application/json"
        assert event.id is not None
        assert event.time is not None
        assert event.nonce is not None

    def test_create_event_with_data(self):
        """Create event with data payload."""
        from integradio.events import SemanticEvent

        data = {"clicked": True, "x": 100, "y": 200}
        event = SemanticEvent(type="ui.click", source="button-1", data=data)

        assert event.data == data

    def test_create_event_with_extensions(self):
        """Create event with Semantic extensions."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(
            type="ui.click",
            source="button-1",
            intent="user triggers action",
            tags=["primary", "submit"],
            correlation_id="corr-123",
        )

        assert event.intent == "user triggers action"
        assert event.tags == ["primary", "submit"]
        assert event.correlation_id == "corr-123"

    def test_to_dict_cloudevents_format(self):
        """to_dict() returns CloudEvents-compliant dict."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(
            type="ui.click",
            source="button-1",
            data={"clicked": True},
            subject="submit-btn",
        )
        result = event.to_dict()

        assert result["specversion"] == "1.0"
        assert result["type"] == "ui.click"
        assert result["source"] == "button-1"
        assert result["id"] == event.id
        assert result["time"] == event.time
        assert result["datacontenttype"] == "application/json"
        assert result["data"] == {"clicked": True}
        assert result["subject"] == "submit-btn"
        assert result["semanticnonce"] == event.nonce

    def test_to_dict_omits_none_values(self):
        """to_dict() omits None optional fields."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="test", source="src")
        result = event.to_dict()

        assert "subject" not in result
        assert "semanticsignature" not in result
        assert "semanticintent" not in result
        assert "semanticcorrelationid" not in result

    def test_to_json_canonical(self):
        """to_json() returns canonical JSON."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="test", source="src", data={"b": 2, "a": 1})
        json_str = event.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["type"] == "test"

        # Keys should be sorted (canonical)
        assert json_str.index('"data"') < json_str.index('"id"')

    def test_from_dict(self):
        """from_dict() creates event from CloudEvents dict."""
        from integradio.events import SemanticEvent

        data = {
            "specversion": "1.0",
            "type": "ui.click",
            "source": "button-1",
            "id": "event-123",
            "time": "2026-01-20T10:00:00Z",
            "data": {"clicked": True},
            "semanticnonce": "abc123",
            "semanticintent": "user action",
            "semantictags": ["tag1", "tag2"],
        }
        event = SemanticEvent.from_dict(data)

        assert event.type == "ui.click"
        assert event.source == "button-1"
        assert event.id == "event-123"
        assert event.data == {"clicked": True}
        assert event.nonce == "abc123"
        assert event.intent == "user action"
        assert event.tags == ["tag1", "tag2"]

    def test_from_json(self):
        """from_json() creates event from JSON string."""
        from integradio.events import SemanticEvent

        json_str = '{"type":"test","source":"src","id":"123"}'
        event = SemanticEvent.from_json(json_str)

        assert event.type == "test"
        assert event.source == "src"
        assert event.id == "123"

    def test_roundtrip_serialization(self):
        """Event survives to_dict -> from_dict roundtrip."""
        from integradio.events import SemanticEvent

        original = SemanticEvent(
            type="ui.click",
            source="button-1",
            data={"count": 42},
            intent="action",
            tags=["a", "b"],
        )

        roundtripped = SemanticEvent.from_dict(original.to_dict())

        assert roundtripped.type == original.type
        assert roundtripped.source == original.source
        assert roundtripped.data == original.data
        assert roundtripped.id == original.id
        assert roundtripped.nonce == original.nonce
        assert roundtripped.intent == original.intent
        assert roundtripped.tags == original.tags


class TestEventPatternMatching:
    """Tests for SemanticEvent.matches_pattern()."""

    def test_exact_match(self):
        """Exact pattern matches exactly."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="ui.component.click", source="src")

        assert event.matches_pattern("ui.component.click") is True
        assert event.matches_pattern("ui.component.hover") is False
        assert event.matches_pattern("ui.component") is False

    def test_single_wildcard(self):
        """Single wildcard (*) matches one level."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="ui.component.click", source="src")

        assert event.matches_pattern("ui.component.*") is True
        assert event.matches_pattern("ui.*.click") is False  # Not supported
        assert event.matches_pattern("*.component.click") is False  # Not supported

    def test_single_wildcard_no_sublevel(self):
        """Single wildcard (*) doesn't match sublevels."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="ui.component.button.click", source="src")

        assert event.matches_pattern("ui.component.*") is False  # Too deep
        assert event.matches_pattern("ui.component.button.*") is True

    def test_multi_level_wildcard(self):
        """Multi-level wildcard (**) matches any depth."""
        from integradio.events import SemanticEvent

        event1 = SemanticEvent(type="ui.component.click", source="src")
        event2 = SemanticEvent(type="ui.component.button.primary.click", source="src")
        event3 = SemanticEvent(type="ui.other", source="src")

        assert event1.matches_pattern("ui.**") is True
        assert event2.matches_pattern("ui.**") is True
        assert event3.matches_pattern("ui.**") is True
        assert event1.matches_pattern("data.**") is False


class TestEventExpiration:
    """Tests for SemanticEvent.is_expired()."""

    def test_fresh_event_not_expired(self):
        """Recently created event is not expired."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="test", source="src")

        assert event.is_expired() is False
        assert event.is_expired(max_age_seconds=300) is False

    def test_old_event_is_expired(self):
        """Event with old timestamp is expired."""
        from integradio.events import SemanticEvent

        old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        event = SemanticEvent(type="test", source="src", time=old_time)

        assert event.is_expired(max_age_seconds=300) is True  # 5 min limit
        assert event.is_expired(max_age_seconds=700) is False  # 11+ min limit

    def test_future_event_is_expired(self):
        """Event with future timestamp is rejected."""
        from integradio.events import SemanticEvent

        future_time = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        event = SemanticEvent(type="test", source="src", time=future_time)

        assert event.is_expired() is True

    def test_invalid_timestamp_is_expired(self):
        """Event with invalid timestamp is treated as expired."""
        from integradio.events import SemanticEvent

        event = SemanticEvent(type="test", source="src", time="invalid-time")

        assert event.is_expired() is True


class TestEventType:
    """Tests for EventType enum."""

    def test_event_type_values(self):
        """EventType enum has expected values."""
        from integradio.events import EventType

        assert EventType.CLICK.value == "ui.interaction.click"
        assert EventType.SUBMIT.value == "ui.interaction.submit"
        assert EventType.DATA_LOADED.value == "data.loaded"
        assert EventType.CONNECTION_OPEN.value == "system.connection.open"

    def test_create_event_uses_enum_value(self):
        """create_event() accepts EventType enum."""
        from integradio.events.event import create_event, EventType

        event = create_event(EventType.CLICK, source="btn")

        assert event.type == "ui.interaction.click"


# =============================================================================
# EventSigner Tests (security.py)
# =============================================================================


class TestEventSigner:
    """Tests for EventSigner HMAC signing."""

    def test_sign_event(self):
        """sign() adds signature to event."""
        from integradio.events import EventSigner, SemanticEvent

        signer = EventSigner("0" * 32)  # 32-byte key
        event = SemanticEvent(type="test", source="src", data={"key": "value"})

        assert event.signature is None
        signer.sign(event)
        assert event.signature is not None
        assert len(event.signature) == 64  # SHA256 hex

    def test_verify_valid_signature(self):
        """verify() returns True for valid signature."""
        from integradio.events import EventSigner, SemanticEvent

        signer = EventSigner("0" * 32)
        event = SemanticEvent(type="test", source="src", data={"key": "value"})

        signer.sign(event)
        assert signer.verify(event) is True

    def test_verify_invalid_signature(self):
        """verify() returns False for tampered event."""
        from integradio.events import EventSigner, SemanticEvent

        signer = EventSigner("0" * 32)
        event = SemanticEvent(type="test", source="src", data={"key": "value"})

        signer.sign(event)

        # Tamper with data
        event.data = {"key": "tampered"}

        assert signer.verify(event) is False

    def test_verify_no_signature(self):
        """verify() returns False for unsigned event."""
        from integradio.events import EventSigner, SemanticEvent

        signer = EventSigner("0" * 32)
        event = SemanticEvent(type="test", source="src")

        assert signer.verify(event) is False

    def test_different_keys_different_signatures(self):
        """Different keys produce different signatures."""
        from integradio.events import EventSigner, SemanticEvent

        signer1 = EventSigner("a" * 32)
        signer2 = EventSigner("b" * 32)

        event1 = SemanticEvent(type="test", source="src", data={"x": 1})
        event2 = SemanticEvent(type="test", source="src", data={"x": 1})
        event2.id = event1.id
        event2.time = event1.time
        event2.nonce = event1.nonce

        signer1.sign(event1)
        signer2.sign(event2)

        assert event1.signature != event2.signature

    def test_generate_key(self):
        """generate_key() produces secure random key."""
        from integradio.events import EventSigner

        key1 = EventSigner.generate_key()
        key2 = EventSigner.generate_key()

        assert len(key1) == 64  # 32 bytes = 64 hex chars
        assert key1 != key2

    def test_short_key_warning(self):
        """Short key triggers warning."""
        from integradio.events import EventSigner
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EventSigner("short")
            assert len(w) == 1
            assert "32 bytes" in str(w[0].message)


# =============================================================================
# RateLimiter Tests (security.py)
# =============================================================================


class TestRateLimiter:
    """Tests for token bucket RateLimiter."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Requests within limit are allowed."""
        from integradio.events import RateLimiter

        limiter = RateLimiter(rate=10.0, burst=10)
        result = await limiter.check("client-1")

        assert result.allowed is True
        assert result.remaining >= 0

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Requests over burst limit are blocked."""
        from integradio.events import RateLimiter

        limiter = RateLimiter(rate=1.0, burst=3)

        # Use up all tokens
        for _ in range(3):
            result = await limiter.check("client-1")
            assert result.allowed is True

        # Next should be blocked
        result = await limiter.check("client-1")
        assert result.allowed is False
        assert result.retry_after is not None
        assert result.retry_after > 0

    @pytest.mark.asyncio
    async def test_tokens_refill(self):
        """Tokens refill over time."""
        from integradio.events import RateLimiter

        limiter = RateLimiter(rate=100.0, burst=2)  # Fast refill

        # Use all tokens
        await limiter.check("client-1")
        await limiter.check("client-1")

        result = await limiter.check("client-1")
        if not result.allowed:
            # Wait for refill
            await asyncio.sleep(0.05)
            result = await limiter.check("client-1")
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_per_client_tracking(self):
        """Different clients have separate buckets."""
        from integradio.events import RateLimiter

        limiter = RateLimiter(rate=1.0, burst=2)

        # Client 1 uses tokens
        await limiter.check("client-1")
        await limiter.check("client-1")

        # Client 2 still has full bucket
        result = await limiter.check("client-2")
        assert result.allowed is True

    def test_reset_client(self):
        """reset() clears client bucket."""
        from integradio.events import RateLimiter

        limiter = RateLimiter()
        limiter._buckets["client-1"] = (0, time.monotonic())

        limiter.reset("client-1")

        assert "client-1" not in limiter._buckets


# =============================================================================
# NonceTracker Tests (security.py)
# =============================================================================


class TestNonceTracker:
    """Tests for replay attack prevention."""

    @pytest.mark.asyncio
    async def test_fresh_nonce_accepted(self):
        """Fresh nonce is accepted."""
        from integradio.events.security import NonceTracker

        tracker = NonceTracker(window_seconds=60)
        result = await tracker.check_and_add("nonce-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_duplicate_nonce_rejected(self):
        """Duplicate nonce is rejected."""
        from integradio.events.security import NonceTracker

        tracker = NonceTracker(window_seconds=60)

        await tracker.check_and_add("nonce-123")
        result = await tracker.check_and_add("nonce-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_old_nonces_cleaned(self):
        """Old nonces are cleaned up."""
        from integradio.events.security import NonceTracker

        tracker = NonceTracker(window_seconds=1)

        await tracker.check_and_add("old-nonce")

        # Manually age the nonce
        tracker._nonces["old-nonce"] = time.time() - 10

        # Adding new nonce triggers cleanup
        await tracker.check_and_add("new-nonce")

        assert "old-nonce" not in tracker._nonces


# =============================================================================
# ConnectionManager Tests (security.py)
# =============================================================================


class TestConnectionManager:
    """Tests for WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_add_connection(self):
        """add() registers new connection."""
        from integradio.events import ConnectionManager

        manager = ConnectionManager()
        info = await manager.add("client-1", "192.168.1.1", "http://example.com", "user-1")

        assert info.client_id == "client-1"
        assert info.ip_address == "192.168.1.1"
        assert info.origin == "http://example.com"
        assert info.user_id == "user-1"
        assert manager.connection_count == 1

    @pytest.mark.asyncio
    async def test_remove_connection(self):
        """remove() unregisters connection."""
        from integradio.events import ConnectionManager

        manager = ConnectionManager()
        await manager.add("client-1", "192.168.1.1")

        info = await manager.remove("client-1")

        assert info is not None
        assert info.client_id == "client-1"
        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_can_connect_global_limit(self):
        """can_connect() rejects when at max connections."""
        from integradio.events import ConnectionManager

        manager = ConnectionManager(max_connections=2)

        await manager.add("client-1", "1.1.1.1")
        await manager.add("client-2", "2.2.2.2")

        allowed, reason = await manager.can_connect("3.3.3.3")

        assert allowed is False
        assert "capacity" in reason.lower()

    @pytest.mark.asyncio
    async def test_can_connect_per_ip_limit(self):
        """can_connect() rejects when IP at limit."""
        from integradio.events import ConnectionManager

        manager = ConnectionManager(max_per_ip=2)

        await manager.add("client-1", "192.168.1.1")
        await manager.add("client-2", "192.168.1.1")

        allowed, reason = await manager.can_connect("192.168.1.1")

        assert allowed is False
        assert "IP" in reason

    @pytest.mark.asyncio
    async def test_update_activity(self):
        """update_activity() updates timestamp and count."""
        from integradio.events import ConnectionManager

        manager = ConnectionManager()
        await manager.add("client-1", "1.1.1.1")

        initial_count = (await manager.get("client-1")).message_count
        await manager.update_activity("client-1")

        info = await manager.get("client-1")
        assert info.message_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_get_idle_connections(self):
        """get_idle_connections() returns idle clients."""
        from integradio.events import ConnectionManager

        manager = ConnectionManager(idle_timeout=1.0)
        await manager.add("client-1", "1.1.1.1")

        # Manually set old activity time
        manager._connections["client-1"].last_activity = time.time() - 10

        idle = await manager.get_idle_connections()

        assert "client-1" in idle


# =============================================================================
# validate_origin Tests (security.py)
# =============================================================================


class TestValidateOrigin:
    """Tests for origin validation."""

    def test_exact_match(self):
        """Exact origin match."""
        from integradio.events import validate_origin

        assert validate_origin("https://example.com", ["https://example.com"]) is True
        assert validate_origin("https://other.com", ["https://example.com"]) is False

    def test_wildcard_subdomain(self):
        """Wildcard subdomain matching."""
        from integradio.events import validate_origin

        allowed = ["*.example.com"]

        assert validate_origin("https://app.example.com", allowed) is True
        assert validate_origin("https://api.example.com", allowed) is True
        assert validate_origin("https://example.com", allowed) is True
        assert validate_origin("https://other.com", allowed) is False

    def test_allow_all_wildcard(self):
        """Star wildcard allows all."""
        from integradio.events import validate_origin

        assert validate_origin("https://any.com", ["*"]) is True

    def test_none_origin(self):
        """None origin handling."""
        from integradio.events import validate_origin

        assert validate_origin(None, ["https://example.com"]) is False
        assert validate_origin(None, ["https://example.com"], allow_none=True) is True


# =============================================================================
# EventMesh Tests (mesh.py)
# =============================================================================


class TestEventMesh:
    """Tests for EventMesh pub/sub hub."""

    @pytest.mark.asyncio
    async def test_emit_and_subscribe(self):
        """Events are delivered to subscribers."""
        from integradio.events import EventMesh

        mesh = EventMesh()
        received = []

        async def handler(event):
            received.append(event)

        mesh.subscribe("test.*", handler)

        async with mesh:
            await mesh.emit("test.event", data={"x": 1})
            await asyncio.sleep(0.1)  # Let worker process

        assert len(received) == 1
        assert received[0].type == "test.event"
        assert received[0].data == {"x": 1}

    @pytest.mark.asyncio
    async def test_pattern_filtering(self):
        """Only matching patterns trigger handlers."""
        from integradio.events import EventMesh

        mesh = EventMesh()
        received = []

        async def handler(event):
            received.append(event)

        mesh.subscribe("ui.*", handler)

        async with mesh:
            await mesh.emit("ui.click", data={})
            await mesh.emit("data.update", data={})  # Shouldn't match
            await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].type == "ui.click"

    @pytest.mark.asyncio
    async def test_decorator_subscription(self):
        """@mesh.on decorator subscribes handler."""
        from integradio.events import EventMesh

        mesh = EventMesh()
        received = []

        @mesh.on("my.event")
        async def handler(event):
            received.append(event)

        async with mesh:
            await mesh.emit("my.event", data={})
            await asyncio.sleep(0.1)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """unsubscribe() removes subscription."""
        from integradio.events import EventMesh

        mesh = EventMesh()
        received = []

        async def handler(event):
            received.append(event)

        sub_id = mesh.subscribe("test.*", handler)
        mesh.unsubscribe(sub_id)

        async with mesh:
            await mesh.emit("test.event", data={})
            await asyncio.sleep(0.1)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Multiple subscribers all receive events."""
        from integradio.events import EventMesh

        mesh = EventMesh()
        received1 = []
        received2 = []

        async def handler1(event):
            received1.append(event)

        async def handler2(event):
            received2.append(event)

        mesh.subscribe("test.*", handler1)
        mesh.subscribe("test.*", handler2)

        async with mesh:
            await mesh.emit("test.event", data={})
            await asyncio.sleep(0.1)

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_event_signing(self):
        """Events are signed when configured."""
        from integradio.events import EventMesh, SemanticEvent

        mesh = EventMesh(secret_key="0" * 32, sign_events=True)
        received = []

        async def handler(event):
            received.append(event)

        mesh.subscribe("test.*", handler)

        async with mesh:
            await mesh.emit("test.event", data={})
            await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].signature is not None

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """get_stats() returns mesh statistics."""
        from integradio.events import EventMesh

        mesh = EventMesh()

        async def handler(event):
            pass

        mesh.subscribe("test.*", handler)

        async with mesh:
            await mesh.emit("test.event", data={})
            await asyncio.sleep(0.1)

            stats = mesh.get_stats()

        assert stats["subscriptions"] == 1
        assert stats["events_published"] >= 1

    @pytest.mark.asyncio
    async def test_replay_buffer(self):
        """replay() sends buffered events."""
        from integradio.events import EventMesh

        mesh = EventMesh(replay_buffer_size=10)
        replayed = []

        async def handler(event):
            replayed.append(event)

        async with mesh:
            # Emit some events
            await mesh.emit("test.a", data={})
            await mesh.emit("test.b", data={})
            await asyncio.sleep(0.1)

            # Replay to new handler
            count = await mesh.replay("test.*", handler)

        assert count == 2
        assert len(replayed) == 2


# =============================================================================
# EventHandler Tests (handlers.py)
# =============================================================================


class TestEventHandler:
    """Tests for EventHandler wrapper."""

    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Async handlers work correctly."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import EventHandler, HandlerConfig

        calls = []

        async def my_handler(event):
            calls.append(event)

        config = HandlerConfig(patterns=["test.*"])
        handler = EventHandler(my_handler, config)

        event = SemanticEvent(type="test.event", source="src")
        await handler(event)

        assert len(calls) == 1
        assert handler.calls == 1

    @pytest.mark.asyncio
    async def test_sync_handler(self):
        """Sync handlers are wrapped in executor."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import EventHandler, HandlerConfig

        calls = []

        def my_handler(event):
            calls.append(event)

        config = HandlerConfig(patterns=["test.*"])
        handler = EventHandler(my_handler, config)

        event = SemanticEvent(type="test.event", source="src")
        await handler(event)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_filter_fn(self):
        """Filter function prevents handler execution."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import EventHandler, HandlerConfig

        calls = []

        async def my_handler(event):
            calls.append(event)

        def filter_fn(event):
            return event.data.get("include", False)

        config = HandlerConfig(patterns=["test.*"], filter_fn=filter_fn)
        handler = EventHandler(my_handler, config)

        event1 = SemanticEvent(type="test.event", source="src", data={"include": False})
        event2 = SemanticEvent(type="test.event", source="src", data={"include": True})

        await handler(event1)
        await handler(event2)

        assert len(calls) == 1  # Only event2

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Handler respects timeout."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import EventHandler, HandlerConfig

        async def slow_handler(event):
            await asyncio.sleep(10)

        config = HandlerConfig(patterns=["test.*"], timeout=0.1)
        handler = EventHandler(slow_handler, config)

        event = SemanticEvent(type="test.event", source="src")
        await handler(event)

        assert handler.errors == 1

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Handler retries on failure."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import EventHandler, HandlerConfig

        attempts = []

        async def failing_handler(event):
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Temporary failure")

        config = HandlerConfig(patterns=["test.*"], max_retries=3)
        handler = EventHandler(failing_handler, config)

        event = SemanticEvent(type="test.event", source="src")
        await handler(event)

        assert len(attempts) == 3  # Initial + 2 retries

    def test_matches_pattern(self):
        """matches() correctly checks patterns."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import EventHandler, HandlerConfig

        async def my_handler(event):
            pass

        config = HandlerConfig(patterns=["ui.*", "data.update"])
        handler = EventHandler(my_handler, config)

        event1 = SemanticEvent(type="ui.click", source="src")
        event2 = SemanticEvent(type="data.update", source="src")
        event3 = SemanticEvent(type="system.event", source="src")

        assert handler.matches(event1) is True
        assert handler.matches(event2) is True
        assert handler.matches(event3) is False

    def test_get_stats(self):
        """get_stats() returns handler metrics."""
        from integradio.events.handlers import EventHandler, HandlerConfig

        async def my_handler(event):
            pass

        config = HandlerConfig(patterns=["test.*"], priority=5)
        handler = EventHandler(my_handler, config)
        handler.calls = 10
        handler.errors = 2
        handler.total_time = 0.5

        stats = handler.get_stats()

        assert stats["name"] == "my_handler"
        assert stats["patterns"] == ["test.*"]
        assert stats["priority"] == 5
        assert stats["calls"] == 10
        assert stats["errors"] == 2
        assert stats["avg_time_ms"] == 50.0


# =============================================================================
# HandlerRegistry Tests (handlers.py)
# =============================================================================


class TestHandlerRegistry:
    """Tests for HandlerRegistry."""

    def test_register_handler(self):
        """register() adds handler to registry."""
        from integradio.events.handlers import HandlerRegistry, EventHandler, HandlerConfig

        registry = HandlerRegistry()

        async def my_handler(event):
            pass

        config = HandlerConfig(patterns=["test.*"])
        handler = EventHandler(my_handler, config)

        registry.register(handler)

        assert len(registry._handlers) == 1

    def test_unregister_handler(self):
        """unregister() removes handler."""
        from integradio.events.handlers import HandlerRegistry, EventHandler, HandlerConfig

        registry = HandlerRegistry()

        async def my_handler(event):
            pass

        config = HandlerConfig(patterns=["test.*"])
        handler = EventHandler(my_handler, config)

        registry.register(handler)
        result = registry.unregister(handler)

        assert result is True
        assert len(registry._handlers) == 0

    def test_priority_ordering(self):
        """Handlers are ordered by priority (higher first)."""
        from integradio.events.handlers import HandlerRegistry, EventHandler, HandlerConfig

        registry = HandlerRegistry()

        async def handler1(event):
            pass

        async def handler2(event):
            pass

        config1 = HandlerConfig(patterns=["test.*"], priority=1)
        config2 = HandlerConfig(patterns=["test.*"], priority=10)

        registry.register(EventHandler(handler1, config1))
        registry.register(EventHandler(handler2, config2))

        assert registry._handlers[0].priority == 10
        assert registry._handlers[1].priority == 1

    def test_get_handlers_for_event(self):
        """get_handlers() returns matching handlers."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import HandlerRegistry, EventHandler, HandlerConfig

        registry = HandlerRegistry()

        async def handler1(event):
            pass

        async def handler2(event):
            pass

        config1 = HandlerConfig(patterns=["ui.*"])
        config2 = HandlerConfig(patterns=["data.*"])

        registry.register(EventHandler(handler1, config1))
        registry.register(EventHandler(handler2, config2))

        event = SemanticEvent(type="ui.click", source="src")
        handlers = registry.get_handlers(event)

        assert len(handlers) == 1
        assert handlers[0].config.patterns == ["ui.*"]

    @pytest.mark.asyncio
    async def test_dispatch_event(self):
        """dispatch() calls all matching handlers."""
        from integradio.events import SemanticEvent
        from integradio.events.handlers import HandlerRegistry, EventHandler, HandlerConfig

        registry = HandlerRegistry()
        calls = []

        async def handler1(event):
            calls.append("h1")

        async def handler2(event):
            calls.append("h2")

        config1 = HandlerConfig(patterns=["test.*"])
        config2 = HandlerConfig(patterns=["test.*"])

        registry.register(EventHandler(handler1, config1))
        registry.register(EventHandler(handler2, config2))

        event = SemanticEvent(type="test.event", source="src")
        count = await registry.dispatch(event)

        assert count == 2
        assert len(calls) == 2


# =============================================================================
# on_event Decorator Tests (handlers.py)
# =============================================================================


class TestOnEventDecorator:
    """Tests for @on_event decorator."""

    def test_on_event_registers_handler(self):
        """@on_event registers to global registry."""
        from integradio.events.handlers import (
            on_event,
            get_global_registry,
            clear_global_registry,
        )

        clear_global_registry()

        @on_event("test.event")
        async def my_handler(event):
            pass

        registry = get_global_registry()
        assert len(registry._handlers) == 1

        clear_global_registry()

    def test_on_event_multiple_patterns(self):
        """@on_event accepts multiple patterns."""
        from integradio.events.handlers import (
            on_event,
            get_global_registry,
            clear_global_registry,
        )

        clear_global_registry()

        @on_event("ui.*", "data.*", "system.*")
        async def my_handler(event):
            pass

        registry = get_global_registry()
        handler = registry._handlers[0]

        assert handler.config.patterns == ["ui.*", "data.*", "system.*"]

        clear_global_registry()

    def test_on_event_with_options(self):
        """@on_event accepts priority and timeout."""
        from integradio.events.handlers import (
            on_event,
            get_global_registry,
            clear_global_registry,
        )

        clear_global_registry()

        @on_event("test.*", priority=5, timeout=10.0, max_retries=2)
        async def my_handler(event):
            pass

        registry = get_global_registry()
        handler = registry._handlers[0]

        assert handler.config.priority == 5
        assert handler.config.timeout == 10.0
        assert handler.config.max_retries == 2

        clear_global_registry()

    def test_on_event_attaches_handler(self):
        """@on_event attaches handler to function."""
        from integradio.events.handlers import on_event, clear_global_registry

        clear_global_registry()

        @on_event("test.*")
        async def my_handler(event):
            pass

        assert hasattr(my_handler, "_event_handler")
        assert my_handler._event_handler is not None

        clear_global_registry()


# =============================================================================
# WebSocketConfig Tests (websocket.py)
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

    def test_custom_values(self):
        """Custom config values override defaults."""
        from integradio.events.websocket import WebSocketConfig

        config = WebSocketConfig(
            allowed_origins=["https://example.com"],
            require_auth=False,
            rate_limit=50.0,
            max_connections=1000,
        )

        assert config.allowed_origins == ["https://example.com"]
        assert config.require_auth is False
        assert config.rate_limit == 50.0
        assert config.max_connections == 1000


# =============================================================================
# Integration Tests
# =============================================================================


class TestEventsIntegration:
    """Integration tests for events module."""

    @pytest.mark.asyncio
    async def test_full_event_flow(self):
        """Complete event flow: create, sign, verify, publish, receive."""
        from integradio.events import EventMesh, SemanticEvent, EventSigner

        secret = EventSigner.generate_key()
        mesh = EventMesh(secret_key=secret, sign_events=True, verify_events=True)

        received = []

        @mesh.on("integration.test")
        async def handler(event):
            received.append(event)

        async with mesh:
            # Create and publish event
            result = await mesh.emit(
                "integration.test",
                data={"message": "hello"},
                source="test",
            )

            assert result is True
            await asyncio.sleep(0.1)

        # Event should be received and signed
        assert len(received) == 1
        assert received[0].data == {"message": "hello"}
        assert received[0].signature is not None

        # Verify signature
        assert mesh.signer.verify(received[0]) is True

    @pytest.mark.asyncio
    async def test_replay_attack_prevention(self):
        """Duplicate nonces are rejected."""
        from integradio.events import EventMesh, SemanticEvent

        mesh = EventMesh(sign_events=False, verify_events=False)

        async with mesh:
            # First event accepted
            event1 = SemanticEvent(type="test", source="src", nonce="same-nonce")
            result1 = await mesh.publish(event1)
            assert result1 is True

            # Same nonce rejected
            event2 = SemanticEvent(type="test", source="src", nonce="same-nonce")
            result2 = await mesh.publish(event2)
            assert result2 is False

    @pytest.mark.asyncio
    async def test_expired_event_rejected(self):
        """Events with old timestamps are rejected."""
        from integradio.events import EventMesh, SemanticEvent

        mesh = EventMesh(max_event_age=60)

        async with mesh:
            # Old event rejected
            old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
            event = SemanticEvent(type="test", source="src", time=old_time)
            result = await mesh.publish(event)

            assert result is False
