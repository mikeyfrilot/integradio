"""
EventMesh - AsyncIO pub/sub hub for real-time event distribution.

Features:
- Pattern-based subscriptions (ui.*, data.**, etc.)
- Async handlers with concurrency control
- Event signing and verification
- Built-in rate limiting
- Event replay buffer for late joiners
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
import fnmatch

from .event import SemanticEvent, create_event
from .security import (
    EventSigner,
    RateLimiter,
    NonceTracker,
    RateLimitResult,
)

logger = logging.getLogger(__name__)

# Type aliases
EventHandler = Callable[[SemanticEvent], Awaitable[None]]
SyncEventHandler = Callable[[SemanticEvent], None]


@dataclass
class Subscription:
    """A subscription to event patterns."""

    id: str
    patterns: list[str]
    handler: EventHandler
    client_id: Optional[str] = None
    created_at: float = field(default_factory=lambda: __import__("time").time())


class EventMesh:
    """
    Central event mesh for pub/sub communication.

    Usage:
        mesh = EventMesh(secret_key="your-32-byte-secret-key")

        # Subscribe to events
        @mesh.on("ui.component.*")
        async def handle_component(event):
            print(event.data)

        # Emit events
        await mesh.emit("ui.component.click", {"id": 123})

        # Or with full event control
        await mesh.publish(SemanticEvent(
            type="ui.component.click",
            source="button-1",
            data={"clicked": True}
        ))
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        sign_events: bool = True,
        verify_events: bool = True,
        rate_limit: float = 1000.0,  # events/second
        rate_burst: int = 2000,
        replay_buffer_size: int = 100,
        max_event_age: int = 300,  # seconds
    ):
        """
        Initialize event mesh.

        Args:
            secret_key: HMAC signing key (auto-generated if None)
            sign_events: Whether to sign outgoing events
            verify_events: Whether to verify incoming signatures
            rate_limit: Global rate limit (events/second)
            rate_burst: Burst allowance
            replay_buffer_size: Events to buffer for replay
            max_event_age: Max age for event timestamps
        """
        # Security components
        if secret_key is None:
            secret_key = EventSigner.generate_key()
            logger.info("Generated new event signing key")

        self._signer = EventSigner(secret_key)
        self._sign_events = sign_events
        self._verify_events = verify_events
        self._rate_limiter = RateLimiter(rate=rate_limit, burst=rate_burst)
        self._nonce_tracker = NonceTracker(window_seconds=max_event_age)
        self._max_event_age = max_event_age

        # Subscriptions
        self._subscriptions: dict[str, Subscription] = {}
        self._sub_counter = 0
        self._sub_lock = asyncio.Lock()

        # Event replay buffer (circular)
        self._replay_buffer: deque[SemanticEvent] = deque(maxlen=replay_buffer_size)

        # Metrics
        self._events_published = 0
        self._events_delivered = 0
        self._events_dropped = 0

        # Running state
        self._running = False
        self._event_queue: asyncio.Queue[SemanticEvent] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    @property
    def signer(self) -> EventSigner:
        """Access the event signer."""
        return self._signer

    async def start(self) -> None:
        """Start the event processing worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("EventMesh started")

    async def stop(self) -> None:
        """Stop the event processing worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("EventMesh stopped")

    async def _worker(self) -> None:
        """Background worker to process events."""
        while self._running:
            try:
                # Wait for event with timeout for graceful shutdown
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                await self._dispatch(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event worker: {e}")

    async def publish(
        self,
        event: SemanticEvent,
        client_id: str = "local",
    ) -> bool:
        """
        Publish an event to the mesh.

        Args:
            event: Event to publish
            client_id: Source client ID for rate limiting

        Returns:
            True if event was accepted
        """
        # Rate limiting
        result = await self._rate_limiter.check(client_id)
        if not result.allowed:
            logger.warning(f"Rate limit exceeded for {client_id}")
            self._events_dropped += 1
            return False

        # Verify incoming event if required
        if self._verify_events and event.signature:
            if not self._signer.verify(event):
                logger.warning(f"Invalid signature on event {event.id}")
                self._events_dropped += 1
                return False

        # Check for replay attacks
        if not await self._nonce_tracker.check_and_add(event.nonce):
            logger.warning(f"Replay detected for nonce {event.nonce}")
            self._events_dropped += 1
            return False

        # Check timestamp
        if event.is_expired(self._max_event_age):
            logger.warning(f"Expired event {event.id}")
            self._events_dropped += 1
            return False

        # Sign if needed
        if self._sign_events and not event.signature:
            self._signer.sign(event)

        # Add to replay buffer
        self._replay_buffer.append(event)

        # Queue for processing
        await self._event_queue.put(event)
        self._events_published += 1

        return True

    async def emit(
        self,
        event_type: str,
        data: Any = None,
        source: str = "mesh",
        **kwargs: Any,
    ) -> bool:
        """
        Convenience method to create and publish an event.

        Args:
            event_type: Event type string
            data: Event payload
            source: Event source
            **kwargs: Additional event attributes

        Returns:
            True if event was accepted
        """
        event = create_event(event_type, source, data, **kwargs)
        return await self.publish(event)

    async def _dispatch(self, event: SemanticEvent) -> None:
        """Dispatch event to matching subscribers."""
        async with self._sub_lock:
            subscriptions = list(self._subscriptions.values())

        # Find matching subscriptions
        tasks = []
        for sub in subscriptions:
            if self._matches_subscription(event, sub):
                tasks.append(self._safe_call(sub.handler, event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._events_delivered += len(tasks)

    def _matches_subscription(
        self, event: SemanticEvent, sub: Subscription
    ) -> bool:
        """Check if event matches subscription patterns."""
        for pattern in sub.patterns:
            if event.matches_pattern(pattern):
                return True
        return False

    async def _safe_call(
        self, handler: EventHandler, event: SemanticEvent
    ) -> None:
        """Call handler with error protection."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Handler error for {event.type}: {e}")

    def subscribe(
        self,
        patterns: str | list[str],
        handler: EventHandler,
        client_id: Optional[str] = None,
    ) -> str:
        """
        Subscribe to event patterns.

        Args:
            patterns: Pattern(s) to match
            handler: Async handler function
            client_id: Optional client identifier

        Returns:
            Subscription ID
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        self._sub_counter += 1
        sub_id = f"sub_{self._sub_counter}"

        sub = Subscription(
            id=sub_id,
            patterns=patterns,
            handler=handler,
            client_id=client_id,
        )
        self._subscriptions[sub_id] = sub

        logger.debug(f"New subscription {sub_id}: {patterns}")
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """
        Remove a subscription.

        Args:
            sub_id: Subscription ID to remove

        Returns:
            True if subscription was found and removed
        """
        if sub_id in self._subscriptions:
            del self._subscriptions[sub_id]
            logger.debug(f"Removed subscription {sub_id}")
            return True
        return False

    def on(
        self, *patterns: str
    ) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator to subscribe a handler to patterns.

        Usage:
            @mesh.on("ui.component.*", "data.update")
            async def handler(event):
                print(event)
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(list(patterns), handler)
            return handler
        return decorator

    async def replay(
        self,
        patterns: str | list[str],
        handler: EventHandler,
        max_events: int = 50,
    ) -> int:
        """
        Replay recent events matching patterns.

        Useful for late-joining clients to catch up.

        Args:
            patterns: Patterns to match
            handler: Handler to call with each event
            max_events: Maximum events to replay

        Returns:
            Number of events replayed
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        count = 0
        for event in list(self._replay_buffer)[-max_events:]:
            for pattern in patterns:
                if event.matches_pattern(pattern):
                    await self._safe_call(handler, event)
                    count += 1
                    break

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get mesh statistics."""
        return {
            "running": self._running,
            "subscriptions": len(self._subscriptions),
            "events_published": self._events_published,
            "events_delivered": self._events_delivered,
            "events_dropped": self._events_dropped,
            "replay_buffer_size": len(self._replay_buffer),
            "queue_size": self._event_queue.qsize(),
        }

    async def __aenter__(self) -> "EventMesh":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()
