"""
Event handlers registry and decorators.

Provides a clean API for registering and managing event handlers
with support for filtering, priority, and error handling.
"""

import asyncio
import functools
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

from .event import SemanticEvent, EventType

logger = logging.getLogger(__name__)

# Type aliases
AsyncHandler = Callable[[SemanticEvent], Awaitable[None]]
SyncHandler = Callable[[SemanticEvent], None]
Handler = Union[AsyncHandler, SyncHandler]
F = TypeVar("F", bound=Handler)


@dataclass
class HandlerConfig:
    """Configuration for an event handler."""

    patterns: list[str]
    priority: int = 0
    filter_fn: Optional[Callable[[SemanticEvent], bool]] = None
    error_handler: Optional[Callable[[Exception, SemanticEvent], None]] = None
    timeout: Optional[float] = None
    max_retries: int = 0
    tags: list[str] = field(default_factory=list)


class EventHandler:
    """
    Wrapper for event handler functions with metadata.

    Provides:
    - Automatic async wrapping
    - Timeout handling
    - Retry logic
    - Error handling
    - Filtering
    """

    def __init__(
        self,
        fn: Handler,
        config: HandlerConfig,
    ):
        self.fn = fn
        self.config = config
        self._is_async = asyncio.iscoroutinefunction(fn)

        # Stats
        self.calls = 0
        self.errors = 0
        self.total_time = 0.0

    @property
    def patterns(self) -> list[str]:
        """Event patterns this handler responds to."""
        return self.config.patterns

    @property
    def priority(self) -> int:
        """Handler priority (higher = called first)."""
        return self.config.priority

    async def __call__(self, event: SemanticEvent) -> None:
        """Execute the handler."""
        # Apply filter
        if self.config.filter_fn and not self.config.filter_fn(event):
            return

        import time
        start = time.perf_counter()

        try:
            await self._execute_with_retry(event)
        except Exception as e:
            self.errors += 1
            if self.config.error_handler:
                self.config.error_handler(e, event)
            else:
                logger.error(
                    f"Handler {self.fn.__name__} failed for {event.type}: {e}"
                )
        finally:
            self.calls += 1
            self.total_time += time.perf_counter() - start

    async def _execute_with_retry(self, event: SemanticEvent) -> None:
        """Execute with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                await self._execute(event)
                return
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    # Exponential backoff
                    delay = 0.1 * (2 ** attempt)
                    await asyncio.sleep(delay)

        if last_error:
            raise last_error

    async def _execute(self, event: SemanticEvent) -> None:
        """Execute the handler with optional timeout."""
        if self._is_async:
            coro = self.fn(event)
        else:
            # Wrap sync function
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, self.fn, event)

        if self.config.timeout:
            await asyncio.wait_for(coro, timeout=self.config.timeout)
        else:
            await coro

    def matches(self, event: SemanticEvent) -> bool:
        """Check if this handler should process an event."""
        for pattern in self.config.patterns:
            if event.matches_pattern(pattern):
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "name": self.fn.__name__,
            "patterns": self.config.patterns,
            "priority": self.config.priority,
            "calls": self.calls,
            "errors": self.errors,
            "avg_time_ms": (
                (self.total_time / self.calls * 1000) if self.calls else 0
            ),
        }


class HandlerRegistry:
    """
    Registry for event handlers.

    Provides centralized handler management with:
    - Priority-based ordering
    - Handler discovery
    - Lifecycle management
    """

    def __init__(self):
        self._handlers: list[EventHandler] = []
        self._by_pattern: dict[str, list[EventHandler]] = {}

    def register(self, handler: EventHandler) -> None:
        """Register a handler."""
        self._handlers.append(handler)
        self._handlers.sort(key=lambda h: -h.priority)  # Higher first

        for pattern in handler.patterns:
            if pattern not in self._by_pattern:
                self._by_pattern[pattern] = []
            self._by_pattern[pattern].append(handler)
            self._by_pattern[pattern].sort(key=lambda h: -h.priority)

    def unregister(self, handler: EventHandler) -> bool:
        """Unregister a handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
            for pattern in handler.patterns:
                if pattern in self._by_pattern:
                    self._by_pattern[pattern].remove(handler)
            return True
        return False

    def get_handlers(self, event: SemanticEvent) -> list[EventHandler]:
        """Get handlers matching an event, in priority order."""
        matching = []
        seen = set()

        for handler in self._handlers:
            if id(handler) not in seen and handler.matches(event):
                matching.append(handler)
                seen.add(id(handler))

        return matching

    async def dispatch(self, event: SemanticEvent) -> int:
        """
        Dispatch event to all matching handlers.

        Args:
            event: Event to dispatch

        Returns:
            Number of handlers called
        """
        handlers = self.get_handlers(event)

        # Execute handlers concurrently
        await asyncio.gather(
            *(handler(event) for handler in handlers),
            return_exceptions=True,
        )

        return len(handlers)

    def get_all_stats(self) -> list[dict[str, Any]]:
        """Get stats for all handlers."""
        return [h.get_stats() for h in self._handlers]


# Global registry for decorator-based handlers
_global_registry = HandlerRegistry()


def on_event(
    *patterns: str,
    priority: int = 0,
    filter_fn: Optional[Callable[[SemanticEvent], bool]] = None,
    timeout: Optional[float] = None,
    max_retries: int = 0,
    tags: Optional[list[str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to register an event handler.

    Usage:
        @on_event("ui.component.click", "ui.button.*")
        async def handle_click(event: SemanticEvent):
            print(f"Clicked: {event.data}")

        @on_event("data.*", priority=10, timeout=5.0)
        async def handle_data(event: SemanticEvent):
            # This runs before lower priority handlers
            process_data(event.data)

    Args:
        *patterns: Event patterns to match
        priority: Handler priority (higher = called first)
        filter_fn: Optional filter function
        timeout: Handler timeout in seconds
        max_retries: Number of retries on failure
        tags: Optional tags for handler discovery
    """
    def decorator(fn: F) -> F:
        config = HandlerConfig(
            patterns=list(patterns),
            priority=priority,
            filter_fn=filter_fn,
            timeout=timeout,
            max_retries=max_retries,
            tags=tags or [],
        )
        handler = EventHandler(fn, config)
        _global_registry.register(handler)

        # Attach handler to function for access
        fn._event_handler = handler  # type: ignore

        return fn

    return decorator


def get_global_registry() -> HandlerRegistry:
    """Get the global handler registry."""
    return _global_registry


def clear_global_registry() -> None:
    """Clear all handlers from the global registry."""
    global _global_registry
    _global_registry = HandlerRegistry()
