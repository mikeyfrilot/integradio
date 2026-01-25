"""
Event Tap - Subscribes to EventMesh and captures UI flows.

Listens to mesh events, assembles them into sessions, and persists flows.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from .config import BehaviorModelerConfig
from .models import Session, FlowEvent
from .store import FlowStore

if TYPE_CHECKING:
    from integradio.events import EventMesh, SemanticEvent
    from integradio.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BehaviorTap:
    """Taps into EventMesh to capture UI flows."""

    # Event types that signal session end
    TERMINAL_EVENTS = {
        "submit",
        "logout",
        "close",
        "navigate_away",
        "complete",
        "connection.close",
    }

    # Event patterns to subscribe to
    SUBSCRIBE_PATTERNS = [
        "ui.**",           # All UI events
        "data.**",         # Data load/update events
        "system.connection.*",  # Connection lifecycle
    ]

    def __init__(
        self,
        store: FlowStore,
        config: Optional[BehaviorModelerConfig] = None,
        registry: Optional["ComponentRegistry"] = None,
    ):
        """
        Initialize the behavior tap.

        Args:
            store: FlowStore for persistence
            config: Configuration
            registry: Optional ComponentRegistry for enrichment
        """
        self.store = store
        self.config = config or BehaviorModelerConfig()
        self.registry = registry

        # Active sessions (session_id -> Session)
        self._sessions: dict[str, Session] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

        # Subscription ID (for cleanup)
        self._subscription_id: Optional[str] = None

        # Background task for session timeout
        self._timeout_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._events_captured = 0
        self._sessions_completed = 0

    async def start(self, mesh: "EventMesh") -> None:
        """
        Start capturing events from the mesh.

        Args:
            mesh: EventMesh instance to subscribe to
        """
        if self._running:
            logger.warning("BehaviorTap already running")
            return

        self._running = True

        # Subscribe to event patterns
        self._subscription_id = mesh.subscribe(
            patterns=self.SUBSCRIBE_PATTERNS,
            handler=self._handle_event,
            client_id="behavior_modeler",
        )

        # Start timeout checker
        self._timeout_task = asyncio.create_task(self._check_timeouts())

        logger.info(f"BehaviorTap started, subscribed to: {self.SUBSCRIBE_PATTERNS}")

    async def stop(self, mesh: Optional["EventMesh"] = None) -> None:
        """
        Stop capturing events.

        Args:
            mesh: EventMesh to unsubscribe from
        """
        self._running = False

        # Cancel timeout task
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe
        if mesh and self._subscription_id:
            mesh.unsubscribe(self._subscription_id)

        # Finalize all active sessions
        for session_id in list(self._sessions.keys()):
            await self._finalize_session(session_id, reason="tap_stopped")

        logger.info(f"BehaviorTap stopped. Captured {self._events_captured} events, {self._sessions_completed} sessions")

    async def _handle_event(self, event: "SemanticEvent") -> None:
        """
        Process an incoming mesh event.

        Args:
            event: SemanticEvent from the mesh
        """
        try:
            # Extract session ID
            session_id = self._extract_session_id(event)
            if not session_id:
                logger.debug(f"Skipping event without session context: {event.type}")
                return

            # Get or create session
            async with self._get_session_lock(session_id):
                if session_id not in self._sessions:
                    self._sessions[session_id] = Session(
                        session_id=session_id,
                        started_at=datetime.now(timezone.utc),
                    )
                    logger.debug(f"New session started: {session_id[:8]}...")

                session = self._sessions[session_id]

                # Check max events limit
                if len(session.events) >= self.config.max_events_per_session:
                    logger.warning(f"Session {session_id[:8]} hit max events limit")
                    await self._finalize_session(session_id, reason="max_events")
                    return

                # Enrich with component metadata
                component_meta = None
                if self.registry:
                    component_id = self._extract_component_id(event)
                    if component_id:
                        component_meta = self.registry.get(component_id)

                # Convert to FlowEvent
                flow_event = FlowEvent.from_semantic_event(event, component_meta)
                session.events.append(flow_event)
                self._events_captured += 1

                # Check for terminal event
                if self._is_terminal_event(event):
                    await self._finalize_session(session_id, reason="terminal_event")

        except Exception as e:
            logger.error(f"Error handling event: {e}", exc_info=True)

    def _extract_session_id(self, event: "SemanticEvent") -> Optional[str]:
        """Extract session ID from event."""
        # Priority: correlation_id > data.session_id > source
        if event.correlation_id:
            return event.correlation_id

        if event.data and isinstance(event.data, dict):
            if "session_id" in event.data:
                return event.data["session_id"]

        # Fall back to source (e.g., WebSocket client ID)
        if event.source and event.source != "mesh":
            return f"source_{event.source}"

        return None

    def _extract_component_id(self, event: "SemanticEvent") -> Optional[int]:
        """Extract component ID from event."""
        if event.data and isinstance(event.data, dict):
            comp_id = event.data.get("component_id")
            if isinstance(comp_id, int):
                return comp_id
        return None

    def _is_terminal_event(self, event: "SemanticEvent") -> bool:
        """Check if event signals session end."""
        event_type = event.type.split(".")[-1]
        return event_type in self.TERMINAL_EVENTS

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for a session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    async def _finalize_session(self, session_id: str, reason: str = "unknown") -> None:
        """
        Finalize and persist a session.

        Args:
            session_id: Session to finalize
            reason: Reason for finalization
        """
        if session_id not in self._sessions:
            return

        session = self._sessions.pop(session_id)
        self._session_locks.pop(session_id, None)

        if not session.events:
            logger.debug(f"Discarding empty session {session_id[:8]}")
            return

        session.ended_at = datetime.now(timezone.utc)

        # Persist
        if self.store.save_session(session):
            self._sessions_completed += 1
            logger.info(
                f"Session finalized: {session_id[:8]}... | "
                f"Events: {len(session.events)} | "
                f"Duration: {session.duration_ms}ms | "
                f"Reason: {reason}"
            )
        else:
            logger.error(f"Failed to save session {session_id[:8]}")

    async def _check_timeouts(self) -> None:
        """Background task to check for timed-out sessions."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                now = datetime.now(timezone.utc)
                timeout_seconds = self.config.session_timeout_seconds

                timed_out = []
                for session_id, session in self._sessions.items():
                    if session.events:
                        last_event_time = session.events[-1].timestamp
                        age = (now - last_event_time).total_seconds()
                        if age > timeout_seconds:
                            timed_out.append(session_id)

                for session_id in timed_out:
                    await self._finalize_session(session_id, reason="timeout")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in timeout checker: {e}")

    # ========== Manual Ingestion ==========

    async def ingest_session(self, session: Session) -> bool:
        """
        Manually ingest a session (for mock data or external sources).

        Args:
            session: Session to ingest

        Returns:
            True if saved successfully
        """
        return self.store.save_session(session)

    async def ingest_events(
        self,
        session_id: str,
        events: list[dict],
    ) -> bool:
        """
        Ingest events from external source (e.g., API).

        Args:
            session_id: Session ID
            events: List of event dictionaries

        Returns:
            True if saved successfully
        """
        flow_events = [FlowEvent.from_dict(e) for e in events]

        if not flow_events:
            return False

        session = Session(
            session_id=session_id,
            started_at=flow_events[0].timestamp,
            ended_at=flow_events[-1].timestamp,
            events=flow_events,
        )

        return self.store.save_session(session)

    # ========== Stats ==========

    def get_stats(self) -> dict:
        """Get tap statistics."""
        return {
            "running": self._running,
            "active_sessions": len(self._sessions),
            "events_captured": self._events_captured,
            "sessions_completed": self._sessions_completed,
        }


async def run_tap_standalone(
    mesh: "EventMesh",
    store: FlowStore,
    config: Optional[BehaviorModelerConfig] = None,
) -> BehaviorTap:
    """
    Convenience function to run tap as standalone service.

    Args:
        mesh: EventMesh to connect to
        store: FlowStore for persistence
        config: Optional configuration

    Returns:
        Running BehaviorTap instance
    """
    tap = BehaviorTap(store=store, config=config)
    await tap.start(mesh)
    return tap
