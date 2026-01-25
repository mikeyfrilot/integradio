"""
Data models for the UI Behavior Modeler.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from integradio.events import SemanticEvent
    from integradio.registry import ComponentMetadata


@dataclass
class FlowEvent:
    """A single event within a user flow."""

    event_id: str
    timestamp: datetime

    # Component context
    component_id: Optional[int] = None
    component_type: str = ""
    component_intent: str = ""

    # Event details
    event_type: str = ""
    event_data: dict[str, Any] = field(default_factory=dict)

    # Semantic context
    intent: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Computed
    vector: Optional[np.ndarray] = None

    @classmethod
    def from_semantic_event(
        cls,
        event: "SemanticEvent",
        component_meta: Optional["ComponentMetadata"] = None,
    ) -> "FlowEvent":
        """Create FlowEvent from Integradio SemanticEvent."""
        # Parse timestamp
        time_str = event.time.replace("Z", "+00:00")
        try:
            timestamp = datetime.fromisoformat(time_str)
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        # Extract component ID from event data
        component_id = None
        if event.data and isinstance(event.data, dict):
            component_id = event.data.get("component_id")

        return cls(
            event_id=event.id,
            timestamp=timestamp,
            component_id=component_id,
            component_type=component_meta.component_type if component_meta else "",
            component_intent=component_meta.intent if component_meta else "",
            event_type=event.type.split(".")[-1],
            event_data=event.data if isinstance(event.data, dict) else {},
            intent=event.intent,
            tags=list(event.tags) if event.tags else [],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "component_id": self.component_id,
            "component_type": self.component_type,
            "component_intent": self.component_intent,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "intent": self.intent,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowEvent":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            event_id=data.get("event_id", ""),
            timestamp=timestamp,
            component_id=data.get("component_id"),
            component_type=data.get("component_type", ""),
            component_intent=data.get("component_intent", ""),
            event_type=data.get("event_type", ""),
            event_data=data.get("event_data", {}),
            intent=data.get("intent"),
            tags=data.get("tags", []),
        )


@dataclass
class Session:
    """A user session containing a sequence of UI events."""

    session_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    user_agent: Optional[str] = None
    events: list[FlowEvent] = field(default_factory=list)

    # Computed after encoding
    vector: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None

    @property
    def duration_ms(self) -> int:
        """Session duration in milliseconds."""
        if not self.events:
            return 0
        first = self.events[0].timestamp
        last = self.events[-1].timestamp
        return int((last - first).total_seconds() * 1000)

    @property
    def is_complete(self) -> bool:
        """Whether session has a natural end."""
        if not self.events:
            return False
        terminal_events = {"submit", "navigate_away", "logout", "close", "complete"}
        return self.events[-1].event_type in terminal_events

    @property
    def event_count(self) -> int:
        """Number of events in session."""
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "user_agent": self.user_agent,
            "events": [e.to_dict() for e in self.events],
            "duration_ms": self.duration_ms,
            "is_complete": self.is_complete,
            "cluster_id": self.cluster_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        started_at = data.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        elif started_at is None:
            started_at = datetime.now(timezone.utc)

        ended_at = data.get("ended_at")
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))

        events = [FlowEvent.from_dict(e) for e in data.get("events", [])]

        return cls(
            session_id=data.get("session_id", ""),
            started_at=started_at,
            ended_at=ended_at,
            user_agent=data.get("user_agent"),
            events=events,
            cluster_id=data.get("cluster_id"),
        )


@dataclass
class BehaviorCluster:
    """A cluster of similar user behavior flows."""

    cluster_id: int
    label: str = ""
    description: str = ""

    # Cluster stats
    centroid: Optional[np.ndarray] = None
    session_count: int = 0
    avg_duration_ms: float = 0.0
    completion_rate: float = 0.0

    # Dominant characteristics
    dominant_components: list[str] = field(default_factory=list)
    dominant_intents: list[str] = field(default_factory=list)
    typical_length: int = 0

    # Classification
    cluster_type: str = "unknown"  # happy_path, edge_case, drop_off, error_flow

    # Sample sessions
    sample_session_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "description": self.description,
            "session_count": self.session_count,
            "avg_duration_ms": self.avg_duration_ms,
            "completion_rate": self.completion_rate,
            "dominant_components": self.dominant_components,
            "dominant_intents": self.dominant_intents,
            "typical_length": self.typical_length,
            "cluster_type": self.cluster_type,
            "sample_session_ids": self.sample_session_ids,
        }


@dataclass
class FlowPrediction:
    """Predicted next action(s) in a user flow."""

    predicted_component: str
    predicted_event: str
    predicted_intent: Optional[str] = None
    confidence: float = 0.0

    # Supporting data
    similar_sessions: list[str] = field(default_factory=list)
    cluster_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_component": self.predicted_component,
            "predicted_event": self.predicted_event,
            "predicted_intent": self.predicted_intent,
            "confidence": self.confidence,
            "similar_sessions": self.similar_sessions,
            "cluster_id": self.cluster_id,
        }


@dataclass
class TestGap:
    """A gap between observed UI behavior and test coverage."""

    gap_id: str
    gap_type: str  # uncovered_flow, uncovered_component, rare_path

    # What's missing
    flow_description: str = ""
    affected_components: list[str] = field(default_factory=list)
    observed_count: int = 0

    # Context for test generation
    sample_session_id: Optional[str] = None
    suggested_test_name: str = ""
    suggested_assertions: list[str] = field(default_factory=list)

    # Priority
    priority: str = "medium"  # low, medium, high, critical
    priority_reason: str = ""

    # Status
    status: str = "open"  # open, acknowledged, resolved
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type,
            "flow_description": self.flow_description,
            "affected_components": self.affected_components,
            "observed_count": self.observed_count,
            "sample_session_id": self.sample_session_id,
            "suggested_test_name": self.suggested_test_name,
            "suggested_assertions": self.suggested_assertions,
            "priority": self.priority,
            "priority_reason": self.priority_reason,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }
