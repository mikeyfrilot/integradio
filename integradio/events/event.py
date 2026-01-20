"""
SemanticEvent - CloudEvents-compliant event format with security extensions.

Based on CloudEvents v1.0 specification with additions for:
- HMAC-SHA256 signatures
- Timestamp validation
- Replay protection via nonce
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import json
import uuid


class EventType(str, Enum):
    """Standard event types for Semantic UI components."""

    # Component lifecycle
    COMPONENT_CREATED = "ui.component.created"
    COMPONENT_UPDATED = "ui.component.updated"
    COMPONENT_DESTROYED = "ui.component.destroyed"

    # User interactions
    CLICK = "ui.interaction.click"
    INPUT = "ui.interaction.input"
    SUBMIT = "ui.interaction.submit"
    SELECT = "ui.interaction.select"
    HOVER = "ui.interaction.hover"

    # Data events
    DATA_LOADED = "data.loaded"
    DATA_UPDATED = "data.updated"
    DATA_ERROR = "data.error"

    # System events
    CONNECTION_OPEN = "system.connection.open"
    CONNECTION_CLOSE = "system.connection.close"
    CONNECTION_ERROR = "system.connection.error"
    HEARTBEAT = "system.heartbeat"

    # Custom (user-defined)
    CUSTOM = "custom"


@dataclass
class SemanticEvent:
    """
    CloudEvents-compliant event with security extensions.

    Attributes:
        type: Event type (dot-notation pattern)
        source: Origin of the event (component ID, service name)
        data: Event payload
        id: Unique event identifier
        time: ISO 8601 timestamp
        subject: Optional subject (e.g., specific component)
        datacontenttype: MIME type of data (default: application/json)
        specversion: CloudEvents spec version
        nonce: Random value for replay protection
        signature: HMAC-SHA256 signature (set by EventSigner)
    """

    type: str
    source: str
    data: Any = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    subject: Optional[str] = None
    datacontenttype: str = "application/json"
    specversion: str = "1.0"

    # Security extensions
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    signature: Optional[str] = None

    # Semantic-specific extensions
    intent: Optional[str] = None  # Semantic intent for routing
    tags: list[str] = field(default_factory=list)
    correlation_id: Optional[str] = None  # For request-response patterns

    def to_dict(self) -> dict[str, Any]:
        """Convert to CloudEvents JSON format."""
        result = {
            "specversion": self.specversion,
            "type": self.type,
            "source": self.source,
            "id": self.id,
            "time": self.time,
            "datacontenttype": self.datacontenttype,
        }

        if self.data is not None:
            result["data"] = self.data
        if self.subject:
            result["subject"] = self.subject

        # Extensions (prefixed per CloudEvents spec)
        result["semanticnonce"] = self.nonce
        if self.signature:
            result["semanticsignature"] = self.signature
        if self.intent:
            result["semanticintent"] = self.intent
        if self.tags:
            result["semantictags"] = self.tags
        if self.correlation_id:
            result["semanticcorrelationid"] = self.correlation_id

        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticEvent":
        """Create event from CloudEvents JSON format."""
        return cls(
            specversion=data.get("specversion", "1.0"),
            type=data["type"],
            source=data["source"],
            id=data.get("id", str(uuid.uuid4())),
            time=data.get("time", datetime.now(timezone.utc).isoformat()),
            datacontenttype=data.get("datacontenttype", "application/json"),
            data=data.get("data"),
            subject=data.get("subject"),
            nonce=data.get("semanticnonce", uuid.uuid4().hex[:16]),
            signature=data.get("semanticsignature"),
            intent=data.get("semanticintent"),
            tags=data.get("semantictags", []),
            correlation_id=data.get("semanticcorrelationid"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SemanticEvent":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def matches_pattern(self, pattern: str) -> bool:
        """
        Check if event type matches a subscription pattern.

        Patterns support:
        - Exact match: "ui.component.click"
        - Wildcard suffix: "ui.component.*"
        - Multi-level wildcard: "ui.**"

        Args:
            pattern: Subscription pattern to match against

        Returns:
            True if event type matches pattern
        """
        if pattern == self.type:
            return True

        if pattern.endswith(".**"):
            prefix = pattern[:-3]
            return self.type.startswith(prefix)

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            # Match one level only
            if not self.type.startswith(prefix + "."):
                return False
            remainder = self.type[len(prefix) + 1 :]
            return "." not in remainder

        return False

    def is_expired(self, max_age_seconds: int = 300) -> bool:
        """
        Check if event timestamp is too old (replay protection).

        Args:
            max_age_seconds: Maximum age in seconds (default 5 minutes)

        Returns:
            True if event is expired
        """
        try:
            event_time = datetime.fromisoformat(self.time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = (now - event_time).total_seconds()
            return age > max_age_seconds or age < -60  # Also reject future times
        except (ValueError, TypeError):
            return True  # Invalid timestamp = expired

    def __repr__(self) -> str:
        return (
            f"SemanticEvent(type={self.type!r}, source={self.source!r}, "
            f"id={self.id[:8]}..., signed={self.signature is not None})"
        )


def create_event(
    event_type: str | EventType,
    source: str,
    data: Any = None,
    **kwargs: Any,
) -> SemanticEvent:
    """
    Factory function to create events.

    Args:
        event_type: Event type string or EventType enum
        source: Event source identifier
        data: Event payload
        **kwargs: Additional event attributes

    Returns:
        New SemanticEvent instance
    """
    if isinstance(event_type, EventType):
        event_type = event_type.value

    return SemanticEvent(type=event_type, source=source, data=data, **kwargs)
