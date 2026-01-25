"""Tests for data models."""

import pytest
from datetime import datetime, timezone, timedelta
import numpy as np

from behavior_modeler.models import (
    FlowEvent,
    Session,
    BehaviorCluster,
    FlowPrediction,
    TestGap,
)


class TestFlowEvent:
    """Tests for FlowEvent model."""

    def test_create_basic_event(self):
        """Test creating a basic flow event."""
        event = FlowEvent(
            event_id="evt_001",
            timestamp=datetime.now(timezone.utc),
            component_type="Button",
            event_type="click",
        )
        assert event.event_id == "evt_001"
        assert event.component_type == "Button"
        assert event.event_type == "click"
        assert event.vector is None

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        now = datetime.now(timezone.utc)
        event = FlowEvent(
            event_id="evt_001",
            timestamp=now,
            component_id=42,
            component_type="SearchBox",
            component_intent="Search for code",
            event_type="input",
            event_data={"value": "test"},
            intent="code_search",
            tags=["search", "input"],
        )

        d = event.to_dict()
        assert d["event_id"] == "evt_001"
        assert d["component_id"] == 42
        assert d["event_data"] == {"value": "test"}
        assert d["tags"] == ["search", "input"]

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        d = {
            "event_id": "evt_002",
            "timestamp": "2026-01-20T10:30:00+00:00",
            "component_type": "Button",
            "event_type": "click",
            "intent": "submit",
        }

        event = FlowEvent.from_dict(d)
        assert event.event_id == "evt_002"
        assert event.event_type == "click"
        assert event.intent == "submit"

    def test_event_roundtrip(self):
        """Test dict roundtrip preserves data."""
        original = FlowEvent(
            event_id="evt_003",
            timestamp=datetime.now(timezone.utc),
            component_id=100,
            component_type="Textbox",
            event_type="input",
            event_data={"value": "hello"},
            tags=["test"],
        )

        restored = FlowEvent.from_dict(original.to_dict())
        assert restored.event_id == original.event_id
        assert restored.component_id == original.component_id
        assert restored.event_data == original.event_data


class TestSession:
    """Tests for Session model."""

    def test_create_empty_session(self):
        """Test creating an empty session."""
        session = Session(
            session_id="sess_001",
            started_at=datetime.now(timezone.utc),
        )
        assert session.session_id == "sess_001"
        assert session.events == []
        assert session.duration_ms == 0
        assert not session.is_complete

    def test_session_duration(self):
        """Test session duration calculation."""
        base = datetime.now(timezone.utc)
        events = [
            FlowEvent(event_id="e1", timestamp=base),
            FlowEvent(event_id="e2", timestamp=base + timedelta(seconds=5)),
            FlowEvent(event_id="e3", timestamp=base + timedelta(seconds=10)),
        ]

        session = Session(
            session_id="sess_002",
            started_at=base,
            events=events,
        )

        assert session.duration_ms == 10000  # 10 seconds

    def test_session_is_complete(self):
        """Test session completion detection."""
        base = datetime.now(timezone.utc)

        # Incomplete session
        incomplete = Session(
            session_id="sess_003",
            started_at=base,
            events=[
                FlowEvent(event_id="e1", timestamp=base, event_type="input"),
                FlowEvent(event_id="e2", timestamp=base, event_type="click"),
            ],
        )
        assert not incomplete.is_complete

        # Complete session (ends with submit)
        complete = Session(
            session_id="sess_004",
            started_at=base,
            events=[
                FlowEvent(event_id="e1", timestamp=base, event_type="input"),
                FlowEvent(event_id="e2", timestamp=base, event_type="submit"),
            ],
        )
        assert complete.is_complete

    def test_session_event_count(self):
        """Test event count property."""
        session = Session(
            session_id="sess_005",
            started_at=datetime.now(timezone.utc),
            events=[
                FlowEvent(event_id=f"e{i}", timestamp=datetime.now(timezone.utc))
                for i in range(5)
            ],
        )
        assert session.event_count == 5

    def test_session_roundtrip(self):
        """Test dict roundtrip preserves session data."""
        base = datetime.now(timezone.utc)
        original = Session(
            session_id="sess_006",
            started_at=base,
            ended_at=base + timedelta(minutes=5),
            user_agent="TestBrowser/1.0",
            events=[
                FlowEvent(event_id="e1", timestamp=base, event_type="input"),
                FlowEvent(event_id="e2", timestamp=base + timedelta(seconds=30), event_type="submit"),
            ],
            cluster_id=3,
        )

        restored = Session.from_dict(original.to_dict())
        assert restored.session_id == original.session_id
        assert restored.user_agent == original.user_agent
        assert restored.cluster_id == original.cluster_id
        assert len(restored.events) == len(original.events)


class TestBehaviorCluster:
    """Tests for BehaviorCluster model."""

    def test_create_cluster(self):
        """Test creating a behavior cluster."""
        cluster = BehaviorCluster(
            cluster_id=1,
            label="Search Flow",
            cluster_type="happy_path",
            session_count=100,
            completion_rate=0.85,
            dominant_components=["SearchBox", "SearchResults"],
        )

        assert cluster.cluster_id == 1
        assert cluster.label == "Search Flow"
        assert cluster.completion_rate == 0.85

    def test_cluster_to_dict(self):
        """Test cluster serialization."""
        cluster = BehaviorCluster(
            cluster_id=2,
            label="Upload Flow",
            dominant_intents=["file_upload", "progress"],
        )

        d = cluster.to_dict()
        assert d["cluster_id"] == 2
        assert d["dominant_intents"] == ["file_upload", "progress"]


class TestFlowPrediction:
    """Tests for FlowPrediction model."""

    def test_create_prediction(self):
        """Test creating a flow prediction."""
        pred = FlowPrediction(
            predicted_component="CodePanel",
            predicted_event="view",
            confidence=0.85,
            cluster_id=1,
        )

        assert pred.predicted_component == "CodePanel"
        assert pred.confidence == 0.85


class TestTestGap:
    """Tests for TestGap model."""

    def test_create_gap(self):
        """Test creating a test gap."""
        gap = TestGap(
            gap_id="gap_001",
            gap_type="uncovered_flow",
            flow_description="Upload error retry flow",
            affected_components=["UploadBox", "ErrorDialog"],
            observed_count=50,
            priority="high",
        )

        assert gap.gap_id == "gap_001"
        assert gap.priority == "high"
        assert len(gap.affected_components) == 2
