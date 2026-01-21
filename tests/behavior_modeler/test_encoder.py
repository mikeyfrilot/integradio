"""Tests for FlowEncoder."""

import pytest
from datetime import datetime, timezone, timedelta
import numpy as np

from behavior_modeler.encoder import FlowEncoder, FallbackEncoder
from behavior_modeler.models import Session, FlowEvent
from behavior_modeler.config import BehaviorModelerConfig


class TestFallbackEncoder:
    """Tests for FallbackEncoder (always available)."""

    @pytest.fixture
    def encoder(self):
        return FallbackEncoder()

    def test_encoder_available(self, encoder):
        """Test fallback encoder is always available."""
        assert encoder.available is True

    def test_encode_event(self, encoder):
        """Test encoding a single event."""
        event = FlowEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            component_type="SearchBox",
            event_type="input",
            intent="code_search",
        )

        vector = encoder.encode_event(event)
        assert vector.shape == (768,)
        assert vector.dtype == np.float32

        # Should be normalized
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01

    def test_encode_event_deterministic(self, encoder):
        """Test encoding is deterministic."""
        event = FlowEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            component_type="Button",
            event_type="click",
        )

        vec1 = encoder.encode_event(event)
        vec2 = encoder.encode_event(event)

        np.testing.assert_array_equal(vec1, vec2)

    def test_encode_different_events(self, encoder):
        """Test different events produce different vectors."""
        event1 = FlowEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            component_type="SearchBox",
            event_type="input",
        )
        event2 = FlowEvent(
            event_id="e2",
            timestamp=datetime.now(timezone.utc),
            component_type="Button",
            event_type="click",
        )

        vec1 = encoder.encode_event(event1)
        vec2 = encoder.encode_event(event2)

        # Should be different
        assert not np.allclose(vec1, vec2)

    def test_encode_events_batch(self, encoder):
        """Test batch encoding."""
        events = [
            FlowEvent(event_id=f"e{i}", timestamp=datetime.now(timezone.utc), event_type="click")
            for i in range(5)
        ]

        vectors = encoder.encode_events_batch(events)
        assert len(vectors) == 5
        assert all(v.shape == (768,) for v in vectors)

    def test_encode_session_empty(self, encoder):
        """Test encoding empty session."""
        session = Session(
            session_id="s1",
            started_at=datetime.now(timezone.utc),
            events=[],
        )

        vector = encoder.encode_session(session)
        assert vector.shape == (768,)
        # Empty session should return zero vector
        assert np.allclose(vector, 0)

    def test_encode_session_with_events(self, encoder):
        """Test encoding session with events."""
        base = datetime.now(timezone.utc)
        session = Session(
            session_id="s1",
            started_at=base,
            events=[
                FlowEvent(event_id="e1", timestamp=base, component_type="SearchBox", event_type="input"),
                FlowEvent(event_id="e2", timestamp=base + timedelta(seconds=5), component_type="Results", event_type="select"),
                FlowEvent(event_id="e3", timestamp=base + timedelta(seconds=10), component_type="CodePanel", event_type="view"),
            ],
        )

        vector = encoder.encode_session(session)
        assert vector.shape == (768,)

        # Should be normalized
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01

    def test_encode_session_pooling_methods(self, encoder):
        """Test different pooling methods produce different results."""
        base = datetime.now(timezone.utc)
        session = Session(
            session_id="s1",
            started_at=base,
            events=[
                FlowEvent(event_id="e1", timestamp=base, event_type="input"),
                FlowEvent(event_id="e2", timestamp=base + timedelta(seconds=5), event_type="click"),
            ],
        )

        vec_weighted = encoder.encode_session(session, method="pooled_weighted")
        vec_avg = encoder.encode_session(session, method="pooled_avg")

        # Methods should produce slightly different results
        # (unless events are identical, which they're not)
        assert not np.allclose(vec_weighted, vec_avg)

    def test_encode_query(self, encoder):
        """Test query encoding."""
        query = "user searches and views code"
        vector = encoder.encode_query(query)

        assert vector.shape == (768,)
        assert np.linalg.norm(vector) > 0

    def test_similarity(self, encoder):
        """Test similarity computation."""
        event1 = FlowEvent(event_id="e1", timestamp=datetime.now(timezone.utc), component_type="Search", event_type="input")
        event2 = FlowEvent(event_id="e2", timestamp=datetime.now(timezone.utc), component_type="Search", event_type="input")
        event3 = FlowEvent(event_id="e3", timestamp=datetime.now(timezone.utc), component_type="Upload", event_type="click")

        vec1 = encoder.encode_event(event1)
        vec2 = encoder.encode_event(event2)
        vec3 = encoder.encode_event(event3)

        # Same event should have similarity 1.0
        sim_same = encoder.similarity(vec1, vec2)
        assert sim_same == 1.0

        # Different events should have lower similarity
        sim_diff = encoder.similarity(vec1, vec3)
        assert sim_diff < 1.0

    def test_encode_partial_flow(self, encoder):
        """Test encoding partial flow for prediction."""
        events = [
            FlowEvent(event_id="e1", timestamp=datetime.now(timezone.utc), event_type="input"),
            FlowEvent(event_id="e2", timestamp=datetime.now(timezone.utc), event_type="click"),
        ]

        vector = encoder.encode_partial_flow(events)
        assert vector.shape == (768,)


class TestFlowEncoderWithoutOllama:
    """Tests for FlowEncoder when Ollama is unavailable."""

    @pytest.fixture
    def encoder(self):
        """Create encoder (Ollama likely unavailable in test)."""
        config = BehaviorModelerConfig(ollama_url="http://localhost:99999")  # Invalid port
        return FlowEncoder(config=config)

    def test_unavailable_returns_zero_vector(self, encoder):
        """Test that unavailable encoder returns zero vectors."""
        if encoder.available:
            pytest.skip("Ollama is available, skipping unavailable test")

        event = FlowEvent(
            event_id="e1",
            timestamp=datetime.now(timezone.utc),
            event_type="click",
        )

        vector = encoder.encode_event(event)
        assert vector.shape == (768,)
        # Should be zero vector when unavailable
        assert np.allclose(vector, 0)


class TestEncoderMaxSequenceLength:
    """Test sequence length limiting."""

    def test_long_session_truncated(self):
        """Test that very long sessions are truncated."""
        encoder = FallbackEncoder(config=BehaviorModelerConfig(max_sequence_length=5))

        base = datetime.now(timezone.utc)
        events = [
            FlowEvent(event_id=f"e{i}", timestamp=base + timedelta(seconds=i), event_type="click")
            for i in range(100)
        ]

        session = Session(session_id="long", started_at=base, events=events)
        vector = encoder.encode_session(session)

        # Should complete without error
        assert vector.shape == (768,)
