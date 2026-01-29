"""Tests for mock flow generator."""

import pytest
from datetime import datetime, timezone

from behavior_modeler.mock import MockFlowGenerator, generate_sample_flows, FLOW_TEMPLATES


class TestMockFlowGenerator:
    """Tests for MockFlowGenerator."""

    def test_generator_deterministic_with_seed(self):
        """Verify generator with same seed produces consistent event counts."""
        gen1 = MockFlowGenerator(seed=42)
        gen2 = MockFlowGenerator(seed=42)
        s1 = gen1.generate_session()
        s2 = gen2.generate_session()
        # UUIDs will differ, but event counts should be similar
        assert len(s1.events) == len(s2.events)
            assert len(s1.events) == len(s2.events)

    def test_generate_single_session(self, mock_generator):
        """Test generating a single session."""
        session = mock_generator.generate_session(template_name="search_to_view")

        assert session.session_id
        assert session.started_at
        assert len(session.events) > 0

        # Check events have required fields
        for event in session.events:
            assert event.event_id
            assert event.timestamp
            assert event.event_type

    def test_generate_all_templates(self, mock_generator):
        """Test generating sessions from all templates."""
        for template_name in FLOW_TEMPLATES:
            session = mock_generator.generate_session(template_name=template_name)
            assert len(session.events) > 0
            assert "search" in session.user_agent.lower() or template_name in session.user_agent.lower()

    def test_generate_batch(self, mock_generator):
        """Test generating a batch of sessions."""
        sessions = mock_generator.generate_batch(100)

        assert len(sessions) == 100

        # Check sessions are sorted by time
        for i in range(len(sessions) - 1):
            assert sessions[i].started_at <= sessions[i + 1].started_at

    def test_batch_template_distribution(self, mock_generator):
        """Test batch respects template weights."""
        sessions = mock_generator.generate_batch(
            count=1000,
            template_weights={
                "search_to_view": 0.8,
                "upload_file": 0.2,
            },
        )

        search_count = sum(
            1 for s in sessions
            if "search_to_view" in (s.user_agent or "")
        )
        upload_count = sum(
            1 for s in sessions
            if "upload_file" in (s.user_agent or "")
        )

        # With 80/20 split, search should be much more common
        assert search_count > upload_count * 2

    def test_session_completion_varies(self, mock_generator):
        """Test that sessions have varying completion status."""
        sessions = mock_generator.generate_batch(100)

        complete = sum(1 for s in sessions if s.is_complete)
        incomplete = sum(1 for s in sessions if not s.is_complete)

        # Both should exist
        assert complete > 0
        assert incomplete > 0

    def test_events_have_timestamps(self, mock_generator):
        """Test events have properly ordered timestamps."""
        session = mock_generator.generate_session()

        for i in range(len(session.events) - 1):
            assert session.events[i].timestamp <= session.events[i + 1].timestamp

    def test_events_have_mock_data(self, mock_generator):
        """Test events have generated mock data."""
        session = mock_generator.generate_session(template_name="search_to_view")

        # Search input should have value
        input_events = [e for e in session.events if e.event_type == "input"]
        if input_events:
            assert "value" in input_events[0].event_data

    def test_custom_session_id(self, mock_generator):
        """Test using custom session ID."""
        session = mock_generator.generate_session(session_id="my_custom_id")
        assert session.session_id == "my_custom_id"

    def test_custom_base_time(self, mock_generator):
        """Test using custom base time."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        session = mock_generator.generate_session(base_time=custom_time)

        # First event should be at or after base time
        assert session.events[0].timestamp >= custom_time


class TestConvenienceFunction:
    """Tests for generate_sample_flows convenience function."""

    def test_generate_sample_flows(self):
        """Verify generate_sample_flows with seed produces consistent counts."""
        sessions1 = generate_sample_flows(count=5, seed=999)
        sessions2 = generate_sample_flows(count=5, seed=999)
        assert len(sessions1) == 5
        assert len(sessions2) == 5
        # Event counts should match
        assert len(sessions1[0].events) == len(sessions2[0].events)

