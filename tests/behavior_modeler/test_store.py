"""Tests for FlowStore."""

import pytest
from datetime import datetime, timezone, timedelta
import numpy as np

from behavior_modeler.store import FlowStore
from behavior_modeler.models import Session, FlowEvent, BehaviorCluster, TestGap


class TestFlowStore:
    """Tests for FlowStore."""

    def test_store_initialization(self, store):
        """Test store initializes correctly."""
        assert store is not None
        stats = store.get_stats()
        assert stats["total_sessions"] == 0
        assert stats["total_events"] == 0

    def test_save_and_retrieve_session(self, store):
        """Test saving and retrieving a session."""
        base = datetime.now(timezone.utc)
        session = Session(
            session_id="test_sess_001",
            started_at=base,
            events=[
                FlowEvent(
                    event_id="evt_001",
                    timestamp=base,
                    component_type="SearchBox",
                    event_type="input",
                    event_data={"value": "test query"},
                ),
                FlowEvent(
                    event_id="evt_002",
                    timestamp=base + timedelta(seconds=5),
                    component_type="SearchResults",
                    event_type="select",
                ),
            ],
        )

        # Save
        assert store.save_session(session)

        # Retrieve
        retrieved = store.get_session("test_sess_001")
        assert retrieved is not None
        assert retrieved.session_id == "test_sess_001"
        assert len(retrieved.events) == 2
        assert retrieved.events[0].component_type == "SearchBox"
        assert retrieved.events[1].event_type == "select"

    def test_save_session_with_vector(self, store):
        """Test saving session with vector embedding."""
        base = datetime.now(timezone.utc)
        vector = np.random.randn(768).astype(np.float32)

        session = Session(
            session_id="test_sess_002",
            started_at=base,
            vector=vector,
            events=[
                FlowEvent(event_id="e1", timestamp=base, event_type="click"),
            ],
        )

        assert store.save_session(session)

        retrieved = store.get_session("test_sess_002")
        assert retrieved.vector is not None
        assert retrieved.vector.shape == (768,)
        np.testing.assert_array_almost_equal(retrieved.vector, vector)

    def test_update_session_vector(self, store):
        """Test updating session vector."""
        base = datetime.now(timezone.utc)
        session = Session(
            session_id="test_sess_003",
            started_at=base,
            events=[FlowEvent(event_id="e1", timestamp=base, event_type="click")],
        )
        store.save_session(session)

        # Update vector
        new_vector = np.random.randn(768).astype(np.float32)
        assert store.update_session_vector("test_sess_003", new_vector)

        retrieved = store.get_session("test_sess_003")
        np.testing.assert_array_almost_equal(retrieved.vector, new_vector)

    def test_update_session_cluster(self, store):
        """Test updating session cluster assignment."""
        base = datetime.now(timezone.utc)
        session = Session(
            session_id="test_sess_004",
            started_at=base,
            events=[FlowEvent(event_id="e1", timestamp=base, event_type="click")],
        )
        store.save_session(session)

        assert store.update_session_cluster("test_sess_004", cluster_id=5)

        retrieved = store.get_session("test_sess_004")
        assert retrieved.cluster_id == 5

    def test_list_sessions_empty(self, store):
        """Test listing sessions when empty."""
        sessions, total = store.list_sessions()
        assert sessions == []
        assert total == 0

    def test_list_sessions_with_filters(self, store, sample_sessions):
        """Test listing sessions with various filters."""
        # Save some sessions
        for session in sample_sessions[:10]:
            store.save_session(session)

        # List all
        sessions, total = store.list_sessions()
        assert total == 10

        # Filter by completion
        complete, _ = store.list_sessions(is_complete=True)
        incomplete, _ = store.list_sessions(is_complete=False)
        assert len(complete) + len(incomplete) == 10

        # Pagination
        page1, _ = store.list_sessions(limit=5, offset=0)
        page2, _ = store.list_sessions(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].session_id != page2[0].session_id

    def test_iter_sessions(self, store, sample_sessions):
        """Test iterating over sessions."""
        for session in sample_sessions[:5]:
            store.save_session(session)

        count = 0
        for session in store.iter_sessions(include_events=True):
            count += 1
            assert len(session.events) > 0

        assert count == 5

    def test_get_session_without_events(self, store):
        """Test retrieving session without loading events."""
        base = datetime.now(timezone.utc)
        session = Session(
            session_id="test_sess_005",
            started_at=base,
            events=[
                FlowEvent(event_id="e1", timestamp=base, event_type="click"),
                FlowEvent(event_id="e2", timestamp=base, event_type="submit"),
            ],
        )
        store.save_session(session)

        # Without events
        retrieved = store.get_session("test_sess_005", include_events=False)
        assert retrieved is not None
        assert len(retrieved.events) == 0

        # With events
        retrieved = store.get_session("test_sess_005", include_events=True)
        assert len(retrieved.events) == 2


class TestClusterOperations:
    """Tests for cluster operations."""

    def test_save_and_get_cluster(self, store):
        """Test saving and retrieving clusters."""
        cluster = BehaviorCluster(
            cluster_id=0,  # Will be auto-assigned
            label="Test Cluster",
            description="A test cluster",
            cluster_type="happy_path",
            session_count=100,
            completion_rate=0.85,
            dominant_components=["SearchBox", "Results"],
        )

        cluster_id = store.save_cluster(cluster)
        assert cluster_id > 0

        clusters = store.get_clusters()
        assert len(clusters) == 1
        assert clusters[0].label == "Test Cluster"
        assert clusters[0].completion_rate == 0.85

    def test_update_cluster(self, store):
        """Test updating an existing cluster."""
        cluster = BehaviorCluster(
            cluster_id=0,
            label="Original",
            session_count=50,
        )
        cluster_id = store.save_cluster(cluster)

        # Update
        cluster.cluster_id = cluster_id
        cluster.label = "Updated"
        cluster.session_count = 100
        store.save_cluster(cluster)

        clusters = store.get_clusters()
        assert len(clusters) == 1
        assert clusters[0].label == "Updated"
        assert clusters[0].session_count == 100


class TestTestGapOperations:
    """Tests for test gap operations."""

    def test_save_and_get_gap(self, store):
        """Test saving and retrieving test gaps."""
        gap = TestGap(
            gap_id="gap_001",
            gap_type="uncovered_flow",
            flow_description="Upload retry flow",
            affected_components=["UploadBox", "RetryButton"],
            observed_count=25,
            priority="high",
        )

        assert store.save_test_gap(gap)

        gaps = store.get_test_gaps()
        assert len(gaps) == 1
        assert gaps[0].gap_id == "gap_001"
        assert gaps[0].priority == "high"

    def test_filter_gaps_by_status(self, store):
        """Test filtering gaps by status."""
        # Create gaps with different statuses
        store.save_test_gap(TestGap(gap_id="g1", gap_type="uncovered", status="open"))
        store.save_test_gap(TestGap(gap_id="g2", gap_type="uncovered", status="open"))
        store.save_test_gap(TestGap(gap_id="g3", gap_type="uncovered", status="resolved"))

        open_gaps = store.get_test_gaps(status="open")
        assert len(open_gaps) == 2

        resolved_gaps = store.get_test_gaps(status="resolved")
        assert len(resolved_gaps) == 1


class TestStoreStats:
    """Tests for store statistics."""

    def test_stats_with_data(self, store, sample_sessions):
        """Test statistics after adding data."""
        # Add sessions
        for session in sample_sessions[:5]:
            store.save_session(session)

        # Add cluster
        store.save_cluster(BehaviorCluster(cluster_id=0, label="Test"))

        # Add gap
        store.save_test_gap(TestGap(gap_id="g1", gap_type="uncovered"))

        stats = store.get_stats()
        assert stats["total_sessions"] == 5
        assert stats["total_events"] > 0
        assert stats["total_clusters"] == 1
        assert stats["open_gaps"] == 1
        assert stats["avg_events_per_session"] > 0
