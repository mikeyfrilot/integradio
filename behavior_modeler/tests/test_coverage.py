"""
Batch 3 continued: Behavior Modeler Coverage Tests (12 tests)

Tests for behavior_modeler/* - HIGH priority
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
import numpy as np


class TestBehaviorStoreInsertAndQuery:
    """Tests for FlowStore insert and query operations."""

    def test_behavior_store_insert_and_query(self, tmp_path):
        """Verify store can insert and retrieve sessions."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.models import Session, FlowEvent

        config = BehaviorModelerConfig(db_path=tmp_path / "test.db")
        store = FlowStore(config)

        # Create session
        session = Session(
            session_id="test-session-1",
            started_at=datetime.now(timezone.utc),
            events=[
                FlowEvent(
                    event_id="evt-1",
                    timestamp=datetime.now(timezone.utc),
                    component_type="Button",
                    event_type="click",
                )
            ],
        )

        # Save
        result = store.save_session(session)
        assert result is True

        # Retrieve
        retrieved = store.get_session("test-session-1")
        assert retrieved is not None
        assert retrieved.session_id == "test-session-1"
        assert len(retrieved.events) == 1

        store.close()

    def test_behavior_store_list_sessions(self, tmp_path):
        """Verify store can list sessions with filters."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.models import Session, FlowEvent

        config = BehaviorModelerConfig(db_path=tmp_path / "test.db")
        store = FlowStore(config)

        # Create multiple sessions
        for i in range(5):
            session = Session(
                session_id=f"session-{i}",
                started_at=datetime.now(timezone.utc) - timedelta(hours=i),
                events=[
                    FlowEvent(
                        event_id=f"evt-{i}",
                        timestamp=datetime.now(timezone.utc),
                        event_type="complete" if i % 2 == 0 else "view",
                    )
                ],
            )
            store.save_session(session)

        # List all
        sessions, total = store.list_sessions(limit=10)
        assert total == 5
        assert len(sessions) == 5

        store.close()


class TestBehaviorStoreDeduplication:
    """Tests for session deduplication."""

    def test_behavior_store_deduplication(self, tmp_path):
        """Verify store handles duplicate session IDs."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.models import Session, FlowEvent

        config = BehaviorModelerConfig(db_path=tmp_path / "test.db")
        store = FlowStore(config)

        # Create original session
        session1 = Session(
            session_id="dup-session",
            started_at=datetime.now(timezone.utc),
            events=[
                FlowEvent(
                    event_id="evt-1",
                    timestamp=datetime.now(timezone.utc),
                    event_type="click",
                )
            ],
        )
        store.save_session(session1)

        # Save with same ID (should replace)
        session2 = Session(
            session_id="dup-session",
            started_at=datetime.now(timezone.utc),
            events=[
                FlowEvent(
                    event_id="evt-2",
                    timestamp=datetime.now(timezone.utc),
                    event_type="submit",
                ),
                FlowEvent(
                    event_id="evt-3",
                    timestamp=datetime.now(timezone.utc),
                    event_type="complete",
                ),
            ],
        )
        store.save_session(session2)

        # Should have replaced
        retrieved = store.get_session("dup-session")
        assert len(retrieved.events) == 2  # From session2

        store.close()


class TestBehaviorModelsValidation:
    """Tests for model validation."""

    def test_behavior_models_validation(self):
        """Verify models validate required fields."""
        from behavior_modeler.models import FlowEvent, Session

        # FlowEvent requires event_id and timestamp
        event = FlowEvent(
            event_id="test-evt",
            timestamp=datetime.now(timezone.utc),
        )
        assert event.event_id == "test-evt"

        # Session requires session_id and started_at
        session = Session(
            session_id="test-session",
            started_at=datetime.now(timezone.utc),
        )
        assert session.session_id == "test-session"

    def test_behavior_models_to_dict(self):
        """Verify models serialize to dict correctly."""
        from behavior_modeler.models import FlowEvent, Session, TestGap

        event = FlowEvent(
            event_id="evt-1",
            timestamp=datetime.now(timezone.utc),
            component_type="Button",
            event_type="click",
            tags=["test"],
        )

        event_dict = event.to_dict()
        assert "event_id" in event_dict
        assert "timestamp" in event_dict
        assert event_dict["tags"] == ["test"]

    def test_behavior_models_from_dict(self):
        """Verify models deserialize from dict correctly."""
        from behavior_modeler.models import FlowEvent, Session

        data = {
            "event_id": "evt-1",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "component_type": "Textbox",
            "event_type": "input",
        }

        event = FlowEvent.from_dict(data)
        assert event.event_id == "evt-1"
        assert event.component_type == "Textbox"


class TestBehaviorMockGeneratesValidSequences:
    """Tests for mock flow generator."""

    def test_behavior_mock_generates_valid_sequences(self):
        """Verify mock generator creates valid sessions."""
        from behavior_modeler.mock import MockFlowGenerator

        generator = MockFlowGenerator(seed=42)
        session = generator.generate_session()

        assert session.session_id is not None
        assert len(session.events) > 0
        assert session.started_at is not None

    def test_behavior_mock_batch_generation(self):
        """Verify batch generation works correctly."""
        from behavior_modeler.mock import MockFlowGenerator

        generator = MockFlowGenerator(seed=42)
        sessions = generator.generate_batch(count=10)

        assert len(sessions) == 10

        # Should be sorted by time
        for i in range(1, len(sessions)):
            assert sessions[i].started_at >= sessions[i - 1].started_at

    def test_behavior_mock_template_usage(self):
        """Verify specific templates can be used."""
        from behavior_modeler.mock import MockFlowGenerator, FLOW_TEMPLATES

        generator = MockFlowGenerator(seed=42)

        for template_name in FLOW_TEMPLATES.keys():
            session = generator.generate_session(template_name=template_name)
            assert session is not None
            assert len(session.events) > 0


class TestBehaviorTapEventIngestion:
    """Tests for BehaviorTap event ingestion."""

    @pytest.mark.asyncio
    async def test_behavior_tap_event_ingestion(self, tmp_path):
        """Verify tap can ingest events."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.tap import BehaviorTap

        config = BehaviorModelerConfig(db_path=tmp_path / "test.db")
        store = FlowStore(config)
        tap = BehaviorTap(store=store, config=config)

        # Ingest events directly
        events = [
            {
                "event_id": "evt-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "click",
                "component_type": "Button",
            },
            {
                "event_id": "evt-2",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "complete",
            },
        ]

        result = await tap.ingest_events("test-session", events)
        assert result is True

        # Verify saved
        session = store.get_session("test-session")
        assert session is not None
        assert len(session.events) == 2

        store.close()

    @pytest.mark.asyncio
    async def test_behavior_tap_session_ingestion(self, tmp_path):
        """Verify tap can ingest full sessions."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.tap import BehaviorTap
        from behavior_modeler.models import Session, FlowEvent

        config = BehaviorModelerConfig(db_path=tmp_path / "test.db")
        store = FlowStore(config)
        tap = BehaviorTap(store=store, config=config)

        session = Session(
            session_id="direct-session",
            started_at=datetime.now(timezone.utc),
            events=[
                FlowEvent(
                    event_id="evt-1",
                    timestamp=datetime.now(timezone.utc),
                    event_type="view",
                )
            ],
        )

        result = await tap.ingest_session(session)
        assert result is True

        store.close()

    def test_behavior_tap_stats(self, tmp_path):
        """Verify tap provides statistics."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.tap import BehaviorTap

        config = BehaviorModelerConfig(db_path=tmp_path / "test.db")
        store = FlowStore(config)
        tap = BehaviorTap(store=store, config=config)

        stats = tap.get_stats()

        assert "running" in stats
        assert "active_sessions" in stats
        assert "events_captured" in stats
        assert "sessions_completed" in stats

        store.close()


class TestBehaviorClusterOperations:
    """Tests for cluster operations in store."""

    def test_behavior_cluster_save_and_retrieve(self, tmp_path):
        """Verify clusters can be saved and retrieved."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.models import BehaviorCluster
        import numpy as np

        config = BehaviorModelerConfig(db_path=tmp_path / "test.db")
        store = FlowStore(config)

        cluster = BehaviorCluster(
            cluster_id=0,  # Will be assigned
            label="Happy Path Users",
            description="Users who complete the main flow",
            cluster_type="happy_path",
            centroid=np.random.rand(768).astype(np.float32),
            session_count=100,
            completion_rate=0.95,
        )

        cluster_id = store.save_cluster(cluster)
        assert cluster_id > 0

        clusters = store.get_clusters()
        assert len(clusters) >= 1

        store.close()
