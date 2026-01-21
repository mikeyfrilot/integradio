"""Tests for the FastAPI API layer."""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from behavior_modeler.api import create_app
from behavior_modeler.config import BehaviorModelerConfig


class TestAPIBasics:
    """Test basic API functionality."""

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        app = create_app(config)
        return TestClient(app)

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestFlowIngestion:
    """Test flow ingestion endpoints."""

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        app = create_app(config)
        return TestClient(app)

    def test_ingest_flow(self, client):
        """Test ingesting a flow."""
        response = client.post("/flows/ingest", json={
            "session_id": "test_session_001",
            "events": [
                {
                    "event_id": "evt_001",
                    "timestamp": "2026-01-20T10:30:00Z",
                    "component_type": "SearchBox",
                    "event_type": "input",
                    "event_data": {"value": "test query"},
                },
                {
                    "event_id": "evt_002",
                    "timestamp": "2026-01-20T10:30:05Z",
                    "component_type": "SearchResults",
                    "event_type": "select",
                    "event_data": {"index": 0},
                },
            ],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["session_id"] == "test_session_001"
        assert data["event_count"] == 2

    def test_ingest_empty_events(self, client):
        """Test ingesting with no events fails."""
        response = client.post("/flows/ingest", json={
            "session_id": "test_session_002",
            "events": [],
        })

        assert response.status_code == 400

    def test_ingest_invalid_timestamp(self, client):
        """Test ingesting with invalid timestamp creates warning."""
        response = client.post("/flows/ingest", json={
            "session_id": "test_session_003",
            "events": [
                {
                    "event_id": "evt_001",
                    "timestamp": "invalid-timestamp",
                    "component_type": "Button",
                    "event_type": "click",
                },
                {
                    "event_id": "evt_002",
                    "timestamp": "2026-01-20T10:30:00Z",
                    "component_type": "Button",
                    "event_type": "click",
                },
            ],
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["warnings"]) > 0


class TestSessionEndpoints:
    """Test session listing and retrieval."""

    @pytest.fixture
    def client_with_data(self, populated_store, config):
        """Create test client with populated store."""
        app = create_app(config)
        return TestClient(app)

    def test_list_sessions(self, client_with_data):
        """Test listing sessions."""
        response = client_with_data.get("/flows/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_sessions_with_limit(self, client_with_data):
        """Test listing with limit."""
        response = client_with_data.get("/flows/sessions?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) <= 5

    def test_get_session(self, client_with_data, populated_store):
        """Test getting a specific session."""
        # Get first session ID
        sessions, _ = populated_store.list_sessions(limit=1)
        session_id = sessions[0].session_id

        response = client_with_data.get(f"/flows/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session"]["session_id"] == session_id
        assert "events" in data["session"]

    def test_get_session_not_found(self, client_with_data):
        """Test getting non-existent session."""
        response = client_with_data.get("/flows/sessions/nonexistent_id")

        assert response.status_code == 404


class TestSearchEndpoints:
    """Test search endpoints."""

    @pytest.fixture
    def client_with_data(self, populated_store, config):
        """Create test client with populated store."""
        app = create_app(config)
        return TestClient(app)

    def test_search_by_query(self, client_with_data):
        """Test searching by query."""
        response = client_with_data.post("/flows/search", json={
            "query": "user searches for code",
            "k": 5,
        })

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 5

    def test_search_by_session_id(self, client_with_data, populated_store):
        """Test searching by similar session."""
        sessions, _ = populated_store.list_sessions(limit=1)
        session_id = sessions[0].session_id

        response = client_with_data.post("/flows/search", json={
            "session_id": session_id,
            "k": 5,
        })

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_search_no_query_fails(self, client_with_data):
        """Test search without query or session_id fails."""
        response = client_with_data.post("/flows/search", json={
            "k": 5,
        })

        assert response.status_code == 400


class TestClusterEndpoints:
    """Test clustering endpoints."""

    @pytest.fixture
    def client_with_data(self, populated_store, config):
        """Create test client with populated store."""
        app = create_app(config)
        return TestClient(app)

    def test_compute_clusters(self, client_with_data):
        """Test running clustering."""
        response = client_with_data.post("/clusters/compute", json={
            "algorithm": "simple",  # Use simple fallback for testing
            "min_cluster_size": 3,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert "clusters_found" in data

    def test_list_clusters(self, client_with_data):
        """Test listing clusters."""
        # First compute clusters
        client_with_data.post("/clusters/compute", json={
            "algorithm": "simple",
            "min_cluster_size": 3,
        })

        response = client_with_data.get("/clusters")

        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @pytest.fixture
    def client_with_data(self, populated_store, config):
        """Create test client with populated store."""
        app = create_app(config)
        return TestClient(app)

    def test_predict_next_action(self, client_with_data):
        """Test next action prediction."""
        response = client_with_data.post("/predict/next", json={
            "events": [
                {
                    "event_id": "e1",
                    "timestamp": "2026-01-20T10:00:00Z",
                    "component_type": "SearchBox",
                    "event_type": "input",
                },
                {
                    "event_id": "e2",
                    "timestamp": "2026-01-20T10:00:05Z",
                    "component_type": "SearchResults",
                    "event_type": "select",
                },
            ],
            "top_k": 3,
        })

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data


class TestPatternEndpoints:
    """Test pattern mining endpoints."""

    @pytest.fixture
    def client_with_data(self, populated_store, config):
        """Create test client with populated store."""
        app = create_app(config)
        return TestClient(app)

    def test_get_patterns(self, client_with_data):
        """Test getting mined patterns."""
        response = client_with_data.get("/patterns?min_support=0.1")

        assert response.status_code == 200
        data = response.json()
        assert "patterns" in data
        assert "n_sessions_analyzed" in data


class TestGapEndpoints:
    """Test gap detection endpoints."""

    @pytest.fixture
    def client_with_data(self, populated_store, config):
        """Create test client with populated store."""
        app = create_app(config)
        return TestClient(app)

    def test_list_gaps(self, client_with_data):
        """Test listing gaps."""
        response = client_with_data.get("/gaps")

        assert response.status_code == 200
        data = response.json()
        assert "gaps" in data
        assert "total_gaps" in data
        assert "gaps_by_priority" in data

    def test_analyze_gaps(self, client_with_data):
        """Test gap analysis."""
        response = client_with_data.post("/gaps/analyze?min_support=0.1")

        assert response.status_code == 200
        data = response.json()
        assert "total_gaps" in data
        assert "coverage_percentage" in data


class TestStatsEndpoint:
    """Test statistics endpoint."""

    @pytest.fixture
    def client_with_data(self, populated_store, config):
        """Create test client with populated store."""
        app = create_app(config)
        return TestClient(app)

    def test_get_stats(self, client_with_data):
        """Test getting stats."""
        response = client_with_data.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "store" in data
