"""Tests for FlowIndex."""

import pytest
from datetime import datetime, timezone, timedelta
import numpy as np

from behavior_modeler.index import FlowIndex, SearchResult
from behavior_modeler.encoder import FallbackEncoder
from behavior_modeler.models import Session, FlowEvent
from behavior_modeler.config import BehaviorModelerConfig


class TestFlowIndex:
    """Tests for FlowIndex."""

    @pytest.fixture
    def encoder(self):
        return FallbackEncoder()

    @pytest.fixture
    def populated_store(self, store, encoder, sample_sessions):
        """Store with encoded sessions."""
        for session in sample_sessions[:20]:
            # Encode the session
            vector = encoder.encode_session(session)
            session.vector = vector
            store.save_session(session)
        return store

    @pytest.fixture
    def index(self, populated_store, encoder, config):
        """Built index."""
        idx = FlowIndex(populated_store, encoder, config)
        idx.build()
        return idx

    def test_index_initialization(self, store, encoder, config):
        """Test index initializes correctly."""
        index = FlowIndex(store, encoder, config)
        assert index.size == 0
        assert index.dimension == 768

    def test_index_build(self, populated_store, encoder, config):
        """Test building index from store."""
        index = FlowIndex(populated_store, encoder, config)
        indexed = index.build()

        assert indexed == 20
        assert index.size == 20

    def test_index_stats(self, index):
        """Test index statistics."""
        stats = index.get_stats()
        assert stats["initialized"] is True
        assert stats["size"] == 20
        assert stats["dimension"] == 768

    def test_search_by_vector(self, index, encoder):
        """Test searching by vector."""
        # Create a query vector similar to search flows
        query_event = FlowEvent(
            event_id="q",
            timestamp=datetime.now(timezone.utc),
            component_type="SearchBox",
            event_type="input",
        )
        query_vector = encoder.encode_event(query_event)

        results = index.search(query_vector, k=5)

        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

        # Results should be sorted by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_search_by_query(self, index):
        """Test searching by natural language query."""
        results = index.search_by_query("user searches for code")

        assert len(results) <= 10
        assert all(r.similarity >= 0 and r.similarity <= 1 for r in results)

    def test_search_with_min_similarity(self, index, encoder):
        """Test filtering by minimum similarity."""
        query_vector = encoder.encode_query("random query")

        # High threshold should return fewer results
        results_high = index.search(query_vector, k=10, min_similarity=0.9)
        results_low = index.search(query_vector, k=10, min_similarity=0.1)

        assert len(results_high) <= len(results_low)
        assert all(r.similarity >= 0.9 for r in results_high)

    def test_search_include_session(self, index):
        """Test loading full session data."""
        results = index.search_by_query("upload file", k=3, include_session=True)

        for result in results:
            if result.session is not None:
                assert result.session.session_id == result.session_id

    def test_search_similar_sessions(self, index, populated_store):
        """Test finding similar sessions."""
        # Get first session
        sessions, _ = populated_store.list_sessions(limit=1)
        session_id = sessions[0].session_id

        results = index.search_similar(session_id, k=5)

        # Should not include the query session itself
        assert all(r.session_id != session_id for r in results)
        assert len(results) <= 5

    def test_add_session_to_index(self, index, encoder):
        """Test adding a single session to existing index."""
        initial_size = index.size

        # Create new session
        base = datetime.now(timezone.utc)
        session = Session(
            session_id="new_session_123",
            started_at=base,
            events=[
                FlowEvent(event_id="e1", timestamp=base, event_type="click"),
            ],
        )
        session.vector = encoder.encode_session(session)

        assert index.add_session(session)
        assert index.size == initial_size + 1

        # Should be searchable
        results = index.search(session.vector, k=1)
        assert len(results) == 1
        assert results[0].session_id == "new_session_123"

    def test_add_session_without_vector(self, index):
        """Test adding session without vector fails gracefully."""
        session = Session(
            session_id="no_vector",
            started_at=datetime.now(timezone.utc),
            events=[],
            vector=None,
        )

        assert not index.add_session(session)

    def test_empty_index_search(self, store, encoder, config):
        """Test searching empty index returns empty results."""
        index = FlowIndex(store, encoder, config)
        index._init_index()

        results = index.search_by_query("anything")
        assert results == []


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating search result."""
        result = SearchResult(
            session_id="sess_001",
            similarity=0.85,
            distance=0.15,
        )

        assert result.session_id == "sess_001"
        assert result.similarity == 0.85
        assert result.distance == 0.15
        assert result.session is None

    def test_search_result_with_session(self):
        """Test search result with session data."""
        session = Session(
            session_id="sess_002",
            started_at=datetime.now(timezone.utc),
        )

        result = SearchResult(
            session_id="sess_002",
            similarity=0.9,
            distance=0.1,
            session=session,
        )

        assert result.session is not None
        assert result.session.session_id == "sess_002"


class TestBruteForceSearch:
    """Test brute-force search fallback when HNSW unavailable."""

    @pytest.fixture
    def index_without_hnsw(self, populated_store, encoder, config):
        """Create index without HNSW."""
        index = FlowIndex(populated_store, encoder, config)
        index._index = None  # Force brute-force mode
        index._initialized = True

        # Load session IDs
        for session in populated_store.iter_sessions(include_events=False):
            if session.vector is not None:
                index._session_ids.append(session.session_id)
                index._id_to_label[session.session_id] = len(index._session_ids) - 1

        return index

    def test_brute_force_search(self, index_without_hnsw, encoder):
        """Test brute-force search works."""
        query_vector = encoder.encode_query("search code")

        results = index_without_hnsw.search(query_vector, k=5)

        assert len(results) <= 5
        # Results should still be sorted by similarity
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity
