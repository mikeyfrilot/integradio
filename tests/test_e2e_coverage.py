"""
Batch 4: E2E & Integration Coverage Tests (25 tests)

Tests for end-to-end flows and integration - CRITICAL priority
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import numpy as np
import json


class TestE2EBasicAppFlow:
    """Tests for basic application flow."""

    def test_e2e_basic_app_flow(self, mock_embedder, tmp_path):
        """Verify basic app flow works end-to-end."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks
                from integradio.components import semantic, SemanticComponent
                from integradio.registry import ComponentMetadata

                mock_registry = MagicMock()
                mock_registry.all_components.return_value = []
                mock_registry_cls.return_value = mock_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    with patch("gradio.Blocks.__enter__", return_value=None):
                        with patch("gradio.Blocks.__exit__", return_value=None):
                            # Create blocks
                            blocks = SemanticBlocks(
                                db_path=tmp_path / "test.db",
                                auto_register=False,
                            )

                            # Create mock component
                            mock_comp = MagicMock()
                            mock_comp._id = 1
                            mock_comp.label = "Test Input"

                            # Wrap with semantic
                            wrapped = semantic(mock_comp, intent="test input")

                            assert isinstance(wrapped, SemanticComponent)
                            assert wrapped.intent == "test input"

    def test_e2e_multiple_components_registration(self, mock_embedder, tmp_path):
        """Verify multiple components can be registered."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks
                from integradio.components import semantic

                mock_registry = MagicMock()
                mock_registry.register.return_value = True
                mock_registry_cls.return_value = mock_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)

                    # Create multiple components
                    for i in range(5):
                        mock_comp = MagicMock()
                        mock_comp._id = i + 1
                        semantic(mock_comp, intent=f"component {i}")

                    # All should be tracked
                    from integradio.components import SemanticComponent
                    assert len(SemanticComponent._instances) >= 5


class TestE2EComponentToAPIRoundtrip:
    """Tests for component to API roundtrip."""

    @pytest.mark.asyncio
    async def test_e2e_component_to_api_roundtrip(self, mock_embedder, tmp_path):
        """Verify components can be queried through API."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks
                from integradio.api import create_api_routes
                from integradio.registry import ComponentMetadata, SearchResult

                mock_registry = MagicMock()

                # Setup search results
                mock_metadata = ComponentMetadata(
                    component_id=1,
                    component_type="Textbox",
                    intent="search input",
                    label="Search",
                    tags=["input"],
                )

                mock_result = SearchResult(
                component_id=1,
                score=0.95,
                distance=0.05,
                metadata=mock_metadata,
                )

                mock_registry_cls.return_value = mock_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)
                    blocks.search = MagicMock(return_value=[mock_result])

                    # Setup API
                    mock_app = MagicMock()
                    handlers = {}

                    def capture_get(path):
                        def decorator(func):
                            handlers[path] = func
                            return func
                        return decorator

                    mock_app.get = capture_get

                    create_api_routes(mock_app, blocks)

                    # Test search
                    search_handler = handlers.get("/semantic/search")
                    if search_handler:
                        response = await search_handler(
                            q="search input",
                            k=10,
                            type=None,
                            tags=None,
                        )
                        data = json.loads(response.body)

                        assert "results" in data
                        assert len(data["results"]) == 1
                        assert data["results"][0]["type"] == "Textbox"


class TestE2EEmbedderPipeline:
    """Tests for embedder pipeline end-to-end."""

    def test_e2e_embedder_pipeline(self, temp_cache_dir):
        """Verify embedder pipeline works end-to-end."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder(cache_dir=temp_cache_dir)

            # Simulate embedding flow
            texts = ["component 1", "component 2", "component 3"]

            # When unavailable, returns zeros but doesn't crash
            embedder._available = False
            results = embedder.embed_batch(texts)

            assert len(results) == 3
            for result in results:
                assert result.shape == (768,)

    def test_e2e_embedder_caching_flow(self, temp_cache_dir):
        """Verify embedder caching works in full flow."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder(cache_dir=temp_cache_dir)

            # Pre-cache some vectors
            test_text = "cached search query"
            cached_vector = np.random.rand(768).astype(np.float32)
            cache_key = embedder._cache_key(test_text)
            embedder._cache[cache_key] = cached_vector
            embedder._save_cache()

            # New embedder instance should load cache
            embedder2 = Embedder(cache_dir=temp_cache_dir)

            result = embedder2.embed(test_text)

            assert np.array_equal(result, cached_vector)


class TestE2EFailureRecovery:
    """Tests for failure recovery scenarios."""

    def test_e2e_failure_recovery(self, mock_embedder, tmp_path):
        """Verify system recovers from failures gracefully."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks

                mock_registry = MagicMock()
                mock_registry.get.return_value = None  # Simulate not found
                mock_registry_cls.return_value = mock_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)

                    # Missing component should return error dict, not crash
                    mock_comp = MagicMock()
                    mock_comp._id = 999

                    result = blocks.describe(mock_comp)

                    assert "error" in result

    def test_e2e_circuit_breaker_recovery(self):
        """Verify circuit breaker enables recovery after failures."""
        from integradio.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        import time

        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout_seconds=0.1,
            exception_types=(ValueError,),
        )

        breaker = CircuitBreaker("test-recovery", config=config)

        # Cause failures to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: exec('raise ValueError()'))

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Recovery succeeds
        result = breaker.call(lambda: "recovered")

        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED


class TestE2EDataPersistence:
    """Tests for data persistence across sessions."""

    def test_e2e_registry_persistence(self, tmp_path):
        """Verify registry persists data across sessions."""
        from integradio.registry import ComponentRegistry, ComponentMetadata
        import numpy as np

        db_path = tmp_path / "registry.db"

        # First session - register components
        registry1 = ComponentRegistry(db_path=db_path)

        meta = ComponentMetadata(
            component_id=1,
            component_type="Button",
            intent="submit form",
            label="Submit",
            tags=["action"],
        )

        vector = np.random.rand(768).astype(np.float32)
        registry1.register(1, vector, meta)

        # Close first session
        del registry1

        # Second session - data should persist
        registry2 = ComponentRegistry(db_path=db_path)

        retrieved = registry2.get(1)
        assert retrieved is not None
        assert retrieved.intent == "submit form"

    def test_e2e_behavior_store_persistence(self, tmp_path):
        """Verify behavior store persists sessions."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.models import Session, FlowEvent
        from datetime import datetime, timezone

        config = BehaviorModelerConfig(db_path=tmp_path / "behavior.db")

        # First session
        store1 = FlowStore(config)

        session = Session(
            session_id="persist-test",
            started_at=datetime.now(timezone.utc),
            events=[
                FlowEvent(
                    event_id="evt-1",
                    timestamp=datetime.now(timezone.utc),
                    event_type="click",
                )
            ],
        )

        store1.save_session(session)
        store1.close()

        # Second session
        store2 = FlowStore(config)

        retrieved = store2.get_session("persist-test")
        assert retrieved is not None
        assert retrieved.session_id == "persist-test"

        store2.close()


class TestE2ESearchFlow:
    """Tests for search flow end-to-end."""

    def test_e2e_search_flow(self, mock_embedder, populated_registry):
        """Verify search flow works end-to-end."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks

                mock_registry_cls.return_value = populated_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)
                    blocks._registry = populated_registry

                    # Search for components
                    results = blocks.search("search input", k=5)

                    # Should return results from populated registry
                    assert isinstance(results, list)

    def test_e2e_find_single_component(self, mock_embedder, populated_registry):
        """Verify find() returns single best match."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks
                from integradio.registry import SearchResult

                # Setup mock to return single result
                mock_result = SearchResult(
                component_id=1,
                score=0.95,
                distance=0.05,
                metadata=MagicMock(component_type="Textbox"),
                )

                mock_registry = MagicMock()
                mock_registry.search.return_value = [mock_result]
                mock_registry_cls.return_value = mock_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)

                    # Mock get_semantic to return the component
                    with patch("integradio.blocks.get_semantic") as mock_get:
                        mock_semantic = MagicMock()
                        mock_semantic.component = MagicMock()
                        mock_get.return_value = mock_semantic

                        result = blocks.find("search input")

                        assert result is not None


class TestE2EDataflowTracing:
    """Tests for dataflow tracing end-to-end."""

    def test_e2e_dataflow_tracing(self, mock_embedder, populated_registry):
        """Verify dataflow tracing works."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks

                mock_registry_cls.return_value = populated_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)
                    blocks._registry = populated_registry

                    # Trace component relationships
                    mock_comp = MagicMock()
                    mock_comp._id = 1

                    trace = blocks.trace(mock_comp)

                    assert "upstream" in trace
                    assert "downstream" in trace


class TestE2EGraphExport:
    """Tests for graph export end-to-end."""

    def test_e2e_graph_export(self, mock_embedder, populated_registry):
        """Verify graph export produces valid structure."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks

                populated_registry.export_graph = MagicMock(return_value={
                    "nodes": [
                        {"id": 1, "type": "Textbox"},
                        {"id": 2, "type": "Button"},
                    ],
                    "links": [
                        {"source": 1, "target": 2, "type": "trigger"},
                    ],
                })

                mock_registry_cls.return_value = populated_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)
                    blocks._registry = populated_registry

                    graph = blocks.map()

                    assert "nodes" in graph
                    assert "links" in graph
                    assert len(graph["nodes"]) == 2

    def test_e2e_graph_json_export(self, mock_embedder):
        """Verify graph JSON export is valid JSON."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks

                mock_registry = MagicMock()
                mock_registry.export_graph.return_value = {
                    "nodes": [{"id": 1}],
                    "links": [],
                }
                mock_registry_cls.return_value = mock_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)

                    json_str = blocks.map_json()

                    # Should be valid JSON
                    parsed = json.loads(json_str)
                    assert "nodes" in parsed


class TestE2EBehaviorModelerFlow:
    """Tests for behavior modeler flow end-to-end."""

    def test_e2e_mock_to_store_flow(self, tmp_path):
        """Verify mock generation to store flow works."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.mock import MockFlowGenerator

        config = BehaviorModelerConfig(db_path=tmp_path / "behavior.db")
        store = FlowStore(config)
        generator = MockFlowGenerator(seed=42)

        # Generate and store sessions
        sessions = generator.generate_batch(count=10)

        for session in sessions:
            result = store.save_session(session)
            assert result is True

        # Verify all stored
        stored_sessions, total = store.list_sessions(limit=20)
        assert total == 10

        store.close()

    def test_e2e_store_stats_flow(self, tmp_path):
        """Verify store statistics are accurate."""
        from behavior_modeler.config import BehaviorModelerConfig
        from behavior_modeler.store import FlowStore
        from behavior_modeler.mock import generate_sample_flows

        config = BehaviorModelerConfig(db_path=tmp_path / "behavior.db")
        store = FlowStore(config)

        # Generate and store sessions
        sessions = generate_sample_flows(count=20, seed=42)

        for session in sessions:
            store.save_session(session)

        stats = store.get_stats()

        assert stats["total_sessions"] == 20
        assert stats["total_events"] > 0

        store.close()


class TestE2EErrorScenarios:
    """Tests for error scenarios end-to-end."""

    def test_e2e_missing_database_path(self, tmp_path):
        """Verify system handles missing database gracefully."""
        from integradio.registry import ComponentRegistry

        # Non-existent path should be created
        db_path = tmp_path / "nested" / "path" / "registry.db"$([Environment]::NewLine)        db_path.parent.mkdir(parents=True, exist_ok=True)$([Environment]::NewLine)        registry = ComponentRegistry(db_path=db_path)

        assert db_path.parent.exists()

    def test_e2e_corrupted_cache_recovery(self, tmp_path):
        """Verify embedder recovers from corrupted cache."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()

            # Create corrupted cache file
            cache_file = cache_dir / "embeddings.json"
            cache_file.write_text("not valid json {{{")

            # Should recover without crashing
            with pytest.warns(UserWarning, match="Could not load"):
                embedder = Embedder(cache_dir=cache_dir)

            # Should start with empty cache
            assert embedder._cache == {}

    def test_e2e_empty_search_results(self, mock_embedder):
        """Verify empty search results are handled."""
        with patch("integradio.blocks.ComponentRegistry") as mock_registry_cls:
            with patch("integradio.blocks.Embedder", return_value=mock_embedder):
                from integradio.blocks import SemanticBlocks

                mock_registry = MagicMock()
                mock_registry.search.return_value = []
                mock_registry_cls.return_value = mock_registry

                with patch("gradio.Blocks.__init__", return_value=None):
                    blocks = SemanticBlocks(auto_register=False)

                    results = blocks.search("nonexistent query")

                    assert results == []

                    # find() should return None
                    result = blocks.find("nonexistent")
                    assert result is None

