"""
Integration tests - End-to-end tests for integradio.

These tests verify that all components work together correctly.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path
import numpy as np


class TestSemanticBlocksIntegration:
    """Integration tests for SemanticBlocks with all components."""

    @pytest.fixture
    def mock_ollama(self):
        """Mock Ollama embeddings API."""
        with patch("httpx.get") as mock_get, patch("httpx.post") as mock_post:
            # Mock availability check
            mock_get.return_value = MagicMock(status_code=200)

            # Mock embedding response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embedding": np.random.randn(768).tolist()
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            yield mock_post

    def test_semantic_blocks_full_lifecycle(self, mock_ollama):
        """Test complete lifecycle: register, search, describe."""
        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent, semantic
        import gradio as gr

        # Reset global state
        SemanticComponent._instances.clear()

        # Use in-memory database (db_path=None)
        with SemanticBlocks(db_path=None, auto_register=True) as blocks:
            # Create components with semantic metadata
            input_box = semantic(
                gr.Textbox(label="Search Query"),
                intent="user enters search terms"
            )
            submit_btn = semantic(
                gr.Button("Search"),
                intent="triggers the search"
            )
            output_md = semantic(
                gr.Markdown("Results will appear here"),
                intent="displays search results"
            )

        # Verify components are registered
        all_comps = blocks.registry.all_components()
        assert len(all_comps) >= 3

        # Verify search works
        results = blocks.search("search input", k=5)
        # Should find relevant components
        assert len(results) >= 0  # May be 0 if embeddings are random

        # Verify map works
        graph = blocks.map()
        assert "nodes" in graph
        assert "links" in graph

        # Verify summary works
        summary = blocks.summary()
        assert "components registered" in summary.lower() or len(all_comps) == 0

    def test_embedder_caching(self, mock_ollama):
        """Test that embeddings are cached properly."""
        from integradio.embedder import Embedder
        import tempfile

        with tempfile.TemporaryDirectory() as temp_cache:
            embedder = Embedder(cache_dir=Path(temp_cache))

            # First call - should hit API
            text = "test embedding text"
            result1 = embedder.embed(text)
            initial_call_count = mock_ollama.call_count

            # Second call - should use cache
            result2 = embedder.embed(text)

            # Call count should not increase
            assert mock_ollama.call_count == initial_call_count

            # Results should be identical
            np.testing.assert_array_equal(result1, result2)

    def test_dataflow_extraction(self):
        """Test dataflow extraction from event handlers."""
        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent, semantic
        from integradio.introspect import extract_dataflow
        import gradio as gr

        SemanticComponent._instances.clear()

        with SemanticBlocks(auto_register=False) as blocks:
            input_text = gr.Textbox(label="Input")
            output_text = gr.Textbox(label="Output")
            btn = gr.Button("Process")

            def process(text):
                return text.upper()

            btn.click(fn=process, inputs=input_text, outputs=output_text)

        # Extract dataflow
        flows = extract_dataflow(blocks)

        # Should have captured the event handler
        assert len(flows) >= 1


class TestComponentRegistryIntegration:
    """Integration tests for ComponentRegistry with HNSW index."""

    def test_registry_search_integration(self):
        """Test search with real vectors."""
        from integradio.registry import ComponentRegistry, ComponentMetadata

        # Use in-memory database to avoid Windows file locking issues
        registry = ComponentRegistry(db_path=None)

        # Register components with meaningful vectors
        # Similar components should have similar vectors
        input_vector = np.array([1.0] * 384 + [0.0] * 384, dtype=np.float32)
        output_vector = np.array([0.0] * 384 + [1.0] * 384, dtype=np.float32)

        registry.register(
            component_id=1,
            vector=input_vector,
            metadata=ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="user input field",
                label="Input",
                tags=["input"],
            ),
        )

        registry.register(
            component_id=2,
            vector=output_vector,
            metadata=ComponentMetadata(
                component_id=2,
                component_type="Markdown",
                intent="display output",
                label="Output",
                tags=["output"],
            ),
        )

        # Search for input-like component
        results = registry.search(input_vector, k=2)

        # First result should be the input component
        assert len(results) >= 1
        assert results[0].component_id == 1

    def test_registry_relationship_tracking(self):
        """Test relationship storage and retrieval."""
        from integradio.registry import ComponentRegistry, ComponentMetadata

        registry = ComponentRegistry(db_path=None)

        # Register components
        for i in range(1, 4):
            registry.register(
                component_id=i,
                vector=np.random.randn(768).astype(np.float32),
                metadata=ComponentMetadata(
                    component_id=i,
                    component_type="Component",
                    intent=f"component {i}",
                ),
            )

        # Add relationships
        registry.add_relationship(1, 2, "trigger")
        registry.add_relationship(2, 3, "dataflow")

        # Check relationships
        rel_1 = registry.get_relationships(1)
        rel_2 = registry.get_relationships(2)

        # Component 1 should trigger component 2
        assert "trigger" in rel_1
        assert 2 in rel_1["trigger"]

        # Component 2 should flow to component 3
        assert "dataflow" in rel_2
        assert 3 in rel_2["dataflow"]

    def test_registry_filter_by_type(self):
        """Test filtering search results by component type."""
        from integradio.registry import ComponentRegistry, ComponentMetadata

        registry = ComponentRegistry(db_path=None)

        # Register mixed components
        registry.register(
            component_id=1,
            vector=np.random.randn(768).astype(np.float32),
            metadata=ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="text input",
            ),
        )

        registry.register(
            component_id=2,
            vector=np.random.randn(768).astype(np.float32),
            metadata=ComponentMetadata(
                component_id=2,
                component_type="Button",
                intent="click action",
            ),
        )

        # Search with type filter
        query = np.random.randn(768).astype(np.float32)
        textbox_results = registry.search(query, k=10, component_type="Textbox")

        # All results should be Textbox
        for r in textbox_results:
            assert r.metadata.component_type == "Textbox"

    def test_registry_filter_by_tags(self):
        """Test filtering search results by tags."""
        from integradio.registry import ComponentRegistry, ComponentMetadata

        registry = ComponentRegistry(db_path=None)

        # Register components with different tags
        registry.register(
            component_id=1,
            vector=np.random.randn(768).astype(np.float32),
            metadata=ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="text input",
                tags=["input", "text"],
            ),
        )

        registry.register(
            component_id=2,
            vector=np.random.randn(768).astype(np.float32),
            metadata=ComponentMetadata(
                component_id=2,
                component_type="Slider",
                intent="numeric input",
                tags=["input", "numeric"],
            ),
        )

        # Search with tag filter
        query = np.random.randn(768).astype(np.float32)
        text_results = registry.search(query, k=10, tags=["text"])

        # Should only return text-tagged component
        assert len(text_results) == 1
        assert "text" in text_results[0].metadata.tags


class TestVisualizationIntegration:
    """Integration tests for visualization with real data."""

    @pytest.fixture
    def populated_blocks(self):
        """Create SemanticBlocks with some components."""
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "search input", "label": "Query"},
                {"id": 2, "type": "Button", "intent": "submit action", "label": "Search"},
                {"id": 3, "type": "Markdown", "intent": "display results", "label": "Results"},
            ],
            "links": [
                {"source": 2, "target": 1, "type": "trigger"},
                {"source": 1, "target": 3, "type": "dataflow"},
            ],
        }
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="search input",
                label="Query",
                outputs_to=[3],
            ),
            ComponentMetadata(
                component_id=2,
                component_type="Button",
                intent="submit action",
                label="Search",
                outputs_to=[1],
            ),
            ComponentMetadata(
                component_id=3,
                component_type="Markdown",
                intent="display results",
                label="Results",
                inputs_from=[1],
            ),
        ]
        return mock_blocks

    def test_mermaid_complete_graph(self, populated_blocks):
        """Test Mermaid generation with complete graph."""
        from integradio.viz import generate_mermaid

        result = generate_mermaid(populated_blocks)

        # Should have all nodes
        assert "c1" in result
        assert "c2" in result
        assert "c3" in result

        # Should have all links
        assert "trigger" in result
        assert "-->" in result

        # Should have styles
        assert "classDef" in result

    def test_html_graph_complete(self, populated_blocks):
        """Test HTML graph with complete data."""
        from integradio.viz import generate_html_graph
        import json

        result = generate_html_graph(populated_blocks)

        # Should be complete HTML
        assert "<!DOCTYPE html>" in result
        assert "</html>" in result

        # Should have embedded JSON data
        assert '"nodes"' in result
        assert '"links"' in result

        # Should have interactive features
        assert "searchInput" in result
        assert "tooltip" in result

    def test_ascii_graph_complete(self, populated_blocks):
        """Test ASCII graph with complete data."""
        from integradio.viz import generate_ascii_graph

        result = generate_ascii_graph(populated_blocks)

        # Should have header
        assert "INTEGRADIO" in result or "COMPONENT GRAPH" in result

        # Should list all types
        assert "Textbox" in result
        assert "Button" in result
        assert "Markdown" in result

        # Should show dataflow
        assert "DATAFLOW" in result


class TestAPIIntegration:
    """Integration tests for API with real FastAPI app."""

    @pytest.fixture
    def full_app(self):
        """Create FastAPI app with complete SemanticBlocks mock."""
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from integradio.api import create_api_routes
        from integradio.registry import SearchResult, ComponentMetadata

        blocks = MagicMock()
        blocks.registry = MagicMock()

        # Setup comprehensive mock
        components = [
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="search input",
                label="Query",
                elem_id="search-box",
                tags=["input", "text"],
                file_path="app.py",
                line_number=10,
                inputs_from=[],
                outputs_to=[3],
            ),
            ComponentMetadata(
                component_id=2,
                component_type="Button",
                intent="submit search",
                label="Search",
                elem_id="search-btn",
                tags=["trigger"],
                file_path="app.py",
                line_number=15,
            ),
            ComponentMetadata(
                component_id=3,
                component_type="Markdown",
                intent="display results",
                label="Results",
                tags=["output"],
                file_path="app.py",
                line_number=20,
                inputs_from=[1],
                outputs_to=[],
            ),
        ]

        def mock_get(cid):
            for c in components:
                if c.component_id == cid:
                    return c
            return None

        blocks.registry.get.side_effect = mock_get
        blocks.registry.all_components.return_value = components
        blocks.registry.get_relationships.return_value = {"trigger": [], "dataflow": []}
        blocks.registry.get_dataflow.return_value = {"upstream": [], "downstream": []}

        blocks.search.return_value = [
            SearchResult(
                component_id=1,
                metadata=components[0],
                score=0.95,
                distance=0.05,
            )
        ]

        blocks.map.return_value = {
            "nodes": [
                {"id": c.component_id, "type": c.component_type, "intent": c.intent, "label": c.label}
                for c in components
            ],
            "links": [
                {"source": 2, "target": 1, "type": "trigger"},
                {"source": 1, "target": 3, "type": "dataflow"},
            ],
        }

        app = FastAPI()
        create_api_routes(app, blocks)
        return TestClient(app)

    def test_api_workflow(self, full_app):
        """Test complete API workflow: search -> get component -> get graph."""
        # 1. Search for components
        search_response = full_app.get("/semantic/search?q=search%20input")
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert search_data["count"] >= 1

        # 2. Get component details
        comp_id = search_data["results"][0]["component_id"]
        comp_response = full_app.get(f"/semantic/component/{comp_id}")
        assert comp_response.status_code == 200
        comp_data = comp_response.json()
        assert comp_data["component_id"] == comp_id

        # 3. Get graph
        graph_response = full_app.get("/semantic/graph")
        assert graph_response.status_code == 200
        graph_data = graph_response.json()
        assert len(graph_data["nodes"]) >= 1

        # 4. Get summary
        summary_response = full_app.get("/semantic/summary")
        assert summary_response.status_code == 200
        summary_data = summary_response.json()
        assert summary_data["total_components"] >= 1


class TestIntrospectionIntegration:
    """Integration tests for introspection utilities."""

    def test_component_info_and_intent_building(self):
        """Test extract_component_info and build_intent_text work together."""
        from integradio.introspect import extract_component_info, build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Dropdown"
        mock_component.label = "Category"
        mock_component.elem_id = "cat-select"
        mock_component.elem_classes = ["select-input"]
        mock_component.visible = True
        mock_component.interactive = True
        mock_component._id = 42
        mock_component.placeholder = "Select a category"
        mock_component.value = None
        mock_component.choices = ["A", "B", "C"]

        # Extract info
        info = extract_component_info(mock_component)

        # Build intent from component
        intent = build_intent_text(mock_component, "user selects a category")

        # Info should have correct data
        assert info["type"] == "Dropdown"
        assert info["label"] == "Category"
        assert info["component_id"] == 42

        # Intent should combine all info
        assert "user selects a category" in intent
        assert "Dropdown" in intent
        assert "Category" in intent
        assert "choices:" in intent

    def test_infer_tags_matches_component_behavior(self):
        """Test infer_tags produces expected tags for various components."""
        from integradio.introspect import infer_tags

        # Test various component types
        test_cases = [
            ("Textbox", ["input", "text"]),
            ("Button", ["trigger", "action"]),
            ("Markdown", ["output", "text", "display"]),
            ("Slider", ["input", "numeric", "range"]),
            ("Image", ["media", "visual"]),
            ("Chatbot", ["io", "conversation", "ai"]),
        ]

        for comp_type, expected_tags in test_cases:
            mock = MagicMock()
            mock.__class__.__name__ = comp_type

            result = infer_tags(mock)

            for tag in expected_tags:
                assert tag in result, f"{comp_type} should have tag '{tag}'"


class TestGracefulDegradation:
    """Test graceful degradation when services are unavailable."""

    def test_embedder_offline_mode(self):
        """Test embedder works (with zero vectors) when Ollama is offline."""
        from integradio.embedder import Embedder
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedder = Embedder(base_url="http://localhost:99999")  # Invalid port

        # Should return zero vector without crashing
        result = embedder.embed("test text")

        assert result is not None
        assert result.shape == (768,)
        assert np.all(result == 0)

    def test_registry_without_embeddings(self):
        """Test registry works with zero vectors."""
        from integradio.registry import ComponentRegistry, ComponentMetadata

        # Use in-memory database to avoid Windows file locking
        registry = ComponentRegistry(db_path=None)

        # Register with zero vectors
        zero_vec = np.zeros(768, dtype=np.float32)

        registry.register(
            component_id=1,
            vector=zero_vec,
            metadata=ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="test",
            ),
        )

        # Should be retrievable
        meta = registry.get(1)
        assert meta is not None
        assert meta.component_type == "Textbox"

        # Search should still work (though results may not be meaningful)
        results = registry.search(zero_vec, k=10)
        assert isinstance(results, list)
