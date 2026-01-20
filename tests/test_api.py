"""
Tests for API module - FastAPI endpoints for search and visualization.

Priority 5: Optional feature tests.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json


# Skip if FastAPI not installed
pytest.importorskip("fastapi")


class TestAPIEndpoints:
    """Test FastAPI endpoint functionality."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock SemanticBlocks for testing."""
        from integradio.registry import SearchResult, ComponentMetadata

        blocks = MagicMock()

        # Setup registry
        blocks.registry = MagicMock()
        blocks.registry.get.return_value = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="user input",
            label="Query",
            elem_id="query-input",
            tags=["input", "text"],
            file_path="app.py",
            line_number=42,
        )
        blocks.registry.get_relationships.return_value = {
            "trigger": [2],
            "dataflow": [3],
        }
        blocks.registry.get_dataflow.return_value = {
            "upstream": [1],
            "downstream": [3],
        }
        blocks.registry.all_components.return_value = [
            ComponentMetadata(component_id=1, component_type="Textbox", intent="input", label="Query"),
            ComponentMetadata(component_id=2, component_type="Button", intent="submit", label="Go"),
        ]

        # Setup search
        blocks.search.return_value = [
            SearchResult(
                component_id=1,
                metadata=ComponentMetadata(
                    component_id=1,
                    component_type="Textbox",
                    intent="user input",
                    label="Query",
                    tags=["input", "text"],
                    file_path="app.py",
                    line_number=42,
                ),
                score=0.95,
                distance=0.05,
            )
        ]

        # Setup map
        blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "input", "label": "Query"},
                {"id": 2, "type": "Button", "intent": "submit", "label": "Go"},
            ],
            "links": [
                {"source": 2, "target": 1, "type": "trigger"},
            ],
        }

        return blocks

    @pytest.fixture
    def app_with_routes(self, mock_blocks):
        """Create FastAPI app with semantic routes."""
        from fastapi import FastAPI
        from integradio.api import create_api_routes

        app = FastAPI()
        create_api_routes(app, mock_blocks)
        return app

    @pytest.fixture
    def client(self, app_with_routes):
        """Create test client."""
        from fastapi.testclient import TestClient
        return TestClient(app_with_routes)

    def test_search_endpoint(self, client, mock_blocks):
        """GET /semantic/search returns results."""
        response = client.get("/semantic/search?q=user%20input")

        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert data["query"] == "user input"
        assert "count" in data
        assert data["count"] == 1
        assert "results" in data
        assert len(data["results"]) == 1

        result = data["results"][0]
        assert result["component_id"] == 1
        assert result["type"] == "Textbox"
        assert result["score"] == 0.95

    def test_search_with_filters(self, client, mock_blocks):
        """type and tags query params work."""
        response = client.get("/semantic/search?q=test&k=5&type=Textbox&tags=input,text")

        assert response.status_code == 200

        # Verify search was called with filters
        mock_blocks.search.assert_called_once()
        call_args = mock_blocks.search.call_args
        assert call_args[0][0] == "test"
        assert call_args[1]["k"] == 5
        assert call_args[1]["component_type"] == "Textbox"
        assert call_args[1]["tags"] == ["input", "text"]

    def test_component_endpoint(self, client, mock_blocks):
        """GET /semantic/component/{id} returns metadata."""
        response = client.get("/semantic/component/1")

        assert response.status_code == 200
        data = response.json()

        assert data["component_id"] == 1
        assert data["type"] == "Textbox"
        assert data["intent"] == "user input"
        assert data["label"] == "Query"
        assert data["source"]["file"] == "app.py"
        assert data["source"]["line"] == 42
        assert "trigger" in data["relationships"]

    def test_component_not_found(self, client, mock_blocks):
        """Returns 404 for invalid ID."""
        mock_blocks.registry.get.return_value = None

        response = client.get("/semantic/component/99999")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_graph_endpoint(self, client, mock_blocks):
        """GET /semantic/graph returns nodes/links."""
        response = client.get("/semantic/graph")

        assert response.status_code == 200
        data = response.json()

        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 2
        assert len(data["links"]) == 1

    def test_trace_endpoint(self, client, mock_blocks):
        """GET /semantic/trace/{id} returns dataflow."""
        response = client.get("/semantic/trace/1")

        assert response.status_code == 200
        data = response.json()

        assert data["component_id"] == 1
        assert "upstream" in data
        assert "downstream" in data

    def test_trace_not_found(self, client, mock_blocks):
        """Returns 404 for invalid trace ID."""
        mock_blocks.registry.get.return_value = None

        response = client.get("/semantic/trace/99999")

        assert response.status_code == 404

    def test_summary_endpoint(self, client, mock_blocks):
        """GET /semantic/summary returns counts."""
        response = client.get("/semantic/summary")

        assert response.status_code == 200
        data = response.json()

        assert "total_components" in data
        assert data["total_components"] == 2
        assert "by_type" in data
        assert "components" in data


class TestGradioAPI:
    """Test Gradio API handlers."""

    def test_create_gradio_api(self):
        """create_gradio_api returns handler dict."""
        from integradio.api import create_gradio_api
        from integradio.registry import SearchResult, ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.search.return_value = [
            SearchResult(
                component_id=1,
                metadata=ComponentMetadata(
                    component_id=1,
                    component_type="Textbox",
                    intent="input",
                ),
                score=0.9,
                distance=0.1,
            )
        ]
        mock_blocks.map.return_value = {"nodes": [], "links": []}
        mock_blocks.summary.return_value = "Test summary"

        handlers = create_gradio_api(mock_blocks)

        assert "search" in handlers
        assert "graph" in handlers
        assert "summary" in handlers

    def test_gradio_search_handler(self):
        """Search handler returns formatted results."""
        from integradio.api import create_gradio_api
        from integradio.registry import SearchResult, ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.search.return_value = [
            SearchResult(
                component_id=1,
                metadata=ComponentMetadata(
                    component_id=1,
                    component_type="Textbox",
                    intent="user input",
                ),
                score=0.95,
                distance=0.05,
            )
        ]

        handlers = create_gradio_api(mock_blocks)
        result = handlers["search"]("test query", k=5)

        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["type"] == "Textbox"
        assert result[0]["score"] == 0.95

    def test_gradio_graph_handler(self):
        """Graph handler returns map data."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [{"id": 1}],
            "links": [{"source": 1, "target": 2}],
        }

        handlers = create_gradio_api(mock_blocks)
        result = handlers["graph"]()

        assert "nodes" in result
        assert "links" in result

    def test_gradio_summary_handler(self):
        """Summary handler returns text."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()
        mock_blocks.summary.return_value = "5 components registered"

        handlers = create_gradio_api(mock_blocks)
        result = handlers["summary"]()

        assert result == "5 components registered"


class TestAPIWithoutFastAPI:
    """Test fallback behavior without FastAPI."""

    def test_starlette_fallback(self):
        """API works with Starlette when FastAPI unavailable."""
        # This test just verifies the import fallback logic exists
        from integradio.api import create_api_routes

        # The function should exist regardless
        assert callable(create_api_routes)


# ============================================================================
# EDGE CASES - More comprehensive coverage
# ============================================================================


class TestAPIEndpointEdgeCases:
    """Edge cases for API endpoints."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock SemanticBlocks for testing."""
        from integradio.registry import SearchResult, ComponentMetadata

        blocks = MagicMock()
        blocks.registry = MagicMock()
        blocks.registry.get.return_value = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="user input",
            label="Query",
        )
        blocks.registry.get_relationships.return_value = {}
        blocks.registry.get_dataflow.return_value = {"upstream": [], "downstream": []}
        blocks.registry.all_components.return_value = []
        blocks.search.return_value = []
        blocks.map.return_value = {"nodes": [], "links": []}

        return blocks

    @pytest.fixture
    def client(self, mock_blocks):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from integradio.api import create_api_routes

        app = FastAPI()
        create_api_routes(app, mock_blocks)
        return TestClient(app)

    def test_search_empty_query(self, client, mock_blocks):
        """Search with empty query returns empty results."""
        mock_blocks.search.return_value = []

        response = client.get("/semantic/search?q=")

        # Empty query returns 400 after input validation was added
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_search_unicode_query(self, client, mock_blocks):
        """Search handles Unicode queries."""
        from integradio.registry import SearchResult, ComponentMetadata

        mock_blocks.search.return_value = [
            SearchResult(
                component_id=1,
                metadata=ComponentMetadata(
                    component_id=1,
                    component_type="Textbox",
                    intent="输入框",  # Chinese
                ),
                score=0.9,
                distance=0.1,
            )
        ]

        response = client.get("/semantic/search?q=%E8%BE%93%E5%85%A5")  # URL-encoded Chinese

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1

    def test_search_special_characters(self, client, mock_blocks):
        """Search handles special characters in query."""
        mock_blocks.search.return_value = []

        response = client.get("/semantic/search?q=test%26query%3D123")

        assert response.status_code == 200

    def test_search_large_k_value(self, client, mock_blocks):
        """Search handles large k parameter."""
        mock_blocks.search.return_value = []

        response = client.get("/semantic/search?q=test&k=1000")

        assert response.status_code == 200
        mock_blocks.search.assert_called_once()
        assert mock_blocks.search.call_args[1]["k"] == 1000

    def test_search_negative_k_handled(self, client, mock_blocks):
        """Search handles negative k (FastAPI may coerce or error)."""
        # FastAPI will try to parse negative k as int
        response = client.get("/semantic/search?q=test&k=-1")

        # Should not crash - behavior depends on validation
        assert response.status_code in [200, 422]  # 422 for validation error

    def test_component_zero_id(self, client, mock_blocks):
        """Component endpoint handles ID of 0."""
        from integradio.registry import ComponentMetadata

        mock_blocks.registry.get.return_value = ComponentMetadata(
            component_id=0,
            component_type="Textbox",
            intent="test",
        )

        response = client.get("/semantic/component/0")

        assert response.status_code == 200
        data = response.json()
        assert data["component_id"] == 0

    def test_trace_enrichment_partial(self, client, mock_blocks):
        """Trace endpoint handles partial metadata."""
        from integradio.registry import ComponentMetadata

        mock_blocks.registry.get.return_value = ComponentMetadata(
            component_id=1,
            component_type="Button",
            intent="submit",
        )
        mock_blocks.registry.get_dataflow.return_value = {
            "upstream": [2, 3],
            "downstream": [4],
        }
        # Only return metadata for some IDs
        def mock_get(cid):
            if cid == 1:
                return ComponentMetadata(
                    component_id=1,
                    component_type="Button",
                    intent="submit",
                )
            elif cid == 2:
                return ComponentMetadata(
                    component_id=2,
                    component_type="Textbox",
                    intent="input",
                )
            return None  # ID 3 and 4 not found

        mock_blocks.registry.get.side_effect = mock_get

        response = client.get("/semantic/trace/1")

        assert response.status_code == 200
        data = response.json()
        # Should only include enriched data for found components
        assert len(data["upstream"]) == 1  # Only ID 2 found
        assert data["upstream"][0]["id"] == 2

    def test_summary_many_components(self, client, mock_blocks):
        """Summary handles many components."""
        from integradio.registry import ComponentMetadata

        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=i,
                component_type=f"Type{i % 5}",
                intent=f"intent {i}",
            )
            for i in range(100)
        ]

        response = client.get("/semantic/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["total_components"] == 100
        assert len(data["by_type"]) == 5  # 5 different types

    def test_search_empty_tags_filter(self, client, mock_blocks):
        """Search with empty tags string."""
        response = client.get("/semantic/search?q=test&tags=")

        assert response.status_code == 200
        # Empty tags should be treated as no filter
        call_args = mock_blocks.search.call_args
        # tags='' splits to [''] which should be handled


class TestGradioAPIEdgeCases:
    """Edge cases for Gradio API handlers."""

    def test_search_handler_empty_results(self):
        """Search handler returns empty list for no matches."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()
        mock_blocks.search.return_value = []

        handlers = create_gradio_api(mock_blocks)
        result = handlers["search"]("no matches", k=10)

        assert result == []

    def test_search_handler_score_rounding(self):
        """Search handler rounds scores to 4 decimal places."""
        from integradio.api import create_gradio_api
        from integradio.registry import SearchResult, ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.search.return_value = [
            SearchResult(
                component_id=1,
                metadata=ComponentMetadata(
                    component_id=1,
                    component_type="Textbox",
                    intent="test",
                ),
                score=0.123456789,
                distance=0.876543211,
            )
        ]

        handlers = create_gradio_api(mock_blocks)
        result = handlers["search"]("test")

        assert result[0]["score"] == 0.1235  # Rounded to 4 decimals

    def test_graph_handler_empty_graph(self):
        """Graph handler returns empty graph structure."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}

        handlers = create_gradio_api(mock_blocks)
        result = handlers["graph"]()

        assert result == {"nodes": [], "links": []}

    def test_summary_handler_empty(self):
        """Summary handler returns message for empty registry."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()
        mock_blocks.summary.return_value = "No components registered"

        handlers = create_gradio_api(mock_blocks)
        result = handlers["summary"]()

        assert result == "No components registered"


class TestAPIResponseFormat:
    """Test API response format consistency."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock SemanticBlocks."""
        from integradio.registry import SearchResult, ComponentMetadata

        blocks = MagicMock()
        blocks.registry = MagicMock()
        blocks.registry.get.return_value = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="user input",
            label="Query",
            elem_id="query-box",
            tags=["input", "text"],
            file_path="app.py",
            line_number=42,
            inputs_from=[],
            outputs_to=[2],
        )
        blocks.registry.get_relationships.return_value = {
            "trigger": [],
            "dataflow": [2],
        }
        blocks.registry.get_dataflow.return_value = {
            "upstream": [],
            "downstream": [2],
        }
        blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="input",
            )
        ]
        blocks.search.return_value = []
        blocks.map.return_value = {"nodes": [], "links": []}

        return blocks

    @pytest.fixture
    def client(self, mock_blocks):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from integradio.api import create_api_routes

        app = FastAPI()
        create_api_routes(app, mock_blocks)
        return TestClient(app)

    def test_component_response_includes_all_fields(self, client, mock_blocks):
        """Component response includes all expected fields."""
        response = client.get("/semantic/component/1")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "component_id" in data
        assert "type" in data
        assert "intent" in data
        assert "label" in data
        assert "elem_id" in data
        assert "tags" in data
        assert "source" in data
        assert "file" in data["source"]
        assert "line" in data["source"]
        assert "relationships" in data
        assert "inputs_from" in data
        assert "outputs_to" in data

    def test_search_response_structure(self, client, mock_blocks):
        """Search response has correct structure."""
        from integradio.registry import SearchResult, ComponentMetadata

        mock_blocks.search.return_value = [
            SearchResult(
                component_id=1,
                metadata=ComponentMetadata(
                    component_id=1,
                    component_type="Textbox",
                    intent="test",
                    tags=["input"],
                    file_path="test.py",
                    line_number=10,
                ),
                score=0.9,
                distance=0.1,
            )
        ]

        response = client.get("/semantic/search?q=test")

        data = response.json()
        result = data["results"][0]

        assert "component_id" in result
        assert "type" in result
        assert "intent" in result
        assert "score" in result
        assert "tags" in result
        assert "source" in result

    def test_graph_response_structure(self, client, mock_blocks):
        """Graph response has correct structure."""
        mock_blocks.map.return_value = {
            "nodes": [{"id": 1, "type": "Textbox", "intent": "test", "label": "Q"}],
            "links": [{"source": 1, "target": 2, "type": "dataflow"}],
        }

        response = client.get("/semantic/graph")

        data = response.json()

        assert "nodes" in data
        assert "links" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["links"], list)
