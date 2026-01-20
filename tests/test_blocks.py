"""
Tests for SemanticBlocks - Extended Gradio Blocks with registry integration.

Priority 3: Integration tests.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np


class TestSemanticBlocksInit:
    """Test SemanticBlocks initialization."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_blocks_creates_registry(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """SemanticBlocks has registry attribute."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        blocks = SemanticBlocks()

        mock_registry_cls.assert_called_once()
        assert hasattr(blocks, "_registry")
        assert hasattr(blocks, "registry")

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_blocks_creates_embedder(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """SemanticBlocks has embedder attribute."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        blocks = SemanticBlocks()

        mock_embedder_cls.assert_called_once()
        assert hasattr(blocks, "_embedder")
        assert hasattr(blocks, "embedder")

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_custom_config_passed(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Custom configuration is passed to dependencies."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from pathlib import Path

        db_path = Path("/custom/db.sqlite")
        cache_dir = Path("/custom/cache")

        blocks = SemanticBlocks(
            db_path=db_path,
            cache_dir=cache_dir,
            ollama_url="http://custom:8080",
            embed_model="custom-model",
        )

        # Verify registry was created with custom db_path
        mock_registry_cls.assert_called_once_with(db_path=db_path)

        # Verify embedder was created with custom config
        mock_embedder_cls.assert_called_once_with(
            model="custom-model",
            base_url="http://custom:8080",
            cache_dir=cache_dir,
        )


class TestContextManager:
    """Test context manager behavior."""

    @patch("integradio.blocks.extract_dataflow")
    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    @patch("gradio.Blocks.__exit__")
    def test_auto_register_on_exit(
        self, mock_exit, mock_blocks_init, mock_registry_cls,
        mock_embedder_cls, mock_extract_dataflow
    ):
        """Components registered when exiting context."""
        mock_blocks_init.return_value = None
        mock_exit.return_value = None
        mock_extract_dataflow.return_value = []

        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent

        # Create mock registry
        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()

        # Simulate a semantic component with _semantic_meta.embedded = False
        mock_component = MagicMock()
        mock_component._semantic_meta.embedded = False

        # Add to instances
        SemanticComponent._instances = {1: mock_component}

        # Call __exit__
        blocks.__exit__(None, None, None)

        # Verify registration was attempted
        mock_component._register_to_registry.assert_called_once()


class TestSearchMethods:
    """Test search functionality on SemanticBlocks."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_search_method(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.search() returns SearchResult list."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.registry import SearchResult, ComponentMetadata

        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        # Setup mock registry with search results
        mock_registry = MagicMock()
        mock_search_result = SearchResult(
            component_id=1,
            metadata=ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="user input",
            ),
            score=0.95,
            distance=0.05,
        )
        mock_registry.search.return_value = [mock_search_result]
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        results = blocks.search("find user input")

        assert len(results) == 1
        assert results[0].component_id == 1
        assert results[0].score == 0.95

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_find_single_component(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.find() returns most relevant component."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.registry import SearchResult, ComponentMetadata
        from integradio.components import SemanticComponent

        # Clear instance tracking
        SemanticComponent._instances.clear()

        # Setup embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        # Setup registry
        mock_registry = MagicMock()
        mock_search_result = SearchResult(
            component_id=42,
            metadata=ComponentMetadata(
                component_id=42,
                component_type="Button",
                intent="submit form",
            ),
            score=0.9,
            distance=0.1,
        )
        mock_registry.search.return_value = [mock_search_result]
        mock_registry_cls.return_value = mock_registry

        # Create a semantic component for ID 42
        mock_component = MagicMock()
        mock_component._id = 42
        SemanticComponent._instances[42] = MagicMock()
        SemanticComponent._instances[42].component = mock_component

        blocks = SemanticBlocks()
        result = blocks.find("submit button")

        # Should return the underlying component
        assert result is mock_component

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_find_returns_none_when_empty(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.find() returns None when no results."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.find("nonexistent")

        assert result is None


class TestTraceMethods:
    """Test dataflow tracing."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_trace_component(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.trace() returns upstream/downstream."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.registry import ComponentMetadata

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.get_dataflow.return_value = {
            "upstream": [1, 2],
            "downstream": [4, 5],
        }
        mock_registry.get.side_effect = lambda id: ComponentMetadata(
            component_id=id,
            component_type="Textbox",
            intent=f"component {id}",
            label=f"Label {id}",
        )
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()

        # Create mock component
        mock_component = MagicMock()
        mock_component._id = 3

        result = blocks.trace(mock_component)

        assert "upstream" in result
        assert "downstream" in result
        assert len(result["upstream"]) == 2
        assert len(result["downstream"]) == 2

        # Check enriched data
        assert result["upstream"][0]["id"] == 1
        assert result["upstream"][0]["type"] == "Textbox"


class TestMapMethods:
    """Test graph mapping functionality."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_map_returns_graph(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.map() returns nodes/links dict."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.export_graph.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "input", "label": "Input"},
                {"id": 2, "type": "Button", "intent": "submit", "label": "Submit"},
            ],
            "links": [
                {"source": 2, "target": 1, "type": "trigger"},
            ],
        }
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.map()

        assert "nodes" in result
        assert "links" in result
        assert len(result["nodes"]) == 2
        assert len(result["links"]) == 1

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_map_json(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.map_json() returns JSON string."""
        mock_blocks_init.return_value = None
        import json

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.export_graph.return_value = {"nodes": [], "links": []}
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.map_json()

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "nodes" in parsed
        assert "links" in parsed


class TestDescribeMethods:
    """Test component description."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_describe_component(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.describe() returns full metadata."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.registry import ComponentMetadata

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.get.return_value = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="user input",
            label="Query",
            elem_id="query-input",
            tags=["input", "text"],
            file_path="app.py",
            line_number=42,
        )
        mock_registry.get_relationships.return_value = {
            "trigger": [2],
            "dataflow": [3],
        }
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()

        mock_component = MagicMock()
        mock_component._id = 1

        result = blocks.describe(mock_component)

        assert result["component_id"] == 1
        assert result["type"] == "Textbox"
        assert result["intent"] == "user input"
        assert result["label"] == "Query"
        assert result["source"]["file"] == "app.py"
        assert result["source"]["line"] == 42
        assert "trigger" in result["relationships"]

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_describe_unregistered_component(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.describe() handles unregistered components."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()

        mock_component = MagicMock()
        mock_component._id = 999

        result = blocks.describe(mock_component)

        assert "error" in result


class TestSummary:
    """Test summary generation."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_summary(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.summary() returns formatted string."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.registry import ComponentMetadata

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.all_components.return_value = [
            ComponentMetadata(component_id=1, component_type="Textbox", intent="input", label="Query"),
            ComponentMetadata(component_id=2, component_type="Button", intent="submit", label="Go"),
            ComponentMetadata(component_id=3, component_type="Textbox", intent="output", label="Result"),
        ]
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.summary()

        assert "3 components registered" in result
        assert "Textbox (2)" in result
        assert "Button (1)" in result

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_summary_empty_registry(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """blocks.summary() handles empty registry."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.all_components.return_value = []
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.summary()

        assert "No components registered" in result


class TestDataflowRegistration:
    """Test dataflow relationship registration."""

    @patch("integradio.blocks.extract_dataflow")
    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    @patch("gradio.Blocks.__exit__")
    def test_click_creates_trigger_relationship(
        self, mock_exit, mock_blocks_init, mock_registry_cls,
        mock_embedder_cls, mock_extract_dataflow
    ):
        """btn.click() creates trigger link."""
        mock_blocks_init.return_value = None
        mock_exit.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent

        SemanticComponent._instances.clear()

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_embedder_cls.return_value = MagicMock()

        # Simulate extracted dataflow with trigger
        mock_extract_dataflow.return_value = [
            {
                "fn_id": 0,
                "triggers": [2],  # Button ID
                "inputs": [1],    # Textbox ID
                "outputs": [3],   # Output ID
            }
        ]

        blocks = SemanticBlocks()
        blocks.__exit__(None, None, None)

        # Verify trigger relationship was added
        calls = mock_registry.add_relationship.call_args_list
        trigger_calls = [c for c in calls if c[0][2] == "trigger"]

        assert len(trigger_calls) > 0
        # Button (2) triggers Textbox (1)
        assert any(c[0] == (2, 1, "trigger") for c in calls)

    @patch("integradio.blocks.extract_dataflow")
    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    @patch("gradio.Blocks.__exit__")
    def test_input_output_creates_dataflow(
        self, mock_exit, mock_blocks_init, mock_registry_cls,
        mock_embedder_cls, mock_extract_dataflow
    ):
        """inputs/outputs create dataflow links."""
        mock_blocks_init.return_value = None
        mock_exit.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent

        SemanticComponent._instances.clear()

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_embedder_cls.return_value = MagicMock()

        mock_extract_dataflow.return_value = [
            {
                "fn_id": 0,
                "triggers": [],
                "inputs": [1],
                "outputs": [3],
            }
        ]

        blocks = SemanticBlocks()
        blocks.__exit__(None, None, None)

        # Verify dataflow relationship was added
        calls = mock_registry.add_relationship.call_args_list
        dataflow_calls = [c for c in calls if c[0][2] == "dataflow"]

        assert len(dataflow_calls) > 0
        # Input (1) flows to Output (3)
        assert any(c[0] == (1, 3, "dataflow") for c in calls)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_search_empty_query(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Search with empty query still works."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.zeros(768, dtype=np.float32)
        mock_embedder_cls.return_value = mock_embedder

        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        results = blocks.search("")

        assert results == []
        mock_embedder.embed_query.assert_called_once_with("")

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_search_unicode_query(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Search with unicode query works."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        results = blocks.search("用户搜索 🔍")

        mock_embedder.embed_query.assert_called_once_with("用户搜索 🔍")

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_trace_component_without_id(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Trace raises ValueError for component without _id."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()
        mock_registry_cls.return_value = MagicMock()

        blocks = SemanticBlocks()

        # Component without _id
        mock_component = object()  # Plain object has no _id

        # After error handling improvements, returns error dict instead of raising
        result = blocks.trace(mock_component)
        assert "error" in result
        assert result["upstream"] == []
        assert result["downstream"] == []

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_describe_component_without_id(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Describe raises ValueError for component without _id."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()
        mock_registry_cls.return_value = MagicMock()

        blocks = SemanticBlocks()

        mock_component = object()

        # After error handling improvements, returns error dict instead of raising
        result = blocks.describe(mock_component)
        assert "error" in result

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_trace_with_semantic_component(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Trace works with SemanticComponent wrapper."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent
        from integradio.registry import ComponentMetadata

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.get_dataflow.return_value = {"upstream": [], "downstream": []}
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()

        # Create a SemanticComponent-like mock
        mock_semantic = MagicMock(spec=SemanticComponent)
        mock_semantic.component._id = 42

        result = blocks.trace(mock_semantic)

        mock_registry.get_dataflow.assert_called_once_with(42)

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_find_with_no_instance_returns_none(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Find returns None if instance not in tracker."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.registry import SearchResult, ComponentMetadata
        from integradio.components import SemanticComponent

        SemanticComponent._instances.clear()

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        mock_registry = MagicMock()
        mock_search_result = SearchResult(
            component_id=999,  # ID not in instances
            metadata=ComponentMetadata(
                component_id=999,
                component_type="Textbox",
                intent="test",
            ),
            score=0.9,
            distance=0.1,
        )
        mock_registry.search.return_value = [mock_search_result]
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.find("test")

        # No instance for ID 999
        assert result is None

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_auto_register_disabled(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """auto_register=False prevents automatic registration."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent

        SemanticComponent._instances.clear()

        mock_embedder_cls.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks(auto_register=False)

        # Add a mock semantic component
        mock_semantic = MagicMock()
        mock_semantic._semantic_meta.embedded = False
        SemanticComponent._instances[1] = mock_semantic

        # Call __exit__
        with patch("gradio.Blocks.__exit__", return_value=None):
            blocks.__exit__(None, None, None)

        # Registration should NOT have been called
        mock_semantic._register_to_registry.assert_not_called()


class TestPropertyAccess:
    """Test property accessors."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_registry_property(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """registry property returns the ComponentRegistry."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_embedder_cls.return_value = MagicMock()

        blocks = SemanticBlocks()

        assert blocks.registry is mock_registry

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_embedder_property(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """embedder property returns the Embedder."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder = MagicMock()
        mock_embedder_cls.return_value = mock_embedder
        mock_registry_cls.return_value = MagicMock()

        blocks = SemanticBlocks()

        assert blocks.embedder is mock_embedder


class TestSearchFilters:
    """Test search with various filters."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_search_with_type_filter(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Search passes component_type filter to registry."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        blocks.search("test", component_type="Button")

        # Verify filter was passed
        call_kwargs = mock_registry.search.call_args[1]
        assert call_kwargs["component_type"] == "Button"

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_search_with_tags_filter(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Search passes tags filter to registry."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        blocks.search("test", tags=["input", "text"])

        call_kwargs = mock_registry.search.call_args[1]
        assert call_kwargs["tags"] == ["input", "text"]

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_search_with_k_parameter(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """Search passes k parameter to registry."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        mock_embedder_cls.return_value = mock_embedder

        mock_registry = MagicMock()
        mock_registry.search.return_value = []
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        blocks.search("test", k=5)

        call_kwargs = mock_registry.search.call_args[1]
        assert call_kwargs["k"] == 5


class TestMapJsonOutput:
    """Test JSON output formatting."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_map_json_valid_json(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """map_json returns valid JSON."""
        mock_blocks_init.return_value = None
        import json

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.export_graph.return_value = {
            "nodes": [{"id": 1, "type": "Textbox"}],
            "links": [],
        }
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.map_json()

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["nodes"][0]["id"] == 1

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_map_json_indented(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """map_json returns indented JSON."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.export_graph.return_value = {"nodes": [], "links": []}
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()
        result = blocks.map_json()

        # Should be indented (contains newlines)
        assert "\n" in result


class TestGlobalReferences:
    """Test that SemanticBlocks sets global references correctly."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_sets_semantic_component_registry(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """SemanticBlocks sets SemanticComponent._registry."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_embedder_cls.return_value = MagicMock()

        blocks = SemanticBlocks()

        assert SemanticComponent._registry is mock_registry

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_sets_semantic_component_embedder(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """SemanticBlocks sets SemanticComponent._embedder."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.components import SemanticComponent

        mock_embedder = MagicMock()
        mock_embedder_cls.return_value = mock_embedder
        mock_registry_cls.return_value = MagicMock()

        blocks = SemanticBlocks()

        assert SemanticComponent._embedder is mock_embedder


class TestTraceEnrichment:
    """Test trace result enrichment."""

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_trace_enriches_with_metadata(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """trace() enriches IDs with component metadata."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks
        from integradio.registry import ComponentMetadata

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.get_dataflow.return_value = {
            "upstream": [1, 2],
            "downstream": [3],
        }

        def get_mock_meta(cid):
            return ComponentMetadata(
                component_id=cid,
                component_type=f"Type{cid}",
                intent=f"Intent {cid}",
                label=f"Label {cid}",
            )

        mock_registry.get.side_effect = get_mock_meta
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()

        mock_component = MagicMock()
        mock_component._id = 10

        result = blocks.trace(mock_component)

        # Verify enrichment
        assert len(result["upstream"]) == 2
        assert result["upstream"][0]["id"] == 1
        assert result["upstream"][0]["type"] == "Type1"
        assert result["upstream"][0]["intent"] == "Intent 1"
        assert result["upstream"][0]["label"] == "Label 1"

    @patch("integradio.blocks.Embedder")
    @patch("integradio.blocks.ComponentRegistry")
    @patch("gradio.Blocks.__init__")
    def test_trace_handles_missing_metadata(self, mock_blocks_init, mock_registry_cls, mock_embedder_cls):
        """trace() skips components with no metadata."""
        mock_blocks_init.return_value = None

        from integradio.blocks import SemanticBlocks

        mock_embedder_cls.return_value = MagicMock()

        mock_registry = MagicMock()
        mock_registry.get_dataflow.return_value = {
            "upstream": [1, 2, 3],
            "downstream": [],
        }
        # Only return metadata for ID 2
        mock_registry.get.side_effect = lambda cid: None if cid != 2 else MagicMock(
            component_type="Textbox",
            intent="test",
            label="Test",
        )
        mock_registry_cls.return_value = mock_registry

        blocks = SemanticBlocks()

        mock_component = MagicMock()
        mock_component._id = 10

        result = blocks.trace(mock_component)

        # Only component 2 should be in result
        assert len(result["upstream"]) == 1
        assert result["upstream"][0]["id"] == 2
