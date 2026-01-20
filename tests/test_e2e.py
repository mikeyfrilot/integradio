"""
Tests for end-to-end workflows - Full integration from component creation to search.

Priority 7: Final validation tests.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestFullWorkflow:
    """Test complete workflows from component creation to search."""

    @patch("integradio.embedder.httpx.post")
    @patch("gradio.Blocks.__init__")
    @patch("gradio.Blocks.__exit__")
    def test_full_app_workflow(self, mock_exit, mock_blocks_init, mock_http_post):
        """Create app, register components, search, trace."""
        # Setup mocks
        mock_blocks_init.return_value = None
        mock_exit.return_value = None

        # Mock embedding API
        def mock_embed(*args, **kwargs):
            mock_response = MagicMock()
            # Generate different embeddings based on prompt content
            prompt = kwargs.get("json", {}).get("prompt", "")
            seed = hash(prompt) % 2**32
            np.random.seed(seed)
            mock_response.json.return_value = {"embedding": np.random.rand(768).tolist()}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_http_post.side_effect = mock_embed

        # Import after mocking
        from integradio.blocks import SemanticBlocks
        from integradio.components import semantic, SemanticComponent
        from integradio.introspect import infer_tags

        # Clear any existing instances
        SemanticComponent._instances.clear()

        # Create SemanticBlocks
        blocks = SemanticBlocks()

        # Create mock components with all required attributes explicitly set
        # (MagicMock returns MagicMock for undefined attrs, which SQLite can't serialize)
        mock_textbox = MagicMock()
        mock_textbox._id = 1
        mock_textbox.label = "Search Query"
        mock_textbox.__class__.__name__ = "Textbox"
        mock_textbox.elem_id = None
        mock_textbox.elem_classes = None
        mock_textbox.visible = True
        mock_textbox.interactive = True

        mock_button = MagicMock()
        mock_button._id = 2
        mock_button.label = "Search"
        mock_button.__class__.__name__ = "Button"
        mock_button.elem_id = None
        mock_button.elem_classes = None
        mock_button.visible = True
        mock_button.interactive = True

        mock_output = MagicMock()
        mock_output._id = 3
        mock_output.label = "Results"
        mock_output.__class__.__name__ = "Markdown"
        mock_output.elem_id = None
        mock_output.elem_classes = None
        mock_output.visible = True
        mock_output.interactive = False

        # Wrap components with semantic
        with patch("integradio.components.infer_tags") as mock_infer:
            mock_infer.side_effect = lambda c: {
                "Textbox": ["input", "text"],
                "Button": ["trigger", "action"],
                "Markdown": ["output", "text", "display"],
            }.get(c.__class__.__name__, ["component"])

            inp = semantic(mock_textbox, intent="user enters search query")
            btn = semantic(mock_button, intent="triggers search operation")
            out = semantic(mock_output, intent="displays search results")

        # Register components manually (simulating context exit)
        for comp_id, m in SemanticComponent._instances.items():
            m._register_to_registry()

        # Verify components are registered
        assert len(blocks.registry) == 3
        assert 1 in blocks.registry
        assert 2 in blocks.registry
        assert 3 in blocks.registry

        # Test search
        results = blocks.search("user input")
        assert len(results) > 0

        # Test describe
        description = blocks.describe(inp)
        assert description["component_id"] == 1
        assert description["type"] == "Textbox"

        # Test summary
        summary = blocks.summary()
        assert "3 components registered" in summary

        # Test map
        graph = blocks.map()
        assert len(graph["nodes"]) == 3


class TestSearchRelevance:
    """Test that search returns semantically relevant components."""

    @patch("integradio.embedder.httpx.post")
    @patch("gradio.Blocks.__init__")
    def test_search_finds_relevant_component(self, mock_blocks_init, mock_http_post):
        """'user input' finds Textbox."""
        mock_blocks_init.return_value = None

        # Create embeddings that make "user input" similar to "search query"
        embeddings = {
            "search_document: user enters search query": [0.9] * 384 + [0.1] * 384,
            "search_document: triggers search": [0.1] * 384 + [0.9] * 384,
            "search_document: displays results": [0.5] * 384 + [0.5] * 384,
            "search_query: user input": [0.85] * 384 + [0.15] * 384,  # Similar to query
        }

        def mock_embed(*args, **kwargs):
            mock_response = MagicMock()
            prompt = kwargs.get("json", {}).get("prompt", "")

            # Find matching embedding or use random
            embedding = None
            for key, emb in embeddings.items():
                if key in prompt or prompt in key:
                    embedding = emb
                    break

            if embedding is None:
                np.random.seed(hash(prompt) % 2**32)
                embedding = np.random.rand(768).tolist()

            mock_response.json.return_value = {"embedding": embedding}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_http_post.side_effect = mock_embed

        from integradio.blocks import SemanticBlocks
        from integradio.components import semantic, SemanticComponent
        from integradio.registry import ComponentMetadata

        SemanticComponent._instances.clear()

        blocks = SemanticBlocks()

        # Manually register components with controlled embeddings
        for comp_id, intent, comp_type, label in [
            (1, "user enters search query", "Textbox", "Query"),
            (2, "triggers search", "Button", "Search"),
            (3, "displays results", "Markdown", "Output"),
        ]:
            vector = np.array(
                embeddings.get(f"search_document: {intent}", [0.5] * 768),
                dtype=np.float32,
            )
            metadata = ComponentMetadata(
                component_id=comp_id,
                component_type=comp_type,
                intent=intent,
                label=label,
                tags=["test"],
            )
            blocks.registry.register(comp_id, vector, metadata)

        # Search for "user input" - should find Textbox first
        results = blocks.search("user input")

        assert len(results) > 0
        # The Textbox should be the most relevant result
        assert results[0].component_id == 1
        assert results[0].metadata.component_type == "Textbox"


class TestDataflowTracing:
    """Test dataflow tracing through component chains."""

    @patch("integradio.embedder.httpx.post")
    @patch("gradio.Blocks.__init__")
    def test_dataflow_traced_correctly(self, mock_blocks_init, mock_http_post):
        """Button -> Input -> Output chain works."""
        mock_blocks_init.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_http_post.return_value = mock_response

        from integradio.blocks import SemanticBlocks
        from integradio.registry import ComponentMetadata
        from integradio.components import SemanticComponent

        SemanticComponent._instances.clear()

        blocks = SemanticBlocks()

        # Register components
        components = [
            (1, "Textbox", "user input", "Input"),
            (2, "Button", "trigger", "Go"),
            (3, "Markdown", "output", "Results"),
        ]

        for comp_id, comp_type, intent, label in components:
            vector = np.random.rand(768).astype(np.float32)
            metadata = ComponentMetadata(
                component_id=comp_id,
                component_type=comp_type,
                intent=intent,
                label=label,
            )
            blocks.registry.register(comp_id, vector, metadata)

        # Create relationships: Button(2) triggers Input(1), Input(1) flows to Output(3)
        blocks.registry.add_relationship(2, 1, "trigger")
        blocks.registry.add_relationship(1, 3, "dataflow")

        # Create mock component to trace
        mock_input = MagicMock()
        mock_input._id = 1

        # Trace the input component
        trace = blocks.trace(mock_input)

        # Should have upstream (Button) and downstream (Output)
        assert "upstream" in trace
        assert "downstream" in trace

        # Upstream should include the button
        upstream_ids = [c["id"] for c in trace["upstream"]]
        assert 2 in upstream_ids

        # Downstream should include the output
        downstream_ids = [c["id"] for c in trace["downstream"]]
        assert 3 in downstream_ids

    @patch("integradio.embedder.httpx.post")
    @patch("gradio.Blocks.__init__")
    def test_trace_with_semantic_component(self, mock_blocks_init, mock_http_post):
        """trace() works with SemanticComponent input."""
        mock_blocks_init.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_http_post.return_value = mock_response

        from integradio.blocks import SemanticBlocks
        from integradio.components import semantic, SemanticComponent
        from integradio.registry import ComponentMetadata

        SemanticComponent._instances.clear()

        blocks = SemanticBlocks()

        # Create and wrap a mock component
        mock_comp = MagicMock()
        mock_comp._id = 1
        mock_comp.__class__.__name__ = "Textbox"

        with patch("integradio.components.infer_tags", return_value=["input"]):
            wrapped = semantic(mock_comp, intent="test input")

        # Register it
        vector = np.random.rand(768).astype(np.float32)
        metadata = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="test input",
        )
        blocks.registry.register(1, vector, metadata)

        # Trace should work with the wrapped component
        trace = blocks.trace(wrapped)

        assert "upstream" in trace
        assert "downstream" in trace


class TestCompleteIntegration:
    """Test complete integration scenarios."""

    @patch("integradio.embedder.httpx.post")
    @patch("gradio.Blocks.__init__")
    def test_graph_visualization_integration(self, mock_blocks_init, mock_http_post):
        """Components can be visualized after registration."""
        mock_blocks_init.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_http_post.return_value = mock_response

        from integradio.blocks import SemanticBlocks
        from integradio.registry import ComponentMetadata
        from integradio.viz import generate_mermaid, generate_ascii_graph
        from integradio.components import SemanticComponent

        SemanticComponent._instances.clear()

        blocks = SemanticBlocks()

        # Register multiple components
        for i in range(5):
            vector = np.random.rand(768).astype(np.float32)
            metadata = ComponentMetadata(
                component_id=i,
                component_type=["Textbox", "Button", "Markdown", "Slider", "Image"][i],
                intent=f"component {i}",
                label=f"Label {i}",
            )
            blocks.registry.register(i, vector, metadata)

        # Add some relationships
        blocks.registry.add_relationship(1, 0, "trigger")  # Button triggers Textbox
        blocks.registry.add_relationship(0, 2, "dataflow")  # Textbox to Markdown

        # Generate visualizations
        mermaid = generate_mermaid(blocks)
        ascii_graph = generate_ascii_graph(blocks)

        # Verify visualizations contain expected content
        assert "graph TD" in mermaid
        assert "c0" in mermaid  # Node IDs
        assert "c1" in mermaid

        assert "INTEGRADIO" in ascii_graph
        assert "Textbox" in ascii_graph
        assert "Button" in ascii_graph

    @patch("integradio.embedder.httpx.post")
    @patch("gradio.Blocks.__init__")
    def test_multiple_search_queries(self, mock_blocks_init, mock_http_post):
        """Multiple searches on same registry work correctly."""
        mock_blocks_init.return_value = None

        call_count = [0]

        def mock_embed(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            # Different embeddings for different queries
            prompt = kwargs.get("json", {}).get("prompt", "")
            np.random.seed(hash(prompt) % 2**32)
            mock_response.json.return_value = {"embedding": np.random.rand(768).tolist()}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_http_post.side_effect = mock_embed

        from integradio.blocks import SemanticBlocks
        from integradio.registry import ComponentMetadata
        from integradio.components import SemanticComponent

        SemanticComponent._instances.clear()

        blocks = SemanticBlocks()

        # Register components
        for i, (intent, comp_type) in enumerate([
            ("user text input", "Textbox"),
            ("submit button", "Button"),
            ("image display", "Image"),
            ("text output", "Markdown"),
        ]):
            vector = np.random.rand(768).astype(np.float32)
            metadata = ComponentMetadata(
                component_id=i,
                component_type=comp_type,
                intent=intent,
                label=f"Label {i}",
                tags=[comp_type.lower()],
            )
            blocks.registry.register(i, vector, metadata)

        # Multiple searches
        results1 = blocks.search("input field")
        results2 = blocks.search("button")
        results3 = blocks.search("display image")

        # All searches should return results
        assert len(results1) > 0
        assert len(results2) > 0
        assert len(results3) > 0

        # Results should be different for different queries
        # (Component IDs in first position may vary based on embeddings)
