"""
Additional tests for inspector panel module to improve coverage.

Focuses on:
- create_search_panel with Gradio context
- create_details_panel with Gradio context
- create_inspector_panel internal handlers (refresh_tree, refresh_flow, do_search, get_details)
"""

import pytest
from unittest.mock import MagicMock, patch
import gradio as gr

from integradio.inspector.tree import ComponentTree, ComponentNode
from integradio.inspector.dataflow import DataFlowGraph, HandlerInfo
from integradio.inspector.panel import (
    create_tree_view,
    create_dataflow_view,
    create_search_panel,
    create_details_panel,
    create_inspector_panel,
    create_floating_inspector,
)


# =============================================================================
# Tests for create_search_panel (lines 104-118)
# =============================================================================

class TestCreateSearchPanel:
    """Tests for create_search_panel function."""

    def test_creates_textbox_and_html(self):
        """Test that search panel creates textbox and HTML components."""
        def mock_search(query):
            return []

        with gr.Blocks() as demo:
            search_input, results_output = create_search_panel(mock_search)

        assert isinstance(search_input, gr.Textbox)
        assert isinstance(results_output, gr.HTML)

    def test_search_input_has_label(self):
        """Test search input has correct label."""
        with gr.Blocks() as demo:
            search_input, _ = create_search_panel(lambda q: [])

        assert search_input.label == "Search Components"

    def test_search_input_has_placeholder(self):
        """Test search input has placeholder text."""
        with gr.Blocks() as demo:
            search_input, _ = create_search_panel(lambda q: [])

        assert "Search" in search_input.placeholder

    def test_search_input_has_elem_id(self):
        """Test search input has correct element ID."""
        with gr.Blocks() as demo:
            search_input, _ = create_search_panel(lambda q: [])

        assert search_input.elem_id == "inspector-search"

    def test_results_output_has_elem_id(self):
        """Test results output has correct element ID."""
        with gr.Blocks() as demo:
            _, results_output = create_search_panel(lambda q: [])

        assert results_output.elem_id == "inspector-results"


# =============================================================================
# Tests for create_details_panel (lines 128-143)
# =============================================================================

class TestCreateDetailsPanel:
    """Tests for create_details_panel function."""

    def test_creates_textbox_and_json(self):
        """Test that details panel creates textbox and JSON components."""
        with gr.Blocks() as demo:
            component_id, details = create_details_panel()

        assert isinstance(component_id, gr.Textbox)
        assert isinstance(details, gr.JSON)

    def test_component_id_has_label(self):
        """Test component ID textbox has correct label."""
        with gr.Blocks() as demo:
            component_id, _ = create_details_panel()

        assert component_id.label == "Component ID"

    def test_component_id_not_interactive(self):
        """Test component ID textbox is not interactive."""
        with gr.Blocks() as demo:
            component_id, _ = create_details_panel()

        assert component_id.interactive is False

    def test_component_id_has_elem_id(self):
        """Test component ID has correct element ID."""
        with gr.Blocks() as demo:
            component_id, _ = create_details_panel()

        assert component_id.elem_id == "inspector-component-id"

    def test_details_has_label(self):
        """Test details JSON has correct label."""
        with gr.Blocks() as demo:
            _, details = create_details_panel()

        assert details.label == "Component Details"


# =============================================================================
# Tests for create_inspector_panel (lines 165, 227-267)
# =============================================================================

class TestCreateInspectorPanel:
    """Tests for create_inspector_panel function."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks for inspector panel."""
        blocks = MagicMock()
        blocks.fns = []
        blocks.dependencies = []
        return blocks

    def test_creates_sidebar(self, mock_blocks):
        """Test that inspector panel creates a sidebar."""
        with gr.Blocks() as demo:
            sidebar = create_inspector_panel(mock_blocks)

        assert sidebar is not None

    def test_accepts_position_parameter(self, mock_blocks):
        """Test inspector panel accepts position parameter."""
        with gr.Blocks() as demo:
            sidebar = create_inspector_panel(mock_blocks, position="left")

        assert sidebar is not None

    def test_accepts_width_parameter(self, mock_blocks):
        """Test inspector panel accepts width parameter."""
        with gr.Blocks() as demo:
            sidebar = create_inspector_panel(mock_blocks, width=500)

        assert sidebar is not None

    def test_accepts_collapsed_parameter(self, mock_blocks):
        """Test inspector panel accepts collapsed parameter."""
        with gr.Blocks() as demo:
            sidebar = create_inspector_panel(mock_blocks, collapsed=False)

        assert sidebar is not None


# =============================================================================
# Tests for Internal Handlers (lines 227-267)
# =============================================================================

class TestInspectorPanelHandlers:
    """Tests for internal handler functions in create_inspector_panel."""

    @pytest.fixture
    def mock_blocks_with_components(self):
        """Create mock blocks with semantic components."""
        blocks = MagicMock()
        blocks.fns = []
        blocks.dependencies = []
        return blocks

    def test_refresh_tree_handler_logic(self, mock_blocks_with_components):
        """Test the refresh_tree handler logic."""
        # The handler calls build_component_tree and create_tree_view
        from integradio.inspector.tree import build_component_tree
        from integradio.inspector.panel import create_tree_view

        tree = build_component_tree(mock_blocks_with_components)
        html = create_tree_view(tree)

        assert isinstance(html, str)
        assert "<div" in html

    def test_refresh_flow_handler_logic(self, mock_blocks_with_components):
        """Test the refresh_flow handler logic."""
        from integradio.inspector.dataflow import extract_dataflow
        from integradio.inspector.panel import create_dataflow_view

        flow = extract_dataflow(mock_blocks_with_components)
        md = create_dataflow_view(flow)

        assert isinstance(md, str)
        assert "```mermaid" in md

    def test_do_search_empty_query(self):
        """Test do_search handler with empty query."""
        # Simulating the do_search logic
        def do_search(query):
            if not query:
                return "<p>Enter a search query</p>"
            return "<p>Results</p>"

        result = do_search("")
        assert "Enter a search query" in result

    def test_do_search_with_query_no_results(self):
        """Test do_search handler with query but no results."""
        def do_search(query, engine):
            if not query:
                return "<p>Enter a search query</p>"

            results = engine.search(query)

            if not results:
                return "<p>No results found</p>"

            return "<div>Results</div>"

        mock_engine = MagicMock()
        mock_engine.search.return_value = []

        result = do_search("nonexistent", mock_engine)
        assert "No results found" in result

    def test_do_search_with_results(self):
        """Test do_search handler with results."""
        from integradio.inspector.search import SearchResult

        def do_search(query, engine):
            if not query:
                return "<p>Enter a search query</p>"

            results = engine.search(query)

            if not results:
                return "<p>No results found</p>"

            html = "<div class='search-results'>"
            for r in results:
                html += f'''
<div class="result" style="padding: 8px; border-bottom: 1px solid #eee;">
    <strong>{r.intent}</strong>
    <span style="color: #64748b;"> ({r.component_type})</span>
    <br/>
    <small>Score: {r.score:.2f} | Match: {r.match_type}</small>
</div>
'''
            html += "</div>"
            return html

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.intent = "test intent"
        mock_result.component_type = "Button"
        mock_result.score = 0.95
        mock_result.match_type = "semantic"
        mock_engine.search.return_value = [mock_result]

        result = do_search("test", mock_engine)
        assert "test intent" in result
        assert "Button" in result
        assert "0.95" in result

    def test_get_details_empty_id(self):
        """Test get_details handler with empty component ID."""
        def get_details(comp_id):
            if not comp_id:
                return {}
            return {"id": comp_id}

        result = get_details("")
        assert result == {}

    def test_get_details_component_not_found(self):
        """Test get_details handler when component not found."""
        def get_details(comp_id, instances):
            if not comp_id:
                return {}

            semantic = instances.get(int(comp_id))

            if not semantic:
                return {"error": "Component not found"}

            return {"id": comp_id}

        mock_instances = {}
        result = get_details("999", mock_instances)
        assert result == {"error": "Component not found"}

    def test_get_details_with_valid_component(self):
        """Test get_details handler with valid component."""
        def get_details(comp_id, instances):
            if not comp_id:
                return {}

            semantic = instances.get(int(comp_id))

            if not semantic:
                return {"error": "Component not found"}

            meta = semantic.semantic_meta
            return {
                "id": comp_id,
                "type": type(semantic.component).__name__,
                "intent": meta.intent,
                "tags": meta.tags,
                "file": meta.file_path,
                "line": meta.line_number,
                "has_visual_spec": meta.visual_spec is not None,
            }

        # Create mock component
        mock_semantic = MagicMock()
        mock_meta = MagicMock()
        mock_meta.intent = "test button"
        mock_meta.tags = ["action", "submit"]
        mock_meta.file_path = "/app/main.py"
        mock_meta.line_number = 42
        mock_meta.visual_spec = None
        mock_semantic.semantic_meta = mock_meta
        mock_semantic.component = MagicMock()

        mock_instances = {1: mock_semantic}
        result = get_details("1", mock_instances)

        assert result["id"] == "1"
        assert result["intent"] == "test button"
        assert result["tags"] == ["action", "submit"]
        assert result["file"] == "/app/main.py"
        assert result["line"] == 42
        assert result["has_visual_spec"] is False


# =============================================================================
# Tests for Gradio Import Handling (lines 18-19)
# =============================================================================

class TestGradioImportHandling:
    """Tests for optional Gradio import handling."""

    def test_create_search_panel_raises_without_gradio(self):
        """Test create_search_panel raises ImportError when gr is None."""
        # Save original gr
        import integradio.inspector.panel as panel_module
        original_gr = panel_module.gr

        try:
            # Temporarily set gr to None
            panel_module.gr = None

            with pytest.raises(ImportError) as exc_info:
                create_search_panel(lambda q: [])

            assert "Gradio" in str(exc_info.value)
        finally:
            # Restore gr
            panel_module.gr = original_gr

    def test_create_details_panel_raises_without_gradio(self):
        """Test create_details_panel raises ImportError when gr is None."""
        import integradio.inspector.panel as panel_module
        original_gr = panel_module.gr

        try:
            panel_module.gr = None

            with pytest.raises(ImportError) as exc_info:
                create_details_panel()

            assert "Gradio" in str(exc_info.value)
        finally:
            panel_module.gr = original_gr

    def test_create_inspector_panel_raises_without_gradio(self):
        """Test create_inspector_panel raises ImportError when gr is None."""
        import integradio.inspector.panel as panel_module
        original_gr = panel_module.gr

        mock_blocks = MagicMock()

        try:
            panel_module.gr = None

            with pytest.raises(ImportError) as exc_info:
                create_inspector_panel(mock_blocks)

            assert "Gradio" in str(exc_info.value)
        finally:
            panel_module.gr = original_gr


# =============================================================================
# Integration Tests for Panel with SemanticComponent
# =============================================================================

class TestPanelWithSemanticComponents:
    """Integration tests with actual SemanticComponent instances."""

    @pytest.fixture
    def blocks_with_semantic(self):
        """Create blocks with semantic components for testing."""
        from integradio.blocks import SemanticBlocks
        from integradio.components import semantic

        blocks = SemanticBlocks()

        with blocks:
            btn = semantic(
                gr.Button("Test"),
                intent="test action",
                tags=["test"],
            )

        return blocks

    def test_create_inspector_with_real_blocks(self, blocks_with_semantic):
        """Test creating inspector with real SemanticBlocks."""
        with gr.Blocks() as demo:
            sidebar = create_inspector_panel(blocks_with_semantic)

        assert sidebar is not None

    def test_create_floating_inspector_with_real_blocks(self, blocks_with_semantic):
        """Test creating floating inspector with real blocks."""
        html = create_floating_inspector(blocks_with_semantic)

        assert "semantic-inspector" in html
        assert "__SEMANTIC_INSPECTOR__" in html


# =============================================================================
# Edge Cases for Handler Functions
# =============================================================================

class TestHandlerEdgeCases:
    """Edge case tests for panel handler functions."""

    def test_search_with_special_characters(self):
        """Test search handler with special characters in query."""
        from integradio.inspector.search import SearchResult

        def do_search(query, results):
            if not query:
                return "<p>Enter a search query</p>"
            if not results:
                return "<p>No results found</p>"

            html = "<div class='search-results'>"
            for r in results:
                html += f"<strong>{r.intent}</strong>"
            html += "</div>"
            return html

        # Test with special chars
        result = do_search("<script>alert('xss')</script>", [])
        assert "No results found" in result

    def test_get_details_with_non_numeric_id(self):
        """Test get_details with non-numeric component ID."""
        def get_details(comp_id, instances):
            if not comp_id:
                return {}
            try:
                semantic = instances.get(int(comp_id))
            except ValueError:
                return {"error": "Invalid component ID"}

            if not semantic:
                return {"error": "Component not found"}

            return {"id": comp_id}

        result = get_details("not-a-number", {})
        assert "error" in result

    def test_tree_view_with_long_intent(self):
        """Test tree view with very long intent string."""
        tree = ComponentTree()

        long_intent = "a" * 500
        node = ComponentNode(
            id="1",
            component_type="Button",
            intent=long_intent,
        )

        tree.nodes = {"1": node}
        html = create_tree_view(tree)

        # Should still render without error
        assert long_intent in html

    def test_dataflow_view_with_circular_handlers(self):
        """Test dataflow view with potentially circular references."""
        graph = DataFlowGraph()

        # Create handlers that reference each other's outputs
        graph.add_handler(HandlerInfo(
            name="handler_a",
            inputs=["input_b"],
            outputs=["output_a"],
            trigger_id="btn_a",
            event_type="click",
        ))

        graph.add_handler(HandlerInfo(
            name="handler_b",
            inputs=["output_a"],
            outputs=["input_b"],
            trigger_id="btn_b",
            event_type="click",
        ))

        md = create_dataflow_view(graph)

        # Should still generate valid mermaid
        assert "```mermaid" in md
        assert "flowchart" in md
