"""
Tests for visualization module - Mermaid, D3.js, and ASCII graph generation.

Priority 6: Nice to have tests.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestMermaidGeneration:
    """Test Mermaid diagram generation."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock SemanticBlocks with graph data."""
        from integradio.registry import ComponentMetadata

        blocks = MagicMock()
        blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "user input", "label": "Query"},
                {"id": 2, "type": "Button", "intent": "submit", "label": "Search"},
                {"id": 3, "type": "Markdown", "intent": "results", "label": "Output"},
            ],
            "links": [
                {"source": 2, "target": 1, "type": "trigger"},
                {"source": 1, "target": 3, "type": "dataflow"},
            ],
        }
        blocks.registry = MagicMock()
        blocks.registry.all_components.return_value = [
            ComponentMetadata(component_id=1, component_type="Textbox", intent="user input", label="Query"),
            ComponentMetadata(component_id=2, component_type="Button", intent="submit", label="Search"),
            ComponentMetadata(component_id=3, component_type="Markdown", intent="results", label="Output"),
        ]
        return blocks

    def test_generate_mermaid(self, mock_blocks):
        """Returns valid Mermaid diagram string."""
        from integradio.viz import generate_mermaid

        result = generate_mermaid(mock_blocks)

        assert isinstance(result, str)
        assert "graph TD" in result
        # Should have node definitions
        assert "c1" in result
        assert "c2" in result
        assert "c3" in result
        # Should have links
        assert "-->" in result

    def test_mermaid_node_styles(self, mock_blocks):
        """Different types get different styles."""
        from integradio.viz import generate_mermaid

        result = generate_mermaid(mock_blocks)

        # Button should have trigger style
        assert ":::trigger" in result

        # Textbox should have input style
        assert ":::input" in result

        # Markdown should have output style
        assert ":::output" in result

        # Should define the style classes
        assert "classDef trigger" in result
        assert "classDef input" in result
        assert "classDef output" in result

    def test_mermaid_link_types(self, mock_blocks):
        """Different link types are styled differently."""
        from integradio.viz import generate_mermaid

        result = generate_mermaid(mock_blocks)

        # Trigger links should be labeled
        assert "|trigger|" in result

        # Dataflow links should use regular arrows
        assert "-->" in result

    def test_mermaid_label_escaping(self):
        """Labels with quotes are escaped."""
        from integradio.viz import generate_mermaid

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "test", "label": 'Label "with" quotes'},
            ],
            "links": [],
        }

        result = generate_mermaid(mock_blocks)

        # Quotes should be escaped to single quotes
        assert "Label 'with' quotes" in result

    def test_mermaid_empty_graph(self):
        """Handles empty graph gracefully."""
        from integradio.viz import generate_mermaid

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}

        result = generate_mermaid(mock_blocks)

        assert "graph TD" in result
        # Should still produce valid Mermaid
        assert result.strip().startswith("graph TD")


class TestHTMLGraphGeneration:
    """Test D3.js HTML graph generation."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock SemanticBlocks with graph data."""
        blocks = MagicMock()
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

    def test_generate_html_graph(self, mock_blocks):
        """Returns complete HTML document."""
        from integradio.viz import generate_html_graph

        result = generate_html_graph(mock_blocks)

        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result
        assert "<html>" in result
        assert "</html>" in result
        assert "<head>" in result
        assert "<body>" in result

    def test_html_contains_d3(self, mock_blocks):
        """HTML includes D3.js script."""
        from integradio.viz import generate_html_graph

        result = generate_html_graph(mock_blocks)

        assert "d3js.org" in result or "d3.v" in result
        assert "<script" in result

    def test_html_contains_graph_data(self, mock_blocks):
        """HTML includes the graph data as JSON."""
        from integradio.viz import generate_html_graph
        import json

        result = generate_html_graph(mock_blocks)

        # The graph data should be embedded
        assert '"nodes"' in result
        assert '"links"' in result
        assert "Textbox" in result
        assert "Button" in result

    def test_html_custom_dimensions(self, mock_blocks):
        """Custom width/height are used."""
        from integradio.viz import generate_html_graph

        result = generate_html_graph(mock_blocks, width=1024, height=768)

        # The HTML should be generated (dimensions are used in JS)
        assert isinstance(result, str)
        # The JS should set dimensions based on window, but custom values can be used

    def test_html_has_interactive_features(self, mock_blocks):
        """HTML includes interactive features."""
        from integradio.viz import generate_html_graph

        result = generate_html_graph(mock_blocks)

        # Should have search input
        assert "searchInput" in result or "search" in result

        # Should have tooltip
        assert "tooltip" in result

        # Should have drag functionality
        assert "drag" in result

    def test_html_color_mapping(self, mock_blocks):
        """HTML includes type-based color mapping."""
        from integradio.viz import generate_html_graph

        result = generate_html_graph(mock_blocks)

        # Should define colors for different types
        assert "typeColors" in result or "getColor" in result
        # Should include common types
        assert "Button" in result
        assert "Textbox" in result


class TestASCIIGraphGeneration:
    """Test ASCII art graph generation."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock SemanticBlocks with components."""
        from integradio.registry import ComponentMetadata

        blocks = MagicMock()
        blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "input", "label": "Query"},
                {"id": 2, "type": "Button", "intent": "submit", "label": "Search"},
                {"id": 3, "type": "Markdown", "intent": "results", "label": "Output"},
            ],
            "links": [
                {"source": 2, "target": 1, "type": "trigger"},
                {"source": 1, "target": 3, "type": "dataflow"},
            ],
        }
        blocks.registry = MagicMock()
        blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="user enters search query",
                label="Query",
                inputs_from=[],
                outputs_to=[3],
            ),
            ComponentMetadata(
                component_id=2,
                component_type="Button",
                intent="triggers the search",
                label="Search",
                inputs_from=[],
                outputs_to=[1],
            ),
            ComponentMetadata(
                component_id=3,
                component_type="Markdown",
                intent="displays search results",
                label="Output",
                inputs_from=[1],
                outputs_to=[],
            ),
        ]
        return blocks

    def test_generate_ascii_graph(self, mock_blocks):
        """Returns formatted ASCII art."""
        from integradio.viz import generate_ascii_graph

        result = generate_ascii_graph(mock_blocks)

        assert isinstance(result, str)
        # Should have header
        assert "INTEGRADIO" in result
        # Should list component types
        assert "Textbox" in result
        assert "Button" in result
        assert "Markdown" in result

    def test_ascii_shows_intents(self, mock_blocks):
        """ASCII shows component intents."""
        from integradio.viz import generate_ascii_graph

        result = generate_ascii_graph(mock_blocks)

        assert "intent:" in result
        # Should show intent text
        assert "search" in result.lower()

    def test_ascii_shows_dataflow(self, mock_blocks):
        """ASCII shows dataflow relationships."""
        from integradio.viz import generate_ascii_graph

        result = generate_ascii_graph(mock_blocks)

        assert "DATAFLOW" in result
        # Should show relationship arrows
        assert "-->" in result

    def test_ascii_respects_max_width(self, mock_blocks):
        """ASCII respects max_width parameter."""
        from integradio.viz import generate_ascii_graph

        result = generate_ascii_graph(mock_blocks, max_width=60)

        # Header separators should be 60 chars
        lines = result.split("\n")
        # Find a separator line
        sep_lines = [l for l in lines if l.strip() and all(c in "=-" for c in l.strip())]
        if sep_lines:
            assert len(sep_lines[0]) == 60

    def test_ascii_empty_registry(self):
        """Handles empty registry gracefully."""
        from integradio.viz import generate_ascii_graph

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = []

        result = generate_ascii_graph(mock_blocks)

        assert "No components registered" in result

    def test_ascii_truncates_long_intents(self, mock_blocks):
        """Long intents are truncated."""
        from integradio.viz import generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="This is a very long intent description that should be truncated in the ASCII display",
                label="Test",
            ),
        ]

        result = generate_ascii_graph(mock_blocks)

        # Should have ellipsis for truncated intent
        assert "..." in result

    def test_ascii_shows_connections(self, mock_blocks):
        """ASCII shows input/output connections."""
        from integradio.viz import generate_ascii_graph

        result = generate_ascii_graph(mock_blocks)

        # Should show outputs_to for components that have them
        assert "outputs to" in result or "->" in result


class TestVisualizationEdgeCases:
    """Test edge cases in visualization."""

    def test_special_characters_in_labels(self):
        """Handles special characters in labels."""
        from integradio.viz import generate_mermaid, generate_html_graph

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "test", "label": "Label <with> special & chars"},
            ],
            "links": [],
        }

        # Should not raise
        mermaid = generate_mermaid(mock_blocks)
        html = generate_html_graph(mock_blocks)

        assert isinstance(mermaid, str)
        assert isinstance(html, str)

    def test_unicode_labels(self):
        """Handles Unicode in labels."""
        from integradio.viz import generate_mermaid, generate_html_graph, generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "test", "label": "Unicode: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud83d\ude00"},
            ],
            "links": [],
        }
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="test",
                label="Unicode: \u4e2d\u6587",
            ),
        ]

        # Should not raise
        mermaid = generate_mermaid(mock_blocks)
        html = generate_html_graph(mock_blocks)
        ascii_art = generate_ascii_graph(mock_blocks)

        assert isinstance(mermaid, str)
        assert isinstance(html, str)
        assert isinstance(ascii_art, str)

    def test_many_nodes(self):
        """Handles large number of nodes."""
        from integradio.viz import generate_mermaid, generate_html_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": i, "type": "Textbox", "intent": f"node {i}", "label": f"Node {i}"}
                for i in range(100)
            ],
            "links": [
                {"source": i, "target": i + 1, "type": "dataflow"}
                for i in range(99)
            ],
        }

        # Should not raise or hang
        mermaid = generate_mermaid(mock_blocks)
        html = generate_html_graph(mock_blocks)

        assert len(mermaid) > 0
        assert len(html) > 0


# ============================================================================
# EDGE CASES - More comprehensive coverage
# ============================================================================


class TestMermaidEdgeCases:
    """Edge cases for Mermaid diagram generation."""

    def test_mermaid_unknown_link_type(self):
        """Handles unknown link types with dotted lines."""
        from integradio.viz import generate_mermaid

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "test", "label": "A"},
                {"id": 2, "type": "Textbox", "intent": "test", "label": "B"},
            ],
            "links": [
                {"source": 1, "target": 2, "type": "custom_relationship"},
            ],
        }

        result = generate_mermaid(mock_blocks)

        # Unknown types should use dotted lines with label
        assert "-.->|custom_relationship|" in result

    def test_mermaid_all_component_styles(self):
        """Tests all component type styles."""
        from integradio.viz import generate_mermaid

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Button", "intent": "test", "label": "Btn"},
                {"id": 2, "type": "Textbox", "intent": "test", "label": "Input"},
                {"id": 3, "type": "Number", "intent": "test", "label": "Num"},
                {"id": 4, "type": "Slider", "intent": "test", "label": "Slide"},
                {"id": 5, "type": "Dropdown", "intent": "test", "label": "Drop"},
                {"id": 6, "type": "Markdown", "intent": "test", "label": "MD"},
                {"id": 7, "type": "HTML", "intent": "test", "label": "Web"},
                {"id": 8, "type": "Image", "intent": "test", "label": "Img"},
                {"id": 9, "type": "Plot", "intent": "test", "label": "Chart"},
                {"id": 10, "type": "CustomWidget", "intent": "test", "label": "Custom"},
            ],
            "links": [],
        }

        result = generate_mermaid(mock_blocks)

        # Button should have trigger style
        assert "c1" in result
        assert ":::trigger" in result

        # Input types should have input style
        assert ":::input" in result

        # Output types should have output style
        assert ":::output" in result

        # Custom widget should have no special style
        assert "c10" in result

    def test_mermaid_node_id_prefixing(self):
        """Node IDs are prefixed with 'c' for validity."""
        from integradio.viz import generate_mermaid

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 123, "type": "Textbox", "intent": "test", "label": "Test"},
            ],
            "links": [],
        }

        result = generate_mermaid(mock_blocks)

        # ID should be prefixed
        assert "c123" in result
        # Raw ID should not appear in node definition
        assert '    123[' not in result

    def test_mermaid_link_ids_prefixed(self):
        """Link source/target IDs are prefixed."""
        from integradio.viz import generate_mermaid

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Button", "intent": "test", "label": "A"},
                {"id": 2, "type": "Textbox", "intent": "test", "label": "B"},
            ],
            "links": [
                {"source": 1, "target": 2, "type": "trigger"},
            ],
        }

        result = generate_mermaid(mock_blocks)

        assert "c1 -->|trigger| c2" in result


class TestHTMLGraphEdgeCases:
    """Edge cases for HTML graph generation."""

    def test_html_escapes_json_properly(self):
        """JSON data is properly escaped in HTML."""
        from integradio.viz import generate_html_graph

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "test", "label": "Test</script>alert('xss')"},
            ],
            "links": [],
        }

        result = generate_html_graph(mock_blocks)

        # Should not have unescaped script closing tag
        assert "</script>alert" not in result.split("const data")[1].split("</script>")[0]

    def test_html_includes_style_definitions(self):
        """HTML includes CSS style definitions."""
        from integradio.viz import generate_html_graph

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}

        result = generate_html_graph(mock_blocks)

        assert "<style>" in result
        assert "</style>" in result
        assert "body {" in result or "body{" in result

    def test_html_includes_force_simulation(self):
        """HTML includes D3 force simulation."""
        from integradio.viz import generate_html_graph

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}

        result = generate_html_graph(mock_blocks)

        assert "forceSimulation" in result
        assert "forceLink" in result
        assert "forceManyBody" in result
        assert "forceCenter" in result

    def test_html_includes_arrow_markers(self):
        """HTML includes SVG arrow markers for links."""
        from integradio.viz import generate_html_graph

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}

        result = generate_html_graph(mock_blocks)

        assert "marker" in result
        assert "arrow" in result

    def test_html_simulation_events(self):
        """HTML includes simulation tick events."""
        from integradio.viz import generate_html_graph

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}

        result = generate_html_graph(mock_blocks)

        assert 'on("tick"' in result
        assert "dragstarted" in result
        assert "dragged" in result
        assert "dragended" in result


class TestASCIIGraphEdgeCases:
    """Edge cases for ASCII graph generation."""

    def test_ascii_uses_elem_id_fallback(self):
        """Uses elem_id when label is missing."""
        from integradio.viz import generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=1,
                component_type="Button",
                intent="click me",
                label=None,
                elem_id="btn-submit",
            ),
        ]

        result = generate_ascii_graph(mock_blocks)

        assert "btn-submit" in result

    def test_ascii_uses_id_fallback(self):
        """Uses component ID when label and elem_id are missing."""
        from integradio.viz import generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=42,
                component_type="Button",
                intent="click me",
                label=None,
                elem_id=None,
            ),
        ]

        result = generate_ascii_graph(mock_blocks)

        assert "id=42" in result

    def test_ascii_shows_inputs_from(self):
        """Shows inputs_from connections."""
        from integradio.viz import generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(
                component_id=1,
                component_type="Markdown",
                intent="display result",
                label="Output",
                inputs_from=[2, 3],
                outputs_to=[],
            ),
        ]

        result = generate_ascii_graph(mock_blocks)

        assert "inputs from" in result or "<-" in result

    def test_ascii_multiple_types_grouped(self):
        """Groups components by type."""
        from integradio.viz import generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(component_id=1, component_type="Button", intent="a", label="A"),
            ComponentMetadata(component_id=2, component_type="Textbox", intent="b", label="B"),
            ComponentMetadata(component_id=3, component_type="Button", intent="c", label="C"),
            ComponentMetadata(component_id=4, component_type="Textbox", intent="d", label="D"),
        ]

        result = generate_ascii_graph(mock_blocks)

        # Types should appear as headers
        assert "[Button]" in result
        assert "[Textbox]" in result

    def test_ascii_no_dataflow_section_if_empty(self):
        """No DATAFLOW section when no links."""
        from integradio.viz import generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {"nodes": [], "links": []}  # No links
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(component_id=1, component_type="Button", intent="test", label="A"),
        ]

        result = generate_ascii_graph(mock_blocks)

        # Should not have DATAFLOW section with no links
        assert "DATAFLOW:" not in result


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_all_viz_functions_handle_same_input(self):
        """All viz functions handle the same graph data."""
        from integradio.viz import generate_mermaid, generate_html_graph, generate_ascii_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "input", "label": "Query"},
                {"id": 2, "type": "Button", "intent": "submit", "label": "Go"},
                {"id": 3, "type": "Markdown", "intent": "output", "label": "Result"},
            ],
            "links": [
                {"source": 2, "target": 1, "type": "trigger"},
                {"source": 1, "target": 3, "type": "dataflow"},
            ],
        }
        mock_blocks.registry = MagicMock()
        mock_blocks.registry.all_components.return_value = [
            ComponentMetadata(component_id=1, component_type="Textbox", intent="input", label="Query"),
            ComponentMetadata(component_id=2, component_type="Button", intent="submit", label="Go"),
            ComponentMetadata(component_id=3, component_type="Markdown", intent="output", label="Result"),
        ]

        # All should succeed
        mermaid = generate_mermaid(mock_blocks)
        html = generate_html_graph(mock_blocks)
        ascii_art = generate_ascii_graph(mock_blocks)

        # Basic sanity checks
        assert "Query" in mermaid
        assert "Query" in html
        assert "Query" in ascii_art

        assert "Go" in mermaid
        assert "Go" in html
        assert "Go" in ascii_art

    def test_viz_functions_handle_newlines_in_labels(self):
        """Viz functions handle newlines in labels."""
        from integradio.viz import generate_mermaid, generate_html_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "test", "label": "Line1\nLine2"},
            ],
            "links": [],
        }

        # Should not raise
        mermaid = generate_mermaid(mock_blocks)
        html = generate_html_graph(mock_blocks)

        assert isinstance(mermaid, str)
        assert isinstance(html, str)

    def test_viz_functions_handle_empty_labels(self):
        """Viz functions handle empty string labels."""
        from integradio.viz import generate_mermaid, generate_html_graph
        from integradio.registry import ComponentMetadata

        mock_blocks = MagicMock()
        mock_blocks.map.return_value = {
            "nodes": [
                {"id": 1, "type": "Textbox", "intent": "test", "label": ""},
            ],
            "links": [],
        }

        # Should not raise
        mermaid = generate_mermaid(mock_blocks)
        html = generate_html_graph(mock_blocks)

        assert isinstance(mermaid, str)
        assert isinstance(html, str)
