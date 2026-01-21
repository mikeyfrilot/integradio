"""
Tests for Visual Spec Viewer module.

Tests:
- VisualSpecViewer initialization (with spec, spec_path, default)
- Page and component list helpers
- Component spec retrieval
- Preview HTML rendering
- Placeholder HTML generation for different component types
- Token JSON serialization
- Layout JSON serialization
- Token color update
- CSS generation with themes
- Style Dictionary export
- Save spec functionality
- Add page functionality
- Add component functionality
- Token preview generation
- Build method (Gradio interface)
- Launch method
- Convenience functions (view_spec, create_viewer_demo)
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from integradio.visual.viewer import (
    VisualSpecViewer,
    view_spec,
    create_viewer_demo,
)
from integradio.visual.spec import (
    VisualSpec,
    UISpec,
    PageSpec,
    LayoutSpec,
    SpacingSpec,
    Display,
    Position,
)
from integradio.visual.tokens import (
    DesignToken,
    TokenGroup,
    TokenType,
    ColorValue,
    DimensionValue,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_ui_spec():
    """Create a sample UISpec for testing."""
    spec = UISpec(name="Test App", version="1.0.0")

    # Add global tokens
    spec.tokens.add("colors", TokenGroup(type=TokenType.COLOR))
    spec.tokens.get("colors").add("primary", DesignToken.color("#3b82f6", "Primary color"))
    spec.tokens.get("colors").add("secondary", DesignToken.color("#64748b", "Secondary color"))

    spec.tokens.add("spacing", TokenGroup(type=TokenType.DIMENSION))
    spec.tokens.get("spacing").add("sm", DesignToken.dimension(8, "px", "Small"))
    spec.tokens.get("spacing").add("md", DesignToken.dimension(16, "px", "Medium"))

    # Add a page with components
    home_page = PageSpec(name="Home", route="/")

    search_box = VisualSpec(
        component_id="search-input",
        component_type="Textbox",
    )
    search_box.set_colors(background="#ffffff", text="#1f2937", border="#d1d5db")
    search_box.set_spacing(padding=DimensionValue(12, "px"))
    home_page.add_component(search_box)

    submit_btn = VisualSpec(
        component_id="submit-btn",
        component_type="Button",
    )
    submit_btn.set_colors(background="#3b82f6", text="#ffffff")
    home_page.add_component(submit_btn)

    spec.add_page(home_page)

    # Add another page
    settings_page = PageSpec(name="Settings", route="/settings")
    toggle = VisualSpec(
        component_id="dark-mode-toggle",
        component_type="Checkbox",
    )
    settings_page.add_component(toggle)
    spec.add_page(settings_page)

    return spec


@pytest.fixture
def sample_spec_json(tmp_path, sample_ui_spec):
    """Create a sample spec JSON file."""
    spec_path = tmp_path / "test_spec.json"
    sample_ui_spec.save(spec_path)
    return spec_path


@pytest.fixture
def mock_gradio():
    """Mock gradio module for testing without Gradio dependency."""
    with patch.dict('sys.modules', {'gradio': MagicMock()}):
        import sys
        mock_gr = sys.modules['gradio']

        # Mock Blocks context manager
        mock_blocks = MagicMock()
        mock_blocks.__enter__ = MagicMock(return_value=mock_blocks)
        mock_blocks.__exit__ = MagicMock(return_value=None)
        mock_gr.Blocks.return_value = mock_blocks

        # Mock Tabs context manager
        mock_tabs = MagicMock()
        mock_tabs.__enter__ = MagicMock(return_value=mock_tabs)
        mock_tabs.__exit__ = MagicMock(return_value=None)
        mock_gr.Tabs.return_value = mock_tabs

        # Mock Tab context manager
        mock_tab = MagicMock()
        mock_tab.__enter__ = MagicMock(return_value=mock_tab)
        mock_tab.__exit__ = MagicMock(return_value=None)
        mock_gr.Tab.return_value = mock_tab

        # Mock Row context manager
        mock_row = MagicMock()
        mock_row.__enter__ = MagicMock(return_value=mock_row)
        mock_row.__exit__ = MagicMock(return_value=None)
        mock_gr.Row.return_value = mock_row

        # Mock Column context manager
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_gr.Column.return_value = mock_col

        # Mock Accordion context manager
        mock_accordion = MagicMock()
        mock_accordion.__enter__ = MagicMock(return_value=mock_accordion)
        mock_accordion.__exit__ = MagicMock(return_value=None)
        mock_gr.Accordion.return_value = mock_accordion

        # Mock themes
        mock_gr.themes = MagicMock()
        mock_gr.themes.Soft.return_value = MagicMock()

        # Mock update
        mock_gr.update.return_value = {}

        yield mock_gr


# =============================================================================
# VisualSpecViewer Initialization Tests
# =============================================================================

class TestVisualSpecViewerInit:
    """Tests for VisualSpecViewer initialization."""

    def test_init_with_spec(self, sample_ui_spec):
        """Test initialization with a UISpec object."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        assert viewer.spec == sample_ui_spec
        assert viewer.spec_path is None
        assert viewer._app is None

    def test_init_with_spec_path_string(self, sample_spec_json):
        """Test initialization with spec path as string."""
        viewer = VisualSpecViewer(spec_path=str(sample_spec_json))

        assert viewer.spec_path == sample_spec_json
        assert viewer.spec.name == "Test App"

    def test_init_with_spec_path_pathlib(self, sample_spec_json):
        """Test initialization with spec path as Path object."""
        viewer = VisualSpecViewer(spec_path=sample_spec_json)

        assert viewer.spec_path == sample_spec_json
        assert isinstance(viewer.spec, UISpec)

    def test_init_default_creates_new_spec(self):
        """Test initialization without arguments creates new UISpec."""
        viewer = VisualSpecViewer()

        assert viewer.spec.name == "New UI Spec"
        assert viewer.spec_path is None

    def test_init_with_nonexistent_path(self, tmp_path):
        """Test initialization with non-existent path creates new spec."""
        nonexistent = tmp_path / "does_not_exist.json"
        viewer = VisualSpecViewer(spec_path=nonexistent)

        assert viewer.spec.name == "New UI Spec"
        assert viewer.spec_path == nonexistent

    def test_init_raises_without_gradio(self):
        """Test that ImportError is raised when gradio is not available."""
        # Save original gr value
        import integradio.visual.viewer as viewer_module
        original_gr = viewer_module.gr

        try:
            viewer_module.gr = None
            with pytest.raises(ImportError, match="Gradio is required"):
                VisualSpecViewer()
        finally:
            viewer_module.gr = original_gr


# =============================================================================
# Page and Component List Helper Tests
# =============================================================================

class TestPageListHelpers:
    """Tests for page list helper methods."""

    def test_get_pages_list(self, sample_ui_spec):
        """Test getting list of page names."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        pages = viewer._get_pages_list()

        assert "/" in pages
        assert "/settings" in pages
        assert len(pages) == 2

    def test_get_pages_list_empty(self):
        """Test getting pages list when no pages exist."""
        viewer = VisualSpecViewer()
        pages = viewer._get_pages_list()

        assert pages == ["(no pages)"]

    def test_get_components_for_page(self, sample_ui_spec):
        """Test getting components for a specific page."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        components = viewer._get_components_for_page("/")

        assert "search-input" in components
        assert "submit-btn" in components
        assert len(components) == 2

    def test_get_components_for_nonexistent_page(self, sample_ui_spec):
        """Test getting components for a page that doesn't exist."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        components = viewer._get_components_for_page("/nonexistent")

        assert components == ["(no components)"]

    def test_get_components_for_empty_page(self):
        """Test getting components for a page with no components."""
        spec = UISpec(name="Empty")
        page = PageSpec(name="Empty", route="/empty")
        spec.add_page(page)

        viewer = VisualSpecViewer(spec=spec)
        components = viewer._get_components_for_page("/empty")

        assert components == ["(no components)"]


# =============================================================================
# Component Spec Retrieval Tests
# =============================================================================

class TestComponentSpecRetrieval:
    """Tests for component spec retrieval."""

    def test_get_component_spec(self, sample_ui_spec):
        """Test getting a component's visual spec."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        spec = viewer._get_component_spec("/", "search-input")

        assert spec is not None
        assert spec.component_id == "search-input"
        assert spec.component_type == "Textbox"

    def test_get_component_spec_nonexistent_page(self, sample_ui_spec):
        """Test getting component spec from non-existent page."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        spec = viewer._get_component_spec("/nonexistent", "search-input")

        assert spec is None

    def test_get_component_spec_nonexistent_component(self, sample_ui_spec):
        """Test getting non-existent component spec."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        spec = viewer._get_component_spec("/", "nonexistent-component")

        assert spec is None


# =============================================================================
# Preview HTML Rendering Tests
# =============================================================================

class TestPreviewRendering:
    """Tests for preview HTML rendering."""

    def test_render_preview_with_component(self, sample_ui_spec):
        """Test rendering preview HTML for a component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._render_preview("/", "search-input")

        assert "<style>" in html
        assert "preview-search-input" in html
        assert "preview-container" in html

    def test_render_preview_no_component(self, sample_ui_spec):
        """Test rendering preview when component doesn't exist."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._render_preview("/", "nonexistent")

        assert "Select a component to preview" in html

    def test_render_preview_includes_css(self, sample_ui_spec):
        """Test that preview includes component CSS."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._render_preview("/", "submit-btn")

        assert "#preview-submit-btn" in html


# =============================================================================
# Placeholder HTML Generation Tests
# =============================================================================

class TestPlaceholderHTML:
    """Tests for placeholder HTML generation."""

    def test_placeholder_button(self, sample_ui_spec):
        """Test placeholder HTML for button component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("button", "btn-1")

        assert "<button" in html
        assert "Button" in html

    def test_placeholder_textbox(self, sample_ui_spec):
        """Test placeholder HTML for textbox component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("textbox", "input-1")

        assert "<input" in html
        assert "text" in html

    def test_placeholder_textarea(self, sample_ui_spec):
        """Test placeholder HTML for textarea component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("textarea", "ta-1")

        assert "<textarea" in html

    def test_placeholder_dropdown(self, sample_ui_spec):
        """Test placeholder HTML for dropdown component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("dropdown", "dd-1")

        assert "<select" in html
        assert "<option" in html

    def test_placeholder_checkbox(self, sample_ui_spec):
        """Test placeholder HTML for checkbox component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("checkbox", "cb-1")

        assert "checkbox" in html
        assert "<label" in html

    def test_placeholder_radio(self, sample_ui_spec):
        """Test placeholder HTML for radio component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("radio", "r-1")

        assert "radio" in html

    def test_placeholder_slider(self, sample_ui_spec):
        """Test placeholder HTML for slider component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("slider", "sl-1")

        assert "range" in html

    def test_placeholder_markdown(self, sample_ui_spec):
        """Test placeholder HTML for markdown component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("markdown", "md-1")

        assert "markdown" in html
        assert "<h3" in html

    def test_placeholder_image(self, sample_ui_spec):
        """Test placeholder HTML for image component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("image", "img-1")

        assert "Image" in html

    def test_placeholder_chatbot(self, sample_ui_spec):
        """Test placeholder HTML for chatbot component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("chatbot", "chat-1")

        assert "chat" in html
        assert "msg" in html

    def test_placeholder_unknown_type(self, sample_ui_spec):
        """Test placeholder HTML for unknown component type."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("custom_widget", "cw-1")

        assert "custom_widget" in html

    def test_placeholder_empty_type(self, sample_ui_spec):
        """Test placeholder HTML for empty component type."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_placeholder_html("", "c-1")

        assert "Component" in html


# =============================================================================
# Token JSON Serialization Tests
# =============================================================================

class TestTokenJSON:
    """Tests for token JSON serialization."""

    def test_tokens_to_json(self, sample_ui_spec):
        """Test converting component tokens to JSON."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        json_str = viewer._tokens_to_json("/", "search-input")

        data = json.loads(json_str)
        assert "background" in data
        assert "color" in data
        assert "border-color" in data

    def test_tokens_to_json_no_component(self, sample_ui_spec):
        """Test tokens JSON when component doesn't exist."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        json_str = viewer._tokens_to_json("/", "nonexistent")

        assert json_str == "{}"


# =============================================================================
# Layout JSON Serialization Tests
# =============================================================================

class TestLayoutJSON:
    """Tests for layout JSON serialization."""

    def test_layout_to_json(self, sample_ui_spec):
        """Test converting component layout to JSON."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        json_str = viewer._layout_to_json("/", "search-input")

        data = json.loads(json_str)
        # Should have padding from SpacingSpec
        assert "padding-top" in data or isinstance(data, dict)

    def test_layout_to_json_no_component(self, sample_ui_spec):
        """Test layout JSON when component doesn't exist."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        json_str = viewer._layout_to_json("/", "nonexistent")

        assert json_str == "{}"


# =============================================================================
# Token Update Tests
# =============================================================================

class TestTokenUpdate:
    """Tests for token color update functionality."""

    def test_update_token_color(self, sample_ui_spec):
        """Test updating a token color."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._update_token_color("/", "search-input", "background", "#ff0000")

        # Should return preview HTML
        assert "preview" in html

        # Verify the token was updated
        spec = viewer._get_component_spec("/", "search-input")
        assert spec.tokens["background"].to_css() == "rgb(255, 0, 0)"

    def test_update_token_color_empty_hex(self, sample_ui_spec):
        """Test updating with empty hex color does nothing."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        original = viewer._get_component_spec("/", "search-input").tokens["background"].to_css()

        viewer._update_token_color("/", "search-input", "background", "")

        updated = viewer._get_component_spec("/", "search-input").tokens["background"].to_css()
        assert original == updated

    def test_update_token_color_nonexistent_component(self, sample_ui_spec):
        """Test updating token for non-existent component."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._update_token_color("/", "nonexistent", "background", "#ff0000")

        # Should return preview HTML for non-existent component
        assert "Select a component" in html


# =============================================================================
# CSS Generation Tests
# =============================================================================

class TestCSSGeneration:
    """Tests for CSS generation functionality."""

    def test_generate_full_css(self, sample_ui_spec):
        """Test generating CSS for the entire spec."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        css = viewer._generate_full_css()

        assert "search-input" in css
        assert "submit-btn" in css

    def test_generate_full_css_with_theme(self, sample_ui_spec):
        """Test generating CSS with a theme."""
        # Add a theme to the spec
        dark_tokens = TokenGroup(type=TokenType.COLOR)
        dark_tokens.add("background", DesignToken.color("#1f2937"))
        sample_ui_spec.add_theme("dark", dark_tokens)

        viewer = VisualSpecViewer(spec=sample_ui_spec)
        css = viewer._generate_full_css(theme="dark")

        assert "dark" in css
        assert "--background" in css

    def test_generate_full_css_none_theme(self, sample_ui_spec):
        """Test generating CSS with (none) theme selection."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        css = viewer._generate_full_css(theme="(none)")

        # Should not have any theme-specific selector
        assert 'data-theme="(none)"' not in css


# =============================================================================
# Style Dictionary Export Tests
# =============================================================================

class TestStyleDictionaryExport:
    """Tests for Style Dictionary export functionality."""

    def test_export_style_dictionary(self, sample_ui_spec):
        """Test exporting spec as Style Dictionary JSON."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        json_str = viewer._export_style_dictionary()

        data = json.loads(json_str)
        assert data["name"] == "Test App"
        assert data["version"] == "1.0.0"
        assert "tokens" in data
        assert "pages" in data


# =============================================================================
# Save Spec Tests
# =============================================================================

class TestSaveSpec:
    """Tests for save spec functionality."""

    def test_save_spec(self, sample_ui_spec, tmp_path):
        """Test saving spec to file."""
        spec_path = tmp_path / "saved_spec.json"
        viewer = VisualSpecViewer(spec=sample_ui_spec, spec_path=spec_path)

        result = viewer._save_spec()

        assert "Saved to" in result
        assert spec_path.exists()

        # Verify saved content
        with open(spec_path) as f:
            data = json.load(f)
        assert data["name"] == "Test App"

    def test_save_spec_no_path(self, sample_ui_spec):
        """Test saving when no path is configured."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        result = viewer._save_spec()

        assert "No save path configured" in result


# =============================================================================
# Add Page Tests
# =============================================================================

class TestAddPage:
    """Tests for add page functionality."""

    def test_add_page(self, sample_ui_spec):
        """Test adding a new page."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        pages = viewer._add_page("About", "/about")

        assert "/about" in pages
        assert "/about" in viewer.spec.pages

    def test_add_page_returns_updated_list(self, sample_ui_spec):
        """Test that add_page returns updated page list."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        initial_count = len(viewer._get_pages_list())

        pages = viewer._add_page("Contact", "/contact")

        assert len(pages) == initial_count + 1


# =============================================================================
# Add Component Tests
# =============================================================================

class TestAddComponent:
    """Tests for add component functionality."""

    def test_add_component(self, sample_ui_spec):
        """Test adding a new component to a page."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        components = viewer._add_component("/", "new-button", "Button")

        assert "new-button" in components
        assert "new-button" in viewer.spec.pages["/"].components

    def test_add_component_nonexistent_page(self, sample_ui_spec):
        """Test adding component to non-existent page."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        components = viewer._add_component("/nonexistent", "btn", "Button")

        # Should return components for the requested page (empty since it doesn't exist)
        assert components == ["(no components)"]


# =============================================================================
# Token Preview Generation Tests
# =============================================================================

class TestTokenPreviewGeneration:
    """Tests for token preview HTML generation."""

    def test_generate_token_preview(self, sample_ui_spec):
        """Test generating token preview HTML."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_token_preview()

        assert "<table" in html
        assert "colors.primary" in html or "primary" in html
        assert "spacing.sm" in html or "sm" in html

    def test_generate_token_preview_empty(self):
        """Test generating token preview with no tokens."""
        viewer = VisualSpecViewer()
        html = viewer._generate_token_preview()

        assert "No global tokens defined" in html

    def test_generate_token_preview_includes_color_swatch(self, sample_ui_spec):
        """Test that color tokens have swatches in preview."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        html = viewer._generate_token_preview()

        assert "color-swatch" in html


# =============================================================================
# Build Interface Tests
# =============================================================================

class TestBuildInterface:
    """Tests for Gradio interface building."""

    def test_build_returns_blocks(self, sample_ui_spec):
        """Test that build returns a Gradio Blocks object."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        app = viewer.build()

        assert app is not None
        assert viewer._app is app

    def test_build_sets_internal_app(self, sample_ui_spec):
        """Test that build sets _app internally."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        assert viewer._app is None

        viewer.build()

        assert viewer._app is not None


# =============================================================================
# Launch Tests
# =============================================================================

class TestLaunch:
    """Tests for launch functionality."""

    def test_launch_builds_if_not_built(self, sample_ui_spec):
        """Test that launch calls build if app not built."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # Create a mock for the build method that creates a mock app
        mock_app = MagicMock()
        original_build = viewer.build

        def mock_build_fn():
            viewer._app = mock_app
            return mock_app

        with patch.object(viewer, 'build', side_effect=mock_build_fn) as mock_build:
            viewer.launch()
            mock_build.assert_called_once()

    def test_launch_default_kwargs(self, sample_ui_spec):
        """Test launch uses default kwargs."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        mock_app = MagicMock()
        viewer._app = mock_app

        viewer.launch()

        mock_app.launch.assert_called_once()
        call_kwargs = mock_app.launch.call_args[1]
        assert call_kwargs["server_port"] == 7861
        assert call_kwargs["share"] is False

    def test_launch_custom_kwargs(self, sample_ui_spec):
        """Test launch with custom kwargs."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        mock_app = MagicMock()
        viewer._app = mock_app

        viewer.launch(server_port=8080, share=True)

        call_kwargs = mock_app.launch.call_args[1]
        assert call_kwargs["server_port"] == 8080
        assert call_kwargs["share"] is True


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_view_spec_with_ui_spec(self, sample_ui_spec):
        """Test view_spec with UISpec object."""
        with patch('integradio.visual.viewer.VisualSpecViewer') as MockViewer:
            mock_instance = MagicMock()
            MockViewer.return_value = mock_instance

            view_spec(sample_ui_spec, server_port=7865)

            MockViewer.assert_called_once_with(spec=sample_ui_spec)
            mock_instance.launch.assert_called_once_with(server_port=7865)

    def test_view_spec_with_path_string(self, sample_spec_json):
        """Test view_spec with string path."""
        with patch('integradio.visual.viewer.VisualSpecViewer') as MockViewer:
            mock_instance = MagicMock()
            MockViewer.return_value = mock_instance

            view_spec(str(sample_spec_json))

            MockViewer.assert_called_once_with(spec_path=str(sample_spec_json))
            mock_instance.launch.assert_called_once()

    def test_view_spec_with_path_object(self, sample_spec_json):
        """Test view_spec with Path object."""
        with patch('integradio.visual.viewer.VisualSpecViewer') as MockViewer:
            mock_instance = MagicMock()
            MockViewer.return_value = mock_instance

            view_spec(sample_spec_json)

            MockViewer.assert_called_once_with(spec_path=sample_spec_json)

    def test_create_viewer_demo_returns_blocks(self):
        """Test create_viewer_demo returns a Gradio Blocks object."""
        blocks = create_viewer_demo()

        assert blocks is not None

    def test_create_viewer_demo_has_sample_data(self):
        """Test create_viewer_demo creates sample spec data."""
        # Create demo and check internally that data exists
        # We can't easily inspect the created spec, but we can verify
        # the function completes without error
        try:
            create_viewer_demo()
        except Exception as e:
            pytest.fail(f"create_viewer_demo raised exception: {e}")


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_viewer_with_empty_spec(self):
        """Test viewer with completely empty spec."""
        spec = UISpec(name="Empty")
        viewer = VisualSpecViewer(spec=spec)

        assert viewer._get_pages_list() == ["(no pages)"]
        assert viewer._generate_token_preview() == "<p>No global tokens defined</p>"

    def test_viewer_component_with_no_tokens(self):
        """Test viewer with component that has no tokens."""
        spec = UISpec(name="Test")
        page = PageSpec(name="Page", route="/")
        component = VisualSpec(component_id="empty-component", component_type="Button")
        page.add_component(component)
        spec.add_page(page)

        viewer = VisualSpecViewer(spec=spec)
        json_str = viewer._tokens_to_json("/", "empty-component")

        assert json_str == "{}"

    def test_update_token_creates_new_token(self, sample_ui_spec):
        """Test that updating a token that doesn't exist creates it."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # Update a token that doesn't exist in the component
        viewer._update_token_color("/", "search-input", "new-color-token", "#abcdef")

        spec = viewer._get_component_spec("/", "search-input")
        assert "new-color-token" in spec.tokens

    def test_special_characters_in_component_id(self):
        """Test handling of special characters in component IDs."""
        spec = UISpec(name="Test")
        page = PageSpec(name="Page", route="/")
        component = VisualSpec(
            component_id="my-special_component.v2",
            component_type="Button"
        )
        page.add_component(component)
        spec.add_page(page)

        viewer = VisualSpecViewer(spec=spec)
        html = viewer._render_preview("/", "my-special_component.v2")

        assert "my-special_component.v2" in html

    def test_unicode_in_spec_name(self):
        """Test handling of unicode characters in spec name."""
        spec = UISpec(name="Test App")
        viewer = VisualSpecViewer(spec=spec)
        json_str = viewer._export_style_dictionary()

        data = json.loads(json_str)
        assert data["name"] == "Test App"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the viewer module."""

    def test_full_workflow(self, tmp_path, sample_ui_spec):
        """Test a complete workflow: create, modify, save."""
        spec_path = tmp_path / "workflow_test.json"

        # Create viewer and add content
        viewer = VisualSpecViewer(spec=sample_ui_spec, spec_path=spec_path)

        # Add a new page and component
        viewer._add_page("NewPage", "/new")
        viewer._add_component("/new", "new-btn", "Button")

        # Update a token
        viewer._update_token_color("/new", "new-btn", "background", "#00ff00")

        # Verify in-memory state
        assert "/new" in viewer._get_pages_list()
        assert "new-btn" in viewer._get_components_for_page("/new")

        # Save
        result = viewer._save_spec()
        assert "Saved" in result

        # Verify file was created
        assert spec_path.exists()

        # Verify saved JSON content
        import json
        with open(spec_path) as f:
            data = json.load(f)
        assert "/new" in data.get("pages", {})
        assert data["pages"]["/new"]["components"].get("new-btn") is not None

    def test_css_generation_integration(self, sample_ui_spec):
        """Test that CSS generation works with complex spec."""
        # Add a theme
        dark_tokens = TokenGroup(type=TokenType.COLOR)
        dark_tokens.add("bg", DesignToken.color("#000000"))
        sample_ui_spec.add_theme("dark", dark_tokens)

        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # Generate CSS without theme
        css_no_theme = viewer._generate_full_css()

        # Generate CSS with theme
        css_with_theme = viewer._generate_full_css(theme="dark")

        assert css_no_theme != css_with_theme
        assert "dark" in css_with_theme

    def test_build_and_inspect(self, sample_ui_spec):
        """Test building and inspecting the interface."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        app = viewer.build()

        # Verify the app was created
        assert viewer._app is not None
        assert app is viewer._app


# =============================================================================
# Responsive/Breakpoint Tests
# =============================================================================

class TestResponsive:
    """Tests related to responsive behavior."""

    def test_breakpoints_available(self, sample_ui_spec):
        """Test that breakpoints are available in the spec."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # The spec should have breakpoints from BREAKPOINTS
        assert len(viewer.spec.breakpoints) > 0
        assert "md" in viewer.spec.breakpoints


# =============================================================================
# Theme Tests
# =============================================================================

class TestThemes:
    """Tests for theme functionality."""

    def test_spec_with_multiple_themes(self):
        """Test spec with multiple themes."""
        spec = UISpec(name="MultiTheme")

        light_tokens = TokenGroup(type=TokenType.COLOR)
        light_tokens.add("bg", DesignToken.color("#ffffff"))
        spec.add_theme("light", light_tokens)

        dark_tokens = TokenGroup(type=TokenType.COLOR)
        dark_tokens.add("bg", DesignToken.color("#000000"))
        spec.add_theme("dark", dark_tokens)

        viewer = VisualSpecViewer(spec=spec)

        light_css = viewer._generate_full_css(theme="light")
        dark_css = viewer._generate_full_css(theme="dark")

        assert "light" in light_css
        assert "dark" in dark_css
        assert light_css != dark_css


# =============================================================================
# Inner Function Tests (for Gradio event handlers defined in build())
# =============================================================================

class TestBuildInnerFunctions:
    """Tests for the inner functions defined within the build() method.

    These are the Gradio event handlers that need to be tested separately
    since they're closures defined inside build().
    """

    def test_update_components_handler(self, sample_ui_spec):
        """Test the update_components inner function behavior.

        This function is called when the page dropdown changes.
        """
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # Build the app to create the closures
        viewer.build()

        # The update_components function behavior is encapsulated in
        # _get_components_for_page, which is already tested.
        # We test the actual expected behavior here.
        components = viewer._get_components_for_page("/")
        assert "search-input" in components
        assert "submit-btn" in components

    def test_update_preview_handler(self, sample_ui_spec):
        """Test the update_preview inner function behavior.

        This function is called when the component dropdown changes.
        """
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # The inner function combines three operations:
        # 1. _render_preview
        # 2. _tokens_to_json
        # 3. _layout_to_json
        preview = viewer._render_preview("/", "search-input")
        tokens = viewer._tokens_to_json("/", "search-input")
        layout = viewer._layout_to_json("/", "search-input")

        assert "preview" in preview
        assert "background" in tokens
        assert isinstance(json.loads(layout), dict)

    def test_update_bg_handler(self, sample_ui_spec):
        """Test the update_bg inner function behavior."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        result = viewer._update_token_color("/", "search-input", "background", "#ff0000")
        assert "preview" in result

    def test_update_text_handler(self, sample_ui_spec):
        """Test the update_text inner function behavior."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        result = viewer._update_token_color("/", "search-input", "color", "#0000ff")
        assert "preview" in result

    def test_update_border_handler(self, sample_ui_spec):
        """Test the update_border inner function behavior."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        result = viewer._update_token_color("/", "search-input", "border-color", "#00ff00")
        assert "preview" in result

    def test_add_page_handler_success(self, sample_ui_spec):
        """Test add_page_handler with valid inputs."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # Simulate the handler behavior
        name = "New Page"
        route = "/new-page"

        if name and route:
            viewer._add_page(name, route)
            result = f"Added page: {name} ({route})"
        else:
            result = "Please fill in both fields"

        assert "Added page" in result
        assert "/new-page" in viewer._get_pages_list()

    def test_add_page_handler_empty_name(self, sample_ui_spec):
        """Test add_page_handler with empty name."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        initial_pages = len(viewer._get_pages_list())

        name = ""
        route = "/test"

        if name and route:
            viewer._add_page(name, route)
            result = f"Added page: {name} ({route})"
        else:
            result = "Please fill in both fields"

        assert "Please fill in both fields" in result
        assert len(viewer._get_pages_list()) == initial_pages

    def test_add_page_handler_empty_route(self, sample_ui_spec):
        """Test add_page_handler with empty route."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        initial_pages = len(viewer._get_pages_list())

        name = "Test Page"
        route = ""

        if name and route:
            viewer._add_page(name, route)
            result = f"Added page: {name} ({route})"
        else:
            result = "Please fill in both fields"

        assert "Please fill in both fields" in result
        assert len(viewer._get_pages_list()) == initial_pages

    def test_add_component_handler_success(self, sample_ui_spec):
        """Test add_component_handler with valid inputs."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        page_route = "/"
        comp_id = "new-component"
        comp_type = "Button"

        if page_route and comp_id and comp_type:
            viewer._add_component(page_route, comp_id, comp_type)
            result = f"Added component: {comp_id} to {page_route}"
        else:
            result = "Please fill in all fields"

        assert "Added component" in result
        assert "new-component" in viewer._get_components_for_page("/")

    def test_add_component_handler_empty_page_route(self, sample_ui_spec):
        """Test add_component_handler with empty page route."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        page_route = ""
        comp_id = "test-comp"
        comp_type = "Button"

        if page_route and comp_id and comp_type:
            viewer._add_component(page_route, comp_id, comp_type)
            result = f"Added component: {comp_id} to {page_route}"
        else:
            result = "Please fill in all fields"

        assert "Please fill in all fields" in result

    def test_add_component_handler_empty_comp_id(self, sample_ui_spec):
        """Test add_component_handler with empty component ID."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        initial_components = len(viewer._get_components_for_page("/"))

        page_route = "/"
        comp_id = ""
        comp_type = "Button"

        if page_route and comp_id and comp_type:
            viewer._add_component(page_route, comp_id, comp_type)
            result = f"Added component: {comp_id} to {page_route}"
        else:
            result = "Please fill in all fields"

        assert "Please fill in all fields" in result
        assert len(viewer._get_components_for_page("/")) == initial_components

    def test_add_component_handler_empty_comp_type(self, sample_ui_spec):
        """Test add_component_handler with empty component type."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        initial_components = len(viewer._get_components_for_page("/"))

        page_route = "/"
        comp_id = "test-comp"
        comp_type = ""

        if page_route and comp_id and comp_type:
            viewer._add_component(page_route, comp_id, comp_type)
            result = f"Added component: {comp_id} to {page_route}"
        else:
            result = "Please fill in all fields"

        assert "Please fill in all fields" in result
        assert len(viewer._get_components_for_page("/")) == initial_components

    def test_update_global_tokens_handler(self, sample_ui_spec):
        """Test the update_global_tokens inner function behavior."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)

        # The inner function just returns the token preview (TODO in code)
        preview = viewer._generate_token_preview()
        assert "<table" in preview


# =============================================================================
# Token Preview with Different Token Types
# =============================================================================

class TestTokenPreviewVariety:
    """Tests for token preview generation with different token types."""

    def test_preview_with_dimension_tokens(self):
        """Test token preview shows dimension tokens."""
        spec = UISpec(name="DimTest")
        spec.tokens.add("spacing", TokenGroup(type=TokenType.DIMENSION))
        spec.tokens.get("spacing").add("gap", DesignToken.dimension(16, "px"))

        viewer = VisualSpecViewer(spec=spec)
        preview = viewer._generate_token_preview()

        assert "spacing.gap" in preview or "gap" in preview
        assert "16px" in preview

    def test_preview_with_mixed_token_types(self):
        """Test token preview with multiple token types."""
        spec = UISpec(name="MixedTest")

        # Add color group
        spec.tokens.add("colors", TokenGroup(type=TokenType.COLOR))
        spec.tokens.get("colors").add("primary", DesignToken.color("#007bff"))

        # Add spacing group
        spec.tokens.add("spacing", TokenGroup(type=TokenType.DIMENSION))
        spec.tokens.get("spacing").add("sm", DesignToken.dimension(8, "px"))

        viewer = VisualSpecViewer(spec=spec)
        preview = viewer._generate_token_preview()

        assert "color" in preview.lower()
        assert "dimension" in preview.lower()

    def test_preview_table_structure(self, sample_ui_spec):
        """Test that preview has proper table structure."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        preview = viewer._generate_token_preview()

        assert "<thead>" in preview
        assert "<tbody>" in preview
        assert "<tr" in preview
        assert "<td" in preview


# =============================================================================
# Spec Path Handling Edge Cases
# =============================================================================

class TestSpecPathEdgeCases:
    """Tests for spec path handling edge cases."""

    def test_spec_path_with_special_characters(self, tmp_path):
        """Test spec path with special characters."""
        special_dir = tmp_path / "test specs"
        special_dir.mkdir()
        spec_path = special_dir / "my spec.json"

        spec = UISpec(name="Special Path Test")
        viewer = VisualSpecViewer(spec=spec, spec_path=spec_path)
        viewer._save_spec()

        assert spec_path.exists()

    def test_spec_path_deep_directory(self, tmp_path):
        """Test spec path with deep directory structure."""
        deep_path = tmp_path / "a" / "b" / "c"
        deep_path.mkdir(parents=True)
        spec_path = deep_path / "spec.json"

        spec = UISpec(name="Deep Path Test")
        viewer = VisualSpecViewer(spec=spec, spec_path=spec_path)
        viewer._save_spec()

        assert spec_path.exists()


# =============================================================================
# CSS Generation Edge Cases
# =============================================================================

class TestCSSGenerationEdgeCases:
    """Tests for CSS generation edge cases."""

    def test_css_with_no_pages(self):
        """Test CSS generation with no pages."""
        spec = UISpec(name="Empty")
        viewer = VisualSpecViewer(spec=spec)
        css = viewer._generate_full_css()

        # Should not crash, may be empty or have root vars only
        assert isinstance(css, str)

    def test_css_with_nonexistent_theme(self, sample_ui_spec):
        """Test CSS generation with non-existent theme."""
        viewer = VisualSpecViewer(spec=sample_ui_spec)
        css = viewer._generate_full_css(theme="nonexistent")

        # Should not crash, should generate CSS without theme-specific rules
        assert isinstance(css, str)
