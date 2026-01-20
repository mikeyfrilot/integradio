"""
Tests for Component Library Generator module.

Tests:
- ComponentStatus and PageType enums
- ComponentVariant, ComponentEntry, CategoryEntry, GuidePage dataclasses
- LibraryConfig dataclass
- ComponentLibrary class operations
- HTML generation functions
- LibrarySiteGenerator
- Convenience functions
- Site generation and file output
"""

import pytest
import json
from pathlib import Path

from integradio.visual.library import (
    # Types
    ComponentStatus,
    PageType,
    # Data classes
    ComponentVariant,
    ComponentEntry,
    CategoryEntry,
    GuidePage,
    LibraryConfig,
    # Library
    ComponentLibrary,
    # HTML generation
    generate_css,
    generate_component_preview_html,
    generate_header_html,
    generate_sidebar_html,
    generate_props_table_html,
    generate_variants_html,
    generate_page_html,
    escape_html,
    # Site generator
    LibrarySiteGenerator,
    # Convenience functions
    generate_library_site,
    create_library_from_specs,
    create_library_from_ui_spec,
    quick_library,
)
from integradio.visual.spec import VisualSpec, UISpec, PageSpec
from integradio.visual.tokens import TokenGroup, TokenType, DesignToken, ColorValue


# =============================================================================
# Enum Tests
# =============================================================================

class TestComponentStatus:
    """Tests for ComponentStatus enum."""

    def test_all_statuses(self):
        """Test all status values exist."""
        assert ComponentStatus.DRAFT == "draft"
        assert ComponentStatus.BETA == "beta"
        assert ComponentStatus.STABLE == "stable"
        assert ComponentStatus.DEPRECATED == "deprecated"

    def test_status_is_string(self):
        """Test that status values are strings."""
        for status in ComponentStatus:
            assert isinstance(status.value, str)


class TestPageType:
    """Tests for PageType enum."""

    def test_all_page_types(self):
        """Test all page type values exist."""
        assert PageType.COMPONENT == "component"
        assert PageType.CATEGORY == "category"
        assert PageType.TOKEN == "token"
        assert PageType.GUIDE == "guide"
        assert PageType.INDEX == "index"


# =============================================================================
# ComponentVariant Tests
# =============================================================================

class TestComponentVariant:
    """Tests for ComponentVariant dataclass."""

    def test_basic_creation(self):
        """Test basic variant creation."""
        variant = ComponentVariant(
            name="Primary",
            description="Primary button style",
        )

        assert variant.name == "Primary"
        assert variant.description == "Primary button style"

    def test_variant_with_props(self):
        """Test variant with props."""
        variant = ComponentVariant(
            name="Large",
            props={"size": "lg", "rounded": True},
        )

        assert variant.props["size"] == "lg"
        assert variant.props["rounded"] is True

    def test_variant_with_code_example(self):
        """Test variant with code example."""
        variant = ComponentVariant(
            name="Outline",
            code_example="<Button variant='outline'>Click</Button>",
        )

        assert "Button" in variant.code_example

    def test_to_dict(self):
        """Test dictionary conversion."""
        variant = ComponentVariant(
            name="Secondary",
            description="Secondary style",
            props={"variant": "secondary"},
            code_example="<Button variant='secondary' />",
        )
        result = variant.to_dict()

        assert result["name"] == "Secondary"
        assert result["description"] == "Secondary style"
        assert result["props"]["variant"] == "secondary"
        assert result["code_example"] == "<Button variant='secondary' />"


# =============================================================================
# ComponentEntry Tests
# =============================================================================

class TestComponentEntry:
    """Tests for ComponentEntry dataclass."""

    @pytest.fixture
    def sample_spec(self):
        """Create sample VisualSpec."""
        spec = VisualSpec(
            component_id="btn-primary",
            component_type="Button",
        )
        spec.set_colors(background="#3b82f6", text="#ffffff")
        return spec

    def test_basic_creation(self, sample_spec):
        """Test basic entry creation."""
        entry = ComponentEntry(
            name="Primary Button",
            description="A primary action button",
            spec=sample_spec,
        )

        assert entry.name == "Primary Button"
        assert entry.description == "A primary action button"
        assert entry.category == "Uncategorized"
        assert entry.status == ComponentStatus.STABLE

    def test_slug_generation(self, sample_spec):
        """Test slug generation from name."""
        entry = ComponentEntry(
            name="Primary Button",
            description="",
            spec=sample_spec,
        )

        assert entry.slug == "primary-button"

    def test_slug_with_underscores(self, sample_spec):
        """Test slug handles underscores."""
        entry = ComponentEntry(
            name="Primary_Button",
            description="",
            spec=sample_spec,
        )

        assert entry.slug == "primary-button"

    def test_add_variant(self, sample_spec):
        """Test adding variants."""
        entry = ComponentEntry(
            name="Button",
            description="",
            spec=sample_spec,
        )

        entry.add_variant("Primary", "Main action")
        entry.add_variant("Secondary", "Secondary action")

        assert len(entry.variants) == 2
        assert entry.variants[0].name == "Primary"

    def test_add_prop(self, sample_spec):
        """Test adding prop documentation."""
        entry = ComponentEntry(
            name="Button",
            description="",
            spec=sample_spec,
        )

        entry.add_prop(
            name="onClick",
            type="() => void",
            description="Click handler",
            required=True,
        )

        assert "onClick" in entry.props
        assert entry.props["onClick"]["type"] == "() => void"
        assert entry.props["onClick"]["required"] is True

    def test_add_changelog_entry(self, sample_spec):
        """Test adding changelog entries."""
        entry = ComponentEntry(
            name="Button",
            description="",
            spec=sample_spec,
        )

        entry.add_changelog_entry("1.0.0", "Initial release")
        entry.add_changelog_entry("1.1.0", "Added variants")

        assert len(entry.changelog) == 2
        assert entry.changelog[0]["version"] == "1.0.0"

    def test_to_dict(self, sample_spec):
        """Test dictionary conversion."""
        entry = ComponentEntry(
            name="Button",
            description="A button",
            spec=sample_spec,
            category="Inputs",
            status=ComponentStatus.BETA,
            tags=["interactive", "form"],
        )
        result = entry.to_dict()

        assert result["name"] == "Button"
        assert result["slug"] == "button"
        assert result["category"] == "Inputs"
        assert result["status"] == "beta"
        assert "interactive" in result["tags"]


# =============================================================================
# CategoryEntry Tests
# =============================================================================

class TestCategoryEntry:
    """Tests for CategoryEntry dataclass."""

    def test_basic_creation(self):
        """Test basic category creation."""
        category = CategoryEntry(
            name="Input Components",
            description="Form input elements",
            icon="📝",
        )

        assert category.name == "Input Components"
        assert category.icon == "📝"

    def test_slug_generation(self):
        """Test slug generation."""
        category = CategoryEntry(name="Input Components")

        assert category.slug == "input-components"

    def test_component_list(self):
        """Test component list."""
        category = CategoryEntry(
            name="Inputs",
            components=["button", "textbox", "dropdown"],
        )

        assert len(category.components) == 3
        assert "button" in category.components

    def test_to_dict(self):
        """Test dictionary conversion."""
        category = CategoryEntry(
            name="Layout",
            description="Layout components",
            icon="📐",
            order=1,
        )
        result = category.to_dict()

        assert result["name"] == "Layout"
        assert result["slug"] == "layout"
        assert result["icon"] == "📐"
        assert result["order"] == 1


# =============================================================================
# GuidePage Tests
# =============================================================================

class TestGuidePage:
    """Tests for GuidePage dataclass."""

    def test_basic_creation(self):
        """Test basic guide creation."""
        guide = GuidePage(
            title="Getting Started",
            slug="getting-started",
            content="# Welcome\n\nThis is the getting started guide.",
        )

        assert guide.title == "Getting Started"
        assert guide.slug == "getting-started"

    def test_to_dict(self):
        """Test dictionary conversion."""
        guide = GuidePage(
            title="Installation",
            slug="installation",
            content="# Installation",
            category="Setup",
            order=1,
        )
        result = guide.to_dict()

        assert result["title"] == "Installation"
        assert result["slug"] == "installation"
        assert result["category"] == "Setup"


# =============================================================================
# LibraryConfig Tests
# =============================================================================

class TestLibraryConfig:
    """Tests for LibraryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LibraryConfig()

        assert config.name == "Component Library"
        assert config.version == "1.0.0"
        assert config.show_theme_toggle is True
        assert config.show_search is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = LibraryConfig(
            name="My Design System",
            version="2.0.0",
            primary_color="#10b981",
            github_url="https://github.com/example/repo",
        )

        assert config.name == "My Design System"
        assert config.primary_color == "#10b981"

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = LibraryConfig(
            name="Test Library",
            description="A test library",
        )
        result = config.to_dict()

        assert result["name"] == "Test Library"
        assert result["description"] == "A test library"
        assert "primary_color" in result


# =============================================================================
# ComponentLibrary Tests
# =============================================================================

class TestComponentLibrary:
    """Tests for ComponentLibrary class."""

    @pytest.fixture
    def sample_library(self):
        """Create sample library with components."""
        library = ComponentLibrary("Test Library")

        # Add components
        button_spec = VisualSpec("btn", "Button")
        library.add_component(
            button_spec,
            name="Button",
            description="A clickable button",
            category="Inputs",
        )

        card_spec = VisualSpec("card", "Card")
        library.add_component(
            card_spec,
            name="Card",
            description="A content card",
            category="Layout",
        )

        return library

    def test_initialization(self):
        """Test library initialization."""
        library = ComponentLibrary("My Library")

        assert library.name == "My Library"
        assert len(library.components) == 0
        assert len(library.categories) == 0

    def test_initialization_with_config(self):
        """Test library with custom config."""
        config = LibraryConfig(
            name="Custom Library",
            version="2.0.0",
        )
        library = ComponentLibrary(config=config)

        assert library.name == "Custom Library"
        assert library.config.version == "2.0.0"

    def test_add_component(self):
        """Test adding a component."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("test-btn", "Button")

        entry = library.add_component(
            spec,
            name="Test Button",
            description="A test button",
            category="Inputs",
        )

        assert entry.name == "Test Button"
        assert "test-button" in library.components
        assert "Inputs" in library.categories

    def test_add_component_creates_category(self):
        """Test that adding component creates category."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("btn", "Button")

        library.add_component(spec, category="New Category")

        assert "New Category" in library.categories
        assert "btn" in library.categories["New Category"].components

    def test_add_category(self, sample_library):
        """Test adding a category."""
        category = sample_library.add_category(
            "New Category",
            description="A new category",
            icon="🆕",
        )

        assert category.name == "New Category"
        assert sample_library.categories["New Category"].icon == "🆕"

    def test_update_existing_category(self, sample_library):
        """Test updating existing category."""
        sample_library.add_category("Inputs", icon="📝", order=1)

        assert sample_library.categories["Inputs"].icon == "📝"
        assert sample_library.categories["Inputs"].order == 1

    def test_add_guide(self, sample_library):
        """Test adding a guide."""
        guide = sample_library.add_guide(
            "Getting Started",
            "# Welcome\n\nStart here.",
        )

        assert guide.title == "Getting Started"
        assert "getting-started" in sample_library.guides

    def test_add_token_group(self, sample_library):
        """Test adding token group."""
        group = TokenGroup(type=TokenType.COLOR)
        group.add("primary", DesignToken.color("#3b82f6"))

        sample_library.add_token_group("colors", group)

        assert "colors" in sample_library.token_groups

    def test_get_component(self, sample_library):
        """Test getting component by slug."""
        button = sample_library.get_component("button")

        assert button is not None
        assert button.name == "Button"

    def test_get_component_not_found(self, sample_library):
        """Test getting non-existent component."""
        result = sample_library.get_component("nonexistent")

        assert result is None

    def test_get_components_by_category(self, sample_library):
        """Test getting components by category."""
        inputs = sample_library.get_components_by_category("Inputs")

        assert len(inputs) == 1
        assert inputs[0].name == "Button"

    def test_search_components_by_name(self, sample_library):
        """Test searching by name."""
        results = sample_library.search_components("button")

        assert len(results) == 1
        assert results[0].name == "Button"

    def test_search_components_by_description(self, sample_library):
        """Test searching by description."""
        results = sample_library.search_components("clickable")

        assert len(results) == 1
        assert results[0].name == "Button"

    def test_search_components_by_tag(self):
        """Test searching by tag."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("btn", "Button")
        library.add_component(spec, tags=["interactive", "form"])

        results = library.search_components("interactive")

        assert len(results) == 1

    def test_to_dict(self, sample_library):
        """Test dictionary export."""
        result = sample_library.to_dict()

        assert "config" in result
        assert "components" in result
        assert "categories" in result
        assert "button" in result["components"]

    def test_to_json(self, sample_library):
        """Test JSON export."""
        json_str = sample_library.to_json()

        parsed = json.loads(json_str)
        assert parsed["config"]["name"] == "Test Library"

    def test_save(self, sample_library, tmp_path):
        """Test saving to file."""
        output_path = tmp_path / "library.json"
        sample_library.save(output_path)

        assert output_path.exists()
        content = json.loads(output_path.read_text())
        assert "components" in content


# =============================================================================
# HTML Generation Tests
# =============================================================================

class TestHTMLGeneration:
    """Tests for HTML generation functions."""

    @pytest.fixture
    def sample_config(self):
        """Create sample config."""
        return LibraryConfig(
            name="Test Library",
            primary_color="#3b82f6",
        )

    @pytest.fixture
    def sample_library(self):
        """Create sample library."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("btn", "Button")
        library.add_component(spec, name="Button", category="Inputs")
        return library

    def test_escape_html(self):
        """Test HTML escaping."""
        assert escape_html("<script>") == "&lt;script&gt;"
        assert escape_html("&") == "&amp;"
        assert escape_html('"') == "&quot;"

    def test_generate_css(self, sample_config):
        """Test CSS generation."""
        css = generate_css(sample_config)

        assert ":root" in css
        assert sample_config.primary_color in css
        assert ".header" in css
        assert ".component-card" in css

    def test_generate_css_with_custom(self):
        """Test CSS with custom CSS."""
        config = LibraryConfig(custom_css=".custom { color: red; }")
        css = generate_css(config)

        assert ".custom { color: red; }" in css

    def test_generate_component_preview_button(self):
        """Test button preview generation."""
        spec = VisualSpec("btn", "Button")
        spec.set_colors(background="#3b82f6", text="#ffffff")

        html = generate_component_preview_html(spec)

        assert "<button" in html
        assert "btn" in html

    def test_generate_component_preview_textbox(self):
        """Test textbox preview generation."""
        spec = VisualSpec("input", "Textbox")

        html = generate_component_preview_html(spec)

        assert "<input" in html
        assert "type=\"text\"" in html

    def test_generate_component_preview_card(self):
        """Test card preview generation."""
        spec = VisualSpec("card", "Card")

        html = generate_component_preview_html(spec)

        assert "<div" in html
        assert "Card" in html

    def test_generate_header_html(self, sample_config):
        """Test header generation."""
        html = generate_header_html(sample_config)

        assert "<header" in html
        assert sample_config.name in html
        assert "theme-toggle" in html  # Theme toggle enabled by default

    def test_generate_header_without_theme_toggle(self):
        """Test header without theme toggle."""
        config = LibraryConfig(show_theme_toggle=False)
        html = generate_header_html(config)

        assert "theme-toggle" not in html

    def test_generate_header_with_github(self):
        """Test header with GitHub link."""
        config = LibraryConfig(github_url="https://github.com/test")
        html = generate_header_html(config)

        assert "GitHub" in html
        assert "https://github.com/test" in html

    def test_generate_sidebar_html(self, sample_library):
        """Test sidebar generation."""
        html = generate_sidebar_html(sample_library)

        assert "<nav" in html
        assert "Inputs" in html
        assert "Button" in html

    def test_generate_sidebar_with_active(self, sample_library):
        """Test sidebar with active component."""
        html = generate_sidebar_html(sample_library, current_slug="button")

        assert 'class="active"' in html

    def test_generate_props_table_empty(self):
        """Test props table with no props."""
        html = generate_props_table_html({})

        assert html == ""

    def test_generate_props_table(self):
        """Test props table generation."""
        props = {
            "onClick": {
                "type": "() => void",
                "description": "Click handler",
                "default": None,
                "required": True,
            },
            "disabled": {
                "type": "boolean",
                "description": "Disabled state",
                "default": False,
                "required": False,
            },
        }
        html = generate_props_table_html(props)

        assert "<table" in html
        assert "onClick" in html
        assert "() =&gt; void" in html  # Escaped
        assert "Required" in html
        assert "Optional" in html

    def test_generate_variants_html_empty(self):
        """Test variants HTML with no variants."""
        html = generate_variants_html([])

        assert html == ""

    def test_generate_variants_html(self):
        """Test variants HTML generation."""
        variants = [
            ComponentVariant("Primary", "Main action"),
            ComponentVariant("Secondary", "Secondary action"),
        ]
        html = generate_variants_html(variants)

        assert "<h3>Variants</h3>" in html
        assert "Primary" in html
        assert "Secondary" in html

    def test_generate_page_html(self, sample_config):
        """Test full page generation."""
        html = generate_page_html(
            sample_config,
            title="Test Page",
            content="<p>Content here</p>",
        )

        assert "<!DOCTYPE html>" in html
        assert "<title>Test Page - Test Library</title>" in html
        assert "Content here" in html
        assert "toggleTheme" in html  # Theme toggle script

    def test_generate_page_html_with_sidebar(self, sample_config):
        """Test page with sidebar."""
        html = generate_page_html(
            sample_config,
            title="Test",
            content="<p>Main</p>",
            sidebar="<nav>Sidebar</nav>",
        )

        assert "Sidebar" in html
        assert "layout" in html


# =============================================================================
# LibrarySiteGenerator Tests
# =============================================================================

class TestLibrarySiteGenerator:
    """Tests for LibrarySiteGenerator class."""

    @pytest.fixture
    def sample_library(self):
        """Create sample library for generation."""
        library = ComponentLibrary("Test Library")

        # Add components
        button_spec = VisualSpec("btn", "Button")
        button_spec.set_colors(background="#3b82f6")
        library.add_component(
            button_spec,
            name="Button",
            description="A button component",
            category="Inputs",
            tags=["interactive"],
        )

        card_spec = VisualSpec("card", "Card")
        library.add_component(
            card_spec,
            name="Card",
            description="A card component",
            category="Layout",
        )

        # Add tokens
        colors = TokenGroup(type=TokenType.COLOR)
        colors.add("primary", DesignToken.color("#3b82f6"))
        colors.add("secondary", DesignToken.color("#64748b"))
        library.add_token_group("colors", colors)

        # Add guide
        library.add_guide("Getting Started", "<p>Welcome!</p>")

        return library

    def test_generator_initialization(self, sample_library):
        """Test generator initialization."""
        generator = LibrarySiteGenerator(sample_library)

        assert generator.library == sample_library
        assert generator.config == sample_library.config

    def test_generate_creates_directory(self, sample_library, tmp_path):
        """Test that generate creates output directory."""
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        assert output_dir.exists()

    def test_generate_creates_index(self, sample_library, tmp_path):
        """Test that generate creates index.html."""
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        assert (output_dir / "index.html").exists()
        content = (output_dir / "index.html").read_text()
        assert "Button" in content
        assert "Card" in content

    def test_generate_creates_component_pages(self, sample_library, tmp_path):
        """Test that generate creates component pages."""
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        assert (output_dir / "button.html").exists()
        assert (output_dir / "card.html").exists()

    def test_generate_creates_tokens_page(self, sample_library, tmp_path):
        """Test that generate creates tokens page."""
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        tokens_path = output_dir / "tokens.html"
        assert tokens_path.exists()
        content = tokens_path.read_text()
        assert "Design Tokens" in content

    def test_generate_creates_guides_page(self, sample_library, tmp_path):
        """Test that generate creates guides page."""
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        guides_path = output_dir / "guides.html"
        assert guides_path.exists()

        # Individual guide page
        assert (output_dir / "guide-getting-started.html").exists()

    def test_generate_creates_library_data(self, sample_library, tmp_path):
        """Test that generate creates library-data.json."""
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        data_path = output_dir / "library-data.json"
        assert data_path.exists()
        data = json.loads(data_path.read_text())
        assert "components" in data

    def test_component_page_content(self, sample_library, tmp_path):
        """Test component page content."""
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        content = (output_dir / "button.html").read_text()
        assert "Button" in content
        assert "A button component" in content
        assert "preview-container" in content

    def test_search_in_index(self, sample_library, tmp_path):
        """Test search functionality in index."""
        sample_library.config.show_search = True
        output_dir = tmp_path / "docs"
        generator = LibrarySiteGenerator(sample_library)

        generator.generate(output_dir)

        content = (output_dir / "index.html").read_text()
        assert "search-input" in content
        assert "filterComponents" in content


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_library_site(self, tmp_path):
        """Test generate_library_site function."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("btn", "Button")
        library.add_component(spec)

        output_path = generate_library_site(library, tmp_path / "docs")

        assert output_path.exists()
        assert (output_path / "index.html").exists()

    def test_create_library_from_specs(self):
        """Test creating library from specs."""
        specs = [
            VisualSpec("btn", "Button"),
            VisualSpec("card", "Card"),
        ]

        library = create_library_from_specs(
            specs,
            name="My Library",
            description="A test library",
        )

        assert library.name == "My Library"
        assert len(library.components) == 2

    def test_create_library_from_ui_spec(self):
        """Test creating library from UISpec."""
        ui_spec = UISpec(name="App", version="1.0.0")

        page = PageSpec(name="Home", route="/")
        page.add_component(VisualSpec("btn", "Button"))
        ui_spec.add_page(page)

        library = create_library_from_ui_spec(ui_spec)

        assert library.name == "App"
        assert len(library.components) == 1

    def test_quick_library(self):
        """Test quick_library helper."""
        library = quick_library(
            VisualSpec("btn", "Button"),
            VisualSpec("card", "Card"),
            name="Quick Lib",
        )

        assert library.name == "Quick Lib"
        assert len(library.components) == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete library creation and generation workflow."""
        # Create library
        library = ComponentLibrary("My Design System")
        library.config.description = "A complete design system"
        library.config.github_url = "https://github.com/example/repo"

        # Add categories
        library.add_category("Inputs", "Form input components", "📝", order=1)
        library.add_category("Layout", "Layout components", "📐", order=2)

        # Add components
        button_spec = VisualSpec("btn-primary", "Button")
        button_spec.set_colors(background="#3b82f6", text="#ffffff")
        button = library.add_component(
            button_spec,
            name="Primary Button",
            description="A primary action button",
            category="Inputs",
            tags=["interactive", "form", "action"],
        )
        button.add_variant("Primary", "Default primary style")
        button.add_variant("Secondary", "Secondary action")
        button.add_prop("onClick", "() => void", "Click handler", required=True)
        button.add_prop("disabled", "boolean", "Disable the button", default=False)

        card_spec = VisualSpec("card", "Card")
        library.add_component(
            card_spec,
            name="Card",
            description="A content container card",
            category="Layout",
        )

        # Add design tokens
        colors = TokenGroup(type=TokenType.COLOR)
        colors.add("primary", DesignToken.color("#3b82f6", "Primary brand color"))
        colors.add("secondary", DesignToken.color("#64748b", "Secondary color"))
        library.add_token_group("colors", colors)

        # Add guide
        library.add_guide(
            "Getting Started",
            "<h2>Installation</h2><p>npm install my-design-system</p>",
        )

        # Generate site
        output_path = generate_library_site(library, tmp_path / "docs")

        # Verify all files
        assert (output_path / "index.html").exists()
        assert (output_path / "primary-button.html").exists()
        assert (output_path / "card.html").exists()
        assert (output_path / "tokens.html").exists()
        assert (output_path / "guides.html").exists()
        assert (output_path / "guide-getting-started.html").exists()
        assert (output_path / "library-data.json").exists()

        # Verify content
        index_content = (output_path / "index.html").read_text(encoding="utf-8")
        assert "My Design System" in index_content
        assert "Primary Button" in index_content
        assert "search-input" in index_content

        button_content = (output_path / "primary-button.html").read_text(encoding="utf-8")
        assert "Primary Button" in button_content
        assert "onClick" in button_content
        assert "Variants" in button_content

        tokens_content = (output_path / "tokens.html").read_text(encoding="utf-8")
        assert "Colors" in tokens_content or "colors" in tokens_content.lower()

    def test_empty_library_generation(self, tmp_path):
        """Test generating site for empty library."""
        library = ComponentLibrary("Empty Library")

        output_path = generate_library_site(library, tmp_path / "docs")

        assert (output_path / "index.html").exists()
        assert (output_path / "tokens.html").exists()

    def test_library_with_many_components(self, tmp_path):
        """Test library with many components."""
        library = ComponentLibrary("Large Library")

        # Add 20 components
        for i in range(20):
            spec = VisualSpec(f"comp-{i}", "Component")
            library.add_component(
                spec,
                name=f"Component {i}",
                description=f"Component number {i}",
                category=f"Category {i % 5}",
            )

        output_path = generate_library_site(library, tmp_path / "docs")

        # All component pages should exist
        for i in range(20):
            assert (output_path / f"component-{i}.html").exists()

        # Check index has all components
        index_content = (output_path / "index.html").read_text()
        assert "Component 0" in index_content
        assert "Component 19" in index_content


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_component_with_special_characters_in_name(self, tmp_path):
        """Test component with special characters."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("btn", "Button")
        entry = library.add_component(
            spec,
            name="Button <Primary>",
            description="A button with <html> & special chars",
        )

        # Slug should be sanitized
        assert entry.slug == "button-primary"

        output_path = generate_library_site(library, tmp_path / "docs")

        # HTML should be properly escaped
        content = (output_path / "index.html").read_text(encoding="utf-8")
        assert "&lt;Primary&gt;" in content

        # Sanitized filename should exist
        assert (output_path / "button-primary.html").exists()

    def test_component_with_unicode(self, tmp_path):
        """Test component with unicode characters."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("btn", "Button")
        library.add_component(
            spec,
            name="按钮 Button",
            description="中文描述 Chinese description",
        )

        output_path = generate_library_site(library, tmp_path / "docs")

        content = (output_path / "index.html").read_text(encoding="utf-8")
        assert "按钮" in content

    def test_empty_token_group(self, tmp_path):
        """Test library with empty token group."""
        library = ComponentLibrary("Test")
        library.add_token_group("empty", TokenGroup(type=TokenType.COLOR))

        output_path = generate_library_site(library, tmp_path / "docs")

        tokens_content = (output_path / "tokens.html").read_text()
        assert "Design Tokens" in tokens_content

    def test_guide_with_html_content(self, tmp_path):
        """Test guide with HTML content."""
        library = ComponentLibrary("Test")
        library.add_guide(
            "HTML Guide",
            "<h2>Heading</h2><p>Paragraph with <strong>bold</strong></p>",
        )

        output_path = generate_library_site(library, tmp_path / "docs")

        guide_content = (output_path / "guide-html-guide.html").read_text()
        assert "<h2>Heading</h2>" in guide_content
        assert "<strong>bold</strong>" in guide_content

    def test_component_without_colors(self, tmp_path):
        """Test component spec without colors."""
        library = ComponentLibrary("Test")
        spec = VisualSpec("btn", "Button")  # No colors set
        library.add_component(spec)

        output_path = generate_library_site(library, tmp_path / "docs")

        # Should still generate without error
        assert (output_path / "btn.html").exists()

    def test_multiple_guides_ordering(self, tmp_path):
        """Test guides are ordered correctly."""
        library = ComponentLibrary("Test")
        library.add_guide("Third", "Content", order=3)
        library.add_guide("First", "Content", order=1)
        library.add_guide("Second", "Content", order=2)

        output_path = generate_library_site(library, tmp_path / "docs")

        guides_content = (output_path / "guides.html").read_text()
        first_pos = guides_content.find("First")
        second_pos = guides_content.find("Second")
        third_pos = guides_content.find("Third")

        assert first_pos < second_pos < third_pos
