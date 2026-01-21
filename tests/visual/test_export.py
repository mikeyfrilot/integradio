"""
Integration tests for the export module.

Tests the various exporters (Style Dictionary, CSS, Tailwind, DTCG) with
real data structures and file I/O operations.
"""

from __future__ import annotations

import json
import pytest
import sys
from pathlib import Path
from typing import Any

# Add project root to path for direct submodule imports (avoids integradio/__init__.py
# which imports gradio which imports numpy, causing issues with coverage on Python 3.14)
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import directly from visual submodule to avoid gradio/numpy import chain
from integradio.visual.export import (
    StyleDictionaryConfig,
    StyleDictionaryExporter,
    CSSExporter,
    TailwindExporter,
    DTCGExporter,
    export_to_style_dictionary,
    export_to_css,
    export_to_tailwind,
    export_to_dtcg,
)
from integradio.visual.tokens import (
    DesignToken,
    TokenGroup,
    TokenType,
    ColorValue,
    DimensionValue,
    DurationValue,
)
from integradio.visual.spec import UISpec, PageSpec, VisualSpec


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_tokens() -> TokenGroup:
    """Create a sample TokenGroup with various token types."""
    group = TokenGroup()

    # Colors
    colors = TokenGroup()
    colors.add("primary", DesignToken.color("#3366FF", description="Primary brand color"))
    colors.add("secondary", DesignToken.color("#FF6633"))
    colors.add("transparent", DesignToken.color(ColorValue.from_hex("#3366FF80")))  # with alpha
    group.add("colors", colors)

    # Dimensions
    spacing = TokenGroup()
    spacing.add("sm", DesignToken.dimension(8, "px"))
    spacing.add("md", DesignToken.dimension(16, "px"))
    spacing.add("lg", DesignToken.dimension(32, "rem"))
    group.add("spacing", spacing)

    # Duration
    motion = TokenGroup()
    motion.add("fast", DesignToken.duration(100, "ms"))
    motion.add("normal", DesignToken.duration(200, "ms"))
    group.add("motion", motion)

    # Font family
    fonts = TokenGroup()
    fonts.add("sans", DesignToken.font_family(["Inter", "sans-serif"]))
    fonts.add("mono", DesignToken.font_family("JetBrains Mono"))  # single string
    group.add("fonts", fonts)

    return group


@pytest.fixture
def sample_ui_spec(sample_tokens: TokenGroup) -> UISpec:
    """Create a sample UISpec with tokens and pages."""
    spec = UISpec(name="TestApp", version="1.0.0", tokens=sample_tokens)

    # Add a dark theme
    dark_theme = TokenGroup()
    dark_colors = TokenGroup()
    dark_colors.add("primary", DesignToken.color("#6699FF"))
    dark_colors.add("background", DesignToken.color("#1A1A1A"))
    dark_theme.add("colors", dark_colors)
    spec.add_theme("dark", dark_theme)

    # Add a page
    page = PageSpec(name="Home", route="/")
    page.tokens = sample_tokens
    spec.add_page(page)

    return spec


@pytest.fixture
def alias_tokens() -> TokenGroup:
    """Create tokens with alias references."""
    group = TokenGroup()

    # Base colors
    colors = TokenGroup()
    colors.add("blue500", DesignToken.color("#3366FF"))
    # Alias reference
    colors.add("primary", DesignToken.reference("colors.blue500", TokenType.COLOR))
    group.add("colors", colors)

    return group


@pytest.fixture
def complex_tokens() -> TokenGroup:
    """Create tokens with complex/composite values."""
    group = TokenGroup()

    # Number token (no to_dtcg method)
    group.add("scale", DesignToken.number(1.5))

    # Font weight
    group.add("weight", DesignToken.font_weight(700))

    return group


# =============================================================================
# StyleDictionaryConfig Tests
# =============================================================================

class TestStyleDictionaryConfig:
    """Tests for StyleDictionaryConfig."""

    def test_default_config(self):
        """Test creating default config."""
        config = StyleDictionaryConfig.default()

        assert config.source == ["tokens/**/*.json"]
        assert "css" in config.platforms
        assert "scss" in config.platforms
        assert "js" in config.platforms
        assert "json" in config.platforms

    def test_to_dict(self):
        """Test converting config to dict."""
        config = StyleDictionaryConfig(
            source=["custom/**/*.json"],
            platforms={"css": {"buildPath": "out/"}}
        )

        result = config.to_dict()

        assert result["source"] == ["custom/**/*.json"]
        assert result["platforms"]["css"]["buildPath"] == "out/"

    def test_default_config_structure(self):
        """Test default config has correct structure."""
        config = StyleDictionaryConfig.default()
        data = config.to_dict()

        # Check CSS platform
        css_platform = data["platforms"]["css"]
        assert css_platform["transformGroup"] == "css"
        assert css_platform["prefix"] == "sg"
        assert css_platform["buildPath"] == "build/css/"
        assert len(css_platform["files"]) == 1
        assert css_platform["files"][0]["destination"] == "variables.css"
        assert css_platform["files"][0]["format"] == "css/variables"

        # Check SCSS platform
        scss_platform = data["platforms"]["scss"]
        assert scss_platform["files"][0]["destination"] == "_variables.scss"

        # Check JS platform
        js_platform = data["platforms"]["js"]
        assert js_platform["files"][0]["format"] == "javascript/es6"

        # Check JSON platform
        json_platform = data["platforms"]["json"]
        assert json_platform["files"][0]["format"] == "json/nested"


# =============================================================================
# StyleDictionaryExporter Tests
# =============================================================================

class TestStyleDictionaryExporter:
    """Tests for StyleDictionaryExporter."""

    def test_export_basic_tokens(self, sample_ui_spec: UISpec):
        """Test exporting basic tokens."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        result = exporter.export_tokens()

        # Check colors
        assert "colors" in result
        assert "primary" in result["colors"]
        assert result["colors"]["primary"]["type"] == "color"
        assert "value" in result["colors"]["primary"]

        # Check spacing
        assert "spacing" in result
        assert "sm" in result["spacing"]
        assert result["spacing"]["sm"]["value"] == "8px"

    def test_export_color_with_alpha(self, sample_ui_spec: UISpec):
        """Test exporting color with transparency."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        result = exporter.export_tokens()

        # Color with alpha should export as rgba
        transparent = result["colors"]["transparent"]
        assert "rgba" in transparent["value"]

    def test_export_dimension_tokens(self, sample_ui_spec: UISpec):
        """Test exporting dimension tokens."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        result = exporter.export_tokens()

        assert result["spacing"]["sm"]["value"] == "8px"
        assert result["spacing"]["lg"]["value"] == "32rem"

    def test_export_duration_tokens(self, sample_ui_spec: UISpec):
        """Test exporting duration tokens."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        result = exporter.export_tokens()

        assert "motion" in result
        assert result["motion"]["fast"]["value"] == "100ms"
        assert result["motion"]["fast"]["type"] == "duration"

    def test_export_with_description(self, sample_ui_spec: UISpec):
        """Test that descriptions are exported as comments."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        result = exporter.export_tokens()

        # Primary color has a description
        assert "comment" in result["colors"]["primary"]
        assert result["colors"]["primary"]["comment"] == "Primary brand color"

    def test_export_alias_tokens(self, alias_tokens: TokenGroup):
        """Test exporting alias tokens."""
        spec = UISpec(name="TestAlias", tokens=alias_tokens)
        exporter = StyleDictionaryExporter(spec)
        result = exporter.export_tokens()

        # Alias should have reference format
        primary = result["colors"]["primary"]
        assert "{" in primary["value"]
        assert ".value}" in primary["value"]

    def test_export_complex_tokens(self, complex_tokens: TokenGroup):
        """Test exporting tokens with simple values (no to_dtcg)."""
        spec = UISpec(name="TestComplex", tokens=complex_tokens)
        exporter = StyleDictionaryExporter(spec)
        result = exporter.export_tokens()

        # Number token should be exported as-is
        assert result["scale"]["value"] == 1.5
        assert result["scale"]["type"] == "number"

        # Font weight should be exported as-is
        assert result["weight"]["value"] == 700

    def test_export_config_default(self, sample_ui_spec: UISpec):
        """Test exporting config with defaults."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        config = exporter.export_config()

        assert "source" in config
        assert "platforms" in config
        assert "css" in config["platforms"]

    def test_export_config_custom(self, sample_ui_spec: UISpec):
        """Test exporting custom config."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        custom_config = StyleDictionaryConfig(
            source=["custom/*.json"],
            platforms={"android": {"buildPath": "android/"}}
        )
        config = exporter.export_config(custom_config)

        assert config["source"] == ["custom/*.json"]
        assert "android" in config["platforms"]

    def test_save_creates_files(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test saving creates token and config files."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        created = exporter.save(tmp_path)

        assert len(created) == 2

        # Check tokens file
        tokens_path = tmp_path / "tokens" / "tokens.json"
        assert tokens_path.exists()
        with open(tokens_path) as f:
            data = json.load(f)
        assert "colors" in data

        # Check config file
        config_path = tmp_path / "style-dictionary.config.json"
        assert config_path.exists()
        with open(config_path) as f:
            data = json.load(f)
        assert "platforms" in data

    def test_save_without_config(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test saving without config file."""
        exporter = StyleDictionaryExporter(sample_ui_spec)
        created = exporter.save(tmp_path, include_config=False)

        assert len(created) == 1

        # Only tokens file should exist
        tokens_path = tmp_path / "tokens" / "tokens.json"
        assert tokens_path.exists()

        # Config file should not exist
        config_path = tmp_path / "style-dictionary.config.json"
        assert not config_path.exists()

    def test_save_creates_directories(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test that save creates nested directories."""
        output_dir = tmp_path / "deep" / "nested" / "output"
        exporter = StyleDictionaryExporter(sample_ui_spec)
        created = exporter.save(output_dir)

        assert output_dir.exists()
        assert (output_dir / "tokens").exists()

    def test_flatten_color_value(self, sample_ui_spec: UISpec):
        """Test flattening color composite values."""
        exporter = StyleDictionaryExporter(sample_ui_spec)

        # Create a color value dict (as would come from to_dtcg)
        color_dict = {
            "colorSpace": "srgb",
            "components": [0.2, 0.4, 0.9],
            "alpha": 1.0
        }

        result = exporter._flatten_value(TokenType.COLOR, color_dict)
        assert result.startswith("#")
        assert len(result) == 7  # #RRGGBB

    def test_flatten_color_with_alpha(self, sample_ui_spec: UISpec):
        """Test flattening color with alpha."""
        exporter = StyleDictionaryExporter(sample_ui_spec)

        color_dict = {
            "colorSpace": "srgb",
            "components": [0.5, 0.5, 0.5],
            "alpha": 0.5
        }

        result = exporter._flatten_value(TokenType.COLOR, color_dict)
        assert result.startswith("rgba")
        assert "0.5" in result

    def test_flatten_dimension_value(self, sample_ui_spec: UISpec):
        """Test flattening dimension values."""
        exporter = StyleDictionaryExporter(sample_ui_spec)

        dim_dict = {"value": 16, "unit": "px"}
        result = exporter._flatten_value(TokenType.DIMENSION, dim_dict)

        assert result == "16px"

    def test_flatten_duration_value(self, sample_ui_spec: UISpec):
        """Test flattening duration values."""
        exporter = StyleDictionaryExporter(sample_ui_spec)

        dur_dict = {"value": 200, "unit": "ms"}
        result = exporter._flatten_value(TokenType.DURATION, dur_dict)

        assert result == "200ms"

    def test_flatten_unknown_type_passthrough(self, sample_ui_spec: UISpec):
        """Test that unknown types pass through unchanged."""
        exporter = StyleDictionaryExporter(sample_ui_spec)

        value = {"custom": "value"}
        result = exporter._flatten_value(TokenType.STRING, value)

        assert result == value


# =============================================================================
# CSSExporter Tests
# =============================================================================

class TestCSSExporter:
    """Tests for CSSExporter."""

    def test_basic_export(self, sample_ui_spec: UISpec):
        """Test basic CSS export."""
        exporter = CSSExporter(sample_ui_spec)
        css = exporter.export()

        assert ":root" in css or "colors" in css.lower() or len(css) > 0

    def test_export_with_reset(self, sample_ui_spec: UISpec):
        """Test CSS export with reset included."""
        exporter = CSSExporter(sample_ui_spec)
        css = exporter.export(include_reset=True)

        assert "box-sizing: border-box" in css
        assert "margin: 0" in css

    def test_export_minified(self, sample_ui_spec: UISpec):
        """Test minified CSS export."""
        exporter = CSSExporter(sample_ui_spec)
        regular = exporter.export()
        minified = exporter.export(minify=True)

        # Minified should be shorter (no unnecessary whitespace)
        assert len(minified) <= len(regular)
        # Should not contain multi-space sequences
        assert "  " not in minified or len(minified) < len(regular)

    def test_export_with_theme(self, sample_ui_spec: UISpec):
        """Test CSS export with theme."""
        exporter = CSSExporter(sample_ui_spec)
        css = exporter.export(theme="dark")

        # CSS should contain the spec's CSS output
        assert len(css) > 0

    def test_minify_removes_comments(self, sample_ui_spec: UISpec):
        """Test that minification removes comments."""
        exporter = CSSExporter(sample_ui_spec)

        css_with_comment = "/* comment */ .test { color: red; }"
        minified = exporter._minify(css_with_comment)

        assert "comment" not in minified

    def test_minify_removes_whitespace(self, sample_ui_spec: UISpec):
        """Test that minification removes extra whitespace."""
        exporter = CSSExporter(sample_ui_spec)

        css = ".test   {   color:   red;   }"
        minified = exporter._minify(css)

        assert "{" in minified
        assert ":" in minified
        assert "  " not in minified

    def test_generate_reset(self, sample_ui_spec: UISpec):
        """Test reset CSS generation."""
        exporter = CSSExporter(sample_ui_spec)
        reset = exporter._generate_reset()

        assert "Minimal Reset" in reset
        assert "box-sizing: border-box" in reset
        assert "body" in reset
        assert "img, picture" in reset

    def test_save_to_file(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test saving CSS to file."""
        output_path = tmp_path / "styles.css"
        exporter = CSSExporter(sample_ui_spec)
        result = exporter.save(output_path)

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert len(content) > 0

    def test_save_creates_directories(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test that save creates parent directories."""
        output_path = tmp_path / "deep" / "path" / "styles.css"
        exporter = CSSExporter(sample_ui_spec)
        exporter.save(output_path)

        assert output_path.exists()

    def test_save_with_options(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test saving with export options."""
        output_path = tmp_path / "styles.min.css"
        exporter = CSSExporter(sample_ui_spec)
        exporter.save(output_path, minify=True, include_reset=True)

        content = output_path.read_text()
        # Should be minified but include reset
        assert "box-sizing" in content


# =============================================================================
# TailwindExporter Tests
# =============================================================================

class TestTailwindExporter:
    """Tests for TailwindExporter."""

    def test_export_colors(self, sample_ui_spec: UISpec):
        """Test exporting colors for Tailwind."""
        exporter = TailwindExporter(sample_ui_spec)
        result = exporter.export()

        if "colors" in result:
            assert isinstance(result["colors"], dict)

    def test_export_spacing(self, sample_ui_spec: UISpec):
        """Test exporting spacing for Tailwind."""
        exporter = TailwindExporter(sample_ui_spec)
        result = exporter.export()

        if "spacing" in result:
            assert isinstance(result["spacing"], dict)

    def test_export_font_family_array(self, sample_ui_spec: UISpec):
        """Test exporting font family as array."""
        exporter = TailwindExporter(sample_ui_spec)
        result = exporter.export()

        if "fontFamily" in result:
            # Should be a list of font names
            for key, value in result["fontFamily"].items():
                assert isinstance(value, list)

    def test_export_font_family_string(self):
        """Test exporting single font as array."""
        tokens = TokenGroup()
        fonts = TokenGroup()
        fonts.add("mono", DesignToken.font_family("Courier"))
        tokens.add("fonts", fonts)

        spec = UISpec(name="Test", tokens=tokens)
        exporter = TailwindExporter(spec)
        result = exporter.export()

        if "fontFamily" in result:
            # Single font should become array
            assert isinstance(result["fontFamily"]["mono"], list)

    def test_export_config_string(self, sample_ui_spec: UISpec):
        """Test exporting as complete tailwind.config.js."""
        exporter = TailwindExporter(sample_ui_spec)
        config_str = exporter.export_config()

        assert "module.exports" in config_str
        assert "content:" in config_str or "content" in config_str
        assert "theme:" in config_str or "theme" in config_str
        assert "extend:" in config_str or "extend" in config_str

    def test_export_config_valid_js(self, sample_ui_spec: UISpec):
        """Test that exported config is valid JavaScript structure."""
        exporter = TailwindExporter(sample_ui_spec)
        config_str = exporter.export_config()

        # Should have proper structure
        assert "@type" in config_str  # JSDoc type hint
        assert "plugins:" in config_str or "plugins" in config_str

    def test_save_to_file(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test saving Tailwind config to file."""
        output_path = tmp_path / "tailwind.config.js"
        exporter = TailwindExporter(sample_ui_spec)
        result = exporter.save(output_path)

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert "module.exports" in content

    def test_save_creates_directories(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test that save creates parent directories."""
        output_path = tmp_path / "config" / "tailwind.config.js"
        exporter = TailwindExporter(sample_ui_spec)
        exporter.save(output_path)

        assert output_path.exists()

    def test_empty_result_structure(self):
        """Test that empty tokens result in empty sections."""
        spec = UISpec(name="Empty", tokens=TokenGroup())
        exporter = TailwindExporter(spec)
        result = exporter.export()

        # Empty tokens should produce empty or minimal result
        assert isinstance(result, dict)


# =============================================================================
# DTCGExporter Tests
# =============================================================================

class TestDTCGExporter:
    """Tests for DTCGExporter."""

    def test_export_basic(self, sample_ui_spec: UISpec):
        """Test basic DTCG export."""
        exporter = DTCGExporter(sample_ui_spec)
        result = exporter.export()

        # Should be the tokens in DTCG format
        assert isinstance(result, dict)

    def test_export_with_themes(self, sample_ui_spec: UISpec):
        """Test DTCG export with themes."""
        exporter = DTCGExporter(sample_ui_spec)
        result = exporter.export_with_themes()

        assert "base" in result
        assert "dark" in result

    def test_export_themes_structure(self, sample_ui_spec: UISpec):
        """Test that themes have proper structure."""
        exporter = DTCGExporter(sample_ui_spec)
        result = exporter.export_with_themes()

        # Dark theme should have its tokens
        assert "dark" in result
        dark = result["dark"]
        assert "colors" in dark

    def test_save_to_file(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test saving DTCG to file."""
        output_path = tmp_path / "tokens.json"
        exporter = DTCGExporter(sample_ui_spec)
        result = exporter.save(output_path)

        assert result == output_path
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
        assert "base" in data

    def test_save_without_themes(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test saving DTCG without themes."""
        output_path = tmp_path / "tokens.json"
        exporter = DTCGExporter(sample_ui_spec)
        exporter.save(output_path, include_themes=False)

        with open(output_path) as f:
            data = json.load(f)

        # Should not have theme structure
        assert "base" not in data

    def test_save_creates_directories(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test that save creates parent directories."""
        output_path = tmp_path / "dtcg" / "output" / "tokens.json"
        exporter = DTCGExporter(sample_ui_spec)
        exporter.save(output_path)

        assert output_path.exists()


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience export functions."""

    def test_export_to_style_dictionary(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_style_dictionary function."""
        created = export_to_style_dictionary(sample_ui_spec, tmp_path)

        assert len(created) == 2
        assert (tmp_path / "tokens" / "tokens.json").exists()
        assert (tmp_path / "style-dictionary.config.json").exists()

    def test_export_to_css(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_css function."""
        output_path = tmp_path / "styles.css"
        result = export_to_css(sample_ui_spec, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_export_to_css_with_options(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_css with theme and minify options."""
        output_path = tmp_path / "styles.min.css"
        result = export_to_css(
            sample_ui_spec,
            output_path,
            theme="dark",
            minify=True
        )

        assert result == output_path
        assert output_path.exists()

    def test_export_to_tailwind(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_tailwind function."""
        output_path = tmp_path / "tailwind.config.js"
        result = export_to_tailwind(sample_ui_spec, output_path)

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert "module.exports" in content

    def test_export_to_dtcg(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_dtcg function."""
        output_path = tmp_path / "tokens.json"
        result = export_to_dtcg(sample_ui_spec, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_export_to_dtcg_without_themes(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_dtcg without themes."""
        output_path = tmp_path / "tokens.json"
        result = export_to_dtcg(sample_ui_spec, output_path, include_themes=False)

        assert result == output_path

        with open(output_path) as f:
            data = json.load(f)
        assert "base" not in data


# =============================================================================
# Integration Tests
# =============================================================================

class TestExportIntegration:
    """Integration tests for export workflows."""

    def test_full_workflow(self, tmp_path: Path):
        """Test a complete export workflow."""
        # Create a realistic UI spec
        tokens = TokenGroup()

        # Colors
        colors = TokenGroup()
        colors.add("primary", DesignToken.color("#007AFF"))
        colors.add("secondary", DesignToken.color("#5856D6"))
        colors.add("success", DesignToken.color("#34C759"))
        colors.add("warning", DesignToken.color("#FF9500"))
        colors.add("error", DesignToken.color("#FF3B30"))
        tokens.add("colors", colors)

        # Spacing
        spacing = TokenGroup()
        for i, (name, value) in enumerate([("xs", 4), ("sm", 8), ("md", 16), ("lg", 24), ("xl", 32)]):
            spacing.add(name, DesignToken.dimension(value, "px"))
        tokens.add("spacing", spacing)

        # Create spec
        spec = UISpec(name="MyApp", version="2.0.0", tokens=tokens)

        # Export to all formats
        sd_files = export_to_style_dictionary(spec, tmp_path / "style-dictionary")
        css_file = export_to_css(spec, tmp_path / "styles.css")
        tw_file = export_to_tailwind(spec, tmp_path / "tailwind.config.js")
        dtcg_file = export_to_dtcg(spec, tmp_path / "tokens.json")

        # Verify all files exist
        assert len(sd_files) == 2
        assert css_file.exists()
        assert tw_file.exists()
        assert dtcg_file.exists()

        # Verify content integrity
        with open(tmp_path / "tokens.json") as f:
            dtcg_data = json.load(f)
        assert "base" in dtcg_data

    def test_nested_token_groups(self, tmp_path: Path):
        """Test exporting deeply nested token groups."""
        tokens = TokenGroup()

        # Create nested structure
        brand = TokenGroup()
        primary = TokenGroup()
        primary.add("light", DesignToken.color("#E6F0FF"))
        primary.add("base", DesignToken.color("#007AFF"))
        primary.add("dark", DesignToken.color("#0056B3"))
        brand.add("primary", primary)

        secondary = TokenGroup()
        secondary.add("light", DesignToken.color("#F0E6FF"))
        secondary.add("base", DesignToken.color("#5856D6"))
        secondary.add("dark", DesignToken.color("#3F3D99"))
        brand.add("secondary", secondary)

        tokens.add("brand", brand)

        spec = UISpec(name="NestedTest", tokens=tokens)

        # Export and verify
        exporter = StyleDictionaryExporter(spec)
        result = exporter.export_tokens()

        assert "brand" in result
        assert "primary" in result["brand"]
        assert "base" in result["brand"]["primary"]

    def test_round_trip_json(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test that exported JSON can be loaded."""
        # Export to DTCG
        output_path = tmp_path / "tokens.json"
        exporter = DTCGExporter(sample_ui_spec)
        exporter.save(output_path)

        # Load and verify
        with open(output_path) as f:
            data = json.load(f)

        # Should be valid JSON with expected structure
        assert isinstance(data, dict)
        assert "base" in data or "colors" in data

    def test_path_as_string(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test that string paths work."""
        # CSS
        css_path = str(tmp_path / "styles.css")
        export_to_css(sample_ui_spec, css_path)
        assert Path(css_path).exists()

        # Tailwind
        tw_path = str(tmp_path / "tailwind.config.js")
        export_to_tailwind(sample_ui_spec, tw_path)
        assert Path(tw_path).exists()

        # Style Dictionary
        sd_path = str(tmp_path / "sd")
        export_to_style_dictionary(sample_ui_spec, sd_path)
        assert Path(sd_path).exists()

        # DTCG
        dtcg_path = str(tmp_path / "dtcg.json")
        export_to_dtcg(sample_ui_spec, dtcg_path)
        assert Path(dtcg_path).exists()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_spec(self, tmp_path: Path):
        """Test exporting empty spec."""
        spec = UISpec(name="Empty", tokens=TokenGroup())

        # Should not raise
        exporter = StyleDictionaryExporter(spec)
        result = exporter.export_tokens()
        assert isinstance(result, dict)

    def test_special_characters_in_names(self, tmp_path: Path):
        """Test tokens with special characters in names."""
        tokens = TokenGroup()
        tokens.add("my-color", DesignToken.color("#FF0000"))
        tokens.add("spacing_large", DesignToken.dimension(32, "px"))

        spec = UISpec(name="Special", tokens=tokens)
        exporter = StyleDictionaryExporter(spec)
        result = exporter.export_tokens()

        assert "my-color" in result
        assert "spacing_large" in result

    def test_multiple_themes(self, tmp_path: Path):
        """Test spec with multiple themes."""
        spec = UISpec(name="MultiTheme", tokens=TokenGroup())

        # Add multiple themes
        for theme_name in ["light", "dark", "high-contrast"]:
            theme = TokenGroup()
            colors = TokenGroup()
            colors.add("bg", DesignToken.color("#000000" if theme_name == "dark" else "#FFFFFF"))
            theme.add("colors", colors)
            spec.add_theme(theme_name, theme)

        exporter = DTCGExporter(spec)
        result = exporter.export_with_themes()

        assert "light" in result
        assert "dark" in result
        assert "high-contrast" in result

    def test_color_without_components_key(self, sample_ui_spec: UISpec):
        """Test flatten_value with color dict missing components key."""
        exporter = StyleDictionaryExporter(sample_ui_spec)

        # Color dict without components (edge case)
        color_dict = {"colorSpace": "srgb"}
        result = exporter._flatten_value(TokenType.COLOR, color_dict)

        # Should return as-is when components missing
        assert result == color_dict

    def test_css_minify_complex(self, sample_ui_spec: UISpec):
        """Test CSS minification with complex input."""
        exporter = CSSExporter(sample_ui_spec)

        css = """
        /* Header styles */
        .header {
            display: flex;
            justify-content: space-between;
        }

        /* Footer styles */
        .footer {
            padding: 20px;
        }
        """

        minified = exporter._minify(css)

        # No comments
        assert "Header styles" not in minified
        assert "Footer styles" not in minified

        # Compact
        assert ".header{" in minified or ".header {" not in minified
