"""
Export - Style Dictionary and other format exports.

Converts visual specifications to various output formats:
- Style Dictionary JSON (for cross-platform token generation)
- CSS custom properties
- Tailwind config
- Figma tokens (W3C DTCG format)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from .tokens import (
    DesignToken,
    TokenGroup,
    TokenType,
    ColorValue,
    DimensionValue,
)
from .spec import UISpec, PageSpec, VisualSpec, BREAKPOINTS


# =============================================================================
# Style Dictionary Export
# =============================================================================

@dataclass
class StyleDictionaryConfig:
    """Style Dictionary configuration."""
    source: list[str] = field(default_factory=lambda: ["tokens/**/*.json"])
    platforms: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Export to dict."""
        return {
            "source": self.source,
            "platforms": self.platforms,
        }

    @classmethod
    def default(cls) -> "StyleDictionaryConfig":
        """Create default multi-platform config."""
        return cls(
            source=["tokens/**/*.json"],
            platforms={
                "css": {
                    "transformGroup": "css",
                    "prefix": "sg",
                    "buildPath": "build/css/",
                    "files": [
                        {
                            "destination": "variables.css",
                            "format": "css/variables",
                        }
                    ],
                },
                "scss": {
                    "transformGroup": "scss",
                    "prefix": "sg",
                    "buildPath": "build/scss/",
                    "files": [
                        {
                            "destination": "_variables.scss",
                            "format": "scss/variables",
                        }
                    ],
                },
                "js": {
                    "transformGroup": "js",
                    "buildPath": "build/js/",
                    "files": [
                        {
                            "destination": "tokens.js",
                            "format": "javascript/es6",
                        }
                    ],
                },
                "json": {
                    "transformGroup": "web",
                    "buildPath": "build/json/",
                    "files": [
                        {
                            "destination": "tokens.json",
                            "format": "json/nested",
                        }
                    ],
                },
            },
        )


class StyleDictionaryExporter:
    """Export UISpec to Style Dictionary format."""

    def __init__(self, spec: UISpec):
        self.spec = spec

    def export_tokens(self) -> dict:
        """
        Export tokens in Style Dictionary format.

        Style Dictionary uses a slightly different format than DTCG:
        - Uses `value` instead of `$value`
        - Uses `type` instead of `$type`

        Returns:
            Dict ready for JSON serialization
        """
        return self._convert_group(self.spec.tokens)

    def _convert_group(self, group: TokenGroup) -> dict:
        """Convert a TokenGroup to Style Dictionary format."""
        result: dict[str, Any] = {}

        for name, item in group.tokens.items():
            if isinstance(item, DesignToken):
                result[name] = self._convert_token(item)
            elif isinstance(item, TokenGroup):
                result[name] = self._convert_group(item)

        return result

    def _convert_token(self, token: DesignToken) -> dict:
        """Convert a DesignToken to Style Dictionary format."""
        result: dict[str, Any] = {}

        # Handle value based on type
        if token.is_alias:
            # Convert DTCG alias format to Style Dictionary format
            # DTCG: "{colors.primary}" -> SD: "{colors.primary.value}"
            ref = token._reference
            result["value"] = f"{{{ref}.value}}"
        elif hasattr(token.value, "to_dtcg"):
            dtcg_value = token.value.to_dtcg()
            # Flatten composite values for Style Dictionary
            if isinstance(dtcg_value, dict):
                result["value"] = self._flatten_value(token.type, dtcg_value)
            else:
                result["value"] = dtcg_value
        else:
            result["value"] = token.value

        # Add type
        result["type"] = token.type.value

        # Add description as comment
        if token.description:
            result["comment"] = token.description

        return result

    def _flatten_value(self, token_type: TokenType, value: dict) -> Any:
        """Flatten composite values for Style Dictionary compatibility."""
        if token_type == TokenType.COLOR:
            # Convert color object to hex string
            if "components" in value:
                r, g, b = value["components"]
                alpha = value.get("alpha", 1.0)
                if alpha < 1.0:
                    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"
                return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            return value

        if token_type == TokenType.DIMENSION:
            return f"{value['value']}{value['unit']}"

        if token_type == TokenType.DURATION:
            return f"{value['value']}{value['unit']}"

        # Return as-is for other types
        return value

    def export_config(self, config: StyleDictionaryConfig | None = None) -> dict:
        """Export Style Dictionary config."""
        if config is None:
            config = StyleDictionaryConfig.default()
        return config.to_dict()

    def save(
        self,
        output_dir: str | Path,
        include_config: bool = True,
    ) -> list[Path]:
        """
        Save tokens and config to files.

        Args:
            output_dir: Directory to save files
            include_config: Whether to include style-dictionary.config.json

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Save tokens
        tokens_dir = output_dir / "tokens"
        tokens_dir.mkdir(exist_ok=True)

        tokens_file = tokens_dir / "tokens.json"
        with open(tokens_file, "w") as f:
            json.dump(self.export_tokens(), f, indent=2)
        created_files.append(tokens_file)

        # Save config
        if include_config:
            config_file = output_dir / "style-dictionary.config.json"
            with open(config_file, "w") as f:
                json.dump(self.export_config(), f, indent=2)
            created_files.append(config_file)

        return created_files


# =============================================================================
# CSS Export
# =============================================================================

class CSSExporter:
    """Export UISpec to CSS."""

    def __init__(self, spec: UISpec):
        self.spec = spec

    def export(
        self,
        theme: str | None = None,
        include_reset: bool = False,
        minify: bool = False,
    ) -> str:
        """
        Export to CSS string.

        Args:
            theme: Theme variant to include
            include_reset: Whether to include a CSS reset
            minify: Whether to minify output

        Returns:
            CSS string
        """
        parts = []

        # Optional reset
        if include_reset:
            parts.append(self._generate_reset())

        # Main CSS
        parts.append(self.spec.to_css(theme))

        css = "\n\n".join(parts)

        if minify:
            css = self._minify(css)

        return css

    def _generate_reset(self) -> str:
        """Generate minimal CSS reset."""
        return """
/* Minimal Reset */
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0; line-height: 1.5; -webkit-font-smoothing: antialiased; }
img, picture, video, canvas, svg { display: block; max-width: 100%; }
input, button, textarea, select { font: inherit; }
p, h1, h2, h3, h4, h5, h6 { overflow-wrap: break-word; }
""".strip()

    def _minify(self, css: str) -> str:
        """Basic CSS minification."""
        import re
        # Remove comments
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
        # Remove whitespace
        css = re.sub(r"\s+", " ", css)
        # Remove space around braces/colons/semicolons
        css = re.sub(r"\s*{\s*", "{", css)
        css = re.sub(r"\s*}\s*", "}", css)
        css = re.sub(r"\s*:\s*", ":", css)
        css = re.sub(r"\s*;\s*", ";", css)
        return css.strip()

    def save(self, path: str | Path, **kwargs) -> Path:
        """Save CSS to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.export(**kwargs))

        return path


# =============================================================================
# Tailwind Export
# =============================================================================

class TailwindExporter:
    """Export UISpec to Tailwind config."""

    def __init__(self, spec: UISpec):
        self.spec = spec

    def export(self) -> dict:
        """
        Export to Tailwind config format.

        Returns:
            Dict for tailwind.config.js theme.extend
        """
        flat_tokens = self.spec.tokens.flatten()

        # Group by type
        colors = {}
        spacing = {}
        font_family = {}
        font_size = {}

        for path, token in flat_tokens.items():
            # Convert path to Tailwind key (dots to dashes)
            key = path.split(".")[-1]  # Use last segment as key

            if token.type == TokenType.COLOR:
                colors[key] = token.to_css()

            elif token.type == TokenType.DIMENSION:
                spacing[key] = token.to_css()

            elif token.type == TokenType.FONT_FAMILY:
                if isinstance(token.value, list):
                    font_family[key] = token.value
                else:
                    font_family[key] = [token.value]

        result: dict[str, Any] = {}

        if colors:
            result["colors"] = colors
        if spacing:
            result["spacing"] = spacing
        if font_family:
            result["fontFamily"] = font_family
        if font_size:
            result["fontSize"] = font_size

        return result

    def export_config(self) -> str:
        """
        Export as complete tailwind.config.js content.

        Returns:
            JavaScript module content
        """
        theme_extend = json.dumps(self.export(), indent=2)

        return f"""/** @type {{import('tailwindcss').Config}} */
module.exports = {{
  content: [
    "./src/**/*.{{js,jsx,ts,tsx,html}}",
  ],
  theme: {{
    extend: {theme_extend},
  }},
  plugins: [],
}}
"""

    def save(self, path: str | Path) -> Path:
        """Save Tailwind config to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.export_config())

        return path


# =============================================================================
# DTCG Export (Figma-compatible)
# =============================================================================

class DTCGExporter:
    """Export UISpec to W3C DTCG format (Figma-compatible)."""

    def __init__(self, spec: UISpec):
        self.spec = spec

    def export(self) -> dict:
        """
        Export to DTCG format.

        This format is compatible with:
        - Figma Tokens plugin
        - Tokens Studio
        - Any W3C DTCG compliant tool

        Returns:
            DTCG-formatted dict
        """
        return self.spec.tokens.to_dtcg()

    def export_with_themes(self) -> dict:
        """
        Export tokens with theme variants.

        Returns:
            Dict with base tokens and theme overrides
        """
        result = {
            "base": self.export(),
        }

        for theme_name, theme_tokens in self.spec.themes.items():
            result[theme_name] = theme_tokens.to_dtcg()

        return result

    def save(self, path: str | Path, include_themes: bool = True) -> Path:
        """Save DTCG tokens to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.export_with_themes() if include_themes else self.export()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path


# =============================================================================
# Convenience Functions
# =============================================================================

def export_to_style_dictionary(
    spec: UISpec,
    output_dir: str | Path,
) -> list[Path]:
    """
    Quick export to Style Dictionary format.

    Args:
        spec: UISpec to export
        output_dir: Output directory

    Returns:
        List of created files
    """
    exporter = StyleDictionaryExporter(spec)
    return exporter.save(output_dir)


def export_to_css(
    spec: UISpec,
    path: str | Path,
    theme: str | None = None,
    minify: bool = False,
) -> Path:
    """
    Quick export to CSS.

    Args:
        spec: UISpec to export
        path: Output file path
        theme: Theme to include
        minify: Whether to minify

    Returns:
        Path to created file
    """
    exporter = CSSExporter(spec)
    return exporter.save(path, theme=theme, minify=minify)


def export_to_tailwind(spec: UISpec, path: str | Path) -> Path:
    """
    Quick export to Tailwind config.

    Args:
        spec: UISpec to export
        path: Output file path

    Returns:
        Path to created file
    """
    exporter = TailwindExporter(spec)
    return exporter.save(path)


def export_to_dtcg(
    spec: UISpec,
    path: str | Path,
    include_themes: bool = True,
) -> Path:
    """
    Quick export to DTCG format.

    Args:
        spec: UISpec to export
        path: Output file path
        include_themes: Whether to include theme variants

    Returns:
        Path to created file
    """
    exporter = DTCGExporter(spec)
    return exporter.save(path, include_themes=include_themes)
