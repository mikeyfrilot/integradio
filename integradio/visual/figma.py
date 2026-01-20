"""
Figma Integration - Bidirectional sync with Figma designs.

This module provides:
1. Export UISpec to Figma (via Figma REST API or Variables API)
2. Import Figma designs as UISpec
3. Sync: detect drift between code and design

Requires: FIGMA_ACCESS_TOKEN environment variable.

Figma APIs used:
- Figma REST API: https://www.figma.com/developers/api
- Figma Variables API: For design tokens (BETA)
- Figma Plugin API: For deeper integration (requires plugin)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import logging

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .tokens import (
    DesignToken,
    TokenType,
    TokenGroup,
    ColorValue,
    DimensionValue,
)
from .spec import UISpec, PageSpec, VisualSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Figma API Client
# =============================================================================

@dataclass
class FigmaConfig:
    """Figma API configuration."""
    access_token: str = ""
    team_id: str = ""
    project_id: str = ""

    @classmethod
    def from_env(cls) -> "FigmaConfig":
        """Load config from environment variables."""
        return cls(
            access_token=os.environ.get("FIGMA_ACCESS_TOKEN", ""),
            team_id=os.environ.get("FIGMA_TEAM_ID", ""),
            project_id=os.environ.get("FIGMA_PROJECT_ID", ""),
        )

    @property
    def is_valid(self) -> bool:
        return bool(self.access_token)


class FigmaAPIError(Exception):
    """Figma API error."""
    pass


class FigmaClient:
    """
    Client for Figma REST API.

    Usage:
        client = FigmaClient.from_env()
        file_data = client.get_file("abc123")
    """

    BASE_URL = "https://api.figma.com/v1"

    def __init__(self, config: FigmaConfig):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for Figma integration. Install with: pip install httpx")

        if not config.is_valid:
            raise FigmaAPIError("Invalid Figma config: access_token required")

        self.config = config
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "X-Figma-Token": config.access_token,
            },
            timeout=30.0,
        )

    @classmethod
    def from_env(cls) -> "FigmaClient":
        """Create client from environment variables."""
        return cls(FigmaConfig.from_env())

    def get_file(self, file_key: str) -> dict:
        """Get a Figma file."""
        response = self._client.get(f"/files/{file_key}")
        if response.status_code != 200:
            raise FigmaAPIError(f"Failed to get file: {response.text}")
        return response.json()

    def get_file_styles(self, file_key: str) -> dict:
        """Get styles from a Figma file."""
        response = self._client.get(f"/files/{file_key}/styles")
        if response.status_code != 200:
            raise FigmaAPIError(f"Failed to get styles: {response.text}")
        return response.json()

    def get_file_components(self, file_key: str) -> dict:
        """Get components from a Figma file."""
        response = self._client.get(f"/files/{file_key}/components")
        if response.status_code != 200:
            raise FigmaAPIError(f"Failed to get components: {response.text}")
        return response.json()

    def get_team_styles(self) -> dict:
        """Get published styles from team library."""
        if not self.config.team_id:
            raise FigmaAPIError("team_id required for get_team_styles")

        response = self._client.get(f"/teams/{self.config.team_id}/styles")
        if response.status_code != 200:
            raise FigmaAPIError(f"Failed to get team styles: {response.text}")
        return response.json()

    def get_local_variables(self, file_key: str) -> dict:
        """
        Get local variables (design tokens) from a file.

        Requires the Variables API (currently in beta).
        """
        response = self._client.get(f"/files/{file_key}/variables/local")
        if response.status_code != 200:
            raise FigmaAPIError(f"Failed to get variables: {response.text}")
        return response.json()

    def post_variables(self, file_key: str, variables: dict) -> dict:
        """
        Create/update variables in a file.

        Requires write access and Variables API.
        """
        response = self._client.post(
            f"/files/{file_key}/variables",
            json=variables,
        )
        if response.status_code != 200:
            raise FigmaAPIError(f"Failed to post variables: {response.text}")
        return response.json()

    def get_images(
        self,
        file_key: str,
        node_ids: list[str],
        scale: float = 1.0,
        format: str = "png",
    ) -> dict:
        """Export images from Figma nodes."""
        params = {
            "ids": ",".join(node_ids),
            "scale": scale,
            "format": format,
        }
        response = self._client.get(f"/images/{file_key}", params=params)
        if response.status_code != 200:
            raise FigmaAPIError(f"Failed to get images: {response.text}")
        return response.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "FigmaClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Figma Data Types
# =============================================================================

@dataclass
class FigmaColor:
    """Figma color representation (0-1 range)."""
    r: float
    g: float
    b: float
    a: float = 1.0

    def to_color_value(self) -> ColorValue:
        """Convert to our ColorValue."""
        return ColorValue(
            color_space="srgb",
            components=(self.r, self.g, self.b),
            alpha=self.a,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FigmaColor":
        return cls(
            r=d.get("r", 0),
            g=d.get("g", 0),
            b=d.get("b", 0),
            a=d.get("a", 1),
        )


@dataclass
class FigmaStyle:
    """A Figma style (color, text, effect, grid)."""
    key: str
    name: str
    style_type: str  # FILL, TEXT, EFFECT, GRID
    description: str = ""


@dataclass
class FigmaNode:
    """A Figma node (frame, component, etc)."""
    id: str
    name: str
    type: str
    children: list["FigmaNode"] = field(default_factory=list)
    absolute_bounding_box: dict | None = None
    fills: list[dict] = field(default_factory=list)
    strokes: list[dict] = field(default_factory=list)
    effects: list[dict] = field(default_factory=list)


# =============================================================================
# Import from Figma
# =============================================================================

class FigmaImporter:
    """
    Import Figma designs as UISpec.

    Extracts:
    - Color styles -> color tokens
    - Text styles -> typography tokens
    - Components -> VisualSpecs
    - Frames/Pages -> PageSpecs
    """

    def __init__(self, client: FigmaClient):
        self.client = client

    def import_file(self, file_key: str) -> UISpec:
        """Import a Figma file as UISpec."""
        file_data = self.client.get_file(file_key)

        spec = UISpec(
            name=file_data.get("name", "Figma Import"),
            version="1.0.0",
        )

        # Import styles as tokens
        self._import_styles(file_key, spec)

        # Import pages
        document = file_data.get("document", {})
        for page_node in document.get("children", []):
            if page_node.get("type") == "CANVAS":
                page_spec = self._import_page(page_node)
                spec.add_page(page_spec)

        return spec

    def _import_styles(self, file_key: str, spec: UISpec) -> None:
        """Import styles as design tokens."""
        try:
            styles_data = self.client.get_file_styles(file_key)
            styles = styles_data.get("meta", {}).get("styles", [])

            colors = TokenGroup(type=TokenType.COLOR)
            text = TokenGroup(type=TokenType.TYPOGRAPHY)

            for style in styles:
                style_type = style.get("style_type", "")
                name = style.get("name", "").replace("/", ".").replace(" ", "-").lower()

                if style_type == "FILL":
                    # We'd need to get the actual color value from the file
                    # For now, create a placeholder
                    colors.add(name, DesignToken.color(
                        "#000000",
                        style.get("description", ""),
                    ))
                elif style_type == "TEXT":
                    # Placeholder for text styles
                    pass

            if colors.tokens:
                spec.tokens.add("colors", colors)

        except FigmaAPIError as e:
            logger.warning(f"Could not import styles: {e}")

    def _import_page(self, page_node: dict) -> PageSpec:
        """Import a Figma page as PageSpec."""
        name = page_node.get("name", "Untitled")
        route = "/" + name.lower().replace(" ", "-")

        page = PageSpec(name=name, route=route)

        # Find component instances and frames
        for child in page_node.get("children", []):
            child_type = child.get("type", "")

            if child_type in ("COMPONENT", "INSTANCE", "FRAME"):
                visual = self._import_node_as_visual(child)
                if visual:
                    page.add_component(visual)

        return page

    def _import_node_as_visual(self, node: dict) -> VisualSpec | None:
        """Convert a Figma node to VisualSpec."""
        node_id = node.get("id", "")
        name = node.get("name", "")
        node_type = node.get("type", "")

        # Clean up name for component_id
        component_id = name.replace(" ", "-").lower()

        visual = VisualSpec(
            component_id=component_id,
            component_type=self._map_figma_type(node_type),
        )

        # Extract colors from fills
        fills = node.get("fills", [])
        for fill in fills:
            if fill.get("type") == "SOLID":
                color_dict = fill.get("color", {})
                color = FigmaColor.from_dict(color_dict)
                visual.tokens["background"] = DesignToken.color(
                    color.to_color_value()
                )
                break

        # Extract dimensions from bounding box
        bbox = node.get("absoluteBoundingBox", {})
        if bbox:
            width = bbox.get("width")
            height = bbox.get("height")
            if width:
                visual.layout.width = DimensionValue(width, "px")
            if height:
                visual.layout.height = DimensionValue(height, "px")

        return visual

    def _map_figma_type(self, figma_type: str) -> str:
        """Map Figma node type to component type."""
        mapping = {
            "FRAME": "Container",
            "GROUP": "Container",
            "COMPONENT": "Component",
            "INSTANCE": "Component",
            "TEXT": "Markdown",
            "RECTANGLE": "Container",
            "ELLIPSE": "Container",
            "VECTOR": "Image",
            "BOOLEAN_OPERATION": "Container",
        }
        return mapping.get(figma_type, "Unknown")

    def import_variables(self, file_key: str) -> TokenGroup:
        """
        Import Figma Variables as design tokens.

        Figma Variables map directly to design tokens.
        """
        var_data = self.client.get_local_variables(file_key)

        tokens = TokenGroup()
        variables = var_data.get("meta", {}).get("variables", {})

        for var_id, var in variables.items():
            name = var.get("name", var_id).replace("/", ".").lower()
            resolved_type = var.get("resolvedType", "")

            # Get the value for the default mode
            value_by_mode = var.get("valuesByMode", {})
            if value_by_mode:
                mode_id = list(value_by_mode.keys())[0]
                value = value_by_mode[mode_id]

                if resolved_type == "COLOR":
                    if isinstance(value, dict):
                        color = FigmaColor.from_dict(value)
                        tokens.add(name, DesignToken.color(color.to_color_value()))
                elif resolved_type == "FLOAT":
                    tokens.add(name, DesignToken.number(value))
                elif resolved_type == "STRING":
                    tokens.add(name, DesignToken(
                        value=value,
                        type=TokenType.STRING,
                    ))

        return tokens


# =============================================================================
# Export to Figma
# =============================================================================

class FigmaExporter:
    """
    Export UISpec to Figma.

    Creates:
    - Figma Variables from design tokens
    - Styles from token groups
    """

    def __init__(self, client: FigmaClient):
        self.client = client

    def export_tokens_as_variables(
        self,
        spec: UISpec,
        file_key: str,
    ) -> dict:
        """
        Export design tokens as Figma Variables.

        Returns the API response.
        """
        flat_tokens = spec.tokens.flatten()

        # Build variables payload
        variables = []
        for path, token in flat_tokens.items():
            var_name = path.replace(".", "/")

            if token.type == TokenType.COLOR and isinstance(token.value, ColorValue):
                r, g, b = token.value.components
                variables.append({
                    "name": var_name,
                    "resolvedType": "COLOR",
                    "valuesByMode": {
                        "default": {
                            "r": r,
                            "g": g,
                            "b": b,
                            "a": token.value.alpha,
                        }
                    },
                    "description": token.description,
                })

            elif token.type == TokenType.DIMENSION and isinstance(token.value, DimensionValue):
                # Figma uses pixels, convert
                px_value = token.value.value
                if token.value.unit == "rem":
                    px_value = token.value.value * 16

                variables.append({
                    "name": var_name,
                    "resolvedType": "FLOAT",
                    "valuesByMode": {
                        "default": px_value,
                    },
                    "description": token.description,
                })

            elif token.type == TokenType.NUMBER:
                variables.append({
                    "name": var_name,
                    "resolvedType": "FLOAT",
                    "valuesByMode": {
                        "default": token.value,
                    },
                    "description": token.description,
                })

        payload = {
            "variables": variables,
        }

        return self.client.post_variables(file_key, payload)

    def export_to_tokens_json(self, spec: UISpec, path: str | Path) -> Path:
        """
        Export as Figma Tokens JSON format.

        Compatible with the Figma Tokens plugin.
        """
        path = Path(path)

        # Figma Tokens uses a different structure than DTCG
        output: dict[str, Any] = {}

        flat_tokens = spec.tokens.flatten()
        for token_path, token in flat_tokens.items():
            parts = token_path.split(".")
            current = output

            # Build nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Add token
            name = parts[-1]
            current[name] = self._token_to_figma_format(token)

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        return path

    def _token_to_figma_format(self, token: DesignToken) -> dict:
        """Convert token to Figma Tokens plugin format."""
        result: dict[str, Any] = {
            "type": self._map_type_to_figma(token.type),
        }

        if isinstance(token.value, ColorValue):
            result["value"] = token.value.to_hex()
        elif isinstance(token.value, DimensionValue):
            result["value"] = token.value.to_css()
        elif hasattr(token.value, "to_css"):
            result["value"] = token.value.to_css()
        else:
            result["value"] = token.value

        if token.description:
            result["description"] = token.description

        return result

    def _map_type_to_figma(self, token_type: TokenType) -> str:
        """Map our token types to Figma Tokens types."""
        mapping = {
            TokenType.COLOR: "color",
            TokenType.DIMENSION: "sizing",
            TokenType.FONT_FAMILY: "fontFamilies",
            TokenType.FONT_WEIGHT: "fontWeights",
            TokenType.FONT_STYLE: "fontStyles",
            TokenType.DURATION: "duration",
            TokenType.NUMBER: "number",
            TokenType.SHADOW: "boxShadow",
            TokenType.BORDER: "border",
            TokenType.TYPOGRAPHY: "typography",
        }
        return mapping.get(token_type, "other")


# =============================================================================
# Sync / Drift Detection
# =============================================================================

@dataclass
class DriftItem:
    """A single drift between code and Figma."""
    path: str
    code_value: str
    figma_value: str
    drift_type: str  # "added", "removed", "changed"


@dataclass
class DriftReport:
    """Report of drift between code and Figma."""
    spec_name: str
    figma_file: str
    items: list[DriftItem] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return len(self.items) > 0

    @property
    def added(self) -> list[DriftItem]:
        return [i for i in self.items if i.drift_type == "added"]

    @property
    def removed(self) -> list[DriftItem]:
        return [i for i in self.items if i.drift_type == "removed"]

    @property
    def changed(self) -> list[DriftItem]:
        return [i for i in self.items if i.drift_type == "changed"]


class FigmaSyncer:
    """
    Detect drift between UISpec and Figma design.

    Compares tokens and reports differences.
    """

    def __init__(self, client: FigmaClient):
        self.client = client
        self.importer = FigmaImporter(client)

    def check_drift(self, spec: UISpec, file_key: str) -> DriftReport:
        """
        Compare UISpec tokens with Figma Variables.

        Returns a report of differences.
        """
        report = DriftReport(
            spec_name=spec.name,
            figma_file=file_key,
        )

        # Get Figma tokens
        try:
            figma_tokens = self.importer.import_variables(file_key)
            figma_flat = figma_tokens.flatten()
        except FigmaAPIError as e:
            logger.error(f"Could not fetch Figma variables: {e}")
            return report

        # Get code tokens
        code_flat = spec.tokens.flatten()

        # Compare
        code_paths = set(code_flat.keys())
        figma_paths = set(figma_flat.keys())

        # Added in code (not in Figma)
        for path in code_paths - figma_paths:
            report.items.append(DriftItem(
                path=path,
                code_value=code_flat[path].to_css(),
                figma_value="",
                drift_type="added",
            ))

        # Removed in code (in Figma but not code)
        for path in figma_paths - code_paths:
            report.items.append(DriftItem(
                path=path,
                code_value="",
                figma_value=figma_flat[path].to_css(),
                drift_type="removed",
            ))

        # Changed (in both but different values)
        for path in code_paths & figma_paths:
            code_css = code_flat[path].to_css()
            figma_css = figma_flat[path].to_css()

            if code_css != figma_css:
                report.items.append(DriftItem(
                    path=path,
                    code_value=code_css,
                    figma_value=figma_css,
                    drift_type="changed",
                ))

        return report

    def sync_from_figma(self, spec: UISpec, file_key: str) -> int:
        """
        Sync code tokens FROM Figma (Figma is source of truth).

        Returns number of tokens updated.
        """
        figma_tokens = self.importer.import_variables(file_key)
        figma_flat = figma_tokens.flatten()

        updated = 0
        for path, token in figma_flat.items():
            parts = path.split(".")

            # Navigate/create nested groups
            current = spec.tokens
            for part in parts[:-1]:
                existing = current.tokens.get(part)
                if existing is None or not isinstance(existing, TokenGroup):
                    current.add(part, TokenGroup())
                current = current.tokens[part]

            # Set token
            current.add(parts[-1], token)
            updated += 1

        return updated


# =============================================================================
# Convenience Functions
# =============================================================================

def import_from_figma(file_key: str, access_token: str | None = None) -> UISpec:
    """
    Quick import from Figma.

    Args:
        file_key: Figma file key (from URL)
        access_token: Figma access token (or uses FIGMA_ACCESS_TOKEN env)

    Returns:
        UISpec from Figma design
    """
    config = FigmaConfig.from_env()
    if access_token:
        config.access_token = access_token

    with FigmaClient(config) as client:
        importer = FigmaImporter(client)
        return importer.import_file(file_key)


def export_to_figma_tokens(spec: UISpec, path: str | Path) -> Path:
    """
    Quick export to Figma Tokens JSON.

    No API needed - just generates the JSON file.
    """
    # Create a fake client just for the exporter (no API calls)
    config = FigmaConfig(access_token="dummy")

    # We can't use the client without httpx, so just do it manually
    from .export import DTCGExporter
    exporter = DTCGExporter(spec)
    return exporter.save(path)


def check_figma_drift(spec: UISpec, file_key: str) -> DriftReport:
    """Check for drift between spec and Figma."""
    config = FigmaConfig.from_env()

    with FigmaClient(config) as client:
        syncer = FigmaSyncer(client)
        return syncer.check_drift(spec, file_key)
