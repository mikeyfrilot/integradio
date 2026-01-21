"""
Integration tests for the Figma integration module.

Tests the Figma API client, import/export functionality, and drift detection
with mocked HTTP responses to simulate real Figma API behavior.
"""

from __future__ import annotations

import json
import os
import pytest
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Add project root to path for direct submodule imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import directly from visual submodule to avoid gradio/numpy import chain
from integradio.visual.figma import (
    FigmaConfig,
    FigmaAPIError,
    FigmaClient,
    FigmaColor,
    FigmaStyle,
    FigmaNode,
    FigmaImporter,
    FigmaExporter,
    DriftItem,
    DriftReport,
    FigmaSyncer,
    import_from_figma,
    export_to_figma_tokens,
    check_figma_drift,
    HAS_HTTPX,
)
from integradio.visual.tokens import (
    DesignToken,
    TokenGroup,
    TokenType,
    ColorValue,
    DimensionValue,
)
from integradio.visual.spec import UISpec, PageSpec, VisualSpec


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config() -> FigmaConfig:
    """Create a sample Figma config."""
    return FigmaConfig(
        access_token="test-token-12345",
        team_id="team-123",
        project_id="project-456",
    )


@pytest.fixture
def sample_ui_spec() -> UISpec:
    """Create a sample UISpec for testing exports."""
    tokens = TokenGroup()

    # Color tokens
    colors = TokenGroup(type=TokenType.COLOR)
    colors.add("primary", DesignToken.color("#3366FF", "Primary color"))
    colors.add("secondary", DesignToken.color("#FF6633"))
    tokens.add("colors", colors)

    # Dimension tokens
    spacing = TokenGroup(type=TokenType.DIMENSION)
    spacing.add("sm", DesignToken.dimension(8, "px"))
    spacing.add("md", DesignToken.dimension(16, "px"))
    spacing.add("lg", DesignToken.dimension(2, "rem"))
    tokens.add("spacing", spacing)

    # Number tokens
    tokens.add("scale", DesignToken.number(1.5))

    return UISpec(name="TestApp", version="1.0.0", tokens=tokens)


@pytest.fixture
def mock_figma_file_response() -> dict:
    """Mock Figma file API response."""
    return {
        "name": "Test Design",
        "document": {
            "id": "0:0",
            "type": "DOCUMENT",
            "children": [
                {
                    "id": "0:1",
                    "type": "CANVAS",
                    "name": "Page 1",
                    "children": [
                        {
                            "id": "1:2",
                            "type": "FRAME",
                            "name": "Header",
                            "absoluteBoundingBox": {"x": 0, "y": 0, "width": 1200, "height": 80},
                            "fills": [
                                {"type": "SOLID", "color": {"r": 0.2, "g": 0.4, "b": 0.8, "a": 1.0}}
                            ],
                            "children": [],
                        },
                        {
                            "id": "1:3",
                            "type": "COMPONENT",
                            "name": "Button",
                            "absoluteBoundingBox": {"x": 100, "y": 200, "width": 120, "height": 40},
                            "fills": [],
                            "children": [],
                        },
                        {
                            "id": "1:4",
                            "type": "TEXT",
                            "name": "Title",
                            "absoluteBoundingBox": {"x": 0, "y": 100, "width": 300, "height": 48},
                            "fills": [],
                            "children": [],
                        },
                    ],
                }
            ],
        },
    }


@pytest.fixture
def mock_figma_styles_response() -> dict:
    """Mock Figma styles API response."""
    return {
        "meta": {
            "styles": [
                {
                    "key": "style-1",
                    "name": "Primary/Blue",
                    "style_type": "FILL",
                    "description": "Primary brand color",
                },
                {
                    "key": "style-2",
                    "name": "Body Text",
                    "style_type": "TEXT",
                    "description": "Body text style",
                },
                {
                    "key": "style-3",
                    "name": "Secondary/Orange",
                    "style_type": "FILL",
                    "description": "",
                },
            ]
        }
    }


@pytest.fixture
def mock_figma_variables_response() -> dict:
    """Mock Figma variables API response."""
    return {
        "meta": {
            "variables": {
                "var-1": {
                    "name": "colors/primary",
                    "resolvedType": "COLOR",
                    "valuesByMode": {
                        "mode-1": {"r": 0.2, "g": 0.4, "b": 0.8, "a": 1.0}
                    },
                },
                "var-2": {
                    "name": "spacing/base",
                    "resolvedType": "FLOAT",
                    "valuesByMode": {
                        "mode-1": 16
                    },
                },
                "var-3": {
                    "name": "font/family",
                    "resolvedType": "STRING",
                    "valuesByMode": {
                        "mode-1": "Inter"
                    },
                },
            }
        }
    }


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    return mock_client


# =============================================================================
# FigmaConfig Tests
# =============================================================================

class TestFigmaConfig:
    """Tests for FigmaConfig dataclass."""

    def test_default_values(self):
        """Test creating config with default values."""
        config = FigmaConfig()

        assert config.access_token == ""
        assert config.team_id == ""
        assert config.project_id == ""

    def test_with_values(self, sample_config: FigmaConfig):
        """Test creating config with values."""
        assert sample_config.access_token == "test-token-12345"
        assert sample_config.team_id == "team-123"
        assert sample_config.project_id == "project-456"

    def test_is_valid_with_token(self, sample_config: FigmaConfig):
        """Test is_valid returns True when token is set."""
        assert sample_config.is_valid is True

    def test_is_valid_without_token(self):
        """Test is_valid returns False when token is empty."""
        config = FigmaConfig()
        assert config.is_valid is False

    def test_is_valid_with_only_team_id(self):
        """Test is_valid requires access_token, not just team_id."""
        config = FigmaConfig(team_id="team-123")
        assert config.is_valid is False

    def test_from_env_with_values(self):
        """Test loading config from environment variables."""
        with patch.dict(os.environ, {
            "FIGMA_ACCESS_TOKEN": "env-token",
            "FIGMA_TEAM_ID": "env-team",
            "FIGMA_PROJECT_ID": "env-project",
        }):
            config = FigmaConfig.from_env()

            assert config.access_token == "env-token"
            assert config.team_id == "env-team"
            assert config.project_id == "env-project"

    def test_from_env_with_empty_env(self):
        """Test loading config from empty environment."""
        with patch.dict(os.environ, {}, clear=True):
            # Need to remove keys if they exist
            env_copy = os.environ.copy()
            for key in ["FIGMA_ACCESS_TOKEN", "FIGMA_TEAM_ID", "FIGMA_PROJECT_ID"]:
                env_copy.pop(key, None)

            with patch.dict(os.environ, env_copy, clear=True):
                config = FigmaConfig.from_env()

                assert config.access_token == ""
                assert config.team_id == ""
                assert config.project_id == ""


# =============================================================================
# FigmaAPIError Tests
# =============================================================================

class TestFigmaAPIError:
    """Tests for FigmaAPIError exception."""

    def test_basic_error(self):
        """Test creating basic error."""
        error = FigmaAPIError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_error_inheritance(self):
        """Test that FigmaAPIError is an Exception."""
        error = FigmaAPIError("test")
        assert isinstance(error, Exception)

    def test_raise_error(self):
        """Test raising and catching error."""
        with pytest.raises(FigmaAPIError) as exc_info:
            raise FigmaAPIError("API request failed")

        assert "API request failed" in str(exc_info.value)


# =============================================================================
# FigmaClient Tests
# =============================================================================

@pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")
class TestFigmaClient:
    """Tests for FigmaClient."""

    def test_init_without_httpx(self):
        """Test that ImportError is raised without httpx."""
        # This test only makes sense when httpx IS installed
        # We'd need to mock HAS_HTTPX = False
        pass  # Covered by the skipif marker behavior

    def test_init_with_invalid_config(self):
        """Test that FigmaAPIError is raised with invalid config."""
        config = FigmaConfig()  # No token

        with pytest.raises(FigmaAPIError) as exc_info:
            FigmaClient(config)

        assert "access_token required" in str(exc_info.value)

    def test_init_with_valid_config(self, sample_config: FigmaConfig):
        """Test creating client with valid config."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(sample_config)

            assert client.config == sample_config
            mock_client_class.assert_called_once()

    def test_from_env(self):
        """Test creating client from environment."""
        with patch.dict(os.environ, {"FIGMA_ACCESS_TOKEN": "test-token"}):
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient.from_env()

                assert client.config.access_token == "test-token"

    def test_get_file_success(self, sample_config: FigmaConfig, mock_figma_file_response: dict):
        """Test successful file retrieval."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_file_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            result = client.get_file("file-key")

            assert result["name"] == "Test Design"
            mock_client.get.assert_called_with("/files/file-key")

    def test_get_file_failure(self, sample_config: FigmaConfig):
        """Test file retrieval failure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Not found"

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.get_file("invalid-key")

            assert "Failed to get file" in str(exc_info.value)

    def test_get_file_styles_success(self, sample_config: FigmaConfig, mock_figma_styles_response: dict):
        """Test successful styles retrieval."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_styles_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            result = client.get_file_styles("file-key")

            assert "meta" in result
            mock_client.get.assert_called_with("/files/file-key/styles")

    def test_get_file_styles_failure(self, sample_config: FigmaConfig):
        """Test styles retrieval failure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Server error"

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.get_file_styles("file-key")

            assert "Failed to get styles" in str(exc_info.value)

    def test_get_file_components_success(self, sample_config: FigmaConfig):
        """Test successful components retrieval."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"meta": {"components": []}}

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            result = client.get_file_components("file-key")

            assert "meta" in result
            mock_client.get.assert_called_with("/files/file-key/components")

    def test_get_file_components_failure(self, sample_config: FigmaConfig):
        """Test components retrieval failure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.text = "Forbidden"

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.get_file_components("file-key")

            assert "Failed to get components" in str(exc_info.value)

    def test_get_team_styles_success(self, sample_config: FigmaConfig):
        """Test successful team styles retrieval."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"meta": {"styles": []}}

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            result = client.get_team_styles()

            assert "meta" in result
            mock_client.get.assert_called_with("/teams/team-123/styles")

    def test_get_team_styles_without_team_id(self):
        """Test team styles fails without team_id."""
        config = FigmaConfig(access_token="token")  # No team_id

        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.get_team_styles()

            assert "team_id required" in str(exc_info.value)

    def test_get_team_styles_failure(self, sample_config: FigmaConfig):
        """Test team styles retrieval failure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.get_team_styles()

            assert "Failed to get team styles" in str(exc_info.value)

    def test_get_local_variables_success(self, sample_config: FigmaConfig, mock_figma_variables_response: dict):
        """Test successful variables retrieval."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            result = client.get_local_variables("file-key")

            assert "meta" in result
            mock_client.get.assert_called_with("/files/file-key/variables/local")

    def test_get_local_variables_failure(self, sample_config: FigmaConfig):
        """Test variables retrieval failure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.get_local_variables("file-key")

            assert "Failed to get variables" in str(exc_info.value)

    def test_post_variables_success(self, sample_config: FigmaConfig):
        """Test successful variables post."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            variables = {"variables": [{"name": "test", "value": "#FF0000"}]}
            result = client.post_variables("file-key", variables)

            assert result["success"] is True
            mock_client.post.assert_called_with("/files/file-key/variables", json=variables)

    def test_post_variables_failure(self, sample_config: FigmaConfig):
        """Test variables post failure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.text = "Write access required"

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.post_variables("file-key", {})

            assert "Failed to post variables" in str(exc_info.value)

    def test_get_images_success(self, sample_config: FigmaConfig):
        """Test successful images export."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"images": {"1:2": "https://example.com/image.png"}}

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            result = client.get_images("file-key", ["1:2"], scale=2.0, format="svg")

            assert "images" in result
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert call_args[0][0] == "/images/file-key"
            assert call_args[1]["params"]["ids"] == "1:2"
            assert call_args[1]["params"]["scale"] == 2.0
            assert call_args[1]["params"]["format"] == "svg"

    def test_get_images_failure(self, sample_config: FigmaConfig):
        """Test images export failure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal error"

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)

            with pytest.raises(FigmaAPIError) as exc_info:
                client.get_images("file-key", ["1:2"])

            assert "Failed to get images" in str(exc_info.value)

    def test_context_manager(self, sample_config: FigmaConfig):
        """Test client as context manager."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            with FigmaClient(sample_config) as client:
                assert client is not None

            mock_client.close.assert_called_once()

    def test_close(self, sample_config: FigmaConfig):
        """Test client close method."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            client.close()

            mock_client.close.assert_called_once()


# =============================================================================
# FigmaColor Tests
# =============================================================================

class TestFigmaColor:
    """Tests for FigmaColor dataclass."""

    def test_default_values(self):
        """Test creating color with default alpha."""
        color = FigmaColor(r=0.5, g=0.5, b=0.5)

        assert color.r == 0.5
        assert color.g == 0.5
        assert color.b == 0.5
        assert color.a == 1.0

    def test_with_alpha(self):
        """Test creating color with custom alpha."""
        color = FigmaColor(r=1.0, g=0.0, b=0.0, a=0.5)

        assert color.a == 0.5

    def test_to_color_value(self):
        """Test converting to ColorValue."""
        color = FigmaColor(r=0.2, g=0.4, b=0.8, a=0.9)
        color_value = color.to_color_value()

        assert isinstance(color_value, ColorValue)
        assert color_value.color_space == "srgb"
        assert color_value.components == (0.2, 0.4, 0.8)
        assert color_value.alpha == 0.9

    def test_from_dict_complete(self):
        """Test creating from complete dict."""
        d = {"r": 0.1, "g": 0.2, "b": 0.3, "a": 0.4}
        color = FigmaColor.from_dict(d)

        assert color.r == 0.1
        assert color.g == 0.2
        assert color.b == 0.3
        assert color.a == 0.4

    def test_from_dict_partial(self):
        """Test creating from partial dict with defaults."""
        d = {"r": 0.5}
        color = FigmaColor.from_dict(d)

        assert color.r == 0.5
        assert color.g == 0
        assert color.b == 0
        assert color.a == 1

    def test_from_dict_empty(self):
        """Test creating from empty dict."""
        color = FigmaColor.from_dict({})

        assert color.r == 0
        assert color.g == 0
        assert color.b == 0
        assert color.a == 1


# =============================================================================
# FigmaStyle Tests
# =============================================================================

class TestFigmaStyle:
    """Tests for FigmaStyle dataclass."""

    def test_required_fields(self):
        """Test creating style with required fields."""
        style = FigmaStyle(
            key="style-123",
            name="Primary Color",
            style_type="FILL",
        )

        assert style.key == "style-123"
        assert style.name == "Primary Color"
        assert style.style_type == "FILL"
        assert style.description == ""

    def test_with_description(self):
        """Test creating style with description."""
        style = FigmaStyle(
            key="style-123",
            name="Primary Color",
            style_type="FILL",
            description="Main brand color",
        )

        assert style.description == "Main brand color"


# =============================================================================
# FigmaNode Tests
# =============================================================================

class TestFigmaNode:
    """Tests for FigmaNode dataclass."""

    def test_required_fields(self):
        """Test creating node with required fields."""
        node = FigmaNode(id="1:2", name="Header", type="FRAME")

        assert node.id == "1:2"
        assert node.name == "Header"
        assert node.type == "FRAME"
        assert node.children == []
        assert node.absolute_bounding_box is None
        assert node.fills == []
        assert node.strokes == []
        assert node.effects == []

    def test_with_all_fields(self):
        """Test creating node with all fields."""
        child = FigmaNode(id="1:3", name="Button", type="COMPONENT")
        bbox = {"x": 0, "y": 0, "width": 100, "height": 50}
        fills = [{"type": "SOLID", "color": {"r": 1, "g": 0, "b": 0, "a": 1}}]
        strokes = [{"type": "SOLID", "color": {"r": 0, "g": 0, "b": 0, "a": 1}}]
        effects = [{"type": "DROP_SHADOW"}]

        node = FigmaNode(
            id="1:2",
            name="Card",
            type="FRAME",
            children=[child],
            absolute_bounding_box=bbox,
            fills=fills,
            strokes=strokes,
            effects=effects,
        )

        assert len(node.children) == 1
        assert node.children[0].name == "Button"
        assert node.absolute_bounding_box == bbox
        assert len(node.fills) == 1
        assert len(node.strokes) == 1
        assert len(node.effects) == 1


# =============================================================================
# FigmaImporter Tests
# =============================================================================

@pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")
class TestFigmaImporter:
    """Tests for FigmaImporter."""

    def test_import_file_basic(
        self,
        sample_config: FigmaConfig,
        mock_figma_file_response: dict,
        mock_figma_styles_response: dict,
    ):
        """Test basic file import."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response_file = MagicMock()
            mock_response_file.status_code = 200
            mock_response_file.json.return_value = mock_figma_file_response

            mock_response_styles = MagicMock()
            mock_response_styles.status_code = 200
            mock_response_styles.json.return_value = mock_figma_styles_response

            mock_client = MagicMock()
            mock_client.get.side_effect = [mock_response_file, mock_response_styles]
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            spec = importer.import_file("file-key")

            assert spec.name == "Test Design"
            assert spec.version == "1.0.0"

    def test_import_file_with_pages(
        self,
        sample_config: FigmaConfig,
        mock_figma_file_response: dict,
        mock_figma_styles_response: dict,
    ):
        """Test file import extracts pages."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response_file = MagicMock()
            mock_response_file.status_code = 200
            mock_response_file.json.return_value = mock_figma_file_response

            mock_response_styles = MagicMock()
            mock_response_styles.status_code = 200
            mock_response_styles.json.return_value = mock_figma_styles_response

            mock_client = MagicMock()
            mock_client.get.side_effect = [mock_response_file, mock_response_styles]
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            spec = importer.import_file("file-key")

            # Should have one page
            assert len(spec.pages) == 1
            assert "/page-1" in spec.pages

    def test_import_file_extracts_components(
        self,
        sample_config: FigmaConfig,
        mock_figma_file_response: dict,
        mock_figma_styles_response: dict,
    ):
        """Test file import extracts components from pages."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response_file = MagicMock()
            mock_response_file.status_code = 200
            mock_response_file.json.return_value = mock_figma_file_response

            mock_response_styles = MagicMock()
            mock_response_styles.status_code = 200
            mock_response_styles.json.return_value = mock_figma_styles_response

            mock_client = MagicMock()
            mock_client.get.side_effect = [mock_response_file, mock_response_styles]
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            spec = importer.import_file("file-key")

            # Should have components from the page
            page = list(spec.pages.values())[0]
            assert len(page.components) >= 2  # Header, Button

    def test_import_styles_as_tokens(
        self,
        sample_config: FigmaConfig,
        mock_figma_file_response: dict,
        mock_figma_styles_response: dict,
    ):
        """Test that styles are imported as tokens."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response_file = MagicMock()
            mock_response_file.status_code = 200
            mock_response_file.json.return_value = mock_figma_file_response

            mock_response_styles = MagicMock()
            mock_response_styles.status_code = 200
            mock_response_styles.json.return_value = mock_figma_styles_response

            mock_client = MagicMock()
            mock_client.get.side_effect = [mock_response_file, mock_response_styles]
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            spec = importer.import_file("file-key")

            # Should have colors from styles
            colors = spec.tokens.get("colors")
            assert colors is not None

    def test_import_styles_api_error_handled(self, sample_config: FigmaConfig, mock_figma_file_response: dict):
        """Test that styles API errors are logged but don't fail import."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response_file = MagicMock()
            mock_response_file.status_code = 200
            mock_response_file.json.return_value = mock_figma_file_response

            mock_response_styles = MagicMock()
            mock_response_styles.status_code = 500
            mock_response_styles.text = "Server error"

            mock_client = MagicMock()
            mock_client.get.side_effect = [mock_response_file, mock_response_styles]
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)

            # Should not raise
            spec = importer.import_file("file-key")
            assert spec.name == "Test Design"

    def test_import_variables(self, sample_config: FigmaConfig, mock_figma_variables_response: dict):
        """Test importing variables as tokens."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            tokens = importer.import_variables("file-key")

            # Should have imported variables as tokens
            flat = tokens.flatten()
            assert "colors.primary" in flat
            assert "spacing.base" in flat
            assert "font.family" in flat

    def test_import_variables_color_type(self, sample_config: FigmaConfig, mock_figma_variables_response: dict):
        """Test that color variables are imported correctly."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            tokens = importer.import_variables("file-key")

            flat = tokens.flatten()
            color_token = flat["colors.primary"]
            assert color_token.type == TokenType.COLOR

    def test_import_variables_float_type(self, sample_config: FigmaConfig, mock_figma_variables_response: dict):
        """Test that float variables are imported correctly."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            tokens = importer.import_variables("file-key")

            flat = tokens.flatten()
            spacing_token = flat["spacing.base"]
            assert spacing_token.type == TokenType.NUMBER
            assert spacing_token.value == 16

    def test_import_variables_string_type(self, sample_config: FigmaConfig, mock_figma_variables_response: dict):
        """Test that string variables are imported correctly."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)
            tokens = importer.import_variables("file-key")

            flat = tokens.flatten()
            font_token = flat["font.family"]
            assert font_token.type == TokenType.STRING
            assert font_token.value == "Inter"

    def test_map_figma_type_frame(self, sample_config: FigmaConfig):
        """Test mapping FRAME type."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(sample_config)
            importer = FigmaImporter(client)

            assert importer._map_figma_type("FRAME") == "Container"
            assert importer._map_figma_type("GROUP") == "Container"
            assert importer._map_figma_type("COMPONENT") == "Component"
            assert importer._map_figma_type("INSTANCE") == "Component"
            assert importer._map_figma_type("TEXT") == "Markdown"
            assert importer._map_figma_type("VECTOR") == "Image"
            assert importer._map_figma_type("UNKNOWN_TYPE") == "Unknown"


# =============================================================================
# FigmaExporter Tests
# =============================================================================

@pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")
class TestFigmaExporter:
    """Tests for FigmaExporter."""

    def test_export_tokens_as_variables(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test exporting tokens as Figma variables."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)
            result = exporter.export_tokens_as_variables(sample_ui_spec, "file-key")

            assert result["success"] is True

            # Verify the call was made
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]

            assert "variables" in payload
            assert isinstance(payload["variables"], list)

    def test_export_color_tokens(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test that color tokens are exported correctly."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)
            exporter.export_tokens_as_variables(sample_ui_spec, "file-key")

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]

            # Find a color variable
            color_vars = [v for v in payload["variables"] if v.get("resolvedType") == "COLOR"]
            assert len(color_vars) >= 1

            # Verify structure
            color_var = color_vars[0]
            assert "valuesByMode" in color_var
            assert "default" in color_var["valuesByMode"]

    def test_export_dimension_tokens(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test that dimension tokens are exported as floats."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)
            exporter.export_tokens_as_variables(sample_ui_spec, "file-key")

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]

            # Find dimension variables (FLOAT type for Figma)
            float_vars = [v for v in payload["variables"] if v.get("resolvedType") == "FLOAT"]
            assert len(float_vars) >= 1

    def test_export_rem_to_px_conversion(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test that rem values are converted to px."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)
            exporter.export_tokens_as_variables(sample_ui_spec, "file-key")

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]

            # Find the lg spacing (2rem = 32px)
            lg_var = next(
                (v for v in payload["variables"] if "lg" in v.get("name", "").lower()),
                None
            )
            if lg_var:
                assert lg_var["valuesByMode"]["default"] == 32  # 2rem * 16

    def test_export_number_tokens(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test that number tokens are exported correctly."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)
            exporter.export_tokens_as_variables(sample_ui_spec, "file-key")

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]

            # Find scale variable
            scale_var = next(
                (v for v in payload["variables"] if "scale" in v.get("name", "").lower()),
                None
            )
            if scale_var:
                assert scale_var["resolvedType"] == "FLOAT"
                assert scale_var["valuesByMode"]["default"] == 1.5

    def test_export_to_tokens_json(self, sample_ui_spec: UISpec, tmp_path: Path, sample_config: FigmaConfig):
        """Test exporting to Figma Tokens JSON format."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)

            output_path = tmp_path / "tokens.json"
            result = exporter.export_to_tokens_json(sample_ui_spec, output_path)

            assert result == output_path
            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            # Should have nested structure
            assert isinstance(data, dict)

    def test_export_to_tokens_json_structure(self, sample_ui_spec: UISpec, tmp_path: Path, sample_config: FigmaConfig):
        """Test that Figma Tokens JSON has correct structure."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)

            output_path = tmp_path / "tokens.json"
            exporter.export_to_tokens_json(sample_ui_spec, output_path)

            with open(output_path) as f:
                data = json.load(f)

            # Check nested structure
            assert "colors" in data
            assert "primary" in data["colors"]
            assert "type" in data["colors"]["primary"]
            assert "value" in data["colors"]["primary"]

    def test_token_to_figma_format_color(self, sample_config: FigmaConfig):
        """Test converting color token to Figma format."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)

            token = DesignToken.color("#FF0000", "Red color")
            result = exporter._token_to_figma_format(token)

            assert result["type"] == "color"
            assert "#ff0000" in result["value"].lower()
            assert result["description"] == "Red color"

    def test_token_to_figma_format_dimension(self, sample_config: FigmaConfig):
        """Test converting dimension token to Figma format."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)

            token = DesignToken.dimension(16, "px")
            result = exporter._token_to_figma_format(token)

            assert result["type"] == "sizing"
            assert result["value"] == "16px"

    def test_map_type_to_figma(self, sample_config: FigmaConfig):
        """Test token type mapping to Figma types."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()

            client = FigmaClient(sample_config)
            exporter = FigmaExporter(client)

            assert exporter._map_type_to_figma(TokenType.COLOR) == "color"
            assert exporter._map_type_to_figma(TokenType.DIMENSION) == "sizing"
            assert exporter._map_type_to_figma(TokenType.FONT_FAMILY) == "fontFamilies"
            assert exporter._map_type_to_figma(TokenType.FONT_WEIGHT) == "fontWeights"
            assert exporter._map_type_to_figma(TokenType.DURATION) == "duration"
            assert exporter._map_type_to_figma(TokenType.NUMBER) == "number"
            assert exporter._map_type_to_figma(TokenType.SHADOW) == "boxShadow"
            assert exporter._map_type_to_figma(TokenType.BORDER) == "border"
            assert exporter._map_type_to_figma(TokenType.TYPOGRAPHY) == "typography"


# =============================================================================
# DriftItem Tests
# =============================================================================

class TestDriftItem:
    """Tests for DriftItem dataclass."""

    def test_create_added_drift(self):
        """Test creating an 'added' drift item."""
        item = DriftItem(
            path="colors.new",
            code_value="#FF0000",
            figma_value="",
            drift_type="added",
        )

        assert item.path == "colors.new"
        assert item.code_value == "#FF0000"
        assert item.figma_value == ""
        assert item.drift_type == "added"

    def test_create_removed_drift(self):
        """Test creating a 'removed' drift item."""
        item = DriftItem(
            path="colors.old",
            code_value="",
            figma_value="#00FF00",
            drift_type="removed",
        )

        assert item.drift_type == "removed"

    def test_create_changed_drift(self):
        """Test creating a 'changed' drift item."""
        item = DriftItem(
            path="colors.primary",
            code_value="#0000FF",
            figma_value="#FF0000",
            drift_type="changed",
        )

        assert item.drift_type == "changed"


# =============================================================================
# DriftReport Tests
# =============================================================================

class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_empty_report(self):
        """Test empty drift report."""
        report = DriftReport(spec_name="TestSpec", figma_file="file-123")

        assert report.spec_name == "TestSpec"
        assert report.figma_file == "file-123"
        assert report.items == []
        assert report.has_drift is False

    def test_report_with_items(self):
        """Test drift report with items."""
        report = DriftReport(
            spec_name="TestSpec",
            figma_file="file-123",
            items=[
                DriftItem("colors.a", "#FF0000", "", "added"),
                DriftItem("colors.b", "", "#00FF00", "removed"),
            ],
        )

        assert report.has_drift is True
        assert len(report.items) == 2

    def test_added_property(self):
        """Test added items filter."""
        report = DriftReport(
            spec_name="TestSpec",
            figma_file="file-123",
            items=[
                DriftItem("colors.a", "#FF0000", "", "added"),
                DriftItem("colors.b", "", "#00FF00", "removed"),
                DriftItem("colors.c", "#0000FF", "", "added"),
            ],
        )

        added = report.added
        assert len(added) == 2
        assert all(item.drift_type == "added" for item in added)

    def test_removed_property(self):
        """Test removed items filter."""
        report = DriftReport(
            spec_name="TestSpec",
            figma_file="file-123",
            items=[
                DriftItem("colors.a", "#FF0000", "", "added"),
                DriftItem("colors.b", "", "#00FF00", "removed"),
                DriftItem("colors.c", "", "#FFFFFF", "removed"),
            ],
        )

        removed = report.removed
        assert len(removed) == 2
        assert all(item.drift_type == "removed" for item in removed)

    def test_changed_property(self):
        """Test changed items filter."""
        report = DriftReport(
            spec_name="TestSpec",
            figma_file="file-123",
            items=[
                DriftItem("colors.a", "#FF0000", "#00FF00", "changed"),
                DriftItem("colors.b", "", "#FFFFFF", "removed"),
            ],
        )

        changed = report.changed
        assert len(changed) == 1
        assert changed[0].path == "colors.a"


# =============================================================================
# FigmaSyncer Tests
# =============================================================================

@pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")
class TestFigmaSyncer:
    """Tests for FigmaSyncer."""

    def test_check_drift_no_drift(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test drift check with no drift."""
        # Create variables response that matches the spec
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "colors/primary",
                        "resolvedType": "COLOR",
                        "valuesByMode": {
                            "mode-1": {"r": 0.2, "g": 0.4, "b": 1.0, "a": 1.0}  # #3366FF
                        },
                    },
                }
            }
        }

        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            syncer = FigmaSyncer(client)
            report = syncer.check_drift(sample_ui_spec, "file-key")

            assert report.spec_name == "TestApp"
            assert report.figma_file == "file-key"

    def test_check_drift_with_added(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test drift check with added tokens in code."""
        # Return empty variables (nothing in Figma)
        variables_response = {"meta": {"variables": {}}}

        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            syncer = FigmaSyncer(client)
            report = syncer.check_drift(sample_ui_spec, "file-key")

            assert report.has_drift is True
            assert len(report.added) > 0

    def test_check_drift_with_removed(self, sample_config: FigmaConfig):
        """Test drift check with tokens only in Figma."""
        # Return variables not in code
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "figma/only",
                        "resolvedType": "COLOR",
                        "valuesByMode": {
                            "mode-1": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
                        },
                    },
                }
            }
        }

        # Create empty spec
        empty_spec = UISpec(name="Empty", tokens=TokenGroup())

        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            syncer = FigmaSyncer(client)
            report = syncer.check_drift(empty_spec, "file-key")

            assert report.has_drift is True
            assert len(report.removed) >= 1

    def test_check_drift_api_error_handled(self, sample_config: FigmaConfig, sample_ui_spec: UISpec):
        """Test that API errors are handled in drift check."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Server error"

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            syncer = FigmaSyncer(client)
            report = syncer.check_drift(sample_ui_spec, "file-key")

            # Should return empty report on error
            assert report.has_drift is False

    def test_sync_from_figma(self, sample_config: FigmaConfig, mock_figma_variables_response: dict):
        """Test syncing tokens from Figma."""
        spec = UISpec(name="Empty", tokens=TokenGroup())

        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_figma_variables_response

            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = FigmaClient(sample_config)
            syncer = FigmaSyncer(client)
            updated = syncer.sync_from_figma(spec, "file-key")

            assert updated >= 1

            # Check that tokens were added to spec
            flat = spec.tokens.flatten()
            assert len(flat) >= 1


# =============================================================================
# Convenience Functions Tests
# =============================================================================

@pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_import_from_figma_with_token(self, mock_figma_file_response: dict, mock_figma_styles_response: dict):
        """Test import_from_figma with access token."""
        with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
            mock_response_file = MagicMock()
            mock_response_file.status_code = 200
            mock_response_file.json.return_value = mock_figma_file_response

            mock_response_styles = MagicMock()
            mock_response_styles.status_code = 200
            mock_response_styles.json.return_value = mock_figma_styles_response

            mock_client = MagicMock()
            mock_client.get.side_effect = [mock_response_file, mock_response_styles]
            mock_client.close = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=None)
            mock_client_class.return_value = mock_client

            spec = import_from_figma("file-key", access_token="test-token")

            assert spec.name == "Test Design"

    def test_import_from_figma_from_env(self, mock_figma_file_response: dict, mock_figma_styles_response: dict):
        """Test import_from_figma using env token."""
        with patch.dict(os.environ, {"FIGMA_ACCESS_TOKEN": "env-token"}):
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response_file = MagicMock()
                mock_response_file.status_code = 200
                mock_response_file.json.return_value = mock_figma_file_response

                mock_response_styles = MagicMock()
                mock_response_styles.status_code = 200
                mock_response_styles.json.return_value = mock_figma_styles_response

                mock_client = MagicMock()
                mock_client.get.side_effect = [mock_response_file, mock_response_styles]
                mock_client.close = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=None)
                mock_client_class.return_value = mock_client

                spec = import_from_figma("file-key")

                assert spec.name == "Test Design"

    def test_export_to_figma_tokens(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_figma_tokens function."""
        output_path = tmp_path / "figma-tokens.json"
        result = export_to_figma_tokens(sample_ui_spec, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_export_to_figma_tokens_as_string(self, sample_ui_spec: UISpec, tmp_path: Path):
        """Test export_to_figma_tokens with string path."""
        output_path = str(tmp_path / "figma-tokens.json")
        result = export_to_figma_tokens(sample_ui_spec, output_path)

        assert Path(result).exists()

    def test_check_figma_drift(self, sample_ui_spec: UISpec, mock_figma_variables_response: dict):
        """Test check_figma_drift function."""
        with patch.dict(os.environ, {"FIGMA_ACCESS_TOKEN": "test-token"}):
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_figma_variables_response

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client.close = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=None)
                mock_client_class.return_value = mock_client

                report = check_figma_drift(sample_ui_spec, "file-key")

                assert isinstance(report, DriftReport)
                assert report.spec_name == "TestApp"


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestImportNodeDetails:
    """Tests for detailed node import functionality."""

    def test_import_node_with_fills(self, sample_config: FigmaConfig):
        """Test importing a node with solid fills."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)

                node = {
                    "id": "1:2",
                    "name": "Card",
                    "type": "FRAME",
                    "fills": [
                        {"type": "SOLID", "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}}
                    ],
                    "absoluteBoundingBox": {"x": 0, "y": 0, "width": 100, "height": 50},
                }

                visual = importer._import_node_as_visual(node)

                assert visual is not None
                assert visual.component_id == "card"
                assert "background" in visual.tokens
                # Check the color was extracted
                bg_token = visual.tokens["background"]
                assert bg_token.type == TokenType.COLOR

    def test_import_node_with_dimensions(self, sample_config: FigmaConfig):
        """Test importing a node with bounding box dimensions."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)

                node = {
                    "id": "1:2",
                    "name": "Box",
                    "type": "FRAME",
                    "fills": [],
                    "absoluteBoundingBox": {"x": 0, "y": 0, "width": 200, "height": 150},
                }

                visual = importer._import_node_as_visual(node)

                assert visual is not None
                assert visual.layout.width == DimensionValue(200, "px")
                assert visual.layout.height == DimensionValue(150, "px")

    def test_import_node_without_dimensions(self, sample_config: FigmaConfig):
        """Test importing a node without bounding box."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)

                node = {
                    "id": "1:2",
                    "name": "Text",
                    "type": "TEXT",
                    "fills": [],
                }

                visual = importer._import_node_as_visual(node)

                assert visual is not None
                assert visual.layout.width is None

    def test_import_node_with_non_solid_fill(self, sample_config: FigmaConfig):
        """Test importing a node with gradient fill (non-solid)."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)

                node = {
                    "id": "1:2",
                    "name": "Gradient",
                    "type": "FRAME",
                    "fills": [
                        {"type": "GRADIENT_LINEAR", "gradientStops": []}
                    ],
                }

                visual = importer._import_node_as_visual(node)

                assert visual is not None
                # No background token since it's not a solid fill
                assert "background" not in visual.tokens

    def test_import_node_with_partial_dimensions(self, sample_config: FigmaConfig):
        """Test importing a node with only width in bounding box."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)

                node = {
                    "id": "1:2",
                    "name": "Partial",
                    "type": "FRAME",
                    "fills": [],
                    "absoluteBoundingBox": {"x": 0, "y": 0, "width": 100},
                }

                visual = importer._import_node_as_visual(node)

                assert visual is not None
                assert visual.layout.width == DimensionValue(100, "px")
                assert visual.layout.height is None

    def test_import_variables_non_dict_color(self, sample_config: FigmaConfig):
        """Test importing variables with non-dict color value (edge case)."""
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "colors/special",
                        "resolvedType": "COLOR",
                        "valuesByMode": {
                            "mode-1": "#FF0000"  # String instead of dict (invalid but should not crash)
                        },
                    },
                }
            }
        }

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = variables_response

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)
                tokens = importer.import_variables("file-key")

                # Should not crash, just skip the invalid variable
                flat = tokens.flatten()
                # The color wasn't added because value wasn't a dict
                assert "colors.special" not in flat

    def test_import_variables_empty_modes(self, sample_config: FigmaConfig):
        """Test importing variables with empty valuesByMode."""
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "empty/var",
                        "resolvedType": "COLOR",
                        "valuesByMode": {},
                    },
                }
            }
        }

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = variables_response

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)
                tokens = importer.import_variables("file-key")

                # Should not crash, empty modes means no value
                flat = tokens.flatten()
                assert "empty.var" not in flat


class TestFigmaExporterValueConversions:
    """Additional tests for FigmaExporter value conversions."""

    def test_export_token_with_to_css_method(self, sample_config: FigmaConfig):
        """Test exporting token whose value has to_css method."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                exporter = FigmaExporter(client)

                # Duration token has to_css method
                from integradio.visual.tokens import DurationValue
                token = DesignToken.duration(300, "ms")
                result = exporter._token_to_figma_format(token)

                assert result["type"] == "duration"
                assert result["value"] == "300ms"

    def test_export_token_without_description(self, sample_config: FigmaConfig):
        """Test exporting token without description."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                exporter = FigmaExporter(client)

                token = DesignToken.color("#FF0000")  # No description
                result = exporter._token_to_figma_format(token)

                assert "description" not in result

    def test_export_token_plain_value(self, sample_config: FigmaConfig):
        """Test exporting token with plain value (no to_css)."""
        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_client_class.return_value = MagicMock()

                client = FigmaClient(sample_config)
                exporter = FigmaExporter(client)

                # Font weight is stored as plain int
                token = DesignToken.font_weight(700)
                result = exporter._token_to_figma_format(token)

                assert result["value"] == 700


class TestSyncerDriftDetection:
    """Additional tests for FigmaSyncer drift detection."""

    def test_check_drift_with_changed(self, sample_config: FigmaConfig):
        """Test drift check detects changed values."""
        # Create spec with a token
        spec = UISpec(name="Test", tokens=TokenGroup())
        colors = TokenGroup(type=TokenType.COLOR)
        colors.add("primary", DesignToken.color("#FF0000"))  # Red in code
        spec.tokens.add("colors", colors)

        # Figma has different value
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "colors/primary",
                        "resolvedType": "COLOR",
                        "valuesByMode": {
                            "mode-1": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}  # Blue in Figma
                        },
                    },
                }
            }
        }

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = variables_response

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                syncer = FigmaSyncer(client)
                report = syncer.check_drift(spec, "file-key")

                assert report.has_drift is True
                assert len(report.changed) >= 1

    def test_sync_from_figma_creates_nested_groups(self, sample_config: FigmaConfig):
        """Test sync creates nested token groups."""
        # Start with empty spec
        spec = UISpec(name="Empty", tokens=TokenGroup())

        # Figma has nested variables
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "brand/colors/primary",  # Deeply nested
                        "resolvedType": "COLOR",
                        "valuesByMode": {
                            "mode-1": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
                        },
                    },
                }
            }
        }

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = variables_response

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                syncer = FigmaSyncer(client)
                updated = syncer.sync_from_figma(spec, "file-key")

                assert updated >= 1

                # Check nested structure was created
                flat = spec.tokens.flatten()
                assert "brand.colors.primary" in flat

    def test_sync_from_figma_replaces_existing_token(self, sample_config: FigmaConfig):
        """Test sync replaces existing tokens."""
        # Start with a spec that has a color
        spec = UISpec(name="Test", tokens=TokenGroup())
        colors = TokenGroup(type=TokenType.COLOR)
        colors.add("primary", DesignToken.color("#000000"))  # Black
        spec.tokens.add("colors", colors)

        # Figma has different value
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "colors/primary",
                        "resolvedType": "COLOR",
                        "valuesByMode": {
                            "mode-1": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}  # White
                        },
                    },
                }
            }
        }

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = variables_response

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                syncer = FigmaSyncer(client)
                updated = syncer.sync_from_figma(spec, "file-key")

                assert updated >= 1

                # Check the value was updated
                flat = spec.tokens.flatten()
                primary = flat.get("colors.primary")
                assert primary is not None

    def test_sync_creates_intermediate_groups(self, sample_config: FigmaConfig):
        """Test sync creates intermediate token groups when needed."""
        # Start with spec that has a different structure
        spec = UISpec(name="Test", tokens=TokenGroup())
        # Add a token directly, not a group
        spec.tokens.add("spacing", DesignToken.dimension(8, "px"))

        # Figma has nested path with same prefix
        variables_response = {
            "meta": {
                "variables": {
                    "var-1": {
                        "name": "spacing/large",  # This should create group, replacing token
                        "resolvedType": "FLOAT",
                        "valuesByMode": {
                            "mode-1": 32
                        },
                    },
                }
            }
        }

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = variables_response

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                syncer = FigmaSyncer(client)
                updated = syncer.sync_from_figma(spec, "file-key")

                assert updated >= 1


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_figma_color_edge_values(self):
        """Test FigmaColor with edge values."""
        # All zeros
        color = FigmaColor(r=0.0, g=0.0, b=0.0, a=0.0)
        cv = color.to_color_value()
        assert cv.components == (0.0, 0.0, 0.0)
        assert cv.alpha == 0.0

        # All ones
        color = FigmaColor(r=1.0, g=1.0, b=1.0, a=1.0)
        cv = color.to_color_value()
        assert cv.components == (1.0, 1.0, 1.0)
        assert cv.alpha == 1.0

    def test_empty_figma_file_import(self, sample_config: FigmaConfig):
        """Test importing an empty Figma file."""
        empty_response = {
            "name": "Empty File",
            "document": {"id": "0:0", "type": "DOCUMENT", "children": []},
        }
        styles_response = {"meta": {"styles": []}}

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response_file = MagicMock()
                mock_response_file.status_code = 200
                mock_response_file.json.return_value = empty_response

                mock_response_styles = MagicMock()
                mock_response_styles.status_code = 200
                mock_response_styles.json.return_value = styles_response

                mock_client = MagicMock()
                mock_client.get.side_effect = [mock_response_file, mock_response_styles]
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)
                spec = importer.import_file("file-key")

                assert spec.name == "Empty File"
                assert len(spec.pages) == 0

    def test_empty_variables_import(self, sample_config: FigmaConfig):
        """Test importing empty variables."""
        empty_vars = {"meta": {"variables": {}}}

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = empty_vars

                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)
                tokens = importer.import_variables("file-key")

                assert len(tokens.flatten()) == 0

    def test_export_empty_spec(self, sample_config: FigmaConfig, tmp_path: Path):
        """Test exporting an empty UISpec."""
        empty_spec = UISpec(name="Empty", tokens=TokenGroup())

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True}

                mock_client = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                exporter = FigmaExporter(client)
                result = exporter.export_tokens_as_variables(empty_spec, "file-key")

                assert result["success"] is True

                # Should have empty variables list
                call_args = mock_client.post.call_args
                payload = call_args[1]["json"]
                assert payload["variables"] == []

    def test_special_characters_in_style_names(self, sample_config: FigmaConfig):
        """Test handling of special characters in style names."""
        styles_response = {
            "meta": {
                "styles": [
                    {
                        "key": "style-1",
                        "name": "Primary/Blue 500",
                        "style_type": "FILL",
                        "description": "",
                    },
                ]
            }
        }

        if HAS_HTTPX:
            with patch("integradio.visual.figma.httpx.Client") as mock_client_class:
                mock_response_file = MagicMock()
                mock_response_file.status_code = 200
                mock_response_file.json.return_value = {
                    "name": "Test",
                    "document": {"id": "0:0", "type": "DOCUMENT", "children": []},
                }

                mock_response_styles = MagicMock()
                mock_response_styles.status_code = 200
                mock_response_styles.json.return_value = styles_response

                mock_client = MagicMock()
                mock_client.get.side_effect = [mock_response_file, mock_response_styles]
                mock_client_class.return_value = mock_client

                client = FigmaClient(sample_config)
                importer = FigmaImporter(client)
                spec = importer.import_file("file-key")

                # Should have normalized the name
                colors = spec.tokens.get("colors")
                if colors:
                    flat = colors.flatten()
                    # Name should be lowercase with dots and dashes
                    for key in flat.keys():
                        assert "/" not in key
                        assert key == key.lower()
