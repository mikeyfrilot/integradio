"""
Pytest fixtures for agent tests.

Reuses fixtures from inspector module and adds agent-specific fixtures.
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Any, Optional

# Import shared fixtures from inspector
from tests.inspector.conftest import (
    MockVisualSpec,
    MockSemanticMetadata,
    MockGradioComponent,
    MockSemanticComponent,
    MockBlocks,
    mock_gradio_components,
    mock_semantic_components,
    mock_blocks,
    populated_blocks,
    patch_semantic_component,
    clean_semantic_instances,
)


@pytest.fixture
def component_with_click():
    """Create a component with click capability."""
    component = MockGradioComponent(
        _id=100,
        component_type="Button",
        label="Click Me",
    )

    semantic = MockSemanticComponent(
        component,
        intent="clickable button",
        tags=["action"],
    )

    return component, semantic


@pytest.fixture
def component_with_value():
    """Create a component with value."""
    component = MockGradioComponent(
        _id=101,
        component_type="Textbox",
        label="Input",
        value="test value",
    )

    semantic = MockSemanticComponent(
        component,
        intent="text input",
        tags=["input"],
    )

    return component, semantic


@pytest.fixture
def component_with_visual_spec():
    """Create a component with visual specification."""
    component = MockGradioComponent(
        _id=102,
        component_type="Button",
        label="Styled",
    )

    visual = MockVisualSpec(
        component_id="102",
        component_type="Button",
        tokens={"background": "#3b82f6"},
    )

    semantic = MockSemanticComponent(
        component,
        intent="styled button",
        tags=["styled"],
        visual_spec=visual,
    )

    return component, semantic


@pytest.fixture
def agent_tools(mock_blocks):
    """Create all agent tools with mock blocks."""
    from integradio.agent.tools import create_all_tools

    return create_all_tools(mock_blocks)


@pytest.fixture
def mcp_server(mock_blocks):
    """Create MCP server with mock blocks."""
    from integradio.agent.mcp import MCPComponentServer

    return MCPComponentServer(mock_blocks, name="test-server")
