"""
Additional tests for LangChain integration to improve coverage.

Focuses on:
- Input schema classes (FindComponentInput, ActionInput, StateInput, FlowInput)
- Tool execution with blocks reference
- Schema validation
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from integradio.agent.langchain import (
    HAS_LANGCHAIN,
    SemanticComponentTool,
    SemanticActionTool,
    SemanticStateTool,
    SemanticFlowTool,
    create_langchain_tools,
)


# =============================================================================
# Tests for Input Schema Classes (lines 43-67)
# =============================================================================

@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestInputSchemas:
    """Tests for Pydantic input schema classes."""

    def test_find_component_input_defaults(self):
        """Test FindComponentInput with default values."""
        from integradio.agent.langchain import FindComponentInput

        input_model = FindComponentInput()
        assert input_model.query is None
        assert input_model.intent is None
        assert input_model.tag is None
        assert input_model.component_type is None
        assert input_model.max_results == 10

    def test_find_component_input_custom(self):
        """Test FindComponentInput with custom values."""
        from integradio.agent.langchain import FindComponentInput

        input_model = FindComponentInput(
            query="search button",
            intent="submit form",
            tag="action",
            component_type="Button",
            max_results=5,
        )
        assert input_model.query == "search button"
        assert input_model.intent == "submit form"
        assert input_model.tag == "action"
        assert input_model.component_type == "Button"
        assert input_model.max_results == 5

    def test_action_input_required_fields(self):
        """Test ActionInput with required fields."""
        from integradio.agent.langchain import ActionInput

        input_model = ActionInput(
            component_id="123",
            action="click",
        )
        assert input_model.component_id == "123"
        assert input_model.action == "click"
        assert input_model.value is None
        assert input_model.event is None

    def test_action_input_all_fields(self):
        """Test ActionInput with all fields."""
        from integradio.agent.langchain import ActionInput

        input_model = ActionInput(
            component_id="456",
            action="set_value",
            value="test input",
            event="change",
        )
        assert input_model.component_id == "456"
        assert input_model.action == "set_value"
        assert input_model.value == "test input"
        assert input_model.event == "change"

    def test_state_input_defaults(self):
        """Test StateInput with default values."""
        from integradio.agent.langchain import StateInput

        input_model = StateInput(component_id="789")
        assert input_model.component_id == "789"
        assert input_model.include_visual_spec is False

    def test_state_input_with_visual_spec(self):
        """Test StateInput with include_visual_spec=True."""
        from integradio.agent.langchain import StateInput

        input_model = StateInput(
            component_id="101",
            include_visual_spec=True,
        )
        assert input_model.include_visual_spec is True

    def test_flow_input_defaults(self):
        """Test FlowInput with default values."""
        from integradio.agent.langchain import FlowInput

        input_model = FlowInput(component_id="202")
        assert input_model.component_id == "202"
        assert input_model.direction == "forward"
        assert input_model.max_depth == 3

    def test_flow_input_custom(self):
        """Test FlowInput with custom values."""
        from integradio.agent.langchain import FlowInput

        input_model = FlowInput(
            component_id="303",
            direction="backward",
            max_depth=5,
        )
        assert input_model.direction == "backward"
        assert input_model.max_depth == 5


# =============================================================================
# Tests for Tool Args Schema (lines 92, 135, 176, 212)
# =============================================================================

@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestToolArgsSchema:
    """Tests for args_schema on LangChain tools."""

    def test_component_tool_args_schema_type(self):
        """Test SemanticComponentTool has correct args_schema type."""
        from integradio.agent.langchain import FindComponentInput

        tool = SemanticComponentTool()
        assert tool.args_schema is FindComponentInput

    def test_action_tool_args_schema_type(self):
        """Test SemanticActionTool has correct args_schema type."""
        from integradio.agent.langchain import ActionInput

        tool = SemanticActionTool()
        assert tool.args_schema is ActionInput

    def test_state_tool_args_schema_type(self):
        """Test SemanticStateTool has correct args_schema type."""
        from integradio.agent.langchain import StateInput

        tool = SemanticStateTool()
        assert tool.args_schema is StateInput

    def test_flow_tool_args_schema_type(self):
        """Test SemanticFlowTool has correct args_schema type."""
        from integradio.agent.langchain import FlowInput

        tool = SemanticFlowTool()
        assert tool.args_schema is FlowInput


# =============================================================================
# Tests for Tool _run Methods (lines 104-114, 146-155, 185-192, 222-230)
# =============================================================================

@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestToolRunMethods:
    """Tests for tool _run methods with mocked underlying tools."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            fns = []
            dependencies = []
        return MockBlocks()

    def test_component_tool_run_delegates(self, mock_blocks):
        """Test SemanticComponentTool._run delegates to ComponentTool."""
        with patch('integradio.agent.langchain.ComponentTool') as MockComponentTool:
            mock_tool_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.to_json.return_value = '{"success": true}'
            mock_tool_instance.run.return_value = mock_result
            MockComponentTool.return_value = mock_tool_instance

            tool = SemanticComponentTool(blocks=mock_blocks)
            result = tool._run(
                query="test",
                intent="submit",
                tag="action",
                component_type="Button",
                max_results=5,
            )

            MockComponentTool.assert_called_once_with(mock_blocks)
            mock_tool_instance.run.assert_called_once_with(
                query="test",
                intent="submit",
                tag="action",
                component_type="Button",
                max_results=5,
            )
            assert result == '{"success": true}'

    def test_action_tool_run_delegates(self, mock_blocks):
        """Test SemanticActionTool._run delegates to ActionTool."""
        with patch('integradio.agent.langchain.ActionTool') as MockActionTool:
            mock_tool_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.to_json.return_value = '{"action": "click"}'
            mock_tool_instance.run.return_value = mock_result
            MockActionTool.return_value = mock_tool_instance

            tool = SemanticActionTool(blocks=mock_blocks)
            result = tool._run(
                component_id="123",
                action="click",
                value="test",
                event="click",
            )

            MockActionTool.assert_called_once_with(mock_blocks)
            mock_tool_instance.run.assert_called_once_with(
                component_id="123",
                action="click",
                value="test",
                event="click",
            )
            assert result == '{"action": "click"}'

    def test_state_tool_run_delegates(self, mock_blocks):
        """Test SemanticStateTool._run delegates to StateTool."""
        with patch('integradio.agent.langchain.StateTool') as MockStateTool:
            mock_tool_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.to_json.return_value = '{"state": "visible"}'
            mock_tool_instance.run.return_value = mock_result
            MockStateTool.return_value = mock_tool_instance

            tool = SemanticStateTool(blocks=mock_blocks)
            result = tool._run(
                component_id="456",
                include_visual_spec=True,
            )

            MockStateTool.assert_called_once_with(mock_blocks)
            mock_tool_instance.run.assert_called_once_with(
                component_id="456",
                include_visual_spec=True,
            )
            assert result == '{"state": "visible"}'

    def test_flow_tool_run_delegates(self, mock_blocks):
        """Test SemanticFlowTool._run delegates to FlowTool."""
        with patch('integradio.agent.langchain.FlowTool') as MockFlowTool:
            mock_tool_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.to_json.return_value = '{"flow": []}'
            mock_tool_instance.run.return_value = mock_result
            MockFlowTool.return_value = mock_tool_instance

            tool = SemanticFlowTool(blocks=mock_blocks)
            result = tool._run(
                component_id="789",
                direction="backward",
                max_depth=5,
            )

            MockFlowTool.assert_called_once_with(mock_blocks)
            mock_tool_instance.run.assert_called_once_with(
                component_id="789",
                direction="backward",
                max_depth=5,
            )
            assert result == '{"flow": []}'


# =============================================================================
# Tests for create_langchain_tools (line 256)
# =============================================================================

@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestCreateLangchainToolsBlocks:
    """Tests for create_langchain_tools with blocks parameter."""

    def test_create_tools_passes_blocks_to_all(self):
        """Test that blocks is passed to all created tools."""
        class MockBlocks:
            fns = []
            dependencies = []

        blocks = MockBlocks()
        tools = create_langchain_tools(blocks)

        assert len(tools) == 4

        for tool in tools:
            assert tool.blocks is blocks

    def test_tools_have_correct_order(self):
        """Test tools are returned in expected order."""
        tools = create_langchain_tools()

        assert tools[0].name == "find_component"
        assert tools[1].name == "component_action"
        assert tools[2].name == "get_state"
        assert tools[3].name == "trace_flow"


# =============================================================================
# Tests for HAS_LANGCHAIN False Branch (lines 26-28, 31-34)
# =============================================================================

class TestLangChainNotInstalledFallback:
    """Tests for fallback behavior when LangChain is not installed."""

    @pytest.mark.skipif(HAS_LANGCHAIN, reason="LangChain IS installed")
    def test_create_langchain_tools_import_error(self):
        """Test create_langchain_tools raises ImportError without LangChain."""
        with pytest.raises(ImportError) as exc_info:
            create_langchain_tools()

        assert "LangChain" in str(exc_info.value)
        assert "pip install langchain-core" in str(exc_info.value)

    def test_tool_classes_inherit_from_object_without_langchain(self):
        """Test that tool classes still exist even without LangChain."""
        # These classes should be defined regardless
        assert SemanticComponentTool is not None
        assert SemanticActionTool is not None
        assert SemanticStateTool is not None
        assert SemanticFlowTool is not None


# =============================================================================
# Integration Tests with Full Tool Execution
# =============================================================================

@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestToolIntegrationWithMocks:
    """Integration tests for tools with mocked dependencies."""

    @pytest.fixture
    def full_mock_blocks(self):
        """Create fully mocked blocks with all needed attributes."""
        blocks = MagicMock()
        blocks.fns = []
        blocks.dependencies = []
        return blocks

    def test_component_tool_full_flow(self, full_mock_blocks):
        """Test full flow through component tool."""
        tool = SemanticComponentTool(blocks=full_mock_blocks)

        # Run the tool - it will call ComponentTool internally
        result = tool._run(query="button")

        # Result should be JSON string
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "success" in parsed or "error" in parsed or "components" in parsed

    def test_action_tool_full_flow(self, full_mock_blocks):
        """Test full flow through action tool."""
        tool = SemanticActionTool(blocks=full_mock_blocks)

        result = tool._run(component_id="1", action="click")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_state_tool_full_flow(self, full_mock_blocks):
        """Test full flow through state tool."""
        tool = SemanticStateTool(blocks=full_mock_blocks)

        result = tool._run(component_id="1")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_flow_tool_full_flow(self, full_mock_blocks):
        """Test full flow through flow tool."""
        tool = SemanticFlowTool(blocks=full_mock_blocks)

        result = tool._run(component_id="1", direction="forward", max_depth=2)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
