"""
Tests for Agent Tools module.

Tests:
- ToolResult and subclasses
- ComponentTool search functionality
- ActionTool action handling
- StateTool state retrieval
- FlowTool dataflow tracing
- Convenience functions
"""

import pytest
import json
from integradio.agent.tools import (
    ToolResult,
    ComponentInfo,
    ActionResult,
    StateResult,
    FlowResult,
    BaseTool,
    ComponentTool,
    ActionTool,
    StateTool,
    FlowTool,
    create_all_tools,
    query_by_intent,
    query_by_tag,
    query_by_type,
    get_component_value,
    set_component_value,
    click_component,
    trace_data_flow,
)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_create_success_result(self):
        """Test creating successful result."""
        result = ToolResult(
            success=True,
            message="Operation completed",
            data={"key": "value"},
        )

        assert result.success is True
        assert result.message == "Operation completed"
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_create_error_result(self):
        """Test creating error result."""
        result = ToolResult(
            success=False,
            message="Operation failed",
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ToolResult(
            success=True,
            message="Test",
            data=[1, 2, 3],
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["message"] == "Test"
        assert data["data"] == [1, 2, 3]
        assert data["error"] is None

    def test_result_to_json(self):
        """Test JSON serialization."""
        result = ToolResult(
            success=True,
            message="Test",
            data={"test": True},
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert parsed["data"]["test"] is True


class TestComponentInfo:
    """Tests for ComponentInfo dataclass."""

    def test_create_component_info(self):
        """Test creating component info."""
        info = ComponentInfo(
            component_id="123",
            component_type="Button",
            intent="submit form",
        )

        assert info.component_id == "123"
        assert info.component_type == "Button"
        assert info.intent == "submit form"

    def test_component_info_defaults(self):
        """Test default values."""
        info = ComponentInfo(
            component_id="1",
            component_type="Textbox",
            intent="input",
        )

        assert info.label is None
        assert info.tags == []
        assert info.is_interactive is True
        assert info.is_visible is True
        assert info.current_value is None

    def test_component_info_to_dict(self):
        """Test component info serialization."""
        info = ComponentInfo(
            component_id="456",
            component_type="Dropdown",
            intent="select option",
            label="Choose one",
            tags=["form", "input"],
            current_value="option1",
        )

        data = info.to_dict()

        assert data["id"] == "456"
        assert data["type"] == "Dropdown"
        assert data["intent"] == "select option"
        assert data["label"] == "Choose one"
        assert data["tags"] == ["form", "input"]
        assert data["value"] == "option1"


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_create_action_result(self):
        """Test creating action result."""
        result = ActionResult(
            success=True,
            message="Value set",
            action="set_value",
            component_id="123",
            previous_value="old",
            new_value="new",
        )

        assert result.success is True
        assert result.action == "set_value"
        assert result.component_id == "123"
        assert result.previous_value == "old"
        assert result.new_value == "new"


class TestStateResult:
    """Tests for StateResult dataclass."""

    def test_create_state_result(self):
        """Test creating state result."""
        result = StateResult(
            success=True,
            message="State retrieved",
            component_id="123",
            state_key="value",
            value="test value",
        )

        assert result.success is True
        assert result.component_id == "123"
        assert result.value == "test value"


class TestFlowResult:
    """Tests for FlowResult dataclass."""

    def test_create_flow_result(self):
        """Test creating flow result."""
        result = FlowResult(
            success=True,
            message="Flow traced",
            source_id="input1",
            direction="forward",
            connected_components=["output1", "output2"],
            handlers=["handler1"],
        )

        assert result.success is True
        assert result.source_id == "input1"
        assert result.direction == "forward"
        assert len(result.connected_components) == 2
        assert "handler1" in result.handlers


class TestComponentTool:
    """Tests for ComponentTool."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = ComponentTool()

        assert tool.name == "find_component"
        assert "Find UI components" in tool.description

    def test_tool_schema(self):
        """Test tool has valid schema."""
        tool = ComponentTool()
        schema = tool.schema

        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "intent" in schema["properties"]
        assert "tag" in schema["properties"]
        assert "component_type" in schema["properties"]

    def test_search_no_components(self):
        """Test search with no components registered."""
        tool = ComponentTool()
        result = tool.run(query="test")

        # Should handle empty case gracefully
        assert isinstance(result, ToolResult)

    def test_search_by_query(self):
        """Test general query search."""
        tool = ComponentTool()
        result = tool.run(query="submit")

        assert isinstance(result, ToolResult)
        assert isinstance(result.data, list)

    def test_search_by_intent(self):
        """Test intent search."""
        tool = ComponentTool()
        result = tool.run(intent="button")

        assert isinstance(result, ToolResult)

    def test_search_by_tag(self):
        """Test tag search."""
        tool = ComponentTool()
        result = tool.run(tag="form")

        assert isinstance(result, ToolResult)

    def test_search_by_type(self):
        """Test type search."""
        tool = ComponentTool()
        result = tool.run(component_type="Button")

        assert isinstance(result, ToolResult)

    def test_search_with_filters(self):
        """Test search with visibility and interactivity filters."""
        tool = ComponentTool()
        result = tool.run(
            query="test",
            interactive_only=True,
            visible_only=True,
            max_results=5,
        )

        assert isinstance(result, ToolResult)
        assert len(result.data) <= 5

    def test_tool_callable(self):
        """Test tool can be called as function."""
        tool = ComponentTool()
        result = tool(query="test")

        assert isinstance(result, ToolResult)


class TestActionTool:
    """Tests for ActionTool."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = ActionTool()

        assert tool.name == "component_action"
        assert "action" in tool.description.lower()

    def test_tool_schema(self):
        """Test tool has valid schema."""
        tool = ActionTool()
        schema = tool.schema

        assert "component_id" in schema["required"]
        assert "action" in schema["required"]
        assert schema["properties"]["action"]["enum"] == ["click", "set_value", "clear", "trigger"]

    def test_invalid_component_id(self):
        """Test action with invalid component ID."""
        tool = ActionTool()
        result = tool.run(component_id="invalid", action="click")

        assert result.success is False
        assert "Invalid" in result.message or "invalid" in result.error.lower()

    def test_component_not_found(self):
        """Test action with non-existent component."""
        tool = ActionTool()
        result = tool.run(component_id="999999", action="click")

        assert result.success is False
        assert "not found" in result.message.lower()

    def test_click_action(self):
        """Test click action structure."""
        tool = ActionTool()
        result = tool.run(component_id="999999", action="click")

        # Should fail (no component) but return valid ActionResult
        assert isinstance(result, ActionResult)
        assert result.action == "click"

    def test_set_value_without_value(self):
        """Test set_value without providing value."""
        tool = ActionTool()
        result = tool.run(component_id="999999", action="set_value")

        # Should succeed in lookup but fail if value missing
        assert isinstance(result, ActionResult)

    def test_trigger_without_event(self):
        """Test trigger without event name."""
        tool = ActionTool()
        result = tool.run(component_id="999999", action="trigger")

        assert isinstance(result, ActionResult)

    def test_unknown_action(self):
        """Test unknown action."""
        tool = ActionTool()
        result = tool.run(component_id="1", action="unknown_action")

        assert isinstance(result, ActionResult)


class TestStateTool:
    """Tests for StateTool."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = StateTool()

        assert tool.name == "get_state"
        assert "state" in tool.description.lower()

    def test_tool_schema(self):
        """Test tool has valid schema."""
        tool = StateTool()
        schema = tool.schema

        assert "component_id" in schema["required"]
        assert "include_visual_spec" in schema["properties"]

    def test_invalid_component_id(self):
        """Test state with invalid component ID."""
        tool = StateTool()
        result = tool.run(component_id="invalid")

        assert result.success is False

    def test_component_not_found(self):
        """Test state with non-existent component."""
        tool = StateTool()
        result = tool.run(component_id="999999")

        assert result.success is False

    def test_include_visual_spec_option(self):
        """Test include_visual_spec option."""
        tool = StateTool()
        result = tool.run(component_id="999999", include_visual_spec=True)

        # Should fail (no component) but accept the option
        assert isinstance(result, StateResult)


class TestFlowTool:
    """Tests for FlowTool."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = FlowTool()

        assert tool.name == "trace_flow"
        assert "dataflow" in tool.description.lower()

    def test_tool_schema(self):
        """Test tool has valid schema."""
        tool = FlowTool()
        schema = tool.schema

        assert "component_id" in schema["required"]
        assert schema["properties"]["direction"]["enum"] == ["forward", "backward"]

    def test_trace_forward(self, mock_blocks):
        """Test forward tracing."""
        tool = FlowTool(mock_blocks)
        result = tool.run(component_id="test", direction="forward")

        assert isinstance(result, FlowResult)
        assert result.direction == "forward"

    def test_trace_backward(self, mock_blocks):
        """Test backward tracing."""
        tool = FlowTool(mock_blocks)
        result = tool.run(component_id="test", direction="backward")

        assert isinstance(result, FlowResult)
        assert result.direction == "backward"

    def test_max_depth(self, mock_blocks):
        """Test max_depth parameter."""
        tool = FlowTool(mock_blocks)
        result = tool.run(component_id="test", max_depth=5)

        assert isinstance(result, FlowResult)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_all_tools(self):
        """Test create_all_tools returns all tools."""
        tools = create_all_tools()

        assert "find_component" in tools
        assert "component_action" in tools
        assert "get_state" in tools
        assert "trace_flow" in tools

        assert isinstance(tools["find_component"], ComponentTool)
        assert isinstance(tools["component_action"], ActionTool)
        assert isinstance(tools["get_state"], StateTool)
        assert isinstance(tools["trace_flow"], FlowTool)

    def test_query_by_intent(self):
        """Test query_by_intent function."""
        results = query_by_intent("submit")
        assert isinstance(results, list)

    def test_query_by_tag(self):
        """Test query_by_tag function."""
        results = query_by_tag("form")
        assert isinstance(results, list)

    def test_query_by_type(self):
        """Test query_by_type function."""
        results = query_by_type("Button")
        assert isinstance(results, list)

    def test_get_component_value(self):
        """Test get_component_value function."""
        value = get_component_value("999999")
        # No component exists, should return None
        assert value is None

    def test_set_component_value(self):
        """Test set_component_value function."""
        success = set_component_value("999999", "new value")
        # No component exists, should return False
        assert isinstance(success, bool)

    def test_click_component(self):
        """Test click_component function."""
        success = click_component("999999")
        assert isinstance(success, bool)

    def test_trace_data_flow(self):
        """Test trace_data_flow function."""
        connected = trace_data_flow("test", direction="forward")
        assert isinstance(connected, list)


class TestToolEdgeCases:
    """Edge case tests for tools."""

    def test_component_tool_empty_query(self):
        """Test component tool with empty strings."""
        tool = ComponentTool()
        result = tool.run(query="", intent="", tag="")

        # Should handle empty strings gracefully
        assert isinstance(result, ToolResult)

    def test_action_tool_empty_value(self):
        """Test action tool with empty value."""
        tool = ActionTool()
        result = tool.run(component_id="1", action="set_value", value="")

        assert isinstance(result, ActionResult)

    def test_state_tool_none_component(self):
        """Test state tool with None-like ID."""
        tool = StateTool()
        result = tool.run(component_id="0")

        assert isinstance(result, StateResult)

    def test_flow_tool_negative_depth(self):
        """Test flow tool with edge case depth."""
        tool = FlowTool()
        result = tool.run(component_id="test", max_depth=0)

        assert isinstance(result, FlowResult)

    def test_result_json_with_complex_data(self):
        """Test JSON serialization with complex data."""
        result = ToolResult(
            success=True,
            message="Test",
            data={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "null": None,
            },
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["data"]["nested"]["key"] == "value"
        assert parsed["data"]["list"] == [1, 2, 3]
        assert parsed["data"]["null"] is None
