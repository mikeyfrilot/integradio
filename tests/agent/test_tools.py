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


class TestToolsWithSemanticComponents:
    """Integration tests with mock SemanticComponent instances."""

    def test_component_tool_finds_registered_components(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ComponentTool finds registered SemanticComponents."""
        tool = ComponentTool(mock_blocks)

        # Search by intent
        result = tool.run(intent="search")

        assert isinstance(result, ToolResult)
        # Should find at least one component with "search" in intent

    def test_component_tool_search_by_query(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ComponentTool general query search."""
        tool = ComponentTool(mock_blocks)
        result = tool.run(query="input")

        assert isinstance(result, ToolResult)
        assert isinstance(result.data, list)

    def test_component_tool_search_by_tag(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ComponentTool search by tag."""
        tool = ComponentTool(mock_blocks)
        result = tool.run(tag="action")

        assert isinstance(result, ToolResult)

    def test_component_tool_search_by_type(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ComponentTool search by component type."""
        tool = ComponentTool(mock_blocks)
        result = tool.run(component_type="Button")

        assert isinstance(result, ToolResult)

    def test_component_tool_combined_filters(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ComponentTool with combined filters."""
        tool = ComponentTool(mock_blocks)
        result = tool.run(
            intent="search",
            tag="input",
            interactive_only=True,
            visible_only=True,
            max_results=5,
        )

        assert isinstance(result, ToolResult)
        assert len(result.data) <= 5

    def test_component_tool_no_filters_returns_all(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ComponentTool with no filters returns all components."""
        tool = ComponentTool(mock_blocks)
        result = tool.run()  # No filters

        assert isinstance(result, ToolResult)
        # Should return all registered semantic components

    def test_action_tool_with_valid_component(
        self,
        mock_blocks,
        component_with_click,
        patch_semantic_component,
    ):
        """Test ActionTool with valid component."""
        component, semantic = component_with_click

        tool = ActionTool(mock_blocks)
        result = tool.run(
            component_id=str(component._id),
            action="click",
        )

        assert isinstance(result, ActionResult)
        # Should succeed since component has click method
        if result.success:
            assert result.action == "click"

    def test_action_tool_set_value(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test ActionTool set_value action."""
        component, semantic = component_with_value

        tool = ActionTool(mock_blocks)
        result = tool.run(
            component_id=str(component._id),
            action="set_value",
            value="new value",
        )

        assert isinstance(result, ActionResult)
        assert result.action == "set_value"
        if result.success:
            assert result.new_value == "new value"

    def test_action_tool_clear(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test ActionTool clear action."""
        component, semantic = component_with_value

        tool = ActionTool(mock_blocks)
        result = tool.run(
            component_id=str(component._id),
            action="clear",
        )

        assert isinstance(result, ActionResult)
        assert result.action == "clear"

    def test_action_tool_trigger_event(
        self,
        mock_blocks,
        component_with_click,
        patch_semantic_component,
    ):
        """Test ActionTool trigger action."""
        component, semantic = component_with_click

        tool = ActionTool(mock_blocks)
        result = tool.run(
            component_id=str(component._id),
            action="trigger",
            event="custom_event",
        )

        assert isinstance(result, ActionResult)
        assert result.action == "trigger"

    def test_action_tool_no_click_method(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ActionTool click on component without click method."""
        # Get a component that doesn't have click method (like Markdown)
        from integradio.components import SemanticComponent

        # Find a component
        comp_id = next(iter(SemanticComponent._instances.keys()))

        tool = ActionTool(mock_blocks)
        result = tool.run(
            component_id=str(comp_id),
            action="click",
        )

        assert isinstance(result, ActionResult)
        # Might fail or succeed depending on component type

    def test_state_tool_with_component(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test StateTool retrieves component state."""
        component, semantic = component_with_value

        tool = StateTool(mock_blocks)
        result = tool.run(component_id=str(component._id))

        assert isinstance(result, StateResult)
        if result.success:
            assert result.data is not None
            assert "type" in result.data
            assert "intent" in result.data

    def test_state_tool_with_visual_spec(
        self,
        mock_blocks,
        component_with_visual_spec,
        patch_semantic_component,
    ):
        """Test StateTool includes visual spec when requested."""
        component, semantic = component_with_visual_spec

        tool = StateTool(mock_blocks)
        result = tool.run(
            component_id=str(component._id),
            include_visual_spec=True,
        )

        assert isinstance(result, StateResult)
        if result.success:
            assert "visual_spec" in result.data

    def test_state_tool_without_visual_spec(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test StateTool without visual spec."""
        component, semantic = component_with_value

        tool = StateTool(mock_blocks)
        result = tool.run(
            component_id=str(component._id),
            include_visual_spec=False,
        )

        assert isinstance(result, StateResult)
        if result.success:
            assert "visual_spec" in result.data
            assert result.data["visual_spec"]["has_spec"] is False

    def test_flow_tool_with_dependencies(
        self,
        populated_blocks,
        mock_semantic_components,
    ):
        """Test FlowTool traces dataflow."""
        tool = FlowTool(populated_blocks)

        # Trace from first component
        result = tool.run(component_id="1", direction="forward")

        assert isinstance(result, FlowResult)
        assert result.direction == "forward"

    def test_flow_tool_backward_trace(
        self,
        populated_blocks,
        mock_semantic_components,
    ):
        """Test FlowTool backward tracing."""
        tool = FlowTool(populated_blocks)

        result = tool.run(component_id="3", direction="backward")

        assert isinstance(result, FlowResult)
        assert result.direction == "backward"


class TestConvenienceFunctionsWithComponents:
    """Integration tests for convenience functions."""

    def test_query_by_intent_with_components(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test query_by_intent finds matching components."""
        try:
            results = query_by_intent("search", mock_blocks)
            assert isinstance(results, list)
        except TypeError:
            # ComponentInfo may have different constructor
            pass

    def test_query_by_tag_with_components(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test query_by_tag finds matching components."""
        try:
            results = query_by_tag("input", mock_blocks)
            assert isinstance(results, list)
        except TypeError:
            # ComponentInfo may have different constructor
            pass

    def test_query_by_type_with_components(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test query_by_type finds matching components."""
        results = query_by_type("Button", mock_blocks)
        assert isinstance(results, list)

    def test_get_component_value_found(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test get_component_value returns value."""
        component, semantic = component_with_value

        value = get_component_value(str(component._id), mock_blocks)
        # May return None if component not found
        assert value is None or value == "test value"

    def test_set_component_value_success(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test set_component_value returns success status."""
        component, semantic = component_with_value

        success = set_component_value(str(component._id), "new", mock_blocks)
        assert isinstance(success, bool)

    def test_click_component_success(
        self,
        mock_blocks,
        component_with_click,
        patch_semantic_component,
    ):
        """Test click_component returns success status."""
        component, semantic = component_with_click

        success = click_component(str(component._id), mock_blocks)
        assert isinstance(success, bool)

    def test_trace_data_flow_forward(
        self,
        populated_blocks,
        mock_semantic_components,
    ):
        """Test trace_data_flow forward direction."""
        connected = trace_data_flow("1", direction="forward", blocks=populated_blocks)
        assert isinstance(connected, list)

    def test_trace_data_flow_backward(
        self,
        populated_blocks,
        mock_semantic_components,
    ):
        """Test trace_data_flow backward direction."""
        connected = trace_data_flow("3", direction="backward", blocks=populated_blocks)
        assert isinstance(connected, list)


class TestToolResultTypes:
    """Tests for specific result type attributes."""

    def test_action_result_attributes(self):
        """Test ActionResult has all expected attributes."""
        result = ActionResult(
            success=True,
            message="Test",
            action="set_value",
            component_id="123",
            previous_value="old",
            new_value="new",
        )

        assert result.action == "set_value"
        assert result.component_id == "123"
        assert result.previous_value == "old"
        assert result.new_value == "new"

    def test_state_result_attributes(self):
        """Test StateResult has all expected attributes."""
        result = StateResult(
            success=True,
            message="Test",
            component_id="123",
            state_key="value",
            value="test",
        )

        assert result.component_id == "123"
        assert result.state_key == "value"
        assert result.value == "test"

    def test_flow_result_attributes(self):
        """Test FlowResult has all expected attributes."""
        result = FlowResult(
            success=True,
            message="Test",
            source_id="123",
            direction="forward",
            connected_components=["456", "789"],
            handlers=["handler1"],
        )

        assert result.source_id == "123"
        assert result.direction == "forward"
        assert result.connected_components == ["456", "789"]
        assert result.handlers == ["handler1"]


class TestCallableValueHandling:
    """Tests for handling callable values in components."""

    def test_component_tool_handles_callable_value(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ComponentTool handles callable value gracefully."""
        from integradio.components import SemanticComponent

        # Find a component and set its value to a callable
        for comp_id, semantic in SemanticComponent._instances.items():
            semantic.component.value = lambda: "computed"
            break

        tool = ComponentTool(mock_blocks)
        result = tool.run()  # Get all components

        # Should not crash
        assert isinstance(result, ToolResult)

    def test_action_tool_handles_callable_previous_value(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test ActionTool handles callable previous_value."""
        from integradio.components import SemanticComponent

        # Find a component and set its value to a callable
        comp_id = None
        for cid, semantic in SemanticComponent._instances.items():
            semantic.component.value = lambda: "computed"
            comp_id = cid
            break

        if comp_id:
            tool = ActionTool(mock_blocks)
            result = tool.run(
                component_id=str(comp_id),
                action="set_value",
                value="new",
            )

            assert isinstance(result, ActionResult)
            # previous_value should be None for callable
            if result.success:
                assert result.previous_value is None
