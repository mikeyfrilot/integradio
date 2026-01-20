"""
Tests for Prompts module.

Tests:
- System prompt constants
- get_system_prompt builder
- Result formatters
- Context builders
"""

import pytest
from integradio.agent.prompts import (
    COMPONENT_QUERY_PROMPT,
    ACTION_PROMPT,
    STATE_PROMPT,
    FLOW_PROMPT,
    get_system_prompt,
    format_component_list,
    format_action_result,
    format_state_result,
    format_flow_result,
    build_component_context,
)


class TestPromptConstants:
    """Tests for prompt constants."""

    def test_component_query_prompt(self):
        """Test COMPONENT_QUERY_PROMPT content."""
        assert "find_component" in COMPONENT_QUERY_PROMPT
        assert "intent" in COMPONENT_QUERY_PROMPT
        assert "tag" in COMPONENT_QUERY_PROMPT.lower()
        assert "type" in COMPONENT_QUERY_PROMPT.lower()

    def test_action_prompt(self):
        """Test ACTION_PROMPT content."""
        assert "action" in ACTION_PROMPT.lower()
        assert "click" in ACTION_PROMPT.lower()
        assert "set_value" in ACTION_PROMPT

    def test_state_prompt(self):
        """Test STATE_PROMPT content."""
        assert "state" in STATE_PROMPT.lower()
        assert "value" in STATE_PROMPT.lower()

    def test_flow_prompt(self):
        """Test FLOW_PROMPT content."""
        assert "dataflow" in FLOW_PROMPT.lower() or "flow" in FLOW_PROMPT.lower()
        assert "forward" in FLOW_PROMPT.lower()
        assert "backward" in FLOW_PROMPT.lower()


class TestGetSystemPrompt:
    """Tests for get_system_prompt function."""

    def test_default_prompt(self):
        """Test default system prompt includes all sections."""
        prompt = get_system_prompt()

        assert "Finding Components" in prompt
        assert "Performing Actions" in prompt
        assert "Reading State" in prompt
        assert "Tracing Dataflow" in prompt
        assert "General Guidelines" in prompt

    def test_exclude_query(self):
        """Test excluding query section."""
        prompt = get_system_prompt(include_query=False)

        assert "Finding Components" not in prompt

    def test_exclude_action(self):
        """Test excluding action section."""
        prompt = get_system_prompt(include_action=False)

        assert "Performing Actions" not in prompt

    def test_exclude_state(self):
        """Test excluding state section."""
        prompt = get_system_prompt(include_state=False)

        assert "Reading State" not in prompt

    def test_exclude_flow(self):
        """Test excluding flow section."""
        prompt = get_system_prompt(include_flow=False)

        assert "Tracing Dataflow" not in prompt

    def test_custom_instructions(self):
        """Test adding custom instructions."""
        custom = "Always be polite to the user."
        prompt = get_system_prompt(custom_instructions=custom)

        assert "Custom Instructions" in prompt
        assert custom in prompt

    def test_minimal_prompt(self):
        """Test prompt with all sections excluded."""
        prompt = get_system_prompt(
            include_query=False,
            include_action=False,
            include_state=False,
            include_flow=False,
        )

        # Should still have intro and guidelines
        assert "AI assistant" in prompt
        assert "General Guidelines" in prompt


class TestFormatComponentList:
    """Tests for format_component_list function."""

    def test_empty_list(self):
        """Test formatting empty list."""
        result = format_component_list([])

        assert "No components found" in result

    def test_single_component(self):
        """Test formatting single component."""
        components = [
            {
                "id": "123",
                "type": "Button",
                "intent": "submit form",
                "tags": ["form", "action"],
                "value": None,
            }
        ]

        result = format_component_list(components)

        assert "1 component" in result
        assert "[123]" in result
        assert "Button" in result
        assert "submit form" in result
        assert "form" in result
        assert "action" in result

    def test_multiple_components(self):
        """Test formatting multiple components."""
        components = [
            {"id": "1", "type": "Button", "intent": "click me", "tags": [], "value": None},
            {"id": "2", "type": "Textbox", "intent": "input text", "tags": ["input"], "value": "hello"},
            {"id": "3", "type": "Dropdown", "intent": "select option", "tags": ["select"], "value": "opt1"},
        ]

        result = format_component_list(components)

        assert "3 component" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_exclude_values(self):
        """Test excluding values."""
        components = [
            {"id": "1", "type": "Textbox", "intent": "input", "value": "secret"},
        ]

        result = format_component_list(components, include_values=False)

        assert "secret" not in result

    def test_exclude_tags(self):
        """Test excluding tags."""
        components = [
            {"id": "1", "type": "Button", "intent": "click", "tags": ["important"]},
        ]

        result = format_component_list(components, include_tags=False)

        assert "Tags:" not in result

    def test_truncate_long_values(self):
        """Test truncating long values."""
        long_value = "a" * 100
        components = [
            {"id": "1", "type": "Textbox", "intent": "input", "value": long_value},
        ]

        result = format_component_list(components)

        # Should be truncated with ...
        assert "..." in result
        assert len(long_value) > 50  # Original was long


class TestFormatActionResult:
    """Tests for format_action_result function."""

    def test_success_result(self):
        """Test formatting successful action."""
        result = {
            "success": True,
            "message": "Value set successfully",
            "new_value": "hello",
        }

        formatted = format_action_result(result)

        assert "Value set successfully" in formatted
        assert "hello" in formatted

    def test_success_without_value(self):
        """Test successful action without new value."""
        result = {
            "success": True,
            "message": "Click registered",
        }

        formatted = format_action_result(result)

        assert "Click registered" in formatted

    def test_error_result(self):
        """Test formatting error action."""
        result = {
            "success": False,
            "message": "Action failed",
            "error": "Component not found",
        }

        formatted = format_action_result(result)

        assert "Action failed" in formatted
        assert "Component not found" in formatted


class TestFormatStateResult:
    """Tests for format_state_result function."""

    def test_success_result(self):
        """Test formatting successful state."""
        result = {
            "success": True,
            "data": {
                "id": "123",
                "type": "Textbox",
                "intent": "user input",
                "label": "Name",
                "value": "John",
                "visible": True,
                "interactive": True,
                "tags": ["form", "input"],
            },
        }

        formatted = format_state_result(result)

        assert "Textbox" in formatted
        assert "ID: 123" in formatted
        assert "user input" in formatted
        assert "Label: Name" in formatted
        assert "Value: John" in formatted
        assert "Visible: True" in formatted
        assert "Interactive: True" in formatted

    def test_error_result(self):
        """Test formatting error state."""
        result = {
            "success": False,
            "message": "Component not found",
        }

        formatted = format_state_result(result)

        assert "Component not found" in formatted

    def test_minimal_data(self):
        """Test formatting with minimal data."""
        result = {
            "success": True,
            "data": {
                "id": "1",
                "type": "Button",
                "intent": "click",
            },
        }

        formatted = format_state_result(result)

        assert "Button" in formatted


class TestFormatFlowResult:
    """Tests for format_flow_result function."""

    def test_success_result_with_connections(self):
        """Test formatting flow with connections."""
        result = {
            "success": True,
            "direction": "forward",
            "source_id": "input1",
            "connected_components": ["process1", "output1", "output2"],
            "handlers": ["handler1", "handler2"],
        }

        formatted = format_flow_result(result)

        assert "forward" in formatted
        assert "input1" in formatted
        assert "3" in formatted or "Connected components: 3" in formatted
        assert "process1" in formatted
        assert "handler1" in formatted

    def test_no_connections(self):
        """Test formatting flow with no connections."""
        result = {
            "success": True,
            "direction": "backward",
            "source_id": "leaf",
            "connected_components": [],
            "handlers": [],
        }

        formatted = format_flow_result(result)

        assert "No connected components" in formatted

    def test_error_result(self):
        """Test formatting error flow."""
        result = {
            "success": False,
            "message": "Could not trace flow",
        }

        formatted = format_flow_result(result)

        assert "Could not trace flow" in formatted

    def test_many_connections(self):
        """Test formatting flow with many connections (truncation)."""
        result = {
            "success": True,
            "direction": "forward",
            "source_id": "root",
            "connected_components": [f"comp{i}" for i in range(20)],
            "handlers": ["handler"],
        }

        formatted = format_flow_result(result)

        # Should mention truncation
        assert "more" in formatted.lower() or "comp9" in formatted


class TestBuildComponentContext:
    """Tests for build_component_context function."""

    def test_empty_components(self):
        """Test context with no components."""
        context = build_component_context([])

        assert "No semantic components" in context

    def test_single_component(self):
        """Test context with single component."""
        components = [
            {"id": "1", "type": "Button", "intent": "submit"},
        ]

        context = build_component_context(components)

        assert "1 semantic component" in context
        assert "Button" in context
        assert "submit" in context

    def test_grouped_by_type(self):
        """Test components are grouped by type."""
        components = [
            {"id": "1", "type": "Button", "intent": "submit"},
            {"id": "2", "type": "Button", "intent": "cancel"},
            {"id": "3", "type": "Textbox", "intent": "input"},
        ]

        context = build_component_context(components)

        assert "Buttons (2)" in context or "Button" in context
        assert "Textbox" in context

    def test_max_components(self):
        """Test limiting number of components."""
        components = [
            {"id": str(i), "type": "Button", "intent": f"button {i}"}
            for i in range(50)
        ]

        context = build_component_context(components, max_components=10)

        # Should mention more components
        assert "more" in context.lower()

    def test_handles_missing_fields(self):
        """Test handling components with missing fields."""
        components = [
            {"id": "1"},  # Missing type and intent
            {"type": "Button"},  # Missing id and intent
        ]

        # Should not raise, handle gracefully
        context = build_component_context(components)
        assert isinstance(context, str)
