"""
Agent Prompts - System prompts and formatters for AI agents.

Provides:
- System prompts for different agent tasks
- Result formatters for human-readable output
- Context builders for agent instructions
"""

from __future__ import annotations

from typing import Any
import json

# =============================================================================
# System Prompts
# =============================================================================

COMPONENT_QUERY_PROMPT = """You are an AI assistant helping users interact with a Gradio UI.

You have access to tools that let you:
1. **find_component**: Search for UI components by intent, tags, or type
2. **get_state**: Get the current state and value of a component
3. **trace_flow**: Understand how data flows between components

When searching for components:
- Use 'intent' for natural language descriptions (e.g., "submit button", "search input")
- Use 'tag' for categorical filters (e.g., "form", "output", "navigation")
- Use 'component_type' for Gradio types (e.g., "Button", "Textbox", "Dropdown")

Always start by finding the relevant component before getting its state or tracing flow.
Component IDs from find_component are needed for other operations.
"""

ACTION_PROMPT = """You are an AI assistant that can interact with a Gradio UI.

You have access to the component_action tool that can:
- **click**: Trigger a button or similar component
- **set_value**: Set the value of an input component
- **clear**: Clear the current value
- **trigger**: Trigger a specific event

Guidelines:
1. First use find_component to locate the target component
2. Use the component_id from the search result
3. For set_value, provide the new value
4. Actions return confirmation, not the actual result

The UI will update based on your actions. Use get_state to verify changes.
"""

STATE_PROMPT = """You are an AI assistant that monitors UI state.

Use get_state to retrieve:
- Current value of inputs
- Visibility and interactivity status
- Component properties (label, placeholder)
- Visual specifications (if requested)

State information helps you:
- Understand the current UI state
- Verify action results
- Debug issues
- Report to users
"""

FLOW_PROMPT = """You are an AI assistant analyzing UI dataflow.

Use trace_flow to understand:
- **forward**: What components receive data from a given component
- **backward**: What components provide data to a given component

This helps you:
- Understand cause and effect in the UI
- Debug unexpected behavior
- Explain how user inputs affect outputs
- Identify dependencies between components
"""


# =============================================================================
# Combined System Prompt
# =============================================================================

def get_system_prompt(
    include_query: bool = True,
    include_action: bool = True,
    include_state: bool = True,
    include_flow: bool = True,
    custom_instructions: str = "",
) -> str:
    """
    Build a system prompt for an agent.

    Args:
        include_query: Include component query instructions
        include_action: Include action instructions
        include_state: Include state instructions
        include_flow: Include flow instructions
        custom_instructions: Additional instructions

    Returns:
        Combined system prompt
    """
    parts = [
        "You are an AI assistant that interacts with a Gradio UI application.",
        "",
        "You have access to tools for finding and interacting with UI components.",
        "",
    ]

    if include_query:
        parts.extend([
            "## Finding Components",
            "Use `find_component` to search by:",
            "- `query`: General text search",
            "- `intent`: Semantic intent description",
            "- `tag`: Category tags",
            "- `component_type`: Gradio component type",
            "",
        ])

    if include_action:
        parts.extend([
            "## Performing Actions",
            "Use `component_action` to:",
            "- `click`: Trigger buttons",
            "- `set_value`: Set input values",
            "- `clear`: Clear values",
            "- `trigger`: Trigger events",
            "",
        ])

    if include_state:
        parts.extend([
            "## Reading State",
            "Use `get_state` to retrieve:",
            "- Current values",
            "- Visibility status",
            "- Component properties",
            "",
        ])

    if include_flow:
        parts.extend([
            "## Tracing Dataflow",
            "Use `trace_flow` to understand:",
            "- Forward flow: outputs from a component",
            "- Backward flow: inputs to a component",
            "",
        ])

    parts.extend([
        "## General Guidelines",
        "1. Always find components before acting on them",
        "2. Use component IDs from find_component for other tools",
        "3. Verify actions with get_state",
        "4. Report results clearly to the user",
        "",
    ])

    if custom_instructions:
        parts.extend([
            "## Custom Instructions",
            custom_instructions,
            "",
        ])

    return "\n".join(parts)


# =============================================================================
# Result Formatters
# =============================================================================

def format_component_list(
    components: list[dict],
    include_values: bool = True,
    include_tags: bool = True,
) -> str:
    """
    Format a list of components for human reading.

    Args:
        components: List of component dicts
        include_values: Include current values
        include_tags: Include tags

    Returns:
        Formatted string
    """
    if not components:
        return "No components found."

    lines = [f"Found {len(components)} component(s):", ""]

    for comp in components:
        line = f"• [{comp.get('id', '?')}] {comp.get('type', 'Unknown')}: {comp.get('intent', 'No intent')}"
        lines.append(line)

        if include_tags and comp.get("tags"):
            lines.append(f"  Tags: {', '.join(comp['tags'])}")

        if include_values and comp.get("value") is not None:
            value = comp["value"]
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            lines.append(f"  Value: {value}")

        lines.append("")

    return "\n".join(lines)


def format_action_result(result: dict) -> str:
    """
    Format an action result for human reading.

    Args:
        result: Action result dict

    Returns:
        Formatted string
    """
    if result.get("success"):
        msg = f"✓ {result.get('message', 'Action completed')}"
        if result.get("new_value") is not None:
            msg += f"\n  New value: {result['new_value']}"
        return msg
    else:
        return f"✗ {result.get('message', 'Action failed')}: {result.get('error', 'Unknown error')}"


def format_state_result(result: dict) -> str:
    """
    Format a state result for human reading.

    Args:
        result: State result dict

    Returns:
        Formatted string
    """
    if not result.get("success"):
        return f"✗ {result.get('message', 'Failed to get state')}"

    data = result.get("data", {})
    lines = [
        f"Component: {data.get('type', 'Unknown')} (ID: {data.get('id', '?')})",
        f"Intent: {data.get('intent', 'None')}",
    ]

    if data.get("label"):
        lines.append(f"Label: {data['label']}")

    if data.get("value") is not None:
        lines.append(f"Value: {data['value']}")

    lines.append(f"Visible: {data.get('visible', 'Unknown')}")
    lines.append(f"Interactive: {data.get('interactive', 'Unknown')}")

    if data.get("tags"):
        lines.append(f"Tags: {', '.join(data['tags'])}")

    return "\n".join(lines)


def format_flow_result(result: dict) -> str:
    """
    Format a flow result for human reading.

    Args:
        result: Flow result dict

    Returns:
        Formatted string
    """
    if not result.get("success"):
        return f"✗ {result.get('message', 'Failed to trace flow')}"

    direction = result.get("direction", "forward")
    connected = result.get("connected_components", [])
    handlers = result.get("handlers", [])

    lines = [f"Dataflow ({direction}) from component {result.get('source_id', '?')}:"]

    if connected:
        lines.append(f"\nConnected components: {len(connected)}")
        for comp_id in connected[:10]:  # Limit display
            lines.append(f"  → {comp_id}")
        if len(connected) > 10:
            lines.append(f"  ... and {len(connected) - 10} more")
    else:
        lines.append("\nNo connected components found.")

    if handlers:
        lines.append(f"\nEvent handlers: {', '.join(handlers)}")

    return "\n".join(lines)


# =============================================================================
# Context Builder
# =============================================================================

def build_component_context(
    components: list[dict],
    max_components: int = 20,
) -> str:
    """
    Build a context string describing available components.

    Args:
        components: List of component dicts
        max_components: Maximum components to include

    Returns:
        Context string for agent
    """
    if not components:
        return "No semantic components are currently registered."

    lines = [
        f"The UI has {len(components)} semantic component(s).",
        "",
        "Available components:",
    ]

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for comp in components[:max_components]:
        comp_type = comp.get("type", "Unknown")
        if comp_type not in by_type:
            by_type[comp_type] = []
        by_type[comp_type].append(comp)

    for comp_type, comps in sorted(by_type.items()):
        lines.append(f"\n{comp_type}s ({len(comps)}):")
        for comp in comps:
            lines.append(f"  • [{comp.get('id')}] {comp.get('intent', 'No intent')}")

    if len(components) > max_components:
        lines.append(f"\n... and {len(components) - max_components} more components")

    return "\n".join(lines)
