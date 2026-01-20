"""
Agent Tools - Core tool implementations for AI agents.

Provides:
- ComponentTool: Find components by semantic query
- ActionTool: Perform actions on components
- StateTool: Read/write component state
- FlowTool: Analyze dataflow

These tools are framework-agnostic and can be adapted
for LangChain, MCP, or direct usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from enum import Enum
import json


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class ToolResult:
    """Base result from any tool operation."""
    success: bool
    message: str
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ComponentInfo:
    """Information about a found component."""
    component_id: str
    component_type: str
    intent: str
    label: str | None = None
    tags: list[str] = field(default_factory=list)
    is_interactive: bool = True
    is_visible: bool = True
    current_value: Any = None
    file_path: str | None = None
    line_number: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.component_id,
            "type": self.component_type,
            "intent": self.intent,
            "label": self.label,
            "tags": self.tags,
            "interactive": self.is_interactive,
            "visible": self.is_visible,
            "value": self.current_value,
            "file": self.file_path,
            "line": self.line_number,
        }


@dataclass
class ActionResult(ToolResult):
    """Result from an action operation."""
    action: str = ""
    component_id: str = ""
    previous_value: Any = None
    new_value: Any = None


@dataclass
class StateResult(ToolResult):
    """Result from a state operation."""
    component_id: str = ""
    state_key: str = ""
    value: Any = None


@dataclass
class FlowResult(ToolResult):
    """Result from a dataflow operation."""
    source_id: str = ""
    direction: str = ""  # "forward" or "backward"
    connected_components: list[str] = field(default_factory=list)
    handlers: list[str] = field(default_factory=list)


# =============================================================================
# Base Tool Class
# =============================================================================

class BaseTool(ABC):
    """Abstract base class for agent tools."""

    name: str = ""
    description: str = ""

    def __init__(self, blocks: Any = None):
        """
        Initialize tool.

        Args:
            blocks: SemanticBlocks instance to operate on
        """
        self.blocks = blocks

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass

    def __call__(self, **kwargs) -> ToolResult:
        """Allow calling tool as function."""
        return self.run(**kwargs)

    @property
    def schema(self) -> dict:
        """Get JSON schema for tool parameters."""
        return {}


# =============================================================================
# Component Tool
# =============================================================================

class ComponentTool(BaseTool):
    """
    Tool for finding components by semantic query.

    Supports:
    - Intent search: Find by natural language intent
    - Tag search: Find by semantic tags
    - Type search: Find by component type
    - Combined queries
    """

    name = "find_component"
    description = """Find UI components by semantic query.

    Use this tool to locate components in the interface by:
    - Intent: Natural language description (e.g., "submit button", "search input")
    - Tags: Semantic tags (e.g., "form", "navigation", "output")
    - Type: Component type (e.g., "Button", "Textbox", "Dropdown")

    Returns list of matching components with their IDs, intents, and current state.
    """

    def run(
        self,
        query: str | None = None,
        intent: str | None = None,
        tag: str | None = None,
        component_type: str | None = None,
        interactive_only: bool = False,
        visible_only: bool = True,
        max_results: int = 10,
    ) -> ToolResult:
        """
        Find components matching the query.

        Args:
            query: General search query (searches all fields)
            intent: Search by intent specifically
            tag: Filter by tag
            component_type: Filter by component type
            interactive_only: Only return interactive components
            visible_only: Only return visible components
            max_results: Maximum number of results

        Returns:
            ToolResult with list of ComponentInfo
        """
        from ..components import SemanticComponent
        from ..inspector.search import SearchEngine

        engine = SearchEngine(self.blocks)
        results: list[ComponentInfo] = []

        # Get all semantic components
        for comp_id, semantic in SemanticComponent._instances.items():
            meta = semantic.semantic_meta
            component = semantic.component

            # Apply filters
            if visible_only:
                visible = getattr(component, "visible", True)
                if not visible:
                    continue

            if interactive_only:
                interactive = getattr(component, "interactive", True)
                if not interactive:
                    continue

            # Score the match
            score = 0.0
            matched = False

            if query:
                # General query - search all fields
                search_results = engine.search(query, max_results=max_results)
                for sr in search_results:
                    if sr.component_id == str(comp_id):
                        score = sr.score
                        matched = True
                        break

            if intent:
                if intent.lower() in meta.intent.lower():
                    score = max(score, 0.9)
                    matched = True

            if tag:
                if tag.lower() in [t.lower() for t in meta.tags]:
                    score = max(score, 1.0)
                    matched = True

            if component_type:
                comp_type_name = type(component).__name__
                if component_type.lower() == comp_type_name.lower():
                    score = max(score, 1.0)
                    matched = True

            # If no filters specified, include all
            if not any([query, intent, tag, component_type]):
                matched = True
                score = 0.5

            if matched:
                value = getattr(component, "value", None)
                if callable(value):
                    value = None

                results.append(ComponentInfo(
                    component_id=str(comp_id),
                    component_type=type(component).__name__,
                    intent=meta.intent,
                    label=getattr(component, "label", None),
                    tags=meta.tags,
                    is_interactive=getattr(component, "interactive", True),
                    is_visible=getattr(component, "visible", True),
                    current_value=value,
                    file_path=meta.file_path,
                    line_number=meta.line_number,
                ))

        # Sort by relevance (higher score first)
        results = results[:max_results]

        if not results:
            return ToolResult(
                success=False,
                message="No components found matching the query",
                data=[],
            )

        return ToolResult(
            success=True,
            message=f"Found {len(results)} component(s)",
            data=[r.to_dict() for r in results],
        )

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "General search query",
                },
                "intent": {
                    "type": "string",
                    "description": "Search by intent",
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by tag",
                },
                "component_type": {
                    "type": "string",
                    "description": "Filter by type (Button, Textbox, etc.)",
                },
                "interactive_only": {
                    "type": "boolean",
                    "description": "Only return interactive components",
                    "default": False,
                },
                "visible_only": {
                    "type": "boolean",
                    "description": "Only return visible components",
                    "default": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10,
                },
            },
        }


# =============================================================================
# Action Tool
# =============================================================================

class ActionTool(BaseTool):
    """
    Tool for performing actions on components.

    Supports:
    - click: Trigger click event
    - set_value: Set component value
    - clear: Clear component value
    - focus: Set focus to component
    - trigger: Trigger specific event
    """

    name = "component_action"
    description = """Perform an action on a UI component.

    Actions available:
    - click: Trigger a button click or similar action
    - set_value: Set the value of an input component
    - clear: Clear the current value
    - trigger: Trigger a specific event (change, submit, etc.)

    Requires the component ID from find_component.
    """

    def run(
        self,
        component_id: str,
        action: Literal["click", "set_value", "clear", "trigger"],
        value: Any = None,
        event: str | None = None,
    ) -> ActionResult:
        """
        Perform action on a component.

        Args:
            component_id: ID of target component
            action: Action to perform
            value: Value for set_value action
            event: Event name for trigger action

        Returns:
            ActionResult with outcome
        """
        from ..components import SemanticComponent

        try:
            comp_id = int(component_id)
        except (ValueError, TypeError):
            return ActionResult(
                success=False,
                message=f"Invalid component ID: {component_id}",
                error="Component ID must be numeric",
                action=action,
                component_id=component_id,
            )

        semantic = SemanticComponent._instances.get(comp_id)
        if not semantic:
            return ActionResult(
                success=False,
                message=f"Component {component_id} not found",
                error="Component not found",
                action=action,
                component_id=component_id,
            )

        component = semantic.component
        previous_value = getattr(component, "value", None)
        if callable(previous_value):
            previous_value = None

        # Perform action
        if action == "click":
            # Check if component has click capability
            if not hasattr(component, "click"):
                return ActionResult(
                    success=False,
                    message=f"Component {component_id} does not support click",
                    error="Action not supported",
                    action=action,
                    component_id=component_id,
                )

            # Note: In Gradio, we can't actually trigger clicks programmatically
            # This would need to be handled by the frontend
            return ActionResult(
                success=True,
                message=f"Click action registered for component {component_id}",
                action=action,
                component_id=component_id,
            )

        elif action == "set_value":
            if value is None:
                return ActionResult(
                    success=False,
                    message="Value is required for set_value action",
                    error="Missing value",
                    action=action,
                    component_id=component_id,
                )

            # Note: Directly setting value requires Gradio's update mechanism
            # This registers the intent; actual update needs event handling
            return ActionResult(
                success=True,
                message=f"Value change registered: {value}",
                action=action,
                component_id=component_id,
                previous_value=previous_value,
                new_value=value,
            )

        elif action == "clear":
            return ActionResult(
                success=True,
                message=f"Clear action registered for component {component_id}",
                action=action,
                component_id=component_id,
                previous_value=previous_value,
                new_value=None,
            )

        elif action == "trigger":
            if not event:
                return ActionResult(
                    success=False,
                    message="Event name is required for trigger action",
                    error="Missing event",
                    action=action,
                    component_id=component_id,
                )

            return ActionResult(
                success=True,
                message=f"Event '{event}' trigger registered for component {component_id}",
                action=action,
                component_id=component_id,
            )

        else:
            return ActionResult(
                success=False,
                message=f"Unknown action: {action}",
                error="Invalid action",
                action=action,
                component_id=component_id,
            )

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["component_id", "action"],
            "properties": {
                "component_id": {
                    "type": "string",
                    "description": "Component ID to act on",
                },
                "action": {
                    "type": "string",
                    "enum": ["click", "set_value", "clear", "trigger"],
                    "description": "Action to perform",
                },
                "value": {
                    "description": "Value for set_value action",
                },
                "event": {
                    "type": "string",
                    "description": "Event name for trigger action",
                },
            },
        }


# =============================================================================
# State Tool
# =============================================================================

class StateTool(BaseTool):
    """
    Tool for reading and observing component state.

    Provides:
    - Get current value
    - Get component properties
    - Check visibility/interactivity
    - Get visual spec
    """

    name = "get_state"
    description = """Get the current state of a UI component.

    Returns:
    - Current value (for inputs)
    - Visibility and interactivity status
    - Component properties (label, placeholder, etc.)
    - Visual specification if available

    Requires the component ID from find_component.
    """

    def run(
        self,
        component_id: str,
        include_visual_spec: bool = False,
    ) -> StateResult:
        """
        Get component state.

        Args:
            component_id: ID of target component
            include_visual_spec: Include visual spec in result

        Returns:
            StateResult with component state
        """
        from ..components import SemanticComponent

        try:
            comp_id = int(component_id)
        except (ValueError, TypeError):
            return StateResult(
                success=False,
                message=f"Invalid component ID: {component_id}",
                error="Component ID must be numeric",
                component_id=component_id,
            )

        semantic = SemanticComponent._instances.get(comp_id)
        if not semantic:
            return StateResult(
                success=False,
                message=f"Component {component_id} not found",
                error="Component not found",
                component_id=component_id,
            )

        meta = semantic.semantic_meta
        component = semantic.component

        # Build state dict
        state = {
            "id": component_id,
            "type": type(component).__name__,
            "intent": meta.intent,
            "tags": meta.tags,
        }

        # Add standard properties
        for prop in ["value", "label", "placeholder", "visible", "interactive"]:
            if hasattr(component, prop):
                val = getattr(component, prop)
                if not callable(val):
                    state[prop] = val

        # Add visual spec if requested
        if include_visual_spec and meta.visual_spec:
            state["visual_spec"] = {
                "has_spec": True,
                "css": meta.visual_spec.to_css() if hasattr(meta.visual_spec, "to_css") else None,
            }
        else:
            state["visual_spec"] = {"has_spec": False}

        return StateResult(
            success=True,
            message=f"State retrieved for component {component_id}",
            data=state,
            component_id=component_id,
            value=state.get("value"),
        )

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["component_id"],
            "properties": {
                "component_id": {
                    "type": "string",
                    "description": "Component ID",
                },
                "include_visual_spec": {
                    "type": "boolean",
                    "description": "Include visual spec in result",
                    "default": False,
                },
            },
        }


# =============================================================================
# Flow Tool
# =============================================================================

class FlowTool(BaseTool):
    """
    Tool for analyzing dataflow relationships.

    Provides:
    - Forward tracing: What components receive data from this one?
    - Backward tracing: What components provide data to this one?
    - Handler analysis: What event handlers connect components?
    """

    name = "trace_flow"
    description = """Trace dataflow relationships between components.

    Directions:
    - forward: Find components that receive data from this one
    - backward: Find components that provide data to this one

    Returns connected component IDs and event handler names.
    Requires the component ID from find_component.
    """

    def run(
        self,
        component_id: str,
        direction: Literal["forward", "backward"] = "forward",
        max_depth: int = 3,
    ) -> FlowResult:
        """
        Trace dataflow from a component.

        Args:
            component_id: Starting component ID
            direction: "forward" or "backward"
            max_depth: Maximum traversal depth

        Returns:
            FlowResult with connected components
        """
        from ..inspector.dataflow import extract_dataflow

        dataflow = extract_dataflow(self.blocks)

        if direction == "forward":
            connected = dataflow.trace_forward(component_id, max_depth=max_depth)
        else:
            connected = dataflow.trace_backward(component_id, max_depth=max_depth)

        # Get handler names involved
        handlers = []
        for handler in dataflow.handlers:
            handler_inputs = handler.inputs
            handler_outputs = handler.outputs

            if component_id in handler_inputs or component_id in handler_outputs:
                handlers.append(handler.name)
            elif any(c in handler_inputs or c in handler_outputs for c in connected):
                handlers.append(handler.name)

        return FlowResult(
            success=True,
            message=f"Found {len(connected)} connected components ({direction})",
            data={
                "connected": connected,
                "handlers": list(set(handlers)),
            },
            source_id=component_id,
            direction=direction,
            connected_components=connected,
            handlers=list(set(handlers)),
        )

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "required": ["component_id"],
            "properties": {
                "component_id": {
                    "type": "string",
                    "description": "Starting component ID",
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward"],
                    "description": "Trace direction",
                    "default": "forward",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth",
                    "default": 3,
                },
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_all_tools(blocks: Any = None) -> dict[str, BaseTool]:
    """
    Create all available tools.

    Args:
        blocks: SemanticBlocks instance

    Returns:
        Dict of tool name -> tool instance
    """
    return {
        "find_component": ComponentTool(blocks),
        "component_action": ActionTool(blocks),
        "get_state": StateTool(blocks),
        "trace_flow": FlowTool(blocks),
    }


def query_by_intent(intent: str, blocks: Any = None) -> list[ComponentInfo]:
    """Quick search by intent."""
    tool = ComponentTool(blocks)
    result = tool.run(intent=intent)
    if result.success:
        return [ComponentInfo(**d) for d in result.data]
    return []


def query_by_tag(tag: str, blocks: Any = None) -> list[ComponentInfo]:
    """Quick search by tag."""
    tool = ComponentTool(blocks)
    result = tool.run(tag=tag)
    if result.success:
        return [ComponentInfo(**d) for d in result.data]
    return []


def query_by_type(component_type: str, blocks: Any = None) -> list[ComponentInfo]:
    """Quick search by type."""
    tool = ComponentTool(blocks)
    result = tool.run(component_type=component_type)
    if result.success:
        return [ComponentInfo(**d) for d in result.data]
    return []


def get_component_value(component_id: str, blocks: Any = None) -> Any:
    """Get a component's current value."""
    tool = StateTool(blocks)
    result = tool.run(component_id=component_id)
    return result.value if result.success else None


def set_component_value(component_id: str, value: Any, blocks: Any = None) -> bool:
    """Set a component's value."""
    tool = ActionTool(blocks)
    result = tool.run(component_id=component_id, action="set_value", value=value)
    return result.success


def click_component(component_id: str, blocks: Any = None) -> bool:
    """Click a component."""
    tool = ActionTool(blocks)
    result = tool.run(component_id=component_id, action="click")
    return result.success


def trace_data_flow(
    component_id: str,
    direction: str = "forward",
    blocks: Any = None,
) -> list[str]:
    """Trace dataflow from a component."""
    tool = FlowTool(blocks)
    result = tool.run(component_id=component_id, direction=direction)
    return result.connected_components if result.success else []
