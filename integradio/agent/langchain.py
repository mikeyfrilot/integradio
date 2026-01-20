"""
LangChain Integration - Tools for LangChain/LangGraph agents.

Provides LangChain-compatible tools:
- SemanticComponentTool: Find components
- SemanticActionTool: Perform actions
- SemanticStateTool: Get/set state

Usage:
    from integradio.agent import create_langchain_tools
    from langchain.agents import AgentExecutor

    tools = create_langchain_tools(blocks)
    agent = AgentExecutor(agent=..., tools=tools)
"""

from __future__ import annotations

from typing import Any, Optional, Type
from dataclasses import dataclass
import json

# Try to import LangChain, but don't fail if not installed
try:
    from langchain_core.tools import BaseTool as LCBaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from pydantic import BaseModel, Field
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    LCBaseTool = object
    CallbackManagerForToolRun = None
    BaseModel = object
    Field = lambda **kwargs: None


# =============================================================================
# Input Schemas (Pydantic models for LangChain)
# =============================================================================

if HAS_LANGCHAIN:

    class FindComponentInput(BaseModel):
        """Input for find_component tool."""
        query: Optional[str] = Field(default=None, description="General search query")
        intent: Optional[str] = Field(default=None, description="Search by intent")
        tag: Optional[str] = Field(default=None, description="Filter by tag")
        component_type: Optional[str] = Field(default=None, description="Filter by type")
        max_results: int = Field(default=10, description="Maximum results")

    class ActionInput(BaseModel):
        """Input for component_action tool."""
        component_id: str = Field(description="Component ID to act on")
        action: str = Field(description="Action: click, set_value, clear, trigger")
        value: Optional[Any] = Field(default=None, description="Value for set_value")
        event: Optional[str] = Field(default=None, description="Event for trigger")

    class StateInput(BaseModel):
        """Input for get_state tool."""
        component_id: str = Field(description="Component ID")
        include_visual_spec: bool = Field(default=False, description="Include visual spec")

    class FlowInput(BaseModel):
        """Input for trace_flow tool."""
        component_id: str = Field(description="Starting component ID")
        direction: str = Field(default="forward", description="forward or backward")
        max_depth: int = Field(default=3, description="Max traversal depth")


# =============================================================================
# LangChain Tools
# =============================================================================

class SemanticComponentTool(LCBaseTool):
    """LangChain tool for finding semantic components."""

    name: str = "find_component"
    description: str = """Find UI components by semantic query.

    Search by:
    - query: General text search across all fields
    - intent: Component's semantic intent (e.g., "submit button")
    - tag: Semantic tags (e.g., "form", "navigation")
    - component_type: Gradio type (e.g., "Button", "Textbox")

    Returns list of matching components with IDs, intents, and values.
    """

    blocks: Any = None

    if HAS_LANGCHAIN:
        args_schema: Type[BaseModel] = FindComponentInput

    def _run(
        self,
        query: str | None = None,
        intent: str | None = None,
        tag: str | None = None,
        component_type: str | None = None,
        max_results: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the tool."""
        from .tools import ComponentTool

        tool = ComponentTool(self.blocks)
        result = tool.run(
            query=query,
            intent=intent,
            tag=tag,
            component_type=component_type,
            max_results=max_results,
        )
        return result.to_json()


class SemanticActionTool(LCBaseTool):
    """LangChain tool for performing component actions."""

    name: str = "component_action"
    description: str = """Perform an action on a UI component.

    Actions:
    - click: Trigger button click
    - set_value: Set input value
    - clear: Clear current value
    - trigger: Trigger specific event

    Requires component_id from find_component tool.
    """

    blocks: Any = None

    if HAS_LANGCHAIN:
        args_schema: Type[BaseModel] = ActionInput

    def _run(
        self,
        component_id: str,
        action: str,
        value: Any = None,
        event: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the tool."""
        from .tools import ActionTool

        tool = ActionTool(self.blocks)
        result = tool.run(
            component_id=component_id,
            action=action,
            value=value,
            event=event,
        )
        return result.to_json()


class SemanticStateTool(LCBaseTool):
    """LangChain tool for reading component state."""

    name: str = "get_state"
    description: str = """Get the current state of a UI component.

    Returns:
    - Current value
    - Visibility and interactivity
    - Label, placeholder, etc.
    - Visual specification (optional)

    Requires component_id from find_component tool.
    """

    blocks: Any = None

    if HAS_LANGCHAIN:
        args_schema: Type[BaseModel] = StateInput

    def _run(
        self,
        component_id: str,
        include_visual_spec: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the tool."""
        from .tools import StateTool

        tool = StateTool(self.blocks)
        result = tool.run(
            component_id=component_id,
            include_visual_spec=include_visual_spec,
        )
        return result.to_json()


class SemanticFlowTool(LCBaseTool):
    """LangChain tool for tracing dataflow."""

    name: str = "trace_flow"
    description: str = """Trace dataflow relationships between components.

    Directions:
    - forward: Find components that receive data from this one
    - backward: Find components that provide data to this one

    Returns connected component IDs and handler names.
    Requires component_id from find_component tool.
    """

    blocks: Any = None

    if HAS_LANGCHAIN:
        args_schema: Type[BaseModel] = FlowInput

    def _run(
        self,
        component_id: str,
        direction: str = "forward",
        max_depth: int = 3,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the tool."""
        from .tools import FlowTool

        tool = FlowTool(self.blocks)
        result = tool.run(
            component_id=component_id,
            direction=direction,
            max_depth=max_depth,
        )
        return result.to_json()


# =============================================================================
# Convenience Function
# =============================================================================

def create_langchain_tools(blocks: Any = None) -> list:
    """
    Create LangChain-compatible tools for Integradio.

    Args:
        blocks: SemanticBlocks instance

    Returns:
        List of LangChain tools

    Raises:
        ImportError: If LangChain is not installed
    """
    if not HAS_LANGCHAIN:
        raise ImportError(
            "LangChain is required for this function. "
            "Install with: pip install langchain-core"
        )

    return [
        SemanticComponentTool(blocks=blocks),
        SemanticActionTool(blocks=blocks),
        SemanticStateTool(blocks=blocks),
        SemanticFlowTool(blocks=blocks),
    ]
