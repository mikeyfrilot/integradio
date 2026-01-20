"""
Agent Module - AI agent integration for Integradio applications.

This module provides tools for AI agents to interact with UI components:
- ComponentTool: Query components by intent, tags, or type
- ActionTool: Perform actions on components (click, set value, etc.)
- StateTool: Read component state and values
- FlowTool: Trace dataflow and dependencies

Designed to work with:
- LangChain/LangGraph agents
- MCP (Model Context Protocol) servers
- Direct API usage

Usage:
    from integradio.agent import (
        ComponentTool,
        ActionTool,
        StateTool,
        create_mcp_server,
        create_langchain_tools,
    )

    # Create tools for LangChain
    tools = create_langchain_tools(blocks)

    # Or create MCP server
    server = create_mcp_server(blocks)
"""

from .tools import (
    # Tool classes
    ComponentTool,
    ActionTool,
    StateTool,
    FlowTool,
    # Tool results
    ToolResult,
    ComponentInfo,
    ActionResult,
    StateResult,
    FlowResult,
    # Convenience functions
    create_all_tools,
    query_by_intent,
    query_by_tag,
    query_by_type,
    get_component_value,
    set_component_value,
    click_component,
    trace_data_flow,
)

from .mcp import (
    # MCP server
    MCPComponentServer,
    create_mcp_server,
    # MCP types
    MCPTool,
    MCPResource,
)

from .langchain import (
    # LangChain integration
    create_langchain_tools,
    SemanticComponentTool,
    SemanticActionTool,
    SemanticStateTool,
)

from .prompts import (
    # System prompts for agents
    COMPONENT_QUERY_PROMPT,
    ACTION_PROMPT,
    STATE_PROMPT,
    FLOW_PROMPT,
    get_system_prompt,
    format_component_list,
    format_action_result,
)

__all__ = [
    # Tools
    "ComponentTool",
    "ActionTool",
    "StateTool",
    "FlowTool",
    # Results
    "ToolResult",
    "ComponentInfo",
    "ActionResult",
    "StateResult",
    "FlowResult",
    # Convenience
    "create_all_tools",
    "query_by_intent",
    "query_by_tag",
    "query_by_type",
    "get_component_value",
    "set_component_value",
    "click_component",
    "trace_data_flow",
    # MCP
    "MCPComponentServer",
    "create_mcp_server",
    "MCPTool",
    "MCPResource",
    # LangChain
    "create_langchain_tools",
    "SemanticComponentTool",
    "SemanticActionTool",
    "SemanticStateTool",
    # Prompts
    "COMPONENT_QUERY_PROMPT",
    "ACTION_PROMPT",
    "STATE_PROMPT",
    "FLOW_PROMPT",
    "get_system_prompt",
    "format_component_list",
    "format_action_result",
]
