"""
MCP Integration - Model Context Protocol server for Integradio.

Provides an MCP server that exposes semantic component tools:
- find_component: Search components by intent/tags/type
- component_action: Perform actions on components
- get_state: Read component state
- trace_flow: Analyze dataflow

Also exposes resources:
- ui://components: List of all components
- ui://tree: Component hierarchy
- ui://dataflow: Dataflow graph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import json


# =============================================================================
# MCP Types
# =============================================================================

@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPPrompt:
    """MCP prompt template."""
    name: str
    description: str
    arguments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


# =============================================================================
# MCP Component Server
# =============================================================================

class MCPComponentServer:
    """
    MCP server for Integradio components.

    Implements the Model Context Protocol to expose UI components
    as tools and resources that AI agents can use.

    Usage:
        server = MCPComponentServer(blocks)

        # Get available tools
        tools = server.list_tools()

        # Call a tool
        result = server.call_tool("find_component", {"intent": "submit"})

        # Get a resource
        resource = server.read_resource("ui://components")
    """

    def __init__(self, blocks: Any = None, name: str = "integradio"):
        """
        Initialize MCP server.

        Args:
            blocks: SemanticBlocks instance
            name: Server name
        """
        self.blocks = blocks
        self.name = name
        self.version = "1.0.0"

        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}

        self._setup_tools()
        self._setup_resources()
        self._setup_prompts()

    def _setup_tools(self) -> None:
        """Register MCP tools."""
        from .tools import ComponentTool, ActionTool, StateTool, FlowTool

        # Find component tool
        comp_tool = ComponentTool(self.blocks)
        self._tools["find_component"] = MCPTool(
            name="find_component",
            description=comp_tool.description,
            input_schema=comp_tool.schema,
            handler=lambda **kwargs: comp_tool.run(**kwargs).to_dict(),
        )

        # Action tool
        action_tool = ActionTool(self.blocks)
        self._tools["component_action"] = MCPTool(
            name="component_action",
            description=action_tool.description,
            input_schema=action_tool.schema,
            handler=lambda **kwargs: action_tool.run(**kwargs).to_dict(),
        )

        # State tool
        state_tool = StateTool(self.blocks)
        self._tools["get_state"] = MCPTool(
            name="get_state",
            description=state_tool.description,
            input_schema=state_tool.schema,
            handler=lambda **kwargs: state_tool.run(**kwargs).to_dict(),
        )

        # Flow tool
        flow_tool = FlowTool(self.blocks)
        self._tools["trace_flow"] = MCPTool(
            name="trace_flow",
            description=flow_tool.description,
            input_schema=flow_tool.schema,
            handler=lambda **kwargs: flow_tool.run(**kwargs).to_dict(),
        )

    def _setup_resources(self) -> None:
        """Register MCP resources."""
        self._resources["ui://components"] = MCPResource(
            uri="ui://components",
            name="Component List",
            description="List of all semantic UI components",
        )

        self._resources["ui://tree"] = MCPResource(
            uri="ui://tree",
            name="Component Tree",
            description="Hierarchical component structure",
        )

        self._resources["ui://dataflow"] = MCPResource(
            uri="ui://dataflow",
            name="Dataflow Graph",
            description="Component dataflow relationships",
        )

        self._resources["ui://intents"] = MCPResource(
            uri="ui://intents",
            name="Intent List",
            description="All unique component intents",
        )

        self._resources["ui://tags"] = MCPResource(
            uri="ui://tags",
            name="Tag List",
            description="All unique component tags",
        )

    def _setup_prompts(self) -> None:
        """Register MCP prompts."""
        self._prompts["find_interactive"] = MCPPrompt(
            name="find_interactive",
            description="Find all interactive components",
            arguments=[],
        )

        self._prompts["describe_component"] = MCPPrompt(
            name="describe_component",
            description="Get detailed description of a component",
            arguments=[
                {"name": "component_id", "description": "Component ID", "required": True}
            ],
        )

        self._prompts["trace_from_input"] = MCPPrompt(
            name="trace_from_input",
            description="Trace dataflow from an input component",
            arguments=[
                {"name": "input_id", "description": "Input component ID", "required": True}
            ],
        )

    # =========================================================================
    # MCP Protocol Methods
    # =========================================================================

    def get_server_info(self) -> dict:
        """Get server capabilities and info."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"listChanged": True, "subscribe": False},
                "prompts": {"listChanged": True},
            },
        }

    def list_tools(self) -> list[dict]:
        """List available tools."""
        return [tool.to_dict() for tool in self._tools.values()]

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result dict
        """
        if name not in self._tools:
            return {
                "content": [
                    {"type": "text", "text": f"Unknown tool: {name}"}
                ],
                "isError": True,
            }

        tool = self._tools[name]
        try:
            result = tool.handler(**(arguments or {}))
            return {
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2)}
                ],
                "isError": not result.get("success", False),
            }
        except Exception as e:
            return {
                "content": [
                    {"type": "text", "text": f"Error: {str(e)}"}
                ],
                "isError": True,
            }

    def list_resources(self) -> list[dict]:
        """List available resources."""
        return [res.to_dict() for res in self._resources.values()]

    def read_resource(self, uri: str) -> dict:
        """
        Read a resource by URI.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        from ..components import SemanticComponent
        from ..inspector.tree import build_component_tree
        from ..inspector.dataflow import extract_dataflow
        from ..inspector.search import list_all_intents, list_all_tags

        if uri not in self._resources:
            return {
                "contents": [
                    {"uri": uri, "mimeType": "text/plain", "text": f"Unknown resource: {uri}"}
                ],
            }

        if uri == "ui://components":
            components = []
            for comp_id, semantic in SemanticComponent._instances.items():
                meta = semantic.semantic_meta
                component = semantic.component
                components.append({
                    "id": str(comp_id),
                    "type": type(component).__name__,
                    "intent": meta.intent,
                    "tags": meta.tags,
                })
            content = json.dumps(components, indent=2)

        elif uri == "ui://tree":
            tree = build_component_tree(self.blocks)
            content = tree.to_json()

        elif uri == "ui://dataflow":
            dataflow = extract_dataflow(self.blocks)
            content = dataflow.to_json()

        elif uri == "ui://intents":
            intents = list_all_intents()
            content = json.dumps(intents, indent=2)

        elif uri == "ui://tags":
            tags = list_all_tags()
            content = json.dumps(tags, indent=2)

        else:
            content = "{}"

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content,
                }
            ],
        }

    def list_prompts(self) -> list[dict]:
        """List available prompts."""
        return [prompt.to_dict() for prompt in self._prompts.values()]

    def get_prompt(self, name: str, arguments: dict | None = None) -> dict:
        """
        Get a prompt by name.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Prompt messages
        """
        if name not in self._prompts:
            return {
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": f"Unknown prompt: {name}"}}
                ],
            }

        arguments = arguments or {}

        if name == "find_interactive":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "Find all interactive components in the UI and list their intents.",
                        },
                    }
                ],
            }

        elif name == "describe_component":
            comp_id = arguments.get("component_id", "")
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Describe the component with ID {comp_id}. Include its type, intent, tags, and current state.",
                        },
                    }
                ],
            }

        elif name == "trace_from_input":
            input_id = arguments.get("input_id", "")
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Trace the dataflow from component {input_id}. What components receive data from it?",
                        },
                    }
                ],
            }

        return {"messages": []}

    # =========================================================================
    # JSON-RPC Handler
    # =========================================================================

    def handle_request(self, request: dict) -> dict:
        """
        Handle a JSON-RPC request.

        Args:
            request: JSON-RPC request

        Returns:
            JSON-RPC response
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        result = None
        error = None

        try:
            if method == "initialize":
                result = self.get_server_info()

            elif method == "tools/list":
                result = {"tools": self.list_tools()}

            elif method == "tools/call":
                name = params.get("name", "")
                arguments = params.get("arguments", {})
                result = self.call_tool(name, arguments)

            elif method == "resources/list":
                result = {"resources": self.list_resources()}

            elif method == "resources/read":
                uri = params.get("uri", "")
                result = self.read_resource(uri)

            elif method == "prompts/list":
                result = {"prompts": self.list_prompts()}

            elif method == "prompts/get":
                name = params.get("name", "")
                arguments = params.get("arguments", {})
                result = self.get_prompt(name, arguments)

            else:
                error = {"code": -32601, "message": f"Method not found: {method}"}

        except Exception as e:
            error = {"code": -32603, "message": str(e)}

        response = {"jsonrpc": "2.0", "id": request_id}
        if error:
            response["error"] = error
        else:
            response["result"] = result

        return response


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mcp_server(blocks: Any = None, name: str = "integradio") -> MCPComponentServer:
    """
    Create an MCP server for Integradio components.

    Args:
        blocks: SemanticBlocks instance
        name: Server name

    Returns:
        MCPComponentServer instance
    """
    return MCPComponentServer(blocks, name)
