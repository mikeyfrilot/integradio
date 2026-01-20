"""
Tests for MCP module.

Tests:
- MCPTool definition
- MCPResource definition
- MCPPrompt definition
- MCPComponentServer methods
- JSON-RPC request handling
"""

import pytest
import json
from integradio.agent.mcp import (
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPComponentServer,
    create_mcp_server,
)


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_create_tool(self):
        """Test creating MCP tool."""
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            handler=lambda: {"result": "ok"},
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object"}

    def test_tool_to_dict(self):
        """Test tool serialization."""
        tool = MCPTool(
            name="my_tool",
            description="My tool description",
            input_schema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            },
            handler=lambda: None,
        )

        data = tool.to_dict()

        assert data["name"] == "my_tool"
        assert data["description"] == "My tool description"
        assert "inputSchema" in data
        assert data["inputSchema"]["type"] == "object"


class TestMCPResource:
    """Tests for MCPResource dataclass."""

    def test_create_resource(self):
        """Test creating MCP resource."""
        resource = MCPResource(
            uri="ui://test",
            name="Test Resource",
            description="A test resource",
        )

        assert resource.uri == "ui://test"
        assert resource.name == "Test Resource"
        assert resource.mime_type == "application/json"

    def test_resource_custom_mime_type(self):
        """Test resource with custom MIME type."""
        resource = MCPResource(
            uri="ui://html",
            name="HTML Resource",
            description="HTML content",
            mime_type="text/html",
        )

        assert resource.mime_type == "text/html"

    def test_resource_to_dict(self):
        """Test resource serialization."""
        resource = MCPResource(
            uri="ui://components",
            name="Components",
            description="List of components",
        )

        data = resource.to_dict()

        assert data["uri"] == "ui://components"
        assert data["name"] == "Components"
        assert data["mimeType"] == "application/json"


class TestMCPPrompt:
    """Tests for MCPPrompt dataclass."""

    def test_create_prompt(self):
        """Test creating MCP prompt."""
        prompt = MCPPrompt(
            name="test_prompt",
            description="A test prompt",
        )

        assert prompt.name == "test_prompt"
        assert prompt.arguments == []

    def test_prompt_with_arguments(self):
        """Test prompt with arguments."""
        prompt = MCPPrompt(
            name="describe",
            description="Describe a component",
            arguments=[
                {"name": "id", "description": "Component ID", "required": True}
            ],
        )

        assert len(prompt.arguments) == 1
        assert prompt.arguments[0]["name"] == "id"

    def test_prompt_to_dict(self):
        """Test prompt serialization."""
        prompt = MCPPrompt(
            name="my_prompt",
            description="My prompt",
            arguments=[
                {"name": "arg1", "description": "First argument"}
            ],
        )

        data = prompt.to_dict()

        assert data["name"] == "my_prompt"
        assert data["description"] == "My prompt"
        assert len(data["arguments"]) == 1


class TestMCPComponentServer:
    """Tests for MCPComponentServer."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    @pytest.fixture
    def server(self, mock_blocks):
        """Create server with mock blocks."""
        return MCPComponentServer(mock_blocks, name="test-server")

    def test_server_creation(self, server):
        """Test server initialization."""
        assert server.name == "test-server"
        assert server.version == "1.0.0"

    def test_get_server_info(self, server):
        """Test server info response."""
        info = server.get_server_info()

        assert "protocolVersion" in info
        assert info["serverInfo"]["name"] == "test-server"
        assert "capabilities" in info
        assert "tools" in info["capabilities"]
        assert "resources" in info["capabilities"]
        assert "prompts" in info["capabilities"]

    def test_list_tools(self, server):
        """Test listing tools."""
        tools = server.list_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 4  # find_component, component_action, get_state, trace_flow

        tool_names = [t["name"] for t in tools]
        assert "find_component" in tool_names
        assert "component_action" in tool_names
        assert "get_state" in tool_names
        assert "trace_flow" in tool_names

    def test_call_tool_success(self, server):
        """Test calling a tool successfully."""
        result = server.call_tool("find_component", {"query": "test"})

        assert "content" in result
        assert len(result["content"]) > 0

    def test_call_tool_unknown(self, server):
        """Test calling unknown tool."""
        result = server.call_tool("unknown_tool")

        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]

    def test_list_resources(self, server):
        """Test listing resources."""
        resources = server.list_resources()

        assert isinstance(resources, list)
        assert len(resources) >= 5

        resource_uris = [r["uri"] for r in resources]
        assert "ui://components" in resource_uris
        assert "ui://tree" in resource_uris
        assert "ui://dataflow" in resource_uris
        assert "ui://intents" in resource_uris
        assert "ui://tags" in resource_uris

    def test_read_resource_components(self, server):
        """Test reading components resource."""
        result = server.read_resource("ui://components")

        assert "contents" in result
        assert len(result["contents"]) > 0
        assert result["contents"][0]["uri"] == "ui://components"
        assert result["contents"][0]["mimeType"] == "application/json"

    def test_read_resource_tree(self, server):
        """Test reading tree resource."""
        result = server.read_resource("ui://tree")

        assert "contents" in result
        content = result["contents"][0]["text"]
        # Should be valid JSON
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_read_resource_dataflow(self, server):
        """Test reading dataflow resource."""
        result = server.read_resource("ui://dataflow")

        assert "contents" in result
        content = result["contents"][0]["text"]
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_read_resource_unknown(self, server):
        """Test reading unknown resource."""
        result = server.read_resource("ui://unknown")

        assert "contents" in result
        assert "Unknown resource" in result["contents"][0]["text"]

    def test_list_prompts(self, server):
        """Test listing prompts."""
        prompts = server.list_prompts()

        assert isinstance(prompts, list)
        assert len(prompts) >= 3

        prompt_names = [p["name"] for p in prompts]
        assert "find_interactive" in prompt_names
        assert "describe_component" in prompt_names
        assert "trace_from_input" in prompt_names

    def test_get_prompt_find_interactive(self, server):
        """Test getting find_interactive prompt."""
        result = server.get_prompt("find_interactive")

        assert "messages" in result
        assert len(result["messages"]) > 0
        assert result["messages"][0]["role"] == "user"

    def test_get_prompt_describe_component(self, server):
        """Test getting describe_component prompt with args."""
        result = server.get_prompt("describe_component", {"component_id": "123"})

        assert "messages" in result
        assert "123" in result["messages"][0]["content"]["text"]

    def test_get_prompt_trace_from_input(self, server):
        """Test getting trace_from_input prompt with args."""
        result = server.get_prompt("trace_from_input", {"input_id": "456"})

        assert "messages" in result
        assert "456" in result["messages"][0]["content"]["text"]

    def test_get_prompt_unknown(self, server):
        """Test getting unknown prompt."""
        result = server.get_prompt("unknown_prompt")

        assert "messages" in result
        assert "Unknown prompt" in result["messages"][0]["content"]["text"]


class TestMCPJsonRPC:
    """Tests for JSON-RPC handling."""

    @pytest.fixture
    def server(self):
        """Create server."""
        class MockBlocks:
            fns = []
            dependencies = []
        return MCPComponentServer(MockBlocks())

    def test_handle_initialize(self, server):
        """Test initialize request."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        }

        response = server.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]

    def test_handle_tools_list(self, server):
        """Test tools/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        }

        response = server.handle_request(request)

        assert "result" in response
        assert "tools" in response["result"]

    def test_handle_tools_call(self, server):
        """Test tools/call request."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "find_component",
                "arguments": {"query": "test"},
            },
        }

        response = server.handle_request(request)

        assert "result" in response
        assert "content" in response["result"]

    def test_handle_resources_list(self, server):
        """Test resources/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
        }

        response = server.handle_request(request)

        assert "result" in response
        assert "resources" in response["result"]

    def test_handle_resources_read(self, server):
        """Test resources/read request."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "ui://components"},
        }

        response = server.handle_request(request)

        assert "result" in response
        assert "contents" in response["result"]

    def test_handle_prompts_list(self, server):
        """Test prompts/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "prompts/list",
        }

        response = server.handle_request(request)

        assert "result" in response
        assert "prompts" in response["result"]

    def test_handle_prompts_get(self, server):
        """Test prompts/get request."""
        request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "prompts/get",
            "params": {"name": "find_interactive"},
        }

        response = server.handle_request(request)

        assert "result" in response
        assert "messages" in response["result"]

    def test_handle_unknown_method(self, server):
        """Test unknown method request."""
        request = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "unknown/method",
        }

        response = server.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_handle_request_preserves_id(self, server):
        """Test that request ID is preserved in response."""
        request = {
            "jsonrpc": "2.0",
            "id": "custom-id-123",
            "method": "tools/list",
        }

        response = server.handle_request(request)

        assert response["id"] == "custom-id-123"


class TestCreateMCPServer:
    """Tests for create_mcp_server convenience function."""

    def test_create_server(self):
        """Test creating server via convenience function."""
        class MockBlocks:
            fns = []
            dependencies = []

        server = create_mcp_server(MockBlocks(), name="my-server")

        assert isinstance(server, MCPComponentServer)
        assert server.name == "my-server"

    def test_create_server_default_name(self):
        """Test creating server with default name."""
        server = create_mcp_server()

        assert server.name == "integradio"

    def test_create_server_no_blocks(self):
        """Test creating server without blocks."""
        server = create_mcp_server()

        assert server.blocks is None
        # Should still work for listing tools
        tools = server.list_tools()
        assert len(tools) >= 4
