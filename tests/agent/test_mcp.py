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


class TestMCPWithSemanticComponents:
    """Integration tests with mock SemanticComponent instances."""

    def test_call_tool_find_component(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test calling find_component tool."""
        server = MCPComponentServer(mock_blocks)

        result = server.call_tool("find_component", {"query": "search"})

        assert "content" in result
        assert len(result["content"]) > 0
        assert "text" in result["content"][0]

    def test_call_tool_find_with_filters(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test find_component with filters."""
        server = MCPComponentServer(mock_blocks)

        result = server.call_tool("find_component", {
            "intent": "search",
            "tag": "input",
            "interactive_only": True,
            "visible_only": True,
            "max_results": 5,
        })

        assert "content" in result

    def test_call_tool_component_action_click(
        self,
        mock_blocks,
        component_with_click,
        patch_semantic_component,
    ):
        """Test calling component_action with click."""
        component, semantic = component_with_click
        server = MCPComponentServer(mock_blocks)

        result = server.call_tool("component_action", {
            "component_id": str(component._id),
            "action": "click",
        })

        assert "content" in result

    def test_call_tool_component_action_set_value(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test calling component_action with set_value."""
        component, semantic = component_with_value
        server = MCPComponentServer(mock_blocks)

        result = server.call_tool("component_action", {
            "component_id": str(component._id),
            "action": "set_value",
            "value": "new value",
        })

        assert "content" in result

    def test_call_tool_get_state(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test calling get_state tool."""
        component, semantic = component_with_value
        server = MCPComponentServer(mock_blocks)

        result = server.call_tool("get_state", {
            "component_id": str(component._id),
        })

        assert "content" in result

    def test_call_tool_get_state_with_visual_spec(
        self,
        mock_blocks,
        component_with_visual_spec,
        patch_semantic_component,
    ):
        """Test calling get_state with visual spec."""
        component, semantic = component_with_visual_spec
        server = MCPComponentServer(mock_blocks)

        result = server.call_tool("get_state", {
            "component_id": str(component._id),
            "include_visual_spec": True,
        })

        assert "content" in result

    def test_call_tool_trace_flow_forward(
        self,
        populated_blocks,
        mock_semantic_components,
    ):
        """Test calling trace_flow forward."""
        server = MCPComponentServer(populated_blocks)

        result = server.call_tool("trace_flow", {
            "component_id": "1",
            "direction": "forward",
        })

        assert "content" in result

    def test_call_tool_trace_flow_backward(
        self,
        populated_blocks,
        mock_semantic_components,
    ):
        """Test calling trace_flow backward."""
        server = MCPComponentServer(populated_blocks)

        result = server.call_tool("trace_flow", {
            "component_id": "3",
            "direction": "backward",
        })

        assert "content" in result


class TestMCPResources:
    """Tests for MCP resource operations."""

    def test_read_resource_intents(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test reading intents resource."""
        server = MCPComponentServer(mock_blocks)

        result = server.read_resource("ui://intents")

        assert "contents" in result
        assert len(result["contents"]) > 0

    def test_read_resource_tags(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test reading tags resource."""
        server = MCPComponentServer(mock_blocks)

        result = server.read_resource("ui://tags")

        assert "contents" in result
        assert len(result["contents"]) > 0

    def test_read_resource_tree(
        self,
        mock_blocks,
        mock_semantic_components,
    ):
        """Test reading tree resource."""
        import json

        server = MCPComponentServer(mock_blocks)

        result = server.read_resource("ui://tree")

        assert "contents" in result
        content = result["contents"][0]["text"]
        # Should be valid JSON
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_read_resource_dataflow(
        self,
        mock_blocks,
        mock_semantic_components,
    ):
        """Test reading dataflow resource."""
        import json

        server = MCPComponentServer(mock_blocks)

        result = server.read_resource("ui://dataflow")

        assert "contents" in result
        content = result["contents"][0]["text"]
        parsed = json.loads(content)
        assert isinstance(parsed, dict)


class TestMCPPromptsWithArgs:
    """Tests for MCP prompt argument handling."""

    def test_prompt_describe_with_component_id(
        self,
        mock_blocks,
        component_with_value,
        patch_semantic_component,
    ):
        """Test describe_component prompt with ID."""
        component, semantic = component_with_value
        server = MCPComponentServer(mock_blocks)

        result = server.get_prompt("describe_component", {
            "component_id": str(component._id),
        })

        assert "messages" in result
        assert str(component._id) in result["messages"][0]["content"]["text"]

    def test_prompt_trace_with_input_id(
        self,
        mock_blocks,
        mock_semantic_components,
    ):
        """Test trace_from_input prompt with ID."""
        server = MCPComponentServer(mock_blocks)

        result = server.get_prompt("trace_from_input", {
            "input_id": "123",
        })

        assert "messages" in result
        assert "123" in result["messages"][0]["content"]["text"]


class TestMCPJsonRPCComplete:
    """Complete JSON-RPC integration tests."""

    def test_full_workflow(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test complete MCP workflow."""
        server = MCPComponentServer(mock_blocks)

        # 1. Initialize
        init_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        })
        assert "result" in init_response

        # 2. List tools
        tools_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        assert "result" in tools_response
        assert "tools" in tools_response["result"]

        # 3. Call a tool
        call_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "find_component",
                "arguments": {"query": "button"},
            },
        })
        assert "result" in call_response

        # 4. List resources
        resources_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
        })
        assert "result" in resources_response
        assert "resources" in resources_response["result"]

        # 5. Read resource
        read_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "ui://tree"},
        })
        assert "result" in read_response

        # 6. Get prompt
        prompt_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "prompts/get",
            "params": {"name": "find_interactive"},
        })
        assert "result" in prompt_response

    def test_error_responses(self, mock_blocks):
        """Test proper error responses."""
        server = MCPComponentServer(mock_blocks)

        # Unknown method
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
        })
        assert "error" in response
        assert response["error"]["code"] == -32601

        # Unknown tool
        response = server.call_tool("unknown_tool")
        assert response["isError"] is True

        # Unknown resource
        response = server.read_resource("ui://unknown")
        assert "Unknown resource" in response["contents"][0]["text"]

        # Unknown prompt
        response = server.get_prompt("unknown_prompt")
        assert "Unknown prompt" in response["messages"][0]["content"]["text"]


class TestMCPProtocolCompliance:
    """Tests for MCP protocol compliance."""

    def test_response_jsonrpc_version(self, mock_blocks):
        """Test responses include jsonrpc version."""
        server = MCPComponentServer(mock_blocks)

        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        })

        assert response["jsonrpc"] == "2.0"

    def test_response_id_preserved(self, mock_blocks):
        """Test request ID is preserved in response."""
        server = MCPComponentServer(mock_blocks)

        # Numeric ID
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/list",
        })
        assert response["id"] == 42

        # String ID
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": "my-custom-id",
            "method": "tools/list",
        })
        assert response["id"] == "my-custom-id"

    def test_capabilities_structure(self, mock_blocks):
        """Test server capabilities structure."""
        server = MCPComponentServer(mock_blocks)
        info = server.get_server_info()

        assert "capabilities" in info
        caps = info["capabilities"]

        # Required capabilities
        assert "tools" in caps
        assert "resources" in caps
        assert "prompts" in caps

    def test_tool_definition_format(self, mock_blocks):
        """Test tool definitions follow MCP format."""
        server = MCPComponentServer(mock_blocks)
        tools = server.list_tools()

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_resource_definition_format(self, mock_blocks):
        """Test resource definitions follow MCP format."""
        server = MCPComponentServer(mock_blocks)
        resources = server.list_resources()

        for resource in resources:
            assert "uri" in resource
            assert "name" in resource
            assert "mimeType" in resource

    def test_prompt_definition_format(self, mock_blocks):
        """Test prompt definitions follow MCP format."""
        server = MCPComponentServer(mock_blocks)
        prompts = server.list_prompts()

        for prompt in prompts:
            assert "name" in prompt
            assert "description" in prompt
            assert "arguments" in prompt
