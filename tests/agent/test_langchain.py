"""
Tests for LangChain integration module.

Tests:
- LangChain tool wrappers
- Input schema validation
- Tool execution
- create_langchain_tools function
"""

import pytest
from integradio.agent.langchain import (
    HAS_LANGCHAIN,
    SemanticComponentTool,
    SemanticActionTool,
    SemanticStateTool,
    SemanticFlowTool,
    create_langchain_tools,
)


class TestLangChainImport:
    """Tests for LangChain import handling."""

    def test_has_langchain_flag(self):
        """Test HAS_LANGCHAIN flag is set."""
        # Should be True or False depending on installation
        assert isinstance(HAS_LANGCHAIN, bool)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestSemanticComponentTool:
    """Tests for SemanticComponentTool."""

    def test_tool_name(self):
        """Test tool has correct name."""
        tool = SemanticComponentTool()
        assert tool.name == "find_component"

    def test_tool_description(self):
        """Test tool has description."""
        tool = SemanticComponentTool()
        assert len(tool.description) > 0
        assert "component" in tool.description.lower()

    def test_tool_run(self):
        """Test running the tool."""
        tool = SemanticComponentTool()
        result = tool._run(query="test")

        # Should return JSON string
        assert isinstance(result, str)
        assert "{" in result  # JSON format

    def test_tool_with_intent(self):
        """Test tool with intent parameter."""
        tool = SemanticComponentTool()
        result = tool._run(intent="submit")

        assert isinstance(result, str)

    def test_tool_with_tag(self):
        """Test tool with tag parameter."""
        tool = SemanticComponentTool()
        result = tool._run(tag="form")

        assert isinstance(result, str)

    def test_tool_with_type(self):
        """Test tool with component_type parameter."""
        tool = SemanticComponentTool()
        result = tool._run(component_type="Button")

        assert isinstance(result, str)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestSemanticActionTool:
    """Tests for SemanticActionTool."""

    def test_tool_name(self):
        """Test tool has correct name."""
        tool = SemanticActionTool()
        assert tool.name == "component_action"

    def test_tool_description(self):
        """Test tool has description."""
        tool = SemanticActionTool()
        assert len(tool.description) > 0
        assert "action" in tool.description.lower()

    def test_tool_run_click(self):
        """Test running click action."""
        tool = SemanticActionTool()
        result = tool._run(component_id="123", action="click")

        assert isinstance(result, str)

    def test_tool_run_set_value(self):
        """Test running set_value action."""
        tool = SemanticActionTool()
        result = tool._run(component_id="123", action="set_value", value="test")

        assert isinstance(result, str)

    def test_tool_run_trigger(self):
        """Test running trigger action."""
        tool = SemanticActionTool()
        result = tool._run(component_id="123", action="trigger", event="change")

        assert isinstance(result, str)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestSemanticStateTool:
    """Tests for SemanticStateTool."""

    def test_tool_name(self):
        """Test tool has correct name."""
        tool = SemanticStateTool()
        assert tool.name == "get_state"

    def test_tool_description(self):
        """Test tool has description."""
        tool = SemanticStateTool()
        assert len(tool.description) > 0
        assert "state" in tool.description.lower()

    def test_tool_run(self):
        """Test running the tool."""
        tool = SemanticStateTool()
        result = tool._run(component_id="123")

        assert isinstance(result, str)

    def test_tool_with_visual_spec(self):
        """Test tool with include_visual_spec."""
        tool = SemanticStateTool()
        result = tool._run(component_id="123", include_visual_spec=True)

        assert isinstance(result, str)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestSemanticFlowTool:
    """Tests for SemanticFlowTool."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            fns = []
            dependencies = []
        return MockBlocks()

    def test_tool_name(self):
        """Test tool has correct name."""
        tool = SemanticFlowTool()
        assert tool.name == "trace_flow"

    def test_tool_description(self):
        """Test tool has description."""
        tool = SemanticFlowTool()
        assert len(tool.description) > 0
        assert "flow" in tool.description.lower()

    def test_tool_run_forward(self, mock_blocks):
        """Test tracing forward."""
        tool = SemanticFlowTool(blocks=mock_blocks)
        result = tool._run(component_id="123", direction="forward")

        assert isinstance(result, str)

    def test_tool_run_backward(self, mock_blocks):
        """Test tracing backward."""
        tool = SemanticFlowTool(blocks=mock_blocks)
        result = tool._run(component_id="123", direction="backward")

        assert isinstance(result, str)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestCreateLangchainTools:
    """Tests for create_langchain_tools function."""

    def test_creates_all_tools(self):
        """Test function creates all tools."""
        tools = create_langchain_tools()

        assert len(tools) == 4

        names = [t.name for t in tools]
        assert "find_component" in names
        assert "component_action" in names
        assert "get_state" in names
        assert "trace_flow" in names

    def test_tools_have_blocks(self):
        """Test tools have blocks reference."""
        class MockBlocks:
            fns = []
            dependencies = []

        blocks = MockBlocks()
        tools = create_langchain_tools(blocks)

        for tool in tools:
            assert tool.blocks == blocks

    def test_tools_are_callable(self):
        """Test tools are callable."""
        tools = create_langchain_tools()

        for tool in tools:
            # All tools should have _run method
            assert hasattr(tool, "_run")
            assert callable(tool._run)


class TestLangChainNotInstalled:
    """Tests for when LangChain is not installed."""

    @pytest.mark.skipif(HAS_LANGCHAIN, reason="LangChain is installed")
    def test_create_langchain_tools_raises(self):
        """Test create_langchain_tools raises ImportError."""
        with pytest.raises(ImportError) as exc_info:
            create_langchain_tools()

        assert "LangChain" in str(exc_info.value)


class TestToolsWithBlocks:
    """Tests for tools with actual blocks."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
    def test_component_tool_with_blocks(self, mock_blocks):
        """Test component tool with blocks."""
        tool = SemanticComponentTool(blocks=mock_blocks)
        result = tool._run(query="test")

        assert isinstance(result, str)

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
    def test_action_tool_with_blocks(self, mock_blocks):
        """Test action tool with blocks."""
        tool = SemanticActionTool(blocks=mock_blocks)
        result = tool._run(component_id="1", action="click")

        assert isinstance(result, str)

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
    def test_state_tool_with_blocks(self, mock_blocks):
        """Test state tool with blocks."""
        tool = SemanticStateTool(blocks=mock_blocks)
        result = tool._run(component_id="1")

        assert isinstance(result, str)

    @pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
    def test_flow_tool_with_blocks(self, mock_blocks):
        """Test flow tool with blocks."""
        tool = SemanticFlowTool(blocks=mock_blocks)
        result = tool._run(component_id="1")

        assert isinstance(result, str)
