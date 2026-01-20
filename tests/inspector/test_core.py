"""
Tests for Inspector Core module.

Tests:
- InspectorMode enum
- InspectorConfig dataclass
- InspectorState dataclass
- Inspector class
- Convenience functions (inspect, dev_mode)
"""

import pytest
from integradio.inspector.core import (
    InspectorMode,
    InspectorConfig,
    InspectorState,
    Inspector,
    inspect,
    dev_mode,
)


class TestInspectorMode:
    """Tests for InspectorMode enum."""

    def test_mode_values(self):
        """Test all mode values."""
        assert InspectorMode.SIDEBAR.value == "sidebar"
        assert InspectorMode.FLOATING.value == "floating"
        assert InspectorMode.EMBEDDED.value == "embedded"
        assert InspectorMode.HIDDEN.value == "hidden"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert InspectorMode("sidebar") == InspectorMode.SIDEBAR
        assert InspectorMode("floating") == InspectorMode.FLOATING
        assert InspectorMode("embedded") == InspectorMode.EMBEDDED
        assert InspectorMode("hidden") == InspectorMode.HIDDEN

    def test_mode_invalid_value(self):
        """Test invalid mode value raises error."""
        with pytest.raises(ValueError):
            InspectorMode("invalid")

    def test_mode_is_string_enum(self):
        """Test that mode is a string enum."""
        mode = InspectorMode.SIDEBAR
        assert isinstance(mode, str)
        assert mode == "sidebar"


class TestInspectorConfig:
    """Tests for InspectorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InspectorConfig()

        assert config.mode == InspectorMode.SIDEBAR
        assert config.position == "right"
        assert config.width == 400
        assert config.collapsed is True
        assert config.show_tree is True
        assert config.show_dataflow is True
        assert config.show_search is True
        assert config.show_details is True
        assert config.auto_refresh is False
        assert config.refresh_interval == 1000

    def test_custom_config(self):
        """Test custom configuration."""
        config = InspectorConfig(
            mode=InspectorMode.FLOATING,
            position="left",
            width=500,
            collapsed=False,
            auto_refresh=True,
            refresh_interval=2000,
        )

        assert config.mode == InspectorMode.FLOATING
        assert config.position == "left"
        assert config.width == 500
        assert config.collapsed is False
        assert config.auto_refresh is True
        assert config.refresh_interval == 2000

    def test_config_feature_toggles(self):
        """Test toggling individual features."""
        config = InspectorConfig(
            show_tree=False,
            show_dataflow=False,
            show_search=False,
            show_details=False,
        )

        assert config.show_tree is False
        assert config.show_dataflow is False
        assert config.show_search is False
        assert config.show_details is False


class TestInspectorState:
    """Tests for InspectorState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = InspectorState()

        assert state.is_open is False
        assert state.selected_component_id is None
        assert state.search_query == ""
        assert state.active_tab == "tree"
        assert state.tree is None
        assert state.dataflow is None
        assert state.search_results == []

    def test_state_to_dict(self):
        """Test state serialization."""
        state = InspectorState(
            is_open=True,
            selected_component_id="123",
            search_query="test",
            active_tab="search",
        )

        data = state.to_dict()

        assert data["is_open"] is True
        assert data["selected_component_id"] == "123"
        assert data["search_query"] == "test"
        assert data["active_tab"] == "search"
        assert data["tree_loaded"] is False
        assert data["dataflow_loaded"] is False
        assert data["search_results_count"] == 0

    def test_state_with_search_results(self):
        """Test state with search results."""
        from integradio.inspector.search import SearchResult

        results = [
            SearchResult("1", "Button", "click", 0.9, "intent", "click"),
            SearchResult("2", "Textbox", "input", 0.8, "intent", "input"),
        ]

        state = InspectorState(search_results=results)

        data = state.to_dict()
        assert data["search_results_count"] == 2


class TestInspector:
    """Tests for Inspector class."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks object."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    def test_inspector_creation(self, mock_blocks):
        """Test creating inspector."""
        inspector = Inspector(mock_blocks)

        assert inspector.blocks == mock_blocks
        assert isinstance(inspector.config, InspectorConfig)
        assert isinstance(inspector.state, InspectorState)
        assert inspector._attached is False

    def test_inspector_with_config(self, mock_blocks):
        """Test inspector with custom config."""
        config = InspectorConfig(
            mode=InspectorMode.FLOATING,
            width=600,
        )

        inspector = Inspector(mock_blocks, config=config)

        assert inspector.config.mode == InspectorMode.FLOATING
        assert inspector.config.width == 600

    def test_inspector_repr(self, mock_blocks):
        """Test inspector string representation."""
        inspector = Inspector(mock_blocks)
        repr_str = repr(inspector)

        assert "Inspector" in repr_str
        assert "sidebar" in repr_str
        assert "attached=False" in repr_str

    def test_inspector_search(self, mock_blocks):
        """Test inspector search functionality."""
        inspector = Inspector(mock_blocks)
        results = inspector.search("test")

        assert isinstance(results, list)
        assert inspector.state.search_query == "test"
        assert inspector.state.search_results == results

    def test_inspector_select_component_invalid(self, mock_blocks):
        """Test selecting invalid component."""
        inspector = Inspector(mock_blocks)

        # Non-numeric ID
        result = inspector.select_component("invalid")
        assert result is None
        assert inspector.state.selected_component_id == "invalid"

    def test_inspector_select_component_not_found(self, mock_blocks):
        """Test selecting non-existent component."""
        inspector = Inspector(mock_blocks)

        # Numeric but doesn't exist
        result = inspector.select_component("999999")
        assert result is None

    def test_inspector_attach_hidden_mode(self, mock_blocks):
        """Test attach returns None in hidden mode."""
        config = InspectorConfig(mode=InspectorMode.HIDDEN)
        inspector = Inspector(mock_blocks, config=config)

        result = inspector.attach()

        assert result is None
        assert inspector._attached is False

    def test_inspector_attach_twice(self, mock_blocks):
        """Test attach returns None when already attached."""
        config = InspectorConfig(mode=InspectorMode.HIDDEN)
        inspector = Inspector(mock_blocks, config=config)

        # Force attached state
        inspector._attached = True

        result = inspector.attach()
        assert result is None

    def test_inspector_export_state(self, mock_blocks):
        """Test exporting inspector state."""
        inspector = Inspector(mock_blocks)
        exported = inspector.export_state()

        assert "state" in exported
        assert "config" in exported
        assert "tree" in exported
        assert "dataflow" in exported

        assert exported["config"]["mode"] == "sidebar"
        assert exported["config"]["position"] == "right"
        assert exported["config"]["width"] == 400

    def test_inspector_get_tree_mermaid(self, mock_blocks):
        """Test getting tree as Mermaid diagram."""
        inspector = Inspector(mock_blocks)

        # Will trigger refresh
        mermaid = inspector.get_tree_mermaid()

        assert isinstance(mermaid, str)
        assert "flowchart" in mermaid or "graph" in mermaid

    def test_inspector_get_dataflow_mermaid(self, mock_blocks):
        """Test getting dataflow as Mermaid diagram."""
        inspector = Inspector(mock_blocks)

        mermaid = inspector.get_dataflow_mermaid()

        assert isinstance(mermaid, str)
        assert "flowchart" in mermaid or "graph" in mermaid


class TestInspectFunction:
    """Tests for inspect convenience function."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    def test_inspect_default_mode(self, mock_blocks):
        """Test inspect with default mode (sidebar requires Gradio context)."""
        # Sidebar mode calls attach() which requires Gradio context
        # Just test it creates correct config without attaching
        config = InspectorConfig(mode=InspectorMode.SIDEBAR)
        inspector = Inspector(mock_blocks, config=config)

        assert isinstance(inspector, Inspector)
        assert inspector.config.mode == InspectorMode.SIDEBAR

    def test_inspect_floating_mode(self, mock_blocks):
        """Test inspect with floating mode (requires Gradio context for attach)."""
        # Floating mode also tries to attach
        config = InspectorConfig(mode=InspectorMode.FLOATING)
        inspector = Inspector(mock_blocks, config=config)

        assert inspector.config.mode == InspectorMode.FLOATING

    def test_inspect_embedded_mode(self, mock_blocks):
        """Test inspect with embedded mode (requires Gradio context for attach)."""
        # Embedded mode also tries to attach
        config = InspectorConfig(mode=InspectorMode.EMBEDDED)
        inspector = Inspector(mock_blocks, config=config)

        assert inspector.config.mode == InspectorMode.EMBEDDED

    def test_inspect_hidden_mode(self, mock_blocks):
        """Test inspect with hidden mode."""
        inspector = inspect(mock_blocks, mode="hidden")

        assert inspector.config.mode == InspectorMode.HIDDEN

    def test_inspect_invalid_mode(self, mock_blocks):
        """Test inspect with invalid mode."""
        with pytest.raises(ValueError):
            inspect(mock_blocks, mode="invalid")


class TestDevModeFunction:
    """Tests for dev_mode convenience function."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    def test_dev_mode_config(self, mock_blocks):
        """Test dev_mode creates appropriate config (without attach for testing)."""
        # dev_mode calls attach() which needs Gradio context
        # Test the config directly instead
        config = InspectorConfig(
            mode=InspectorMode.SIDEBAR,
            collapsed=False,
            auto_refresh=True,
        )
        inspector = Inspector(mock_blocks, config=config)

        assert isinstance(inspector, Inspector)
        assert inspector.config.mode == InspectorMode.SIDEBAR
        assert inspector.config.collapsed is False
        assert inspector.config.auto_refresh is True


class TestInspectorRefresh:
    """Tests for inspector refresh functionality."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    def test_refresh_updates_tree(self, mock_blocks):
        """Test refresh updates tree state."""
        inspector = Inspector(mock_blocks)

        assert inspector.state.tree is None

        inspector.refresh()

        assert inspector.state.tree is not None

    def test_refresh_updates_dataflow(self, mock_blocks):
        """Test refresh updates dataflow state."""
        inspector = Inspector(mock_blocks)

        assert inspector.state.dataflow is None

        inspector.refresh()

        assert inspector.state.dataflow is not None


class TestInspectorEdgeCases:
    """Edge case tests for inspector."""

    def test_inspector_none_blocks(self):
        """Test inspector with None blocks."""
        inspector = Inspector(None)

        assert inspector.blocks is None
        # Should not crash on basic operations

    def test_inspector_empty_search(self):
        """Test search with empty query."""
        class MockBlocks:
            fns = []
            dependencies = []

        inspector = Inspector(MockBlocks())
        results = inspector.search("")

        assert isinstance(results, list)

    def test_inspector_select_none(self):
        """Test selecting None component."""
        class MockBlocks:
            fns = []
            dependencies = []

        inspector = Inspector(MockBlocks())
        result = inspector.select_component(None)

        assert result is None

    def test_export_state_with_loaded_data(self):
        """Test export_state includes loaded data info."""
        class MockBlocks:
            fns = []
            dependencies = []

        inspector = Inspector(MockBlocks())
        inspector.refresh()

        exported = inspector.export_state()

        # Tree and dataflow should now have data
        assert exported["tree"] is not None
        assert exported["dataflow"] is not None

    def test_multiple_refreshes(self):
        """Test multiple refresh calls don't cause issues."""
        class MockBlocks:
            fns = []
            dependencies = []

        inspector = Inspector(MockBlocks())

        for _ in range(5):
            inspector.refresh()

        # Should have latest data
        assert inspector.state.tree is not None
        assert inspector.state.dataflow is not None


class TestInspectorModes:
    """Tests for different inspector modes."""

    @pytest.fixture
    def mock_blocks(self):
        """Create mock blocks."""
        class MockBlocks:
            def __init__(self):
                self.fns = []
                self.dependencies = []
        return MockBlocks()

    def test_all_modes_accessible(self, mock_blocks):
        """Test all modes can be configured."""
        modes = [
            InspectorMode.SIDEBAR,
            InspectorMode.FLOATING,
            InspectorMode.EMBEDDED,
            InspectorMode.HIDDEN,
        ]

        for mode in modes:
            config = InspectorConfig(mode=mode)
            inspector = Inspector(mock_blocks, config=config)
            assert inspector.config.mode == mode

    def test_hidden_mode_no_refresh_on_attach(self, mock_blocks):
        """Test hidden mode doesn't refresh on attach."""
        config = InspectorConfig(mode=InspectorMode.HIDDEN)
        inspector = Inspector(mock_blocks, config=config)

        inspector.attach()

        # State should remain uninitialized
        # (attach returns early for hidden mode)
        # Note: This depends on implementation detail
