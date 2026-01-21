"""
Pytest fixtures for inspector and agent tests.

Provides mock SemanticComponent instances and Gradio-like blocks
for comprehensive integration testing.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MockVisualSpec:
    """Mock VisualSpec for testing."""
    component_id: str = ""
    component_type: str = ""
    tokens: dict = field(default_factory=dict)

    def to_css(self, selector: str | None = None) -> str:
        return f"#{self.component_id} {{ color: red; }}"


@dataclass
class MockSemanticMetadata:
    """Mock SemanticMetadata for testing."""
    intent: str = ""
    tags: list = field(default_factory=list)
    file_path: str | None = None
    line_number: int | None = None
    embedded: bool = False
    extra: dict = field(default_factory=dict)
    visual_spec: Optional[MockVisualSpec] = None


class MockGradioComponent:
    """Mock Gradio component for testing."""

    def __init__(
        self,
        _id: int,
        component_type: str = "Button",
        label: str | None = None,
        value: Any = None,
        visible: bool = True,
        interactive: bool = True,
        elem_id: str | None = None,
        placeholder: str | None = None,
        children: list | None = None,
    ):
        self._id = _id
        self._type = component_type
        self.label = label
        self.value = value
        self.visible = visible
        self.interactive = interactive
        self.elem_id = elem_id
        self.placeholder = placeholder
        self.children = children or []

    def click(self):
        """Mock click method for buttons."""
        pass


class MockSemanticComponent:
    """Mock SemanticComponent for testing."""

    _instances: dict = {}
    _registry = None
    _embedder = None

    def __init__(
        self,
        component: MockGradioComponent,
        intent: str = "",
        tags: list | None = None,
        visual_spec: MockVisualSpec | None = None,
    ):
        self._component = component
        self._semantic_meta = MockSemanticMetadata(
            intent=intent,
            tags=tags or [],
            file_path="test.py",
            line_number=10,
            visual_spec=visual_spec,
        )
        # Register in class-level dict
        MockSemanticComponent._instances[component._id] = self

    @property
    def component(self) -> MockGradioComponent:
        return self._component

    @property
    def semantic_meta(self) -> MockSemanticMetadata:
        return self._semantic_meta


class MockBlocks:
    """Mock SemanticBlocks for testing."""

    def __init__(self):
        self.fns = []
        self.dependencies = []
        self.blocks = {}
        self.children = []
        self.title = "Test App"
        self._blocks = self

    def add_component(self, component: MockGradioComponent):
        """Add a component to the blocks."""
        self.blocks[component._id] = component
        self.children.append(component)

    def add_dependency(
        self,
        fn,
        trigger: list,
        inputs: list,
        outputs: list,
        trigger_event: str = "click",
    ):
        """Add a dependency (event handler)."""
        self.dependencies.append({
            "fn": fn,
            "trigger": trigger,
            "inputs": inputs,
            "outputs": outputs,
            "trigger_event": trigger_event,
        })


@pytest.fixture
def mock_gradio_components():
    """Create a set of mock Gradio components."""
    components = {
        "textbox": MockGradioComponent(
            _id=1,
            component_type="Textbox",
            label="Search Query",
            value="",
            placeholder="Enter search...",
        ),
        "button": MockGradioComponent(
            _id=2,
            component_type="Button",
            label="Search",
        ),
        "output": MockGradioComponent(
            _id=3,
            component_type="Markdown",
            label="Results",
            value="",
        ),
        "slider": MockGradioComponent(
            _id=4,
            component_type="Slider",
            label="Max Results",
            value=10,
        ),
        "dropdown": MockGradioComponent(
            _id=5,
            component_type="Dropdown",
            label="Category",
            value="all",
        ),
        "hidden": MockGradioComponent(
            _id=6,
            component_type="Textbox",
            label="Hidden",
            visible=False,
        ),
        "readonly": MockGradioComponent(
            _id=7,
            component_type="Textbox",
            label="Readonly",
            interactive=False,
        ),
    }
    return components


@pytest.fixture
def mock_semantic_components(mock_gradio_components):
    """Create mock SemanticComponents wrapping Gradio components."""
    # Clear existing instances
    MockSemanticComponent._instances = {}

    components = {
        "textbox": MockSemanticComponent(
            mock_gradio_components["textbox"],
            intent="user search input",
            tags=["input", "text", "search"],
        ),
        "button": MockSemanticComponent(
            mock_gradio_components["button"],
            intent="trigger search action",
            tags=["action", "submit"],
            visual_spec=MockVisualSpec(component_id="2", component_type="Button"),
        ),
        "output": MockSemanticComponent(
            mock_gradio_components["output"],
            intent="display search results",
            tags=["output", "display"],
        ),
        "slider": MockSemanticComponent(
            mock_gradio_components["slider"],
            intent="control result count",
            tags=["input", "control"],
        ),
        "dropdown": MockSemanticComponent(
            mock_gradio_components["dropdown"],
            intent="select category filter",
            tags=["input", "filter"],
        ),
    }
    return components


@pytest.fixture
def mock_blocks(mock_gradio_components):
    """Create mock blocks with components and dependencies."""
    blocks = MockBlocks()

    for component in mock_gradio_components.values():
        blocks.add_component(component)

    # Add some dependencies (event handlers)
    def search_fn(query, max_results):
        return f"Results for: {query}"

    def filter_fn(category):
        return f"Filtered by: {category}"

    blocks.add_dependency(
        fn=search_fn,
        trigger=[mock_gradio_components["button"]._id],
        inputs=[
            mock_gradio_components["textbox"],
            mock_gradio_components["slider"],
        ],
        outputs=[mock_gradio_components["output"]],
        trigger_event="click",
    )

    blocks.add_dependency(
        fn=filter_fn,
        trigger=[mock_gradio_components["dropdown"]._id],
        inputs=[mock_gradio_components["dropdown"]],
        outputs=[mock_gradio_components["output"]],
        trigger_event="change",
    )

    return blocks


@pytest.fixture
def populated_blocks(mock_blocks, mock_semantic_components):
    """Blocks with semantic components registered."""
    return mock_blocks


@pytest.fixture
def patch_semantic_component(mock_semantic_components):
    """Patch SemanticComponent._instances with mock components."""
    from integradio.components import SemanticComponent

    original_instances = SemanticComponent._instances.copy()

    # Copy mock instances to real class
    for comp_id, semantic in MockSemanticComponent._instances.items():
        SemanticComponent._instances[comp_id] = semantic

    yield

    # Restore original
    SemanticComponent._instances = original_instances


@pytest.fixture
def clean_semantic_instances():
    """Clean SemanticComponent instances before and after test."""
    from integradio.components import SemanticComponent

    original_instances = SemanticComponent._instances.copy()
    SemanticComponent._instances.clear()
    MockSemanticComponent._instances.clear()

    yield

    SemanticComponent._instances = original_instances
    MockSemanticComponent._instances.clear()


@pytest.fixture
def component_with_click():
    """Create a component with click capability."""
    component = MockGradioComponent(
        _id=100,
        component_type="Button",
        label="Click Me",
    )

    semantic = MockSemanticComponent(
        component,
        intent="clickable button",
        tags=["action"],
    )

    return component, semantic


@pytest.fixture
def component_with_value():
    """Create a component with value."""
    component = MockGradioComponent(
        _id=101,
        component_type="Textbox",
        label="Input",
        value="test value",
    )

    semantic = MockSemanticComponent(
        component,
        intent="text input",
        tags=["input"],
    )

    return component, semantic


@pytest.fixture
def component_with_visual_spec():
    """Create a component with visual specification."""
    component = MockGradioComponent(
        _id=102,
        component_type="Button",
        label="Styled",
    )

    visual = MockVisualSpec(
        component_id="102",
        component_type="Button",
        tokens={"background": "#3b82f6"},
    )

    semantic = MockSemanticComponent(
        component,
        intent="styled button",
        tags=["styled"],
        visual_spec=visual,
    )

    return component, semantic
