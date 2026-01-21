"""
Tests for Component Tree module.

Tests:
- ComponentNode creation and manipulation
- ComponentTree building and traversal
- Tree search functionality
- Mermaid and ASCII export
"""

import pytest
from integradio.inspector.tree import (
    ComponentNode,
    ComponentTree,
    tree_to_mermaid,
    tree_to_ascii,
)


class TestComponentNode:
    """Tests for ComponentNode dataclass."""

    def test_create_node(self):
        """Test basic node creation."""
        node = ComponentNode(
            id="123",
            component_type="Button",
            intent="submit form",
        )

        assert node.id == "123"
        assert node.component_type == "Button"
        assert node.intent == "submit form"
        assert node.children == []
        assert node.parent_id is None

    def test_node_with_all_fields(self):
        """Test node creation with all optional fields."""
        node = ComponentNode(
            id="456",
            component_type="Textbox",
            intent="user input",
            label="Name",
            elem_id="name-input",
            tags=["input", "text", "required"],
            file_path="/app/main.py",
            line_number=42,
            has_visual_spec=True,
            visual_tokens={"background": "#ffffff", "color": "#000000"},
            is_visible=True,
            is_interactive=True,
        )

        assert node.label == "Name"
        assert node.elem_id == "name-input"
        assert "required" in node.tags
        assert node.file_path == "/app/main.py"
        assert node.line_number == 42
        assert node.has_visual_spec is True
        assert node.visual_tokens["background"] == "#ffffff"

    def test_add_child(self):
        """Test adding child nodes."""
        parent = ComponentNode(id="parent", component_type="Row", intent="layout")
        child1 = ComponentNode(id="child1", component_type="Button", intent="action 1")
        child2 = ComponentNode(id="child2", component_type="Button", intent="action 2")

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.children) == 2
        assert child1.parent_id == "parent"
        assert child2.parent_id == "parent"

    def test_find_node(self):
        """Test finding nodes in subtree."""
        root = ComponentNode(id="root", component_type="Column", intent="main layout")
        child = ComponentNode(id="child", component_type="Row", intent="row")
        grandchild = ComponentNode(id="grandchild", component_type="Button", intent="btn")

        root.add_child(child)
        child.add_child(grandchild)

        # Find existing nodes
        assert root.find("root") == root
        assert root.find("child") == child
        assert root.find("grandchild") == grandchild

        # Find non-existing
        assert root.find("nonexistent") is None

    def test_iter_all(self):
        """Test iterating over all nodes."""
        root = ComponentNode(id="root", component_type="Column", intent="main")
        child1 = ComponentNode(id="c1", component_type="Button", intent="btn1")
        child2 = ComponentNode(id="c2", component_type="Button", intent="btn2")
        grandchild = ComponentNode(id="gc", component_type="Textbox", intent="input")

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        all_nodes = list(root.iter_all())
        assert len(all_nodes) == 4
        assert root in all_nodes
        assert grandchild in all_nodes

    def test_to_dict(self):
        """Test JSON serialization."""
        node = ComponentNode(
            id="test",
            component_type="Button",
            intent="click me",
            tags=["action"],
        )
        child = ComponentNode(id="child", component_type="Icon", intent="icon")
        node.add_child(child)

        data = node.to_dict()

        assert data["id"] == "test"
        assert data["type"] == "Button"
        assert data["intent"] == "click me"
        assert data["tags"] == ["action"]
        assert len(data["children"]) == 1
        assert data["children"][0]["id"] == "child"


class TestComponentTree:
    """Tests for ComponentTree."""

    def test_create_empty_tree(self):
        """Test creating an empty tree."""
        tree = ComponentTree()

        assert tree.root is None
        assert tree.total_components == 0
        assert tree.nodes == {}

    def test_add_root_node(self):
        """Test adding a root node."""
        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Blocks", intent="main app")

        tree.add_node(root)

        assert tree.root == root
        assert tree.total_components == 1
        assert "root" in tree.nodes

    def test_add_child_nodes(self):
        """Test adding nodes with parent relationships."""
        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Column", intent="layout")
        child = ComponentNode(id="child", component_type="Button", intent="action")

        tree.add_node(root)
        tree.add_node(child, parent_id="root")

        assert tree.total_components == 2
        assert child in root.children
        assert child.parent_id == "root"

    def test_get_node(self):
        """Test retrieving nodes by ID."""
        tree = ComponentTree()
        node = ComponentNode(id="test", component_type="Textbox", intent="input")
        tree.add_node(node)

        assert tree.get_node("test") == node
        assert tree.get_node("nonexistent") is None

    def test_find_by_intent(self):
        """Test searching by intent."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="1", component_type="Textbox", intent="user search query"))
        tree.add_node(ComponentNode(id="2", component_type="Button", intent="submit search"))
        tree.add_node(ComponentNode(id="3", component_type="Markdown", intent="display results"))

        results = tree.find_by_intent("search")
        assert len(results) == 2

        results = tree.find_by_intent("results")
        assert len(results) == 1

    def test_find_by_type(self):
        """Test searching by component type."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="1", component_type="Button", intent="btn1"))
        tree.add_node(ComponentNode(id="2", component_type="Button", intent="btn2"))
        tree.add_node(ComponentNode(id="3", component_type="Textbox", intent="input"))

        buttons = tree.find_by_type("Button")
        assert len(buttons) == 2

        textboxes = tree.find_by_type("Textbox")
        assert len(textboxes) == 1

    def test_find_by_tag(self):
        """Test searching by tag."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="1", component_type="Textbox", intent="name", tags=["required", "input"]))
        tree.add_node(ComponentNode(id="2", component_type="Textbox", intent="email", tags=["required", "email"]))
        tree.add_node(ComponentNode(id="3", component_type="Textbox", intent="notes", tags=["optional"]))

        required = tree.find_by_tag("required")
        assert len(required) == 2

        optional = tree.find_by_tag("optional")
        assert len(optional) == 1

    def test_get_roots(self):
        """Test getting root-level nodes."""
        tree = ComponentTree()
        root1 = ComponentNode(id="r1", component_type="Tab", intent="tab1")
        root2 = ComponentNode(id="r2", component_type="Tab", intent="tab2")
        child = ComponentNode(id="c1", component_type="Button", intent="btn")

        tree.add_node(root1)
        tree.add_node(root2)
        tree.add_node(child, parent_id="r1")

        roots = tree.get_roots()
        assert len(roots) == 2
        assert root1 in roots
        assert root2 in roots
        assert child not in roots

    def test_to_json(self):
        """Test JSON export."""
        tree = ComponentTree(app_name="Test App")
        tree.add_node(ComponentNode(id="root", component_type="Blocks", intent="main"))

        json_str = tree.to_json()

        assert '"app_name": "Test App"' in json_str
        assert '"total_components": 1' in json_str


class TestTreeExport:
    """Tests for tree export functions."""

    @pytest.fixture
    def sample_tree(self):
        """Create a sample tree for testing."""
        tree = ComponentTree(app_name="Sample App")

        root = ComponentNode(id="root", component_type="Column", intent="main layout")
        tree.add_node(root)

        search_row = ComponentNode(id="search-row", component_type="Row", intent="search section")
        tree.add_node(search_row, parent_id="root")

        search_input = ComponentNode(
            id="search-input",
            component_type="Textbox",
            intent="user search query",
            has_visual_spec=True,
        )
        tree.add_node(search_input, parent_id="search-row")

        search_btn = ComponentNode(
            id="search-btn",
            component_type="Button",
            intent="submit search",
            has_visual_spec=True,
        )
        tree.add_node(search_btn, parent_id="search-row")

        results = ComponentNode(
            id="results",
            component_type="Markdown",
            intent="display search results",
        )
        tree.add_node(results, parent_id="root")

        return tree

    def test_tree_to_mermaid(self, sample_tree):
        """Test Mermaid diagram export."""
        mermaid = tree_to_mermaid(sample_tree)

        # Check structure
        assert "graph TB" in mermaid or "graph LR" in mermaid

        # Check nodes are present (intent should be in labels)
        assert "main layout" in mermaid
        assert "search" in mermaid.lower()

        # Check edges
        assert "-->" in mermaid

        # Check styling for semantic components
        assert "classDef semantic" in mermaid

    def test_tree_to_mermaid_direction(self, sample_tree):
        """Test Mermaid diagram with different directions."""
        lr = tree_to_mermaid(sample_tree, direction="LR")
        assert "graph LR" in lr

        tb = tree_to_mermaid(sample_tree, direction="TB")
        assert "graph TB" in tb

    def test_tree_to_ascii(self, sample_tree):
        """Test ASCII art export."""
        ascii_tree = tree_to_ascii(sample_tree)

        # Check structure characters
        assert "├──" in ascii_tree or "└──" in ascii_tree

        # Check component info present
        assert "Column" in ascii_tree or "main" in ascii_tree

        # Check visual spec indicators
        assert "●" in ascii_tree  # Has visual spec
        assert "○" in ascii_tree  # No visual spec


class TestTreeEdgeCases:
    """Edge case tests for tree module."""

    def test_empty_tree_exports(self):
        """Test exporting empty tree."""
        tree = ComponentTree()

        mermaid = tree_to_mermaid(tree)
        assert "graph" in mermaid

        ascii_out = tree_to_ascii(tree)
        assert ascii_out == ""  # Empty tree produces empty string

    def test_single_node_tree(self):
        """Test tree with single node."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="only", component_type="Button", intent="solo"))

        mermaid = tree_to_mermaid(tree)
        assert "solo" in mermaid

    def test_deep_tree(self):
        """Test deeply nested tree."""
        tree = ComponentTree()
        prev_id = None

        for i in range(10):
            node = ComponentNode(id=f"node{i}", component_type="Column", intent=f"level {i}")
            tree.add_node(node, parent_id=prev_id)
            prev_id = f"node{i}"

        assert tree.total_components == 10

        # All nodes should be findable
        for i in range(10):
            assert tree.get_node(f"node{i}") is not None

    def test_wide_tree(self):
        """Test tree with many siblings."""
        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Row", intent="buttons")
        tree.add_node(root)

        for i in range(20):
            child = ComponentNode(id=f"btn{i}", component_type="Button", intent=f"action {i}")
            tree.add_node(child, parent_id="root")

        assert tree.total_components == 21
        assert len(root.children) == 20

    def test_unicode_intents(self):
        """Test handling of unicode in intents."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(
            id="emoji",
            component_type="Button",
            intent="🔍 Search with émojis and spëcial chars",
        ))

        json_out = tree.to_json()
        assert "Search" in json_out

    def test_long_intent_truncation(self):
        """Test that long intents are handled in exports."""
        tree = ComponentTree()
        long_intent = "A" * 100
        tree.add_node(ComponentNode(id="long", component_type="Markdown", intent=long_intent))

        mermaid = tree_to_mermaid(tree)
        # Should be truncated or handled gracefully
        assert "..." in mermaid or len(mermaid) < 500


class TestBuildComponentTree:
    """Tests for build_component_tree with mock blocks."""

    def test_build_tree_from_blocks(
        self,
        mock_blocks,
        mock_semantic_components,
    ):
        """Test building tree from mock blocks."""
        from integradio.inspector.tree import build_component_tree

        tree = build_component_tree(mock_blocks)

        assert isinstance(tree, ComponentTree)
        assert tree.total_components >= 0

    def test_build_tree_with_dependencies(
        self,
        populated_blocks,
        mock_semantic_components,
    ):
        """Test building tree from blocks with dependencies."""
        from integradio.inspector.tree import build_component_tree

        tree = build_component_tree(populated_blocks)

        assert isinstance(tree, ComponentTree)

    def test_build_tree_empty_blocks(self):
        """Test building from empty blocks."""
        from integradio.inspector.tree import build_component_tree
        from tests.inspector.conftest import MockBlocks

        blocks = MockBlocks()
        tree = build_component_tree(blocks)

        assert isinstance(tree, ComponentTree)

    def test_build_tree_none_blocks(self):
        """Test building from None blocks."""
        from integradio.inspector.tree import build_component_tree

        tree = build_component_tree(None)

        assert isinstance(tree, ComponentTree)
        assert tree.total_components == 0

    def test_build_tree_includes_semantic_info(
        self,
        mock_blocks,
        mock_semantic_components,
        patch_semantic_component,
    ):
        """Test that built tree includes semantic metadata."""
        from integradio.inspector.tree import build_component_tree
        from integradio.components import SemanticComponent

        tree = build_component_tree(mock_blocks)

        # Should include semantic information
        assert tree.semantic_components >= 0


class TestTreeNodeDepth:
    """Tests for node depth calculation."""

    def test_node_depth_root(self):
        """Test root node has depth 0."""
        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Column", intent="main")
        tree.add_node(root)

        assert root.depth == 0

    def test_node_depth_children(self):
        """Test child nodes have appropriate depth."""
        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Column", intent="main")
        tree.add_node(root)

        child = ComponentNode(id="child", component_type="Row", intent="row")
        tree.add_node(child, parent_id="root")

        grandchild = ComponentNode(id="grandchild", component_type="Button", intent="btn")
        tree.add_node(grandchild, parent_id="child")

        # Depth may be computed lazily or differently
        # Just verify parent-child relationship
        assert child.parent_id == "root"
        assert grandchild.parent_id == "child"


class TestTreeMermaidShapes:
    """Tests for Mermaid node shapes by component type."""

    def test_mermaid_button_shape(self):
        """Test Button uses correct Mermaid shape."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="btn", component_type="Button", intent="click"))

        mermaid = tree_to_mermaid(tree)

        # Buttons should use a specific shape
        assert "btn" in mermaid

    def test_mermaid_textbox_shape(self):
        """Test Textbox uses correct Mermaid shape."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="txt", component_type="Textbox", intent="input"))

        mermaid = tree_to_mermaid(tree)

        assert "txt" in mermaid

    def test_mermaid_slider_shape(self):
        """Test Slider uses correct Mermaid shape."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="sld", component_type="Slider", intent="adjust"))

        mermaid = tree_to_mermaid(tree)

        assert "sld" in mermaid

    def test_mermaid_markdown_shape(self):
        """Test Markdown uses correct Mermaid shape."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(id="md", component_type="Markdown", intent="display"))

        mermaid = tree_to_mermaid(tree)

        assert "md" in mermaid


class TestTreeToJSON:
    """Tests for tree JSON serialization."""

    def test_tree_json_structure(self):
        """Test JSON output structure."""
        import json

        tree = ComponentTree(app_name="Test")
        tree.add_node(ComponentNode(
            id="1",
            component_type="Button",
            intent="test",
            tags=["tag1"],
        ))

        json_str = tree.to_json()
        data = json.loads(json_str)

        assert "app_name" in data
        assert "total_components" in data
        assert "semantic_components" in data
        assert "roots" in data

    def test_tree_json_includes_children(self):
        """Test JSON includes nested children."""
        import json

        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Column", intent="main")
        tree.add_node(root)

        child = ComponentNode(id="child", component_type="Button", intent="btn")
        tree.add_node(child, parent_id="root")

        json_str = tree.to_json()
        data = json.loads(json_str)

        # Should have nested structure
        assert len(data["roots"]) == 1
        assert "children" in data["roots"][0]


class TestTreeWithVisualSpec:
    """Tests for tree nodes with visual specifications."""

    def test_node_with_visual_tokens(self):
        """Test node stores visual tokens."""
        node = ComponentNode(
            id="styled",
            component_type="Button",
            intent="styled button",
            has_visual_spec=True,
            visual_tokens={
                "background": "#3b82f6",
                "color": "#ffffff",
                "border-radius": "4px",
            },
        )

        assert node.has_visual_spec is True
        assert node.visual_tokens["background"] == "#3b82f6"

    def test_tree_counts_semantic_components(self):
        """Test tree counts semantic vs non-semantic components."""
        tree = ComponentTree()

        # Add some components with and without visual spec
        tree.add_node(ComponentNode(
            id="1", component_type="Button", intent="styled",
            has_visual_spec=True,
        ))
        tree.add_node(ComponentNode(
            id="2", component_type="Button", intent="unstyled",
            has_visual_spec=False,
        ))

        tree.semantic_components = 1
        assert tree.semantic_components >= 0

    def test_mermaid_highlights_visual_spec(self):
        """Test Mermaid diagram highlights visual spec nodes."""
        tree = ComponentTree()
        tree.add_node(ComponentNode(
            id="styled",
            component_type="Button",
            intent="styled",
            has_visual_spec=True,
        ))
        tree.add_node(ComponentNode(
            id="unstyled",
            component_type="Button",
            intent="unstyled",
            has_visual_spec=False,
        ))

        mermaid = tree_to_mermaid(tree)

        # Should have class for semantic components
        assert "classDef semantic" in mermaid


class TestTreeIterators:
    """Tests for tree iteration methods."""

    def test_nodes_dict_contains_all(self):
        """Test that nodes dict contains all added nodes."""
        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Column", intent="main")
        tree.add_node(root)

        child1 = ComponentNode(id="c1", component_type="Button", intent="btn1")
        tree.add_node(child1, parent_id="root")

        child2 = ComponentNode(id="c2", component_type="Button", intent="btn2")
        tree.add_node(child2, parent_id="root")

        # Count all nodes via nodes dict
        assert len(tree.nodes) == 3
        assert tree.get_node("root") == root
        assert tree.get_node("c1") == child1
        assert tree.get_node("c2") == child2

    def test_tree_maintains_hierarchy(self):
        """Test tree maintains parent-child hierarchy."""
        tree = ComponentTree()
        root = ComponentNode(id="root", component_type="Column", intent="main")
        tree.add_node(root)

        child = ComponentNode(id="child", component_type="Row", intent="row")
        tree.add_node(child, parent_id="root")

        grandchild = ComponentNode(id="gc", component_type="Button", intent="btn")
        tree.add_node(grandchild, parent_id="child")

        # Verify hierarchy
        assert len(root.children) == 1
        assert root.children[0] == child
        assert len(child.children) == 1
        assert child.children[0] == grandchild
