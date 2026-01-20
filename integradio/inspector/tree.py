"""
Component Tree - Build and visualize component hierarchies.

Creates a tree representation of all semantic components in a Gradio app,
including their relationships, intents, and visual specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator
import json


@dataclass
class ComponentNode:
    """A node in the component tree."""
    id: str
    component_type: str
    intent: str
    label: str | None = None
    elem_id: str | None = None
    tags: list[str] = field(default_factory=list)
    children: list["ComponentNode"] = field(default_factory=list)
    parent_id: str | None = None

    # Source location
    file_path: str | None = None
    line_number: int | None = None

    # Visual spec reference
    has_visual_spec: bool = False
    visual_tokens: dict[str, str] = field(default_factory=dict)

    # Runtime state (for live inspection)
    is_visible: bool = True
    is_interactive: bool = True
    current_value: Any = None

    def add_child(self, child: "ComponentNode") -> None:
        """Add a child node."""
        child.parent_id = self.id
        self.children.append(child)

    def find(self, node_id: str) -> "ComponentNode | None":
        """Find a node by ID in this subtree."""
        if self.id == node_id:
            return self
        for child in self.children:
            found = child.find(node_id)
            if found:
                return found
        return None

    def iter_all(self) -> Iterator["ComponentNode"]:
        """Iterate over all nodes in this subtree."""
        yield self
        for child in self.children:
            yield from child.iter_all()

    @property
    def depth(self) -> int:
        """Get the depth of this node in the tree."""
        if self.parent_id is None:
            return 0
        # This is a simplified version - actual depth would need tree traversal
        return 1

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "type": self.component_type,
            "intent": self.intent,
            "label": self.label,
            "elem_id": self.elem_id,
            "tags": self.tags,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "has_visual_spec": self.has_visual_spec,
            "visual_tokens": self.visual_tokens,
            "is_visible": self.is_visible,
            "is_interactive": self.is_interactive,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ComponentTree:
    """
    Complete component tree for a Gradio app.

    Represents the hierarchy of all semantic components.
    """
    root: ComponentNode | None = None
    nodes: dict[str, ComponentNode] = field(default_factory=dict)

    # Metadata
    app_name: str = ""
    total_components: int = 0
    semantic_components: int = 0

    def add_node(self, node: ComponentNode, parent_id: str | None = None) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].add_child(node)
        elif self.root is None:
            self.root = node

        self.total_components = len(self.nodes)

    def get_node(self, node_id: str) -> ComponentNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def find_by_intent(self, query: str) -> list[ComponentNode]:
        """Find nodes whose intent contains the query."""
        query_lower = query.lower()
        return [
            node for node in self.nodes.values()
            if query_lower in node.intent.lower()
        ]

    def find_by_type(self, component_type: str) -> list[ComponentNode]:
        """Find nodes by component type."""
        type_lower = component_type.lower()
        return [
            node for node in self.nodes.values()
            if type_lower in node.component_type.lower()
        ]

    def find_by_tag(self, tag: str) -> list[ComponentNode]:
        """Find nodes that have a specific tag."""
        tag_lower = tag.lower()
        return [
            node for node in self.nodes.values()
            if any(tag_lower in t.lower() for t in node.tags)
        ]

    def get_roots(self) -> list[ComponentNode]:
        """Get all root-level nodes (no parent)."""
        return [n for n in self.nodes.values() if n.parent_id is None]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "app_name": self.app_name,
            "total_components": self.total_components,
            "semantic_components": self.semantic_components,
            "roots": [r.to_dict() for r in self.get_roots()],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def build_component_tree(blocks: Any) -> ComponentTree:
    """
    Build a component tree from a SemanticBlocks instance.

    Args:
        blocks: SemanticBlocks instance

    Returns:
        ComponentTree representing all components
    """
    from ..components import SemanticComponent

    tree = ComponentTree()
    tree.app_name = getattr(blocks, "title", "Gradio App")

    # Get the underlying Gradio blocks
    gr_blocks = getattr(blocks, "_blocks", blocks)

    # Track which components we've processed
    processed: set[str] = set()

    def process_component(component: Any, parent_id: str | None = None) -> str | None:
        """Process a single component and return its node ID."""
        comp_id = str(getattr(component, "_id", id(component)))

        if comp_id in processed:
            return comp_id
        processed.add(comp_id)

        # Check if it's a semantic component
        semantic = SemanticComponent._instances.get(getattr(component, "_id", None))

        if semantic:
            tree.semantic_components += 1
            meta = semantic.semantic_meta

            # Get visual spec info
            has_visual = meta.visual_spec is not None
            visual_tokens = {}
            if has_visual and meta.visual_spec:
                visual_tokens = {
                    k: v.to_css() for k, v in meta.visual_spec.tokens.items()
                }

            node = ComponentNode(
                id=comp_id,
                component_type=type(component).__name__,
                intent=meta.intent,
                label=getattr(component, "label", None),
                elem_id=getattr(component, "elem_id", None),
                tags=meta.tags,
                file_path=meta.file_path,
                line_number=meta.line_number,
                has_visual_spec=has_visual,
                visual_tokens=visual_tokens,
                is_interactive=getattr(component, "interactive", True),
            )
        else:
            # Non-semantic component
            node = ComponentNode(
                id=comp_id,
                component_type=type(component).__name__,
                intent=getattr(component, "label", "") or type(component).__name__,
                label=getattr(component, "label", None),
                elem_id=getattr(component, "elem_id", None),
                is_interactive=getattr(component, "interactive", True),
            )

        tree.add_node(node, parent_id)

        # Process children (for layout components)
        children = getattr(component, "children", [])
        if children:
            for child in children:
                if hasattr(child, "_id"):
                    process_component(child, comp_id)

        return comp_id

    # Process all blocks
    if hasattr(gr_blocks, "blocks"):
        # Gradio Blocks has a .blocks dict
        for block_id, block in gr_blocks.blocks.items():
            process_component(block)
    elif hasattr(gr_blocks, "children"):
        for child in gr_blocks.children:
            process_component(child)

    return tree


def tree_to_mermaid(tree: ComponentTree, direction: str = "TB") -> str:
    """
    Convert component tree to Mermaid diagram syntax.

    Args:
        tree: ComponentTree to convert
        direction: Diagram direction (TB, BT, LR, RL)

    Returns:
        Mermaid diagram string
    """
    lines = [f"graph {direction}"]

    def node_shape(node: ComponentNode) -> str:
        """Get Mermaid node shape based on component type."""
        comp_type = node.component_type.lower()
        label = node.intent[:30] + "..." if len(node.intent) > 30 else node.intent

        if "button" in comp_type:
            return f'{node.id}(["{label}"])'  # Stadium shape
        elif "textbox" in comp_type or "input" in comp_type:
            return f'{node.id}[/"{label}"/]'  # Parallelogram
        elif "row" in comp_type or "column" in comp_type or "group" in comp_type:
            return f'{node.id}{{"{label}"}}'  # Diamond
        elif "markdown" in comp_type or "html" in comp_type:
            return f'{node.id}>"{label}"]'  # Asymmetric
        else:
            return f'{node.id}["{label}"]'  # Rectangle

    def add_node(node: ComponentNode, indent: str = "    ") -> None:
        """Add a node and its children to the diagram."""
        lines.append(f"{indent}{node_shape(node)}")

        for child in node.children:
            lines.append(f"{indent}{node.id} --> {child.id}")
            add_node(child, indent)

    # Add all root nodes
    for root in tree.get_roots():
        add_node(root)

    # Add styling
    lines.append("")
    lines.append("    %% Styling")

    # Style semantic vs non-semantic components
    semantic_ids = [n.id for n in tree.nodes.values() if n.has_visual_spec]
    if semantic_ids:
        lines.append(f"    classDef semantic fill:#3b82f6,color:#fff")
        lines.append(f"    class {','.join(semantic_ids)} semantic")

    return "\n".join(lines)


def tree_to_json(tree: ComponentTree, indent: int = 2) -> str:
    """
    Convert component tree to JSON.

    Args:
        tree: ComponentTree to convert
        indent: JSON indentation

    Returns:
        JSON string
    """
    return tree.to_json(indent)


def tree_to_ascii(tree: ComponentTree) -> str:
    """
    Convert component tree to ASCII art representation.

    Returns:
        ASCII tree string
    """
    lines = []

    def add_node(node: ComponentNode, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        icon = "●" if node.has_visual_spec else "○"
        type_short = node.component_type[:10]

        lines.append(f"{prefix}{connector}{icon} [{type_short}] {node.intent}")

        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            add_node(child, new_prefix, i == len(node.children) - 1)

    for i, root in enumerate(tree.get_roots()):
        add_node(root, "", i == len(tree.get_roots()) - 1)

    return "\n".join(lines)
