"""
Inspector Core - Main inspector class and configuration.

The Inspector is the main entry point for adding development tools
to an Integradio application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import json

from .tree import ComponentTree, build_component_tree
from .dataflow import DataFlowGraph, extract_dataflow
from .search import SearchEngine, SearchResult


class InspectorMode(str, Enum):
    """Inspector display modes."""
    SIDEBAR = "sidebar"      # Collapsible sidebar
    FLOATING = "floating"    # Floating button + panel
    EMBEDDED = "embedded"    # Embedded in a tab
    HIDDEN = "hidden"        # Disabled


@dataclass
class InspectorConfig:
    """Configuration for the inspector."""
    mode: InspectorMode = InspectorMode.SIDEBAR
    position: str = "right"  # "left" or "right"
    width: int = 400
    collapsed: bool = True   # Start collapsed
    show_tree: bool = True
    show_dataflow: bool = True
    show_search: bool = True
    show_details: bool = True
    auto_refresh: bool = False  # Auto-refresh on changes
    refresh_interval: int = 1000  # ms


@dataclass
class InspectorState:
    """Runtime state of the inspector."""
    is_open: bool = False
    selected_component_id: str | None = None
    search_query: str = ""
    active_tab: str = "tree"

    # Cached data
    tree: ComponentTree | None = None
    dataflow: DataFlowGraph | None = None
    search_results: list[SearchResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_open": self.is_open,
            "selected_component_id": self.selected_component_id,
            "search_query": self.search_query,
            "active_tab": self.active_tab,
            "tree_loaded": self.tree is not None,
            "dataflow_loaded": self.dataflow is not None,
            "search_results_count": len(self.search_results),
        }


class Inspector:
    """
    Live component inspector for Integradio applications.

    Usage:
        from integradio import SemanticBlocks
        from integradio.inspector import Inspector

        with SemanticBlocks() as demo:
            # Your components...
            inp = semantic(gr.Textbox(), intent="user input")
            btn = semantic(gr.Button("Go"), intent="submit")

            # Attach inspector (dev mode only)
            if DEV_MODE:
                inspector = Inspector(demo)
                inspector.attach()

        demo.launch()
    """

    def __init__(
        self,
        blocks: Any,
        config: InspectorConfig | None = None,
    ):
        """
        Initialize the inspector.

        Args:
            blocks: SemanticBlocks instance to inspect
            config: Inspector configuration
        """
        self.blocks = blocks
        self.config = config or InspectorConfig()
        self.state = InspectorState()
        self._search_engine = SearchEngine(blocks)
        self._attached = False

    def refresh(self) -> None:
        """Refresh the inspector data."""
        self.state.tree = build_component_tree(self.blocks)
        self.state.dataflow = extract_dataflow(self.blocks)

    def search(self, query: str) -> list[SearchResult]:
        """
        Search for components.

        Args:
            query: Search query

        Returns:
            List of matching components
        """
        self.state.search_query = query
        self.state.search_results = self._search_engine.search(query)
        return self.state.search_results

    def select_component(self, component_id: str) -> dict | None:
        """
        Select a component and get its details.

        Args:
            component_id: ID of component to select

        Returns:
            Component details dict or None
        """
        from ..components import SemanticComponent

        self.state.selected_component_id = component_id

        # Get component
        try:
            comp_id = int(component_id)
        except (ValueError, TypeError):
            return None

        semantic = SemanticComponent._instances.get(comp_id)
        if not semantic:
            return None

        meta = semantic.semantic_meta
        component = semantic.component

        # Build details
        details = {
            "id": component_id,
            "type": type(component).__name__,
            "intent": meta.intent,
            "tags": meta.tags,
            "file_path": meta.file_path,
            "line_number": meta.line_number,
            "embedded": meta.embedded,
        }

        # Add visual spec info
        if meta.visual_spec:
            details["visual_spec"] = {
                "has_spec": True,
                "tokens": {k: v.to_css() for k, v in meta.visual_spec.tokens.items()},
                "css": meta.visual_spec.to_css(),
            }
        else:
            details["visual_spec"] = {"has_spec": False}

        # Add component-specific properties
        for attr in ["label", "placeholder", "value", "interactive", "visible"]:
            if hasattr(component, attr):
                val = getattr(component, attr)
                # Don't include functions
                if not callable(val):
                    details[attr] = val

        # Add dataflow info
        if self.state.dataflow:
            inputs = self.state.dataflow.get_inputs_for(component_id)
            outputs = self.state.dataflow.get_outputs_for(component_id)
            details["dataflow"] = {
                "inputs_from": inputs,
                "outputs_to": outputs,
            }

        return details

    def attach(self) -> Any:
        """
        Attach the inspector panel to the Gradio app.

        Returns:
            The created panel component
        """
        if self._attached:
            return None

        if self.config.mode == InspectorMode.HIDDEN:
            return None

        # Refresh data
        self.refresh()

        # Create panel based on mode
        if self.config.mode == InspectorMode.SIDEBAR:
            from .panel import create_inspector_panel
            panel = create_inspector_panel(
                self.blocks,
                position=self.config.position,
                width=self.config.width,
                collapsed=self.config.collapsed,
            )
            self._attached = True
            return panel

        elif self.config.mode == InspectorMode.FLOATING:
            from .panel import create_floating_inspector
            html = create_floating_inspector(self.blocks)
            # Inject HTML into the app
            import gradio as gr
            panel = gr.HTML(html, visible=True)
            self._attached = True
            return panel

        elif self.config.mode == InspectorMode.EMBEDDED:
            # Create as a tab
            import gradio as gr
            from .panel import create_tree_view, create_dataflow_view

            with gr.Tab("🔍 Inspector"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Component Tree")
                        gr.HTML(create_tree_view(self.state.tree))

                    with gr.Column(scale=1):
                        gr.Markdown("### Data Flow")
                        gr.Markdown(create_dataflow_view(self.state.dataflow))

                with gr.Row():
                    search_input = gr.Textbox(label="Search")
                    results = gr.JSON(label="Results")

                    search_input.change(
                        lambda q: [r.to_dict() for r in self.search(q)],
                        inputs=[search_input],
                        outputs=[results],
                    )

            self._attached = True
            return None

        return None

    def get_tree_mermaid(self) -> str:
        """Get component tree as Mermaid diagram."""
        if self.state.tree is None:
            self.refresh()
        from .tree import tree_to_mermaid
        return tree_to_mermaid(self.state.tree)

    def get_dataflow_mermaid(self) -> str:
        """Get dataflow as Mermaid diagram."""
        if self.state.dataflow is None:
            self.refresh()
        from .dataflow import dataflow_to_mermaid
        return dataflow_to_mermaid(self.state.dataflow)

    def export_state(self) -> dict:
        """
        Export current inspector state as JSON-serializable dict.

        Useful for debugging or persistence.
        """
        return {
            "state": self.state.to_dict(),
            "config": {
                "mode": self.config.mode.value,
                "position": self.config.position,
                "width": self.config.width,
            },
            "tree": self.state.tree.to_dict() if self.state.tree else None,
            "dataflow": self.state.dataflow.to_dict() if self.state.dataflow else None,
        }

    def __repr__(self) -> str:
        return f"Inspector(mode={self.config.mode.value}, attached={self._attached})"


# =============================================================================
# Convenience Functions
# =============================================================================

def inspect(blocks: Any, mode: str = "sidebar") -> Inspector:
    """
    Quick way to add an inspector to a Gradio app.

    Args:
        blocks: SemanticBlocks instance
        mode: "sidebar", "floating", "embedded", or "hidden"

    Returns:
        Inspector instance (already attached)
    """
    mode_enum = InspectorMode(mode)
    config = InspectorConfig(mode=mode_enum)
    inspector = Inspector(blocks, config)
    inspector.attach()
    return inspector


def dev_mode(blocks: Any) -> Inspector:
    """
    Enable development mode with full inspector.

    Args:
        blocks: SemanticBlocks instance

    Returns:
        Inspector instance
    """
    config = InspectorConfig(
        mode=InspectorMode.SIDEBAR,
        collapsed=False,
        auto_refresh=True,
    )
    inspector = Inspector(blocks, config)
    inspector.attach()
    return inspector
