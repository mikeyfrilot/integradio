"""
Live Component Inspector - Development tools for Integradio apps.

Provides a Gradio-based inspector panel that shows:
- Real-time component tree with semantic intents
- Live dataflow visualization
- Visual spec overlay
- Search across UI by intent

Usage:
    from integradio import SemanticBlocks
    from integradio.inspector import Inspector

    with SemanticBlocks() as demo:
        # Your components here...

        # Add inspector panel (dev mode only)
        inspector = Inspector(demo)
        inspector.attach()

    demo.launch()
"""

from .core import (
    Inspector,
    InspectorConfig,
    InspectorState,
)

from .tree import (
    ComponentNode,
    ComponentTree,
    build_component_tree,
    tree_to_mermaid,
    tree_to_json,
)

from .dataflow import (
    DataFlowEdge,
    DataFlowGraph,
    extract_dataflow,
    dataflow_to_mermaid,
)

from .search import (
    SearchResult,
    SearchEngine,
    search_by_intent,
    search_by_tag,
    search_by_type,
)

from .panel import (
    create_inspector_panel,
    create_tree_view,
    create_dataflow_view,
    create_search_panel,
    create_details_panel,
)

__all__ = [
    # Core
    "Inspector",
    "InspectorConfig",
    "InspectorState",
    # Tree
    "ComponentNode",
    "ComponentTree",
    "build_component_tree",
    "tree_to_mermaid",
    "tree_to_json",
    # Dataflow
    "DataFlowEdge",
    "DataFlowGraph",
    "extract_dataflow",
    "dataflow_to_mermaid",
    # Search
    "SearchResult",
    "SearchEngine",
    "search_by_intent",
    "search_by_tag",
    "search_by_type",
    # Panel
    "create_inspector_panel",
    "create_tree_view",
    "create_dataflow_view",
    "create_search_panel",
    "create_details_panel",
]
