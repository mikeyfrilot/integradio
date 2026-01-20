"""
Inspector Panel - Gradio UI components for the inspector.

Provides embeddable panels for:
- Component tree visualization
- Dataflow diagrams
- Search interface
- Component details
"""

from __future__ import annotations

from typing import Any, Callable
import json

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore

from .tree import ComponentTree, build_component_tree, tree_to_mermaid, tree_to_ascii
from .dataflow import DataFlowGraph, extract_dataflow, dataflow_to_mermaid
from .search import SearchEngine, SearchResult


def create_tree_view(tree: ComponentTree) -> str:
    """
    Create an HTML tree view of components.

    Args:
        tree: ComponentTree to visualize

    Returns:
        HTML string
    """
    def node_to_html(node: Any, depth: int = 0) -> str:
        indent = "  " * depth
        icon = "🔵" if node.has_visual_spec else "⚪"
        type_badge = f'<span class="type-badge">{node.component_type}</span>'

        html = f'''
{indent}<div class="tree-node" data-id="{node.id}" style="margin-left: {depth * 20}px;">
{indent}  <span class="node-icon">{icon}</span>
{indent}  <span class="node-intent">{node.intent}</span>
{indent}  {type_badge}
{indent}  <span class="node-tags">{", ".join(node.tags)}</span>
{indent}</div>'''

        for child in node.children:
            html += node_to_html(child, depth + 1)

        return html

    css = '''
<style>
.tree-view { font-family: monospace; font-size: 13px; }
.tree-node { padding: 4px 8px; cursor: pointer; border-radius: 4px; }
.tree-node:hover { background: #f0f0f0; }
.node-icon { margin-right: 8px; }
.node-intent { font-weight: 500; }
.type-badge {
    background: #e2e8f0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    margin-left: 8px;
}
.node-tags { color: #64748b; font-size: 11px; margin-left: 8px; }
</style>
'''

    html = f'{css}\n<div class="tree-view">'
    for root in tree.get_roots():
        html += node_to_html(root)
    html += '</div>'

    return html


def create_dataflow_view(graph: DataFlowGraph) -> str:
    """
    Create a Mermaid diagram view of dataflow.

    Args:
        graph: DataFlowGraph to visualize

    Returns:
        Mermaid diagram string (for gr.Markdown)
    """
    mermaid = dataflow_to_mermaid(graph)
    return f"```mermaid\n{mermaid}\n```"


def create_search_panel(on_search: Callable[[str], list[SearchResult]]) -> tuple:
    """
    Create a search panel with results.

    Args:
        on_search: Callback when search is performed

    Returns:
        Tuple of (search_input, results_output) components
    """
    if gr is None:
        raise ImportError("Gradio is required for inspector panels")

    search_input = gr.Textbox(
        label="Search Components",
        placeholder="Search by intent, tag, or type...",
        elem_id="inspector-search",
    )

    results_output = gr.HTML(
        label="Results",
        elem_id="inspector-results",
    )

    return search_input, results_output


def create_details_panel() -> tuple:
    """
    Create a component details panel.

    Returns:
        Tuple of (component_id_input, details_output) components
    """
    if gr is None:
        raise ImportError("Gradio is required for inspector panels")

    component_id = gr.Textbox(
        label="Component ID",
        placeholder="Select a component...",
        elem_id="inspector-component-id",
        interactive=False,
    )

    details = gr.JSON(
        label="Component Details",
        elem_id="inspector-details",
    )

    return component_id, details


def create_inspector_panel(
    blocks: Any,
    position: str = "right",
    width: int = 400,
    collapsed: bool = True,
) -> Any:
    """
    Create a complete inspector panel that can be attached to a Gradio app.

    Args:
        blocks: SemanticBlocks instance to inspect
        position: Panel position ("left" or "right")
        width: Panel width in pixels
        collapsed: Start collapsed

    Returns:
        Gradio Sidebar component
    """
    if gr is None:
        raise ImportError("Gradio is required for inspector panels")

    # Build initial data
    tree = build_component_tree(blocks)
    dataflow = extract_dataflow(blocks)
    engine = SearchEngine(blocks)

    with gr.Sidebar(
        position=position,
        width=width,
        open=not collapsed,
    ) as sidebar:
        gr.Markdown("## 🔍 Component Inspector")

        with gr.Tabs():
            # Tree tab
            with gr.Tab("Tree"):
                tree_html = gr.HTML(
                    value=create_tree_view(tree),
                    label="Component Tree",
                )
                refresh_tree_btn = gr.Button("🔄 Refresh", size="sm")

            # Dataflow tab
            with gr.Tab("Flow"):
                flow_md = gr.Markdown(
                    value=create_dataflow_view(dataflow),
                    label="Data Flow",
                )
                refresh_flow_btn = gr.Button("🔄 Refresh", size="sm")

            # Search tab
            with gr.Tab("Search"):
                search_input = gr.Textbox(
                    label="Query",
                    placeholder="Search by intent, tag, type...",
                )
                search_results = gr.HTML(
                    value="<p>Enter a search query</p>",
                )

            # Details tab
            with gr.Tab("Details"):
                selected_id = gr.Textbox(
                    label="Component ID",
                    interactive=True,
                )
                component_details = gr.JSON(
                    label="Details",
                )

        # Stats
        gr.Markdown(f"""
---
**Stats:**
- Total Components: {tree.total_components}
- Semantic Components: {tree.semantic_components}
- Event Handlers: {len(dataflow.handlers)}
        """)

        # Event handlers
        def refresh_tree():
            new_tree = build_component_tree(blocks)
            return create_tree_view(new_tree)

        def refresh_flow():
            new_flow = extract_dataflow(blocks)
            return create_dataflow_view(new_flow)

        def do_search(query):
            if not query:
                return "<p>Enter a search query</p>"

            results = engine.search(query)

            if not results:
                return "<p>No results found</p>"

            html = "<div class='search-results'>"
            for r in results:
                html += f'''
<div class="result" style="padding: 8px; border-bottom: 1px solid #eee;">
    <strong>{r.intent}</strong>
    <span style="color: #64748b;"> ({r.component_type})</span>
    <br/>
    <small>Score: {r.score:.2f} | Match: {r.match_type}</small>
</div>
'''
            html += "</div>"
            return html

        def get_details(comp_id):
            if not comp_id:
                return {}

            from ..components import SemanticComponent
            semantic = SemanticComponent._instances.get(int(comp_id))

            if not semantic:
                return {"error": "Component not found"}

            meta = semantic.semantic_meta
            return {
                "id": comp_id,
                "type": type(semantic.component).__name__,
                "intent": meta.intent,
                "tags": meta.tags,
                "file": meta.file_path,
                "line": meta.line_number,
                "has_visual_spec": meta.visual_spec is not None,
            }

        refresh_tree_btn.click(refresh_tree, outputs=[tree_html])
        refresh_flow_btn.click(refresh_flow, outputs=[flow_md])
        search_input.change(do_search, inputs=[search_input], outputs=[search_results])
        selected_id.change(get_details, inputs=[selected_id], outputs=[component_details])

    return sidebar


def create_floating_inspector(blocks: Any) -> str:
    """
    Create a floating inspector that can be injected into any app.

    Returns HTML/JS that creates a floating inspector panel.

    Args:
        blocks: SemanticBlocks instance

    Returns:
        HTML/JS string to inject
    """
    tree = build_component_tree(blocks)
    dataflow = extract_dataflow(blocks)

    tree_json = tree.to_json()
    flow_json = dataflow.to_json()

    return f'''
<div id="semantic-inspector" style="
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
">
    <button id="inspector-toggle" style="
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 20px;
    ">🔍</button>

    <div id="inspector-panel" style="
        display: none;
        position: absolute;
        bottom: 60px;
        right: 0;
        width: 400px;
        max-height: 500px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        overflow: auto;
        padding: 16px;
    ">
        <h3 style="margin: 0 0 16px 0;">Component Inspector</h3>

        <div id="inspector-tabs">
            <button class="tab active" data-tab="tree">Tree</button>
            <button class="tab" data-tab="flow">Flow</button>
            <button class="tab" data-tab="search">Search</button>
        </div>

        <div id="inspector-content">
            <pre id="tree-content" style="font-size: 12px; overflow: auto;">
{tree_to_ascii(tree)}
            </pre>
        </div>
    </div>
</div>

<script>
(function() {{
    const toggle = document.getElementById('inspector-toggle');
    const panel = document.getElementById('inspector-panel');

    toggle.addEventListener('click', () => {{
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }});

    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {{
        tab.addEventListener('click', (e) => {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
        }});
    }});

    // Store data for JS access
    window.__SEMANTIC_INSPECTOR__ = {{
        tree: {tree_json},
        dataflow: {flow_json}
    }};
}})();
</script>

<style>
#inspector-tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}}
#inspector-tabs .tab {{
    padding: 6px 12px;
    border: 1px solid #e2e8f0;
    background: white;
    border-radius: 4px;
    cursor: pointer;
}}
#inspector-tabs .tab.active {{
    background: #3b82f6;
    color: white;
    border-color: #3b82f6;
}}
</style>
'''
