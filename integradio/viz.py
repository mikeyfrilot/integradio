"""
Visualization module - Component graph rendering.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .blocks import SemanticBlocks

logger = logging.getLogger(__name__)


def _safe_label(label: Any) -> str:
    """Safely convert a label to string, handling None and special characters."""
    if label is None:
        return "unnamed"
    label_str = str(label)
    # Escape quotes and newlines for Mermaid compatibility
    return label_str.replace('"', "'").replace("\n", " ").replace("\r", "")


def _validate_graph(graph: Any) -> dict:
    """Validate and normalize graph structure."""
    if not isinstance(graph, dict):
        logger.warning("Invalid graph structure: expected dict")
        return {"nodes": [], "links": []}

    nodes = graph.get("nodes", [])
    links = graph.get("links", [])

    if not isinstance(nodes, list):
        logger.warning("Invalid nodes structure: expected list")
        nodes = []
    if not isinstance(links, list):
        logger.warning("Invalid links structure: expected list")
        links = []

    return {"nodes": nodes, "links": links}


def generate_mermaid(blocks: "SemanticBlocks") -> str:
    """
    Generate Mermaid diagram from component graph.

    Args:
        blocks: SemanticBlocks instance

    Returns:
        Mermaid diagram string
    """
    graph = _validate_graph(blocks.map())
    lines = ["graph TD"]

    # Define nodes
    for node in graph["nodes"]:
        if not isinstance(node, dict):
            continue
        node_id = f"c{node.get('id', 0)}"
        label = _safe_label(node.get("label"))
        node_type = node.get("type", "")

        # Style by type
        if "Button" in node_type:
            lines.append(f'    {node_id}["{label}"]:::trigger')
        elif any(t in node_type for t in ["Textbox", "Number", "Slider", "Dropdown"]):
            lines.append(f'    {node_id}("{label}"):::input')
        elif any(t in node_type for t in ["Markdown", "HTML", "Image", "Plot"]):
            lines.append(f'    {node_id}[["{label}"]]:::output')
        else:
            lines.append(f'    {node_id}("{label}")')

    # Define links
    for link in graph["links"]:
        if not isinstance(link, dict):
            continue
        source = f"c{link.get('source', 0)}"
        target = f"c{link.get('target', 0)}"
        link_type = link.get("type", "")

        if link_type == "trigger":
            lines.append(f"    {source} -->|trigger| {target}")
        elif link_type == "dataflow":
            lines.append(f"    {source} --> {target}")
        else:
            lines.append(f"    {source} -.->|{link_type}| {target}")

    # Add styles
    lines.extend([
        "",
        "    classDef trigger fill:#f9a825,stroke:#f57f17",
        "    classDef input fill:#4caf50,stroke:#2e7d32",
        "    classDef output fill:#2196f3,stroke:#1565c0",
    ])

    return "\n".join(lines)


def generate_html_graph(blocks: "SemanticBlocks", width: int = 800, height: int = 600) -> str:
    """
    Generate interactive D3.js force graph HTML.

    Args:
        blocks: SemanticBlocks instance
        width: Graph width in pixels
        height: Graph height in pixels

    Returns:
        Complete HTML document with D3 visualization
    """
    graph = _validate_graph(blocks.map())
    try:
        graph_json = json.dumps(graph)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize graph to JSON: {e}")
        graph_json = '{"nodes": [], "links": []}'

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Integradio Component Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            background: #1a1a2e;
            color: #eee;
        }}
        #graph {{
            width: 100vw;
            height: 100vh;
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            stroke: #fff;
            stroke-width: 2px;
        }}
        .node text {{
            font-size: 12px;
            fill: #fff;
            pointer-events: none;
        }}
        .link {{
            stroke: #666;
            stroke-opacity: 0.6;
        }}
        .link.trigger {{
            stroke: #f9a825;
            stroke-dasharray: 5,5;
        }}
        .link.dataflow {{
            stroke: #4caf50;
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            border: 1px solid #444;
            border-radius: 4px;
            padding: 10px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            max-width: 300px;
        }}
        #search {{
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 100;
        }}
        #search input {{
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            background: #2a2a4e;
            color: #fff;
            width: 250px;
        }}
        #search input::placeholder {{
            color: #888;
        }}
    </style>
</head>
<body>
    <div id="search">
        <input type="text" placeholder="Search components..." id="searchInput">
    </div>
    <div id="graph"></div>
    <div id="tooltip"></div>

    <script>
        const data = {graph_json};

        const width = window.innerWidth;
        const height = window.innerHeight;

        // Color by type
        const typeColors = {{
            'Button': '#f9a825',
            'Textbox': '#4caf50',
            'Markdown': '#2196f3',
            'Image': '#9c27b0',
            'Slider': '#00bcd4',
            'Dropdown': '#8bc34a',
            'default': '#607d8b'
        }};

        function getColor(type) {{
            for (const [key, color] of Object.entries(typeColors)) {{
                if (type.includes(key)) return color;
            }}
            return typeColors.default;
        }}

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        // Arrow markers
        svg.append("defs").selectAll("marker")
            .data(["dataflow", "trigger"])
            .enter().append("marker")
            .attr("id", d => `arrow-${{d}}`)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("fill", d => d === "trigger" ? "#f9a825" : "#4caf50")
            .attr("d", "M0,-5L10,0L0,5");

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", d => `link ${{d.type}}`)
            .attr("marker-end", d => `url(#arrow-${{d.type}})`);

        const node = svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", 12)
            .attr("fill", d => getColor(d.type));

        node.append("text")
            .attr("dx", 15)
            .attr("dy", 4)
            .text(d => d.label);

        // Tooltip
        const tooltip = d3.select("#tooltip");

        node.on("mouseover", function(event, d) {{
            tooltip
                .style("display", "block")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px")
                .html(`
                    <strong>${{d.label}}</strong><br>
                    Type: ${{d.type}}<br>
                    Intent: ${{d.intent}}<br>
                    ID: ${{d.id}}
                `);
        }}).on("mouseout", function() {{
            tooltip.style("display", "none");
        }});

        // Search
        d3.select("#searchInput").on("input", function() {{
            const query = this.value.toLowerCase();
            node.style("opacity", d => {{
                if (!query) return 1;
                const match = d.label.toLowerCase().includes(query) ||
                              d.intent.toLowerCase().includes(query) ||
                              d.type.toLowerCase().includes(query);
                return match ? 1 : 0.2;
            }});
        }});

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>"""


def generate_ascii_graph(blocks: "SemanticBlocks", max_width: int = 80) -> str:
    """
    Generate ASCII representation of component graph.

    Args:
        blocks: SemanticBlocks instance
        max_width: Maximum line width

    Returns:
        ASCII art representation
    """
    graph = _validate_graph(blocks.map())
    components = blocks.registry.all_components()

    if not components:
        return "No components registered"

    lines = [
        "=" * max_width,
        "INTEGRADIO COMPONENT GRAPH".center(max_width),
        "=" * max_width,
        "",
    ]

    # Group by type
    by_type: dict[str, list] = {}
    for meta in components:
        comp_type = meta.component_type or "Unknown"
        if comp_type not in by_type:
            by_type[comp_type] = []
        by_type[comp_type].append(meta)

    for comp_type, metas in sorted(by_type.items()):
        lines.append(f"[{comp_type}]")
        for meta in metas:
            label = meta.label or meta.elem_id or f"id={meta.component_id}"
            # Handle None intent
            intent = meta.intent or ""
            intent_preview = intent[:40] + "..." if len(intent) > 40 else intent
            lines.append(f"  +-- {label}")
            lines.append(f"  |   intent: {intent_preview}")

            # Show connections
            if meta.inputs_from:
                lines.append(f"  |   <- inputs from: {meta.inputs_from}")
            if meta.outputs_to:
                lines.append(f"  |   -> outputs to: {meta.outputs_to}")

            lines.append("  |")

        lines.append("")

    # Show relationships
    if graph["links"]:
        lines.append("-" * max_width)
        lines.append("DATAFLOW:")
        for link in graph["links"]:
            lines.append(f"  {link['source']} --[{link['type']}]--> {link['target']}")

    lines.append("=" * max_width)
    return "\n".join(lines)
