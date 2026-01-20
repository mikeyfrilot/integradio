"""
Dataflow Visualization - Track and visualize data flow between components.

Shows how data moves through a Gradio app:
- Input → Function → Output connections
- Event triggers (click, change, etc.)
- State dependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import json


class EdgeType(str, Enum):
    """Types of dataflow edges."""
    INPUT = "input"       # Component provides input to handler
    OUTPUT = "output"     # Handler writes to component
    TRIGGER = "trigger"   # Component triggers the handler
    STATE = "state"       # State dependency


@dataclass
class DataFlowEdge:
    """An edge in the dataflow graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    event_name: str = ""  # click, change, submit, etc.
    handler_name: str = ""  # Function name

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "event": self.event_name,
            "handler": self.handler_name,
        }


@dataclass
class HandlerInfo:
    """Information about an event handler."""
    name: str
    inputs: list[str]  # Component IDs
    outputs: list[str]  # Component IDs
    trigger_id: str  # Component that triggers this
    event_type: str  # click, change, etc.

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "trigger": self.trigger_id,
            "event": self.event_type,
        }


@dataclass
class DataFlowGraph:
    """
    Complete dataflow graph for a Gradio app.

    Represents how data flows between components through event handlers.
    """
    edges: list[DataFlowEdge] = field(default_factory=list)
    handlers: list[HandlerInfo] = field(default_factory=list)

    # Component lookup
    _components: dict[str, str] = field(default_factory=dict)  # id -> intent

    def add_edge(self, edge: DataFlowEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def add_handler(self, handler: HandlerInfo) -> None:
        """Add a handler and create edges."""
        self.handlers.append(handler)

        # Create trigger edge
        self.add_edge(DataFlowEdge(
            source_id=handler.trigger_id,
            target_id=f"fn:{handler.name}",
            edge_type=EdgeType.TRIGGER,
            event_name=handler.event_type,
            handler_name=handler.name,
        ))

        # Create input edges
        for inp_id in handler.inputs:
            self.add_edge(DataFlowEdge(
                source_id=inp_id,
                target_id=f"fn:{handler.name}",
                edge_type=EdgeType.INPUT,
                handler_name=handler.name,
            ))

        # Create output edges
        for out_id in handler.outputs:
            self.add_edge(DataFlowEdge(
                source_id=f"fn:{handler.name}",
                target_id=out_id,
                edge_type=EdgeType.OUTPUT,
                handler_name=handler.name,
            ))

    def get_inputs_for(self, component_id: str) -> list[str]:
        """Get all components that provide input to a component."""
        result = []
        for edge in self.edges:
            if edge.target_id == component_id and edge.edge_type == EdgeType.OUTPUT:
                # Find what feeds into this handler
                handler_id = edge.source_id
                for e2 in self.edges:
                    if e2.target_id == handler_id and e2.edge_type == EdgeType.INPUT:
                        result.append(e2.source_id)
        return result

    def get_outputs_for(self, component_id: str) -> list[str]:
        """Get all components that receive output from a component."""
        result = []
        for edge in self.edges:
            if edge.source_id == component_id and edge.edge_type == EdgeType.INPUT:
                # Find what this handler outputs to
                handler_id = edge.target_id
                for e2 in self.edges:
                    if e2.source_id == handler_id and e2.edge_type == EdgeType.OUTPUT:
                        result.append(e2.target_id)
        return result

    def trace_forward(self, component_id: str, max_depth: int = 5) -> list[str]:
        """Trace data flow forward from a component."""
        visited = set()
        result = []

        def trace(cid: str, depth: int) -> None:
            if depth > max_depth or cid in visited:
                return
            visited.add(cid)

            outputs = self.get_outputs_for(cid)
            for out_id in outputs:
                result.append(out_id)
                trace(out_id, depth + 1)

        trace(component_id, 0)
        return result

    def trace_backward(self, component_id: str, max_depth: int = 5) -> list[str]:
        """Trace data flow backward to find sources."""
        visited = set()
        result = []

        def trace(cid: str, depth: int) -> None:
            if depth > max_depth or cid in visited:
                return
            visited.add(cid)

            inputs = self.get_inputs_for(cid)
            for inp_id in inputs:
                result.append(inp_id)
                trace(inp_id, depth + 1)

        trace(component_id, 0)
        return result

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "edges": [e.to_dict() for e in self.edges],
            "handlers": [h.to_dict() for h in self.handlers],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def extract_dataflow(blocks: Any) -> DataFlowGraph:
    """
    Extract dataflow graph from a Gradio Blocks instance.

    Args:
        blocks: Gradio Blocks or SemanticBlocks instance

    Returns:
        DataFlowGraph representing all data flows
    """
    graph = DataFlowGraph()

    # Get the underlying Gradio blocks
    gr_blocks = getattr(blocks, "_blocks", blocks)

    # Gradio stores dependencies in the blocks
    if hasattr(gr_blocks, "dependencies"):
        for dep in gr_blocks.dependencies:
            # Extract handler info
            fn = dep.get("fn")
            fn_name = getattr(fn, "__name__", "anonymous") if fn else "anonymous"

            # Get trigger component
            trigger_ids = dep.get("trigger", [])
            trigger_id = str(trigger_ids[0]) if trigger_ids else "unknown"

            # Get event type
            event_type = dep.get("trigger_event", "click")

            # Get inputs
            inputs = dep.get("inputs", [])
            input_ids = [str(getattr(i, "_id", i)) for i in inputs]

            # Get outputs
            outputs = dep.get("outputs", [])
            output_ids = [str(getattr(o, "_id", o)) for o in outputs]

            handler = HandlerInfo(
                name=fn_name,
                inputs=input_ids,
                outputs=output_ids,
                trigger_id=trigger_id,
                event_type=event_type,
            )
            graph.add_handler(handler)

    # Also check for fns attribute (older Gradio versions)
    elif hasattr(gr_blocks, "fns"):
        for fn_info in gr_blocks.fns.values():
            fn = fn_info.get("fn")
            fn_name = getattr(fn, "__name__", "anonymous") if fn else "anonymous"

            inputs = fn_info.get("inputs", [])
            input_ids = [str(getattr(i, "_id", i)) for i in inputs]

            outputs = fn_info.get("outputs", [])
            output_ids = [str(getattr(o, "_id", o)) for o in outputs]

            # Try to determine trigger
            trigger_id = input_ids[0] if input_ids else "unknown"

            handler = HandlerInfo(
                name=fn_name,
                inputs=input_ids,
                outputs=output_ids,
                trigger_id=trigger_id,
                event_type="change",
            )
            graph.add_handler(handler)

    return graph


def dataflow_to_mermaid(graph: DataFlowGraph, direction: str = "LR") -> str:
    """
    Convert dataflow graph to Mermaid diagram.

    Args:
        graph: DataFlowGraph to convert
        direction: Diagram direction (LR, TB, etc.)

    Returns:
        Mermaid diagram string
    """
    lines = [f"flowchart {direction}"]

    # Collect unique nodes
    nodes: set[str] = set()
    for edge in graph.edges:
        nodes.add(edge.source_id)
        nodes.add(edge.target_id)

    # Add node definitions
    for node_id in nodes:
        if node_id.startswith("fn:"):
            # Function node
            fn_name = node_id[3:]
            lines.append(f'    {node_id.replace(":", "_")}{{{{"{fn_name}"}}}}')
        else:
            # Component node
            lines.append(f'    {node_id}["{node_id}"]')

    # Add edges
    for edge in graph.edges:
        source = edge.source_id.replace(":", "_")
        target = edge.target_id.replace(":", "_")

        if edge.edge_type == EdgeType.TRIGGER:
            label = edge.event_name or "trigger"
            lines.append(f'    {source} -->|{label}| {target}')
        elif edge.edge_type == EdgeType.INPUT:
            lines.append(f'    {source} -.->|in| {target}')
        elif edge.edge_type == EdgeType.OUTPUT:
            lines.append(f'    {target} -->|out| {source}')

    # Styling
    lines.append("")
    lines.append("    %% Styling")
    lines.append("    classDef fn fill:#22c55e,color:#fff")

    fn_nodes = [n.replace(":", "_") for n in nodes if n.startswith("fn:")]
    if fn_nodes:
        lines.append(f"    class {','.join(fn_nodes)} fn")

    return "\n".join(lines)


def dataflow_to_ascii(graph: DataFlowGraph) -> str:
    """
    Convert dataflow to ASCII representation.

    Returns:
        ASCII diagram string
    """
    lines = ["Data Flow:"]
    lines.append("=" * 40)

    for handler in graph.handlers:
        # Show inputs
        if handler.inputs:
            inputs_str = ", ".join(handler.inputs)
            lines.append(f"  [{inputs_str}]")
            lines.append("        │")
            lines.append("        ▼")

        # Show handler
        lines.append(f"  ┌─────────────────────┐")
        lines.append(f"  │ {handler.name:<19} │ ← {handler.event} on {handler.trigger}")
        lines.append(f"  └─────────────────────┘")

        # Show outputs
        if handler.outputs:
            lines.append("        │")
            lines.append("        ▼")
            outputs_str = ", ".join(handler.outputs)
            lines.append(f"  [{outputs_str}]")

        lines.append("")

    return "\n".join(lines)
