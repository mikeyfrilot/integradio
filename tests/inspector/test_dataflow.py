"""
Tests for Dataflow module.

Tests:
- DataFlowEdge creation
- DataFlowGraph building and traversal
- Forward/backward tracing
- Mermaid export
"""

import pytest
from integradio.inspector.dataflow import (
    DataFlowEdge,
    DataFlowGraph,
    HandlerInfo,
    EdgeType,
    dataflow_to_mermaid,
)


class TestDataFlowEdge:
    """Tests for DataFlowEdge dataclass."""

    def test_create_edge(self):
        """Test basic edge creation."""
        edge = DataFlowEdge(
            source_id="input1",
            target_id="fn:process",
            edge_type=EdgeType.INPUT,
        )

        assert edge.source_id == "input1"
        assert edge.target_id == "fn:process"
        assert edge.edge_type == EdgeType.INPUT

    def test_edge_with_metadata(self):
        """Test edge with event and handler info."""
        edge = DataFlowEdge(
            source_id="button",
            target_id="fn:submit",
            edge_type=EdgeType.TRIGGER,
            event_name="click",
            handler_name="submit_form",
        )

        assert edge.event_name == "click"
        assert edge.handler_name == "submit_form"

    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = DataFlowEdge(
            source_id="src",
            target_id="dst",
            edge_type=EdgeType.OUTPUT,
            event_name="change",
            handler_name="update",
        )

        data = edge.to_dict()

        assert data["source"] == "src"
        assert data["target"] == "dst"
        assert data["type"] == "output"
        assert data["event"] == "change"

    def test_all_edge_types(self):
        """Test all edge type values."""
        assert EdgeType.INPUT.value == "input"
        assert EdgeType.OUTPUT.value == "output"
        assert EdgeType.TRIGGER.value == "trigger"
        assert EdgeType.STATE.value == "state"


class TestHandlerInfo:
    """Tests for HandlerInfo dataclass."""

    def test_create_handler(self):
        """Test handler creation."""
        handler = HandlerInfo(
            name="process_input",
            inputs=["inp1", "inp2"],
            outputs=["out1"],
            trigger_id="btn",
            event_type="click",
        )

        assert handler.name == "process_input"
        assert len(handler.inputs) == 2
        assert len(handler.outputs) == 1
        assert handler.trigger_id == "btn"
        assert handler.event_type == "click"

    def test_handler_to_dict(self):
        """Test handler serialization."""
        handler = HandlerInfo(
            name="submit",
            inputs=["form"],
            outputs=["result", "status"],
            trigger_id="submit_btn",
            event_type="click",
        )

        data = handler.to_dict()

        assert data["name"] == "submit"
        assert data["inputs"] == ["form"]
        assert data["outputs"] == ["result", "status"]


class TestDataFlowGraph:
    """Tests for DataFlowGraph."""

    def test_create_empty_graph(self):
        """Test creating empty graph."""
        graph = DataFlowGraph()

        assert len(graph.edges) == 0
        assert len(graph.handlers) == 0

    def test_add_edge(self):
        """Test adding edges."""
        graph = DataFlowGraph()

        edge = DataFlowEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.INPUT,
        )
        graph.add_edge(edge)

        assert len(graph.edges) == 1
        assert graph.edges[0].source_id == "a"

    def test_add_handler_creates_edges(self):
        """Test that adding handler creates appropriate edges."""
        graph = DataFlowGraph()

        handler = HandlerInfo(
            name="process",
            inputs=["inp1", "inp2"],
            outputs=["out1"],
            trigger_id="btn",
            event_type="click",
        )
        graph.add_handler(handler)

        assert len(graph.handlers) == 1

        # Should create edges:
        # - trigger edge (btn -> fn:process)
        # - input edges (inp1 -> fn:process, inp2 -> fn:process)
        # - output edge (fn:process -> out1)
        assert len(graph.edges) == 4

        # Check trigger edge
        trigger_edges = [e for e in graph.edges if e.edge_type == EdgeType.TRIGGER]
        assert len(trigger_edges) == 1
        assert trigger_edges[0].source_id == "btn"

        # Check input edges
        input_edges = [e for e in graph.edges if e.edge_type == EdgeType.INPUT]
        assert len(input_edges) == 2

        # Check output edge
        output_edges = [e for e in graph.edges if e.edge_type == EdgeType.OUTPUT]
        assert len(output_edges) == 1

    def test_get_inputs_for(self):
        """Test getting input sources for a component."""
        graph = DataFlowGraph()

        # Create a handler: input1, input2 -> process -> output1
        graph.add_handler(HandlerInfo(
            name="process",
            inputs=["input1", "input2"],
            outputs=["output1"],
            trigger_id="input1",
            event_type="change",
        ))

        inputs = graph.get_inputs_for("output1")
        assert "input1" in inputs
        assert "input2" in inputs

    def test_get_outputs_for(self):
        """Test getting output destinations for a component."""
        graph = DataFlowGraph()

        # input1 -> process -> output1, output2
        graph.add_handler(HandlerInfo(
            name="process",
            inputs=["input1"],
            outputs=["output1", "output2"],
            trigger_id="btn",
            event_type="click",
        ))

        outputs = graph.get_outputs_for("input1")
        assert "output1" in outputs
        assert "output2" in outputs

    def test_trace_forward(self):
        """Test forward tracing through dataflow."""
        graph = DataFlowGraph()

        # Chain: a -> b -> c
        graph.add_handler(HandlerInfo(
            name="step1",
            inputs=["a"],
            outputs=["b"],
            trigger_id="a",
            event_type="change",
        ))
        graph.add_handler(HandlerInfo(
            name="step2",
            inputs=["b"],
            outputs=["c"],
            trigger_id="b",
            event_type="change",
        ))

        trace = graph.trace_forward("a")
        assert "b" in trace
        assert "c" in trace

    def test_trace_backward(self):
        """Test backward tracing through dataflow."""
        graph = DataFlowGraph()

        # Chain: a -> b -> c
        graph.add_handler(HandlerInfo(
            name="step1",
            inputs=["a"],
            outputs=["b"],
            trigger_id="a",
            event_type="change",
        ))
        graph.add_handler(HandlerInfo(
            name="step2",
            inputs=["b"],
            outputs=["c"],
            trigger_id="b",
            event_type="change",
        ))

        trace = graph.trace_backward("c")
        assert "b" in trace
        assert "a" in trace

    def test_trace_max_depth(self):
        """Test that tracing respects max depth."""
        graph = DataFlowGraph()

        # Long chain: a -> b -> c -> d -> e -> f
        for i in range(5):
            graph.add_handler(HandlerInfo(
                name=f"step{i}",
                inputs=[chr(ord('a') + i)],
                outputs=[chr(ord('a') + i + 1)],
                trigger_id=chr(ord('a') + i),
                event_type="change",
            ))

        # With max_depth=2, should not reach all nodes
        trace = graph.trace_forward("a", max_depth=2)
        assert "b" in trace
        assert "c" in trace
        # May not reach d, e, f

    def test_to_json(self):
        """Test JSON export."""
        graph = DataFlowGraph()
        graph.add_handler(HandlerInfo(
            name="test",
            inputs=["inp"],
            outputs=["out"],
            trigger_id="btn",
            event_type="click",
        ))

        json_str = graph.to_json()

        assert "edges" in json_str
        assert "handlers" in json_str
        assert "test" in json_str


class TestDataflowExport:
    """Tests for dataflow export functions."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample dataflow graph."""
        graph = DataFlowGraph()

        # Search flow: query -> search -> results
        graph.add_handler(HandlerInfo(
            name="search",
            inputs=["query"],
            outputs=["results"],
            trigger_id="search_btn",
            event_type="click",
        ))

        # Filter flow: results, filter -> filter_results -> filtered
        graph.add_handler(HandlerInfo(
            name="filter_results",
            inputs=["results", "filter"],
            outputs=["filtered"],
            trigger_id="filter",
            event_type="change",
        ))

        return graph

    def test_dataflow_to_mermaid(self, sample_graph):
        """Test Mermaid diagram export."""
        mermaid = dataflow_to_mermaid(sample_graph)

        # Check structure
        assert "flowchart" in mermaid

        # Check function nodes use different shape
        assert "{{" in mermaid  # Diamond shape for functions

        # Check edges
        assert "-->" in mermaid
        assert "-.->|in|" in mermaid  # Input edges

        # Check styling
        assert "classDef fn" in mermaid

    def test_dataflow_to_mermaid_direction(self, sample_graph):
        """Test Mermaid diagram with different directions."""
        lr = dataflow_to_mermaid(sample_graph, direction="LR")
        assert "flowchart LR" in lr

        tb = dataflow_to_mermaid(sample_graph, direction="TB")
        assert "flowchart TB" in tb


class TestDataflowEdgeCases:
    """Edge case tests for dataflow module."""

    def test_empty_graph_export(self):
        """Test exporting empty graph."""
        graph = DataFlowGraph()
        mermaid = dataflow_to_mermaid(graph)

        assert "flowchart" in mermaid

    def test_handler_no_inputs(self):
        """Test handler with no inputs."""
        graph = DataFlowGraph()
        graph.add_handler(HandlerInfo(
            name="init",
            inputs=[],
            outputs=["status"],
            trigger_id="load_btn",
            event_type="click",
        ))

        # Should have trigger and output edges only
        assert len([e for e in graph.edges if e.edge_type == EdgeType.INPUT]) == 0
        assert len([e for e in graph.edges if e.edge_type == EdgeType.OUTPUT]) == 1

    def test_handler_no_outputs(self):
        """Test handler with no outputs (side effect only)."""
        graph = DataFlowGraph()
        graph.add_handler(HandlerInfo(
            name="log",
            inputs=["data"],
            outputs=[],
            trigger_id="data",
            event_type="change",
        ))

        assert len([e for e in graph.edges if e.edge_type == EdgeType.OUTPUT]) == 0

    def test_cyclic_flow(self):
        """Test handling of cyclic data flow."""
        graph = DataFlowGraph()

        # Create cycle: a -> b -> a
        graph.add_handler(HandlerInfo(
            name="step1",
            inputs=["a"],
            outputs=["b"],
            trigger_id="a",
            event_type="change",
        ))
        graph.add_handler(HandlerInfo(
            name="step2",
            inputs=["b"],
            outputs=["a"],
            trigger_id="b",
            event_type="change",
        ))

        # Trace should not infinite loop (max_depth protects)
        trace = graph.trace_forward("a", max_depth=5)
        # Should terminate, not hang
        assert len(trace) < 100

    def test_multiple_handlers_same_trigger(self):
        """Test multiple handlers triggered by same component."""
        graph = DataFlowGraph()

        graph.add_handler(HandlerInfo(
            name="handler1",
            inputs=["shared"],
            outputs=["out1"],
            trigger_id="btn",
            event_type="click",
        ))
        graph.add_handler(HandlerInfo(
            name="handler2",
            inputs=["shared"],
            outputs=["out2"],
            trigger_id="btn",
            event_type="click",
        ))

        # Both outputs should be reachable from shared
        outputs = graph.get_outputs_for("shared")
        assert "out1" in outputs
        assert "out2" in outputs

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency graph."""
        graph = DataFlowGraph()

        # a -> b, a -> c, b -> d, c -> d (diamond)
        graph.add_handler(HandlerInfo(
            name="split1", inputs=["a"], outputs=["b"], trigger_id="a", event_type="change"))
        graph.add_handler(HandlerInfo(
            name="split2", inputs=["a"], outputs=["c"], trigger_id="a", event_type="change"))
        graph.add_handler(HandlerInfo(
            name="join1", inputs=["b"], outputs=["d"], trigger_id="b", event_type="change"))
        graph.add_handler(HandlerInfo(
            name="join2", inputs=["c"], outputs=["d"], trigger_id="c", event_type="change"))

        # Trace forward from a should reach all
        forward = graph.trace_forward("a")
        assert "b" in forward
        assert "c" in forward
        assert "d" in forward

        # Trace backward from d should reach a
        backward = graph.trace_backward("d")
        assert "a" in backward
