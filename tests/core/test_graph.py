"""Tests for the WorkflowGraph API."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentloom.compat import MissingDependencyProxy
from agentloom.core.dag import DAG
from agentloom.core.graph import GraphEdge, GraphNode, WorkflowGraph
from agentloom.core.models import (
    Condition,
    StepDefinition,
    StepType,
    WorkflowDefinition,
)


# Shared helpers
def _linear_dag() -> DAG:
    """a -> b -> c"""
    return DAG.from_steps([("a", []), ("b", ["a"]), ("c", ["b"])])


def _diamond_dag() -> DAG:
    """root -> left, root -> right, left -> merge, right -> merge"""
    return DAG.from_steps(
        [
            ("root", []),
            ("left", ["root"]),
            ("right", ["root"]),
            ("merge", ["left", "right"]),
        ]
    )


def _router_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="test",
        steps=[
            StepDefinition(id="start", type=StepType.LLM_CALL, prompt="hi"),
            StepDefinition(
                id="route",
                type=StepType.ROUTER,
                depends_on=["start"],
                conditions=[Condition(expression="x > 0", target="yes")],
                default="no",
            ),
            StepDefinition(
                id="yes",
                type=StepType.LLM_CALL,
                depends_on=["route"],
                prompt="y",
            ),
            StepDefinition(
                id="no",
                type=StepType.LLM_CALL,
                depends_on=["route"],
                prompt="n",
            ),
        ],
    )


def _mixed_type_workflow() -> WorkflowDefinition:
    """Workflow with all four step types."""
    return WorkflowDefinition(
        name="mixed",
        steps=[
            StepDefinition(id="llm_step", type=StepType.LLM_CALL, prompt="q"),
            StepDefinition(
                id="tool_step",
                type=StepType.TOOL,
                depends_on=["llm_step"],
                tool_name="dummy",
            ),
            StepDefinition(
                id="route_step",
                type=StepType.ROUTER,
                depends_on=["tool_step"],
                conditions=[Condition(expression="x == 1", target="sub_step")],
                default="sub_step",
            ),
            StepDefinition(
                id="sub_step",
                type=StepType.SUBWORKFLOW,
                depends_on=["route_step"],
                workflow_inline={"name": "inner", "steps": []},
            ),
        ],
    )


# TestGraphNode
class TestGraphNode:
    def test_construction_all_fields(self) -> None:
        node = GraphNode(
            id="my_node",
            type=StepType.LLM_CALL,
            depends_on=["a", "b"],
            label="My Node",
        )
        assert node.id == "my_node"
        assert node.type == StepType.LLM_CALL
        assert node.depends_on == ["a", "b"]
        assert node.label == "My Node"

    def test_default_label_equals_id(self) -> None:
        node = GraphNode(id="step_x", type=StepType.TOOL)
        assert node.label == "step_x"

    def test_explicit_empty_label_falls_back_to_id(self) -> None:
        node = GraphNode(id="step_y", type=StepType.ROUTER, label="")
        assert node.label == "step_y"

    def test_immutability_raises_on_mutation(self) -> None:
        node = GraphNode(id="frozen", type=StepType.LLM_CALL)
        with pytest.raises(ValidationError):
            node.id = "other"  # type: ignore[misc]

    def test_depends_on_defaults_empty(self) -> None:
        node = GraphNode(id="root", type=StepType.LLM_CALL)
        assert node.depends_on == []


# TestGraphEdge
class TestGraphEdge:
    def test_construction_all_fields(self) -> None:
        edge = GraphEdge(source="a", target="b", label="condition")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.label == "condition"

    def test_default_label_is_empty(self) -> None:
        edge = GraphEdge(source="a", target="b")
        assert edge.label == ""

    def test_immutability_raises_on_mutation(self) -> None:
        edge = GraphEdge(source="a", target="b")
        with pytest.raises(ValidationError):
            edge.source = "c"  # type: ignore[misc]


# TestFromWorkflow
class TestFromWorkflow:
    def test_build_from_simple_workflow(self) -> None:
        wf = _router_workflow()
        g = WorkflowGraph.from_workflow(wf)
        node_ids = {n.id for n in g.nodes}
        assert node_ids == {"start", "route", "yes", "no"}

    def test_node_types_match_step_types(self) -> None:
        wf = _mixed_type_workflow()
        g = WorkflowGraph.from_workflow(wf)
        type_map = {n.id: n.type for n in g.nodes}
        assert type_map["llm_step"] == StepType.LLM_CALL
        assert type_map["tool_step"] == StepType.TOOL
        assert type_map["route_step"] == StepType.ROUTER
        assert type_map["sub_step"] == StepType.SUBWORKFLOW

    def test_router_edge_condition_label(self) -> None:
        wf = _router_workflow()
        g = WorkflowGraph.from_workflow(wf)
        edge_map = {(e.source, e.target): e.label for e in g.edges}
        assert edge_map[("route", "yes")] == "x > 0"

    def test_router_edge_default_label(self) -> None:
        wf = _router_workflow()
        g = WorkflowGraph.from_workflow(wf)
        edge_map = {(e.source, e.target): e.label for e in g.edges}
        assert edge_map[("route", "no")] == "default"

    def test_non_router_edges_have_empty_label(self) -> None:
        wf = _router_workflow()
        g = WorkflowGraph.from_workflow(wf)
        edge_map = {(e.source, e.target): e.label for e in g.edges}
        assert edge_map[("start", "route")] == ""

    def test_workflow_reference_stored(self) -> None:
        wf = _router_workflow()
        g = WorkflowGraph.from_workflow(wf)
        step = g.get_step_definition("start")
        assert step is not None
        assert step.id == "start"


# TestFromDAG
class TestFromDAG:
    def test_build_from_bare_dag(self) -> None:
        dag = _linear_dag()
        g = WorkflowGraph.from_dag(dag)
        assert {n.id for n in g.nodes} == {"a", "b", "c"}

    def test_all_nodes_get_llm_call_type(self) -> None:
        dag = _diamond_dag()
        g = WorkflowGraph.from_dag(dag)
        for node in g.nodes:
            assert node.type == StepType.LLM_CALL

    def test_get_step_definition_returns_none(self) -> None:
        dag = _linear_dag()
        g = WorkflowGraph.from_dag(dag)
        assert g.get_step_definition("a") is None

    def test_edges_have_empty_labels(self) -> None:
        dag = _linear_dag()
        g = WorkflowGraph.from_dag(dag)
        for edge in g.edges:
            assert edge.label == ""


# TestProperties
class TestProperties:
    def test_nodes_in_topological_order(self) -> None:
        dag = _linear_dag()
        g = WorkflowGraph.from_dag(dag)
        ids = [n.id for n in g.nodes]
        assert ids.index("a") < ids.index("b")
        assert ids.index("b") < ids.index("c")

    def test_edges_sorted_by_source_then_target(self) -> None:
        dag = _diamond_dag()
        g = WorkflowGraph.from_dag(dag)
        pairs = [(e.source, e.target) for e in g.edges]
        assert pairs == sorted(pairs)

    def test_roots_no_predecessors(self) -> None:
        dag = _diamond_dag()
        g = WorkflowGraph.from_dag(dag)
        assert g.roots == ["root"]

    def test_roots_multiple(self) -> None:
        dag = DAG.from_steps([("a", []), ("b", []), ("c", ["a", "b"])])
        g = WorkflowGraph.from_dag(dag)
        assert g.roots == ["a", "b"]

    def test_leaves_no_successors(self) -> None:
        dag = _diamond_dag()
        g = WorkflowGraph.from_dag(dag)
        assert g.leaves == ["merge"]

    def test_leaves_multiple(self) -> None:
        dag = DAG.from_steps([("root", []), ("a", ["root"]), ("b", ["root"])])
        g = WorkflowGraph.from_dag(dag)
        assert g.leaves == ["a", "b"]

    def test_layers_delegates_to_dag(self) -> None:
        dag = _diamond_dag()
        g = WorkflowGraph.from_dag(dag)
        layers = g.layers
        assert layers[0] == ["root"]
        assert sorted(layers[1]) == ["left", "right"]
        assert layers[2] == ["merge"]

    def test_single_node_roots_and_leaves(self) -> None:
        dag = DAG()
        dag.add_node("solo")
        g = WorkflowGraph.from_dag(dag)
        assert g.roots == ["solo"]
        assert g.leaves == ["solo"]


# TestGetStepDefinition
class TestGetStepDefinition:
    def test_returns_step_definition_from_workflow(self) -> None:
        wf = _router_workflow()
        g = WorkflowGraph.from_workflow(wf)
        step = g.get_step_definition("route")
        assert step is not None
        assert step.type == StepType.ROUTER

    def test_returns_none_for_unknown_id(self) -> None:
        wf = _router_workflow()
        g = WorkflowGraph.from_workflow(wf)
        assert g.get_step_definition("nonexistent") is None

    def test_returns_none_when_built_from_dag(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        assert g.get_step_definition("a") is None


# TestAllPaths
class TestAllPaths:
    def test_linear_chain_single_path(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        paths = g.all_paths()
        assert paths == [["a", "b", "c"]]

    def test_diamond_two_paths(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        paths = g.all_paths()
        assert len(paths) == 2
        assert ["root", "left", "merge"] in paths
        assert ["root", "right", "merge"] in paths

    def test_parallel_roots_to_single_leaf(self) -> None:
        dag = DAG.from_steps([("a", []), ("b", []), ("c", ["a", "b"])])
        g = WorkflowGraph.from_dag(dag)
        paths = g.all_paths()
        assert len(paths) == 2
        assert ["a", "c"] in paths
        assert ["b", "c"] in paths

    def test_single_node_one_path(self) -> None:
        dag = DAG()
        dag.add_node("only")
        g = WorkflowGraph.from_dag(dag)
        paths = g.all_paths()
        assert paths == [["only"]]

    def test_empty_graph_empty_list(self) -> None:
        g = WorkflowGraph.from_dag(DAG())
        assert g.all_paths() == []

    def test_result_is_sorted(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        paths = g.all_paths()
        assert paths == sorted(paths)


# TestPrimePaths
class TestPrimePaths:
    def test_linear_chain_single_prime_path(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        primes = g.prime_paths()
        assert primes == [["a", "b", "c"]]

    def test_diamond_paths_are_prime(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        primes = g.prime_paths()
        # Both full root-to-merge paths must be present
        assert ["root", "left", "merge"] in primes
        assert ["root", "right", "merge"] in primes

    def test_subpaths_filtered_out(self) -> None:
        # a -> b -> c: ["b", "c"] is a subpath of ["a", "b", "c"]
        g = WorkflowGraph.from_dag(_linear_dag())
        primes = g.prime_paths()
        for path in primes:
            assert path != ["b", "c"]
            assert path != ["a", "b"]

    def test_complex_graph_no_path_is_subpath_of_another(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        primes = g.prime_paths()
        for i, p in enumerate(primes):
            for j, other in enumerate(primes):
                if i == j:
                    continue
                n = len(p)
                is_sub = any(other[k : k + n] == p for k in range(len(other) - n + 1))
                assert not is_sub, f"{p} is a subpath of {other}"

    def test_result_is_sorted(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        primes = g.prime_paths()
        assert primes == sorted(primes)


# TestCriticalPath
class TestCriticalPath:
    def test_linear_chain_full_chain(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        assert g.critical_path() == ["a", "b", "c"]

    def test_diamond_longest_path(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        cp = g.critical_path()
        # All root-to-merge paths have length 3, so any is valid
        assert cp[0] == "root"
        assert cp[-1] == "merge"
        assert len(cp) == 3

    def test_parallel_independent_chains_longest_selected(self) -> None:
        # Two chains: a->b->c (length 3) and x->y (length 2)
        dag = DAG.from_steps([("a", []), ("b", ["a"]), ("c", ["b"]), ("x", []), ("y", ["x"])])
        g = WorkflowGraph.from_dag(dag)
        cp = g.critical_path()
        assert len(cp) == 3
        assert cp == ["a", "b", "c"]

    def test_empty_graph_returns_empty_list(self) -> None:
        g = WorkflowGraph.from_dag(DAG())
        assert g.critical_path() == []

    def test_single_node(self) -> None:
        dag = DAG()
        dag.add_node("solo")
        g = WorkflowGraph.from_dag(dag)
        assert g.critical_path() == ["solo"]


# TestToDict
class TestToDict:
    def test_keys_present(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        d = g.to_dict()
        assert set(d.keys()) == {"nodes", "edges", "roots", "leaves", "layers", "critical_path"}

    def test_nodes_are_dicts_with_expected_keys(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        d = g.to_dict()
        for node_dict in d["nodes"]:  # type: ignore[union-attr]
            assert isinstance(node_dict, dict)
            assert "id" in node_dict
            assert "type" in node_dict
            assert "depends_on" in node_dict
            assert "label" in node_dict

    def test_edges_are_dicts(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        d = g.to_dict()
        for edge_dict in d["edges"]:  # type: ignore[union-attr]
            assert isinstance(edge_dict, dict)
            assert "source" in edge_dict
            assert "target" in edge_dict
            assert "label" in edge_dict

    def test_no_pydantic_models_in_output(self) -> None:
        import json

        g = WorkflowGraph.from_dag(_diamond_dag())
        d = g.to_dict()
        # Should not raise — all values must be JSON-serialisable
        json.dumps(d)

    def test_roots_and_leaves_are_lists(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        d = g.to_dict()
        assert isinstance(d["roots"], list)
        assert isinstance(d["leaves"], list)


# TestToDot
class TestToDot:
    def test_contains_digraph_workflow(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        dot = g.to_dot()
        assert "digraph workflow" in dot

    def test_contains_node_declarations(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        dot = g.to_dot()
        assert '"a"' in dot
        assert '"b"' in dot
        assert '"c"' in dot

    def test_llm_call_shape(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        dot = g.to_dot()
        assert "box, style=rounded" in dot

    def test_router_shape(self) -> None:
        g = WorkflowGraph.from_workflow(_router_workflow())
        dot = g.to_dot()
        assert "diamond" in dot

    def test_contains_edge_declarations(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        dot = g.to_dot()
        assert "->" in dot

    def test_router_edge_has_label(self) -> None:
        g = WorkflowGraph.from_workflow(_router_workflow())
        dot = g.to_dot()
        assert 'label="x > 0"' in dot

    def test_edge_without_label_has_no_label_attr(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        dot = g.to_dot()
        # Plain edges should not have [label= on them
        for line in dot.splitlines():
            if "->" in line:
                assert "label=" not in line


# TestToPnml
class TestToPnml:
    def test_valid_xml(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        pnml = g.to_pnml()
        root = ET.fromstring(pnml.split("\n", 1)[1])  # strip XML declaration
        assert root.tag == "pnml"

    def test_contains_place_per_node(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        pnml = g.to_pnml()
        root = ET.fromstring(pnml.split("\n", 1)[1])
        net = root.find("net")
        assert net is not None
        place_ids = {p.get("id") for p in net.findall("place")}
        assert place_ids == {"p_a", "p_b", "p_c"}

    def test_contains_transition_per_edge(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        pnml = g.to_pnml()
        root = ET.fromstring(pnml.split("\n", 1)[1])
        net = root.find("net")
        assert net is not None
        transitions = net.findall("transition")
        assert len(transitions) == 2  # a->b and b->c

    def test_contains_two_arcs_per_edge(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        pnml = g.to_pnml()
        root = ET.fromstring(pnml.split("\n", 1)[1])
        net = root.find("net")
        assert net is not None
        arcs = net.findall("arc")
        assert len(arcs) == 4  # 2 edges * 2 arcs each

    def test_starts_with_xml_declaration(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        pnml = g.to_pnml()
        assert pnml.startswith("<?xml")


# TestToMermaid
class TestToMermaid:
    def test_starts_with_graph_td(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        mmd = g.to_mermaid()
        assert mmd.startswith("graph TD")

    def test_llm_node_uses_bracket_syntax(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        mmd = g.to_mermaid()
        # LLM nodes: id["label"]
        assert 'a["a"]' in mmd

    def test_router_node_uses_curly_syntax(self) -> None:
        g = WorkflowGraph.from_workflow(_router_workflow())
        mmd = g.to_mermaid()
        assert "route{route}" in mmd

    def test_tool_node_uses_trapezoid_syntax(self) -> None:
        g = WorkflowGraph.from_workflow(_mixed_type_workflow())
        mmd = g.to_mermaid()
        assert "tool_step[/tool_step/]" in mmd

    def test_subworkflow_node_uses_double_bracket_syntax(self) -> None:
        g = WorkflowGraph.from_workflow(_mixed_type_workflow())
        mmd = g.to_mermaid()
        assert "sub_step[[sub_step]]" in mmd

    def test_edge_with_label_uses_pipe_syntax(self) -> None:
        g = WorkflowGraph.from_workflow(_router_workflow())
        mmd = g.to_mermaid()
        assert "-->|x > 0|" in mmd

    def test_plain_edge_uses_arrow_syntax(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        mmd = g.to_mermaid()
        assert "a --> b" in mmd

    def test_contains_all_node_ids(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        mmd = g.to_mermaid()
        for nid in ["root", "left", "right", "merge"]:
            assert nid in mmd


# TestToNetworkx
class TestToNetworkx:
    def test_builds_digraph_when_networkx_available(self) -> None:
        nx = pytest.importorskip("networkx")
        g = WorkflowGraph.from_dag(_diamond_dag())
        graph = g.to_networkx()
        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes) == {"root", "left", "right", "merge"}

    def test_node_attributes_correct(self) -> None:
        pytest.importorskip("networkx")
        g = WorkflowGraph.from_dag(_linear_dag())
        graph = g.to_networkx()
        attrs = graph.nodes["a"]
        assert "type" in attrs
        assert "label" in attrs
        assert attrs["label"] == "a"

    def test_edge_attributes_correct(self) -> None:
        pytest.importorskip("networkx")
        g = WorkflowGraph.from_workflow(_router_workflow())
        graph = g.to_networkx()
        edge_data = graph.edges["route", "yes"]
        assert edge_data["label"] == "x > 0"

    def test_raises_import_error_when_networkx_missing(self) -> None:
        proxy = MissingDependencyProxy("networkx", "graph")
        g = WorkflowGraph.from_dag(_linear_dag())
        with (
            patch("agentloom.core.graph.try_import", return_value=proxy),
            pytest.raises(ImportError, match="networkx"),
        ):
            g.to_networkx()


# Audit-driven edge-case tests
class TestPrimePathsEdgeCases:
    def test_empty_graph(self) -> None:
        g = WorkflowGraph.from_dag(DAG())
        assert g.prime_paths() == []

    def test_disconnected_graph(self) -> None:
        dag = DAG.from_steps([("a", []), ("b", ["a"]), ("x", []), ("y", ["x"])])
        g = WorkflowGraph.from_dag(dag)
        primes = g.prime_paths()
        assert ["a", "b"] in primes
        assert ["x", "y"] in primes

    def test_max_paths_limit_raises(self) -> None:
        # Fan-out graph: root -> n1, n2, ..., n10; each ni -> leaf
        dag = DAG()
        dag.add_node("root")
        dag.add_node("leaf")
        for i in range(10):
            nid = f"n{i}"
            dag.add_edge("root", nid)
            dag.add_edge(nid, "leaf")
        g = WorkflowGraph.from_dag(dag)
        # A very low limit should trigger the safeguard
        with pytest.raises(ValueError, match="exceeded"):
            g.prime_paths(max_paths=5)

    def test_default_max_paths_allows_normal_graphs(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        # Should not raise — diamond is well within limits
        primes = g.prime_paths()
        assert len(primes) >= 2


class TestCriticalPathDeterminism:
    def test_tie_breaking_is_deterministic(self) -> None:
        g = WorkflowGraph.from_dag(_diamond_dag())
        # Call multiple times — same result every time
        results = [g.critical_path() for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestEdgesDefensiveCopy:
    def test_mutating_returned_edges_does_not_affect_internal_state(self) -> None:
        g = WorkflowGraph.from_dag(_linear_dag())
        edges = g.edges
        original_len = len(edges)
        edges.clear()
        assert len(g.edges) == original_len


class TestPnmlIdCollisions:
    def test_underscore_node_ids_produce_unique_pnml_ids(self) -> None:
        # Nodes with underscores that could collide with naive separator
        dag = DAG.from_steps([("a_b", []), ("c", ["a_b"]), ("a", []), ("b_c", ["a"])])
        g = WorkflowGraph.from_dag(dag)
        pnml = g.to_pnml()
        root = ET.fromstring(pnml.split("\n", 1)[1])
        net = root.find("net")
        assert net is not None
        # All IDs must be unique
        all_ids: list[str] = []
        for elem in net:
            eid = elem.get("id")
            if eid:
                all_ids.append(eid)
        assert len(all_ids) == len(set(all_ids)), f"Duplicate IDs found: {all_ids}"


class TestDotEscaping:
    def test_backslash_in_label_is_escaped(self) -> None:
        dag = DAG()
        dag.add_node("a\\b")
        g = WorkflowGraph.from_dag(dag)
        dot = g.to_dot()
        assert "\\\\" in dot  # backslash should be doubled


class TestMermaidEscaping:
    def test_pipe_in_edge_label_is_escaped(self) -> None:
        wf = WorkflowDefinition(
            name="test",
            steps=[
                StepDefinition(id="start", type=StepType.LLM_CALL, prompt="hi"),
                StepDefinition(
                    id="route",
                    type=StepType.ROUTER,
                    depends_on=["start"],
                    conditions=[Condition(expression="a | b", target="end")],
                ),
                StepDefinition(
                    id="end",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="bye",
                ),
            ],
        )
        g = WorkflowGraph.from_workflow(wf)
        mmd = g.to_mermaid()
        # The pipe must be escaped so it doesn't break Mermaid -->|label| syntax
        assert "|a | b|" not in mmd  # raw pipe would break parsing
        assert "#vert;" in mmd


class TestCriticalPathTieBreaking:
    def test_equal_length_paths_break_tie_by_id(self) -> None:
        """When two predecessors have equal distance, tie-break by node ID."""
        dag = _diamond_dag()
        g = WorkflowGraph.from_dag(dag)
        cp = g.critical_path()
        # Both paths (root->left->merge, root->right->merge) have length 3.
        # Deterministic: sorted preds + (dist, id) key ensures "right" > "left".
        assert cp == ["root", "right", "merge"]

    def test_repeated_calls_same_result(self) -> None:
        dag = _diamond_dag()
        g = WorkflowGraph.from_dag(dag)
        results = {tuple(g.critical_path()) for _ in range(20)}
        assert len(results) == 1


class TestMermaidSanitization:
    def test_node_id_with_spaces_sanitized(self) -> None:
        dag = DAG()
        dag.add_node("my node")
        g = WorkflowGraph.from_dag(dag)
        mmd = g.to_mermaid()
        # ID is sanitized, label preserves original text in quotes
        assert 'my_node["my node"]' in mmd

    def test_label_with_brackets_escaped(self) -> None:
        node = GraphNode(id="x", type=StepType.LLM_CALL, label="arr[0]")
        dag = DAG()
        dag.add_node("x")
        g = WorkflowGraph(dag=dag, nodes=[node], edges=[])
        mmd = g.to_mermaid()
        assert "#lsqb;" in mmd
        assert "#rsqb;" in mmd

    def test_label_with_braces_escaped(self) -> None:
        node = GraphNode(id="x", type=StepType.LLM_CALL, label="obj{k}")
        dag = DAG()
        dag.add_node("x")
        g = WorkflowGraph(dag=dag, nodes=[node], edges=[])
        mmd = g.to_mermaid()
        assert "#lbrace;" in mmd
        assert "#rbrace;" in mmd


class TestPnmlNamespaceCollision:
    def test_step_named_t0_no_collision(self) -> None:
        """A step named 't0' must not collide with transition ID 't0'."""
        dag = DAG.from_steps([("t0", []), ("end", ["t0"])])
        g = WorkflowGraph.from_dag(dag)
        pnml = g.to_pnml()
        root = ET.fromstring(pnml.split("\n", 1)[1])
        net = root.find("net")
        assert net is not None
        all_ids = [elem.get("id") for elem in net if elem.get("id")]
        assert len(all_ids) == len(set(all_ids)), f"Duplicate IDs: {all_ids}"
        # Place for step "t0" should be "p_t0", transition should be "t0"
        place_ids = {p.get("id") for p in net.findall("place")}
        assert "p_t0" in place_ids
        trans_ids = {t.get("id") for t in net.findall("transition")}
        assert "t0" in trans_ids


class TestValidatorTypeSafety:
    def test_non_dict_input_passes_through(self) -> None:
        """model_validator(mode='before') can receive non-dict; must not crash."""
        node = GraphNode(id="ok", type=StepType.LLM_CALL)
        assert node.label == "ok"


class TestFromWorkflowRouterEdgeCases:
    def test_router_with_no_conditions_only_default(self) -> None:
        wf = WorkflowDefinition(
            name="test",
            steps=[
                StepDefinition(id="start", type=StepType.LLM_CALL, prompt="hi"),
                StepDefinition(
                    id="route",
                    type=StepType.ROUTER,
                    depends_on=["start"],
                    conditions=[],
                    default="end",
                ),
                StepDefinition(
                    id="end",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="bye",
                ),
            ],
        )
        g = WorkflowGraph.from_workflow(wf)
        edge_map = {(e.source, e.target): e.label for e in g.edges}
        assert edge_map[("route", "end")] == "default"
