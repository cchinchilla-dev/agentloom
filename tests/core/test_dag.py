"""Tests for the DAG (Directed Acyclic Graph) module."""

from __future__ import annotations

import pytest

from agentloom.core.dag import DAG
from agentloom.exceptions import ValidationError


class TestDAGConstruction:
    """Test DAG node and edge construction."""

    def test_add_node(self) -> None:
        dag = DAG()
        dag.add_node("a")
        assert "a" in dag.nodes

    def test_add_multiple_nodes(self) -> None:
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        assert dag.nodes == {"a", "b", "c"}

    def test_add_duplicate_node(self) -> None:
        dag = DAG()
        dag.add_node("a")
        dag.add_node("a")
        assert dag.nodes == {"a"}

    def test_add_edge_creates_nodes(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        assert "a" in dag.nodes
        assert "b" in dag.nodes

    def test_predecessors(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("c", "b")
        assert dag.predecessors("b") == {"a", "c"}

    def test_predecessors_empty(self) -> None:
        dag = DAG()
        dag.add_node("a")
        assert dag.predecessors("a") == set()

    def test_successors(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        assert dag.successors("a") == {"b", "c"}

    def test_successors_empty(self) -> None:
        dag = DAG()
        dag.add_node("a")
        assert dag.successors("a") == set()

    def test_from_steps(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", ["a"]),
                ("c", ["a"]),
                ("d", ["b", "c"]),
            ]
        )
        assert dag.nodes == {"a", "b", "c", "d"}
        assert dag.predecessors("d") == {"b", "c"}
        assert dag.successors("a") == {"b", "c"}


class TestTopologicalSort:
    """Test topological sort functionality."""

    def test_linear_chain(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", ["a"]),
                ("c", ["b"]),
            ]
        )
        order = dag.topological_sort()
        assert order == ["a", "b", "c"]

    def test_parallel_steps(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", []),
                ("c", ["a", "b"]),
            ]
        )
        order = dag.topological_sort()
        # a and b have no deps, should come first (sorted alphabetically)
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_single_node(self) -> None:
        dag = DAG()
        dag.add_node("a")
        assert dag.topological_sort() == ["a"]

    def test_empty_dag(self) -> None:
        dag = DAG()
        assert dag.topological_sort() == []

    def test_cycle_raises_validation_error(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(ValidationError, match="cycle"):
            dag.topological_sort()


class TestCycleDetection:
    """Test cycle detection in the DAG."""

    def test_no_cycle(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", ["a"]),
                ("c", ["b"]),
            ]
        )
        errors = dag.validate()
        assert errors == []

    def test_simple_cycle(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        errors = dag.validate()
        assert len(errors) > 0
        assert any("Cycle" in e or "cycle" in e.lower() for e in errors)

    def test_three_node_cycle(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        errors = dag.validate()
        assert len(errors) > 0

    def test_self_loop(self) -> None:
        dag = DAG()
        dag.add_node("a")
        dag.add_edge("a", "a")
        errors = dag.validate()
        assert len(errors) > 0


class TestExecutionLayers:
    """Test execution layer computation for parallel execution."""

    def test_linear_chain_produces_one_step_per_layer(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", ["a"]),
                ("c", ["b"]),
            ]
        )
        layers = dag.execution_layers()
        assert layers == [["a"], ["b"], ["c"]]

    def test_parallel_steps_in_same_layer(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", []),
                ("c", ["a", "b"]),
            ]
        )
        layers = dag.execution_layers()
        assert len(layers) == 2
        assert sorted(layers[0]) == ["a", "b"]
        assert layers[1] == ["c"]

    def test_diamond_dependency(self) -> None:
        dag = DAG.from_steps(
            [
                ("root", []),
                ("left", ["root"]),
                ("right", ["root"]),
                ("merge", ["left", "right"]),
            ]
        )
        layers = dag.execution_layers()
        assert layers[0] == ["root"]
        assert sorted(layers[1]) == ["left", "right"]
        assert layers[2] == ["merge"]

    def test_empty_dag(self) -> None:
        dag = DAG()
        layers = dag.execution_layers()
        assert layers == []

    def test_single_node(self) -> None:
        dag = DAG()
        dag.add_node("only")
        layers = dag.execution_layers()
        assert layers == [["only"]]

    def test_cycle_raises_validation_error(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(ValidationError):
            dag.execution_layers()


class TestGetReadyNodes:
    """Test get_ready_nodes for incremental execution."""

    def test_initial_ready_nodes(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", []),
                ("c", ["a", "b"]),
            ]
        )
        ready = dag.get_ready_nodes(completed=set())
        assert sorted(ready) == ["a", "b"]

    def test_after_partial_completion(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", []),
                ("c", ["a", "b"]),
            ]
        )
        ready = dag.get_ready_nodes(completed={"a"})
        # b is still ready, c needs both a and b
        assert ready == ["b"]

    def test_after_all_deps_completed(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", []),
                ("c", ["a", "b"]),
            ]
        )
        ready = dag.get_ready_nodes(completed={"a", "b"})
        assert ready == ["c"]

    def test_all_completed(self) -> None:
        dag = DAG.from_steps(
            [
                ("a", []),
                ("b", ["a"]),
            ]
        )
        ready = dag.get_ready_nodes(completed={"a", "b"})
        assert ready == []

    def test_no_nodes(self) -> None:
        dag = DAG()
        ready = dag.get_ready_nodes(completed=set())
        assert ready == []


class TestIterativeCycleDetection:
    """The cycle-detection DFS is iterative so deep chains do not hit Python's recursion limit."""

    def test_cycle_detection_handles_deep_graph(self) -> None:
        dag = DAG()
        for i in range(2000):
            dag.add_edge(f"n{i}", f"n{i + 1}")
        # No cycle; must validate without raising RecursionError.
        assert dag.validate() == []

    def test_cycle_detection_message_is_accurate(self) -> None:
        # Two independent cycles. Both should be reported with their own
        # path, not leak nodes from the first traversal into the second.
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        dag.add_edge("c", "d")
        dag.add_edge("d", "c")

        errors = dag.validate()
        assert any("a -> b -> a" in e or "b -> a -> b" in e for e in errors)
        assert any("c -> d -> c" in e or "d -> c -> d" in e for e in errors)


class TestTransitiveSuccessors:
    def test_transitive_successors_single_root(self) -> None:
        dag = DAG()
        dag.add_edge("r", "a")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        assert dag.transitive_successors({"a"}) == {"a", "b", "c"}

    def test_transitive_successors_multiple_roots(self) -> None:
        dag = DAG()
        dag.add_edge("r", "a")
        dag.add_edge("r", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")
        assert dag.transitive_successors({"a", "b"}) == {"a", "b", "c"}

    def test_transitive_successors_unknown_root_returns_empty(self) -> None:
        dag = DAG()
        dag.add_edge("a", "b")
        assert dag.transitive_successors({"not_in_graph"}) == set()
