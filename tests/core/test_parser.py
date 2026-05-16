"""Tests for the WorkflowParser module."""

from __future__ import annotations

import pytest

from agentloom.core.parser import WorkflowParser
from agentloom.exceptions import ValidationError

# defined here for now, will move to conftest later
SIMPLE_YAML = """
name: yaml-test
version: "1.0"
config:
  provider: mock
  model: mock-model
state:
  question: "What is Python?"
steps:
  - id: answer
    type: llm_call
    prompt: "Answer: {state.question}"
    output: answer
"""

INVALID_YAML_CYCLE = """
name: cycle-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
    depends_on: [b]
  - id: b
    type: llm_call
    prompt: "b"
    depends_on: [a]
"""

INVALID_YAML_MISSING_REF = """
name: missing-ref-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
    depends_on: [nonexistent]
"""


class TestYAMLParsing:
    """Test parsing valid YAML into WorkflowDefinition."""

    def test_parse_simple_yaml(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert workflow.name == "yaml-test"

    def test_parse_simple_yaml_steps(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert len(workflow.steps) == 1
        assert workflow.steps[0].id == "answer"

    def test_parse_simple_yaml_config(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert workflow.config.provider == "mock"
        assert workflow.config.model == "mock-model"

    def test_parse_simple_yaml_state(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert workflow.state["question"] == "What is Python?"

    def test_parse_from_dict(self) -> None:
        data = {
            "name": "dict-test",
            "steps": [{"id": "s1", "type": "llm_call", "prompt": "test"}],
        }
        workflow = WorkflowParser.from_dict(data)
        assert workflow.name == "dict-test"


class TestValidationErrors:
    """Test that invalid YAML raises proper errors."""

    def test_invalid_yaml_syntax(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml("invalid: yaml: content: [")

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml("name: test\n")

    def test_non_mapping_yaml(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml("- just\n- a\n- list\n")

    def test_invalid_step_type(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(
                "name: t\nsteps:\n  - id: x\n    type: invalid\n    prompt: y\n"
            )


class TestMissingReferences:
    def test_missing_depends_on_reference(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(INVALID_YAML_MISSING_REF)

    def test_valid_depends_on_reference(self) -> None:
        yaml_str = """
name: valid-ref
steps:
  - id: a
    type: llm_call
    prompt: "a"
  - id: b
    type: llm_call
    prompt: "b"
    depends_on: [a]
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        assert len(workflow.steps) == 2


class TestDAGBuild:
    def test_build_dag_simple(self) -> None:
        yaml_str = """
name: dag-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
  - id: b
    type: llm_call
    prompt: "b"
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        dag = WorkflowParser.build_dag(workflow)
        assert set(dag.nodes) == {"a", "b"}

    def test_build_dag_with_deps(self) -> None:
        yaml_str = """
name: dag-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
  - id: b
    type: llm_call
    prompt: "b"
    depends_on: [a]
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        dag = WorkflowParser.build_dag(workflow)
        assert dag.predecessors("b") == {"a"}

    def test_build_dag_cycle_raises_error(self) -> None:
        """Cycle YAML should fail at the reference validation or DAG validation stage."""
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(INVALID_YAML_CYCLE)


class TestParseTimeInvariants:
    """#055 regression — parse-time validation of step ids and concurrency.

    Three flavours of footgun became hard errors in 0.5.0: duplicate step
    ids (used to shadow silently), ``max_concurrent_steps <= 0`` (used to
    deadlock or surface a cryptic ``total_tokens must be >= 0``), and
    parallel-eligible steps sharing an ``output:`` key (used to drop most
    writers under last-writer-wins).
    """

    def test_rejects_duplicate_step_ids(self) -> None:
        yaml_str = """
name: dup
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: first}
  - {id: a, type: llm_call, prompt: second}
"""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowParser.from_yaml(yaml_str)
        msg = str(exc_info.value)
        # Error must name the offending id and the indices so the workflow
        # author can find both copies in the source.
        assert "a" in msg
        assert "[0, 1]" in msg

    @pytest.mark.parametrize("bad", [0, -1, -100, 9999999])
    def test_rejects_invalid_max_concurrent_steps(self, bad: int) -> None:
        yaml_str = f"""
name: bad-concurrent
config:
  provider: mock
  model: x
  max_concurrent_steps: {bad}
steps:
  - {{id: a, type: llm_call, prompt: x}}
"""
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(yaml_str)

    def test_accepts_boundary_max_concurrent_steps_1(self) -> None:
        yaml_str = """
name: boundary
config: {provider: mock, model: x, max_concurrent_steps: 1}
steps:
  - {id: a, type: llm_call, prompt: x}
"""
        wf = WorkflowParser.from_yaml(yaml_str)
        assert wf.config.max_concurrent_steps == 1

    def test_warns_on_parallel_steps_sharing_output_key(self) -> None:
        import warnings

        yaml_str = """
name: dup-output
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared}
"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WorkflowParser.from_yaml(yaml_str)
            messages = [str(w.message) for w in caught]
        assert any("shared" in m and "'a'" in m and "'b'" in m for m in messages)

    def test_no_warning_when_owners_chain_via_depends_on(self) -> None:
        """Sequential overwrite is intentional — no warning."""
        import warnings

        yaml_str = """
name: seq
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared, depends_on: [a]}
"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WorkflowParser.from_yaml(yaml_str)
            messages = [str(w.message) for w in caught if "shared" in str(w.message)]
        assert messages == []

    def test_strict_outputs_promotes_warning_to_error(self) -> None:
        yaml_str = """
name: strict
config: {provider: mock, model: x, strict_outputs: true}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared}
"""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowParser.from_yaml(yaml_str)
        assert "shared" in str(exc_info.value)
