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
