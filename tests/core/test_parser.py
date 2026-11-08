"""Tests for the WorkflowParser module."""

from __future__ import annotations

import pytest

from agentloom.core.parser import WorkflowParser
from agentloom.exceptions import ValidationError

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
        """A cycle fails at the reference or the DAG validation stage."""
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(INVALID_YAML_CYCLE)


class TestParseTimeInvariants:
    """Parse-time checks on step ids, ``max_concurrent_steps`` and shared ``output:`` keys.

    A duplicate id would shadow the other step, and parallel steps sharing an ``output:`` key
    would lose writes to last-writer-wins.
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
        # The id and both indices let the author find the two copies.
        assert "a" in msg
        assert "[0, 1]" in msg

    @pytest.mark.parametrize(
        "yaml_str",
        [
            "name: w\nconfig: {provider: mock, model: x, max_retries: -1}\n"
            "steps:\n  - {id: a, type: llm_call, prompt: x}\n",
            "name: w\nconfig: {provider: mock, model: x}\n"
            "steps:\n  - {id: a, type: llm_call, prompt: x, retry: {max_retries: -1}}\n",
        ],
        ids=["workflow", "step"],
    )
    def test_rejects_negative_max_retries(self, yaml_str: str) -> None:
        # A negative count would run the step zero times
        with pytest.raises(ValidationError, match="max_retries"):
            WorkflowParser.from_yaml(yaml_str)

    def test_shared_step_keeps_each_workflow_max_retries(self) -> None:
        from agentloom.core.models import StepDefinition, WorkflowConfig, WorkflowDefinition

        step = StepDefinition(id="a", type="llm_call", prompt="x")
        none = WorkflowDefinition(name="w0", config=WorkflowConfig(max_retries=0), steps=[step])
        two = WorkflowDefinition(name="w2", config=WorkflowConfig(max_retries=2), steps=[step])
        assert none.steps[0].retry.max_retries == 0
        assert two.steps[0].retry.max_retries == 2
        assert "max_retries" not in step.retry.model_fields_set

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

    def test_warns_on_partial_chain_owners(self) -> None:
        """Three owners with a partial dependency chain still collide.

        ``b`` depends on ``a`` but can run alongside ``c``, so the warning names ``b`` and ``c``
        even though ``b`` has ``a`` in its ancestry.
        """
        import warnings

        yaml_str = """
name: partial-chain
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared, depends_on: [a]}
  - {id: c, type: llm_call, prompt: x, output: shared}
"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WorkflowParser.from_yaml(yaml_str)
            messages = [str(w.message) for w in caught if "shared" in str(w.message)]
        assert any("'b'" in m and "'c'" in m for m in messages)


class TestStrictExtraKeys:
    """``StepDefinition`` and ``WorkflowConfig`` reject unknown keys at parse time.

    A typo fails ``agentloom validate`` with the field named instead of being dropped silently.
    """

    def test_step_definition_rejects_unknown_key(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from agentloom.core.models import StepDefinition, StepType

        with pytest.raises(PydanticValidationError, match="workflow"):
            # ``workflow:`` is a typo for ``workflow_inline:``.
            StepDefinition.model_validate(
                {"id": "s", "type": StepType.SUBWORKFLOW, "workflow": {"name": "child"}}
            )

    def test_workflow_config_rejects_unknown_key(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from agentloom.core.models import WorkflowConfig

        with pytest.raises(PydanticValidationError, match="responses"):
            # Mock responses come from ``responses_file:``; an inline ``responses:`` is refused.
            WorkflowConfig.model_validate({"provider": "mock", "responses": ["a", "b"]})

    def test_workflow_yaml_with_typo_rejected_at_parse(self) -> None:
        yaml = """\
name: typo-test
config:
  provider: mock
  model: mock-model
steps:
  - id: sub
    type: subworkflow
    workflow:
      name: child
"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            # from_yaml wraps Pydantic and parse failures in agentloom's own ``ValidationError``.
            with pytest.raises(ValidationError):
                WorkflowParser.from_yaml(f.name)
