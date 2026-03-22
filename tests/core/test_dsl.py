"""Tests for the Python DSL workflow builder."""

from __future__ import annotations

from agentloom.core.dsl import WorkflowBuilder, workflow
from agentloom.core.models import StepType


class TestWorkflowFactory:
    def test_creates_builder(self) -> None:
        builder = workflow("test")
        assert isinstance(builder, WorkflowBuilder)
        assert builder.name == "test"

    def test_config_kwargs_passed(self) -> None:
        builder = workflow("test", provider="ollama", model="phi4")
        assert builder.config.provider == "ollama"
        assert builder.config.model == "phi4"


class TestBuilderChaining:
    def test_set_state_returns_self(self) -> None:
        builder = workflow("test")
        result = builder.set_state(question="hi")
        assert result is builder

    def test_add_llm_step_returns_self(self) -> None:
        builder = workflow("test")
        result = builder.add_llm_step("s1", prompt="hello")
        assert result is builder

    def test_full_chain(self) -> None:
        wf = (
            workflow("chain-test", provider="mock", model="mock-model")
            .set_state(question="What?")
            .add_llm_step("answer", prompt="{question}", output="answer")
            .build()
        )
        assert wf.name == "chain-test"
        assert wf.state["question"] == "What?"
        assert len(wf.steps) == 1


class TestBuildOutput:
    def test_llm_step(self) -> None:
        wf = (
            workflow("test")
            .add_llm_step("s1", prompt="hi", system_prompt="be helpful", output="out")
            .build()
        )
        step = wf.steps[0]
        assert step.id == "s1"
        assert step.type == StepType.LLM_CALL
        assert step.prompt == "hi"
        assert step.system_prompt == "be helpful"
        assert step.output == "out"

    def test_tool_step(self) -> None:
        wf = (
            workflow("test")
            .add_tool_step("t1", tool_name="http_request", tool_args={"url": "http://x"})
            .build()
        )
        step = wf.steps[0]
        assert step.type == StepType.TOOL
        assert step.tool_name == "http_request"

    def test_router_step(self) -> None:
        wf = (
            workflow("test")
            .add_router_step(
                "r1",
                conditions=[("state.x == 1", "target_a")],
                default="target_b",
            )
            .build()
        )
        step = wf.steps[0]
        assert step.type == StepType.ROUTER
        assert len(step.conditions) == 1
        assert step.default == "target_b"

    def test_subworkflow_step(self) -> None:
        wf = (
            workflow("test")
            .add_subworkflow_step("sub", workflow_path="child.yaml", output="sub_out")
            .build()
        )
        step = wf.steps[0]
        assert step.type == StepType.SUBWORKFLOW
        assert step.workflow_path == "child.yaml"

    def test_dependencies(self) -> None:
        wf = (
            workflow("test")
            .add_llm_step("a", prompt="a")
            .add_llm_step("b", prompt="b", depends_on=["a"])
            .build()
        )
        assert wf.steps[1].depends_on == ["a"]

    def test_workflow_config(self) -> None:
        wf = workflow("test", budget_usd=1.0, max_retries=5).build()
        assert wf.config.budget_usd == 1.0
        assert wf.config.max_retries == 5
