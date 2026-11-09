"""Tests for the Python DSL workflow builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentloom.core.dsl import WorkflowBuilder, workflow
from agentloom.core.models import (
    AgentDefinition,
    StateKeyConfig,
    StepType,
    WebhookConfig,
)
from agentloom.exceptions import ValidationError


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
            .add_llm_step("target_a", prompt="a", depends_on=["r1"])
            .add_llm_step("target_b", prompt="b", depends_on=["r1"])
            .build()
        )
        step = wf.steps[0]
        assert step.type == StepType.ROUTER
        assert len(step.conditions) == 1
        assert step.default == "target_b"

    def test_subworkflow_step(self, tmp_path: Path) -> None:
        child = tmp_path / "child.yaml"
        child.write_text("name: child\nsteps: []\n", encoding="utf-8")
        wf = (
            workflow("test")
            .add_subworkflow_step("sub", workflow_path=str(child), output="sub_out")
            .build()
        )
        step = wf.steps[0]
        assert step.type == StepType.SUBWORKFLOW
        assert step.workflow_path == str(child)

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


class TestNewStepMethods:
    def test_embed_step(self) -> None:
        wf = (
            workflow("test")
            .add_embed_step(
                "e", inputs="state.docs", model="m", dimensions=8, output="vectors", depends_on=[]
            )
            .build()
        )
        step = wf.steps[0]
        assert step.type == StepType.EMBED
        assert (step.inputs, step.model, step.dimensions) == ("state.docs", "m", 8)
        assert step.output == "vectors"

    def test_approval_gate_step(self) -> None:
        notify = WebhookConfig(url="https://example.test/hook")
        wf = (
            workflow("test")
            .add_llm_step("a", prompt="p")
            .add_approval_gate_step(
                "g",
                timeout_seconds=60,
                on_timeout="reject",
                notify=notify,
                output="decision",
                depends_on=["a"],
            )
            .build()
        )
        step = wf.steps[1]
        assert step.type == StepType.APPROVAL_GATE
        assert (step.timeout_seconds, step.on_timeout) == (60, "reject")
        assert step.notify == notify
        assert step.output == "decision"
        assert step.depends_on == ["a"]

    def test_approval_gate_needs_nothing(self) -> None:
        assert workflow("test").add_approval_gate_step("g").build().steps[0].notify is None

    def test_on_timeout_without_timeout_seconds_fails_at_the_call(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            workflow("test").add_approval_gate_step("g", on_timeout="approve")

    def test_agent_step_from_a_definition(self) -> None:
        agent = AgentDefinition(name="helper")
        wf = (
            workflow("test")
            .add_agent_step(
                "a", agent, prompt="hi", conversation="state.chat", output="reply", depends_on=[]
            )
            .build()
        )
        step = wf.steps[0]
        assert step.type == StepType.AGENT
        assert step.agent == agent
        assert (step.prompt, step.conversation, step.output) == ("hi", "state.chat", "reply")

    def test_agent_step_from_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "helper.yaml"
        path.write_text("name: helper\n", encoding="utf-8")
        step = workflow("test").add_agent_step("a", str(path), prompt="hi").build().steps[0]
        assert step.agent is not None
        assert step.agent.name == "helper"

    def test_state_schema(self) -> None:
        wf = (
            workflow("test")
            .set_state_schema({"api_key": {"redact": True}})
            .set_state_schema({"*token*": StateKeyConfig(redact=True)})
            .add_llm_step("a", prompt="p")
            .build()
        )
        assert sorted(wf.redaction_patterns()) == ["*token*", "api_key"]

    def test_state_schema_typo_fails_at_the_call(self) -> None:
        with pytest.raises(ValueError, match="redacted"):
            workflow("test").set_state_schema({"api_key": {"redacted": True}})


class TestBuildValidates:
    def test_cycle(self) -> None:
        builder = (
            workflow("test")
            .add_llm_step("a", prompt="a", depends_on=["b"])
            .add_llm_step("b", prompt="b", depends_on=["a"])
        )
        with pytest.raises(ValidationError, match="Cycle detected"):
            builder.build()

    def test_missing_dependency(self) -> None:
        builder = workflow("test").add_llm_step("a", prompt="a", depends_on=["nope"])
        with pytest.raises(ValidationError, match="non-existent step 'nope'"):
            builder.build()

    def test_missing_router_target(self) -> None:
        builder = workflow("test").add_router_step("r", conditions=[("state.x", "nope")])
        with pytest.raises(ValidationError, match="non-existent target 'nope'"):
            builder.build()

    def test_missing_router_default(self) -> None:
        builder = workflow("test").add_router_step("r", conditions=[], default="nope")
        with pytest.raises(ValidationError, match="non-existent default target 'nope'"):
            builder.build()

    def test_missing_subworkflow_file(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "child.yaml")
        builder = workflow("test").add_subworkflow_step("sub", workflow_path=missing)
        with pytest.raises(ValidationError, match="non-existent file"):
            builder.build()

    def test_two_turns_racing_on_a_conversation(self) -> None:
        builder = (
            workflow("test")
            .add_llm_step("a", prompt="a", conversation="state.chat")
            .add_llm_step("b", prompt="b", conversation="state.chat")
        )
        with pytest.raises(ValidationError, match="'chat'"):
            builder.build()

    def test_valid_build_returns_the_workflow(self) -> None:
        wf = (
            workflow("test")
            .add_router_step("r", conditions=[("state.x", "a")], default="b")
            .add_llm_step("a", prompt="a", depends_on=["r"])
            .add_llm_step("b", prompt="b", depends_on=["r"])
            .build()
        )
        assert wf.step_ids() == ["r", "a", "b"]
