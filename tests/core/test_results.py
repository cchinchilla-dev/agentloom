"""Tests for result models."""

from __future__ import annotations

from agentloom.core.results import (
    StepResult,
    StepStatus,
    TokenUsage,
    WorkflowResult,
    WorkflowStatus,
)


class TestTokenUsage:
    def test_defaults(self) -> None:
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.total_tokens == 0

    def test_custom_values(self) -> None:
        t = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert t.total_tokens == 30


class TestStepResult:
    def test_success(self) -> None:
        r = StepResult(step_id="s1", status=StepStatus.SUCCESS, output="hello")
        assert r.status == StepStatus.SUCCESS
        assert r.output == "hello"
        assert r.error is None

    def test_failure(self) -> None:
        r = StepResult(step_id="s1", status=StepStatus.FAILED, error="boom")
        assert r.error == "boom"
        assert r.output is None

    def test_defaults(self) -> None:
        r = StepResult(step_id="s1", status=StepStatus.PENDING)
        assert r.cost_usd == 0.0
        assert r.duration_ms == 0.0
        assert r.token_usage.total_tokens == 0

    def test_serialization(self) -> None:
        r = StepResult(
            step_id="s1",
            status=StepStatus.SUCCESS,
            output="result",
            cost_usd=0.001,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        data = r.model_dump()
        restored = StepResult.model_validate(data)
        assert restored.step_id == "s1"
        assert restored.cost_usd == 0.001
        assert restored.token_usage.total_tokens == 30


class TestWorkflowResult:
    def test_success(self) -> None:
        r = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.SUCCESS,
            total_tokens=100,
            total_cost_usd=0.01,
        )
        assert r.status == WorkflowStatus.SUCCESS
        assert r.error is None

    def test_failure_with_error(self) -> None:
        r = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.FAILED,
            error="something broke",
        )
        assert r.error == "something broke"

    def test_step_results_dict(self) -> None:
        r = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.SUCCESS,
            step_results={
                "s1": StepResult(step_id="s1", status=StepStatus.SUCCESS),
                "s2": StepResult(step_id="s2", status=StepStatus.SKIPPED),
            },
        )
        assert len(r.step_results) == 2
        assert r.step_results["s2"].status == StepStatus.SKIPPED


class TestStepStatus:
    def test_all_values(self) -> None:
        assert StepStatus.PENDING == "pending"
        assert StepStatus.SUCCESS == "success"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"
        assert StepStatus.TIMEOUT == "timeout"


class TestWorkflowStatus:
    def test_all_values(self) -> None:
        assert WorkflowStatus.SUCCESS == "success"
        assert WorkflowStatus.FAILED == "failed"
        assert WorkflowStatus.BUDGET_EXCEEDED == "budget_exceeded"
