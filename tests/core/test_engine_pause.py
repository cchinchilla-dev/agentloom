"""Tests for the workflow pause mechanism (Issue #40)."""

from __future__ import annotations

from pathlib import Path

from agentloom.checkpointing.base import CheckpointData
from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import WorkflowEngine, _extract_pause_error
from agentloom.core.models import (
    Condition,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepResult, StepStatus, WorkflowStatus
from agentloom.exceptions import PauseRequestedError
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.base import BaseStep, StepContext
from agentloom.steps.registry import StepRegistry, create_default_registry
from tests.conftest import MockProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gateway(provider: MockProvider | None = None) -> ProviderGateway:
    gw = ProviderGateway()
    gw.register(provider or MockProvider(), priority=0)
    return gw


def _three_step_workflow() -> WorkflowDefinition:
    """Linear workflow: step_a → step_b → step_c."""
    return WorkflowDefinition(
        name="pause-test",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"input": "hello"},
        steps=[
            StepDefinition(
                id="step_a",
                type=StepType.LLM_CALL,
                prompt="Process: {state.input}",
                output="result_a",
            ),
            StepDefinition(
                id="step_b",
                type=StepType.LLM_CALL,
                depends_on=["step_a"],
                prompt="Continue: {state.result_a}",
                output="result_b",
            ),
            StepDefinition(
                id="step_c",
                type=StepType.LLM_CALL,
                depends_on=["step_b"],
                prompt="Finish: {state.result_b}",
                output="result_c",
            ),
        ],
    )


class PausingStep(BaseStep):
    """Step executor that raises PauseRequestedError."""

    async def execute(self, context: StepContext) -> StepResult:
        raise PauseRequestedError(context.step_definition.id)


class PausingOnSecondCallStep(BaseStep):
    """LLM-call step that pauses on a specific step_id.

    First call runs the real LLM executor; when the step_id matches
    ``pause_on``, it raises ``PauseRequestedError`` instead.
    """

    pause_on: str = ""

    async def execute(self, context: StepContext) -> StepResult:
        if context.step_definition.id == self.__class__.pause_on:
            raise PauseRequestedError(context.step_definition.id)
        # Delegate to the real LLM executor
        from agentloom.steps.llm_call import LLMCallStep

        return await LLMCallStep().execute(context)


def _registry_that_pauses_on(step_id: str) -> StepRegistry:
    """Build a step registry whose LLM_CALL executor pauses on *step_id*."""

    class _Pauser(PausingOnSecondCallStep):
        pause_on = step_id

    reg = create_default_registry()
    reg.register(StepType.LLM_CALL, _Pauser)
    return reg


# ---------------------------------------------------------------------------
# Test _extract_pause_error helper
# ---------------------------------------------------------------------------


class TestExtractPauseError:
    def test_bare_pause_error(self) -> None:
        err = PauseRequestedError("s1")
        assert _extract_pause_error(err) is err

    def test_wrapped_in_exception_group(self) -> None:
        err = PauseRequestedError("s1")
        group = ExceptionGroup("tg", [err])
        assert _extract_pause_error(group) is err

    def test_nested_exception_group(self) -> None:
        err = PauseRequestedError("s1")
        inner = ExceptionGroup("inner", [err])
        outer = ExceptionGroup("outer", [inner])
        assert _extract_pause_error(outer) is err

    def test_no_pause_error(self) -> None:
        group = ExceptionGroup("tg", [RuntimeError("boom")])
        assert _extract_pause_error(group) is None

    def test_unrelated_exception(self) -> None:
        assert _extract_pause_error(RuntimeError("x")) is None


# ---------------------------------------------------------------------------
# Engine pause tests
# ---------------------------------------------------------------------------


class TestEnginePause:
    """Test that the engine handles PauseRequestedError correctly."""

    async def test_pause_returns_paused_status(self) -> None:
        """Engine returns WorkflowStatus.PAUSED when a step raises PauseRequestedError."""
        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.PAUSED
        assert "step_b" in (result.error or "")

    async def test_pause_saves_checkpoint(self, tmp_path: Path) -> None:
        """Checkpoint is saved with status='paused' and paused_step_id."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
            checkpointer=checkpointer,
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.PAUSED

        loaded = await checkpointer.load(engine.run_id)
        assert loaded.status == "paused"
        assert loaded.paused_step_id == "step_b"

    async def test_pause_preserves_completed_steps(self, tmp_path: Path) -> None:
        """Steps before the pause point are in completed_steps; paused step is not."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
            checkpointer=checkpointer,
        )
        await engine.run()

        loaded = await checkpointer.load(engine.run_id)
        assert "step_a" in loaded.completed_steps
        assert "step_b" not in loaded.completed_steps
        assert "step_c" not in loaded.completed_steps

    async def test_pause_step_result_is_paused(self) -> None:
        """The paused step's result has StepStatus.PAUSED."""
        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
        )
        result = await engine.run()

        assert result.step_results["step_a"].status == StepStatus.SUCCESS
        assert result.step_results["step_b"].status == StepStatus.PAUSED

    async def test_pause_state_has_completed_outputs(self) -> None:
        """State includes outputs from steps completed before the pause."""
        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
        )
        result = await engine.run()

        assert "result_a" in result.final_state
        assert "result_b" not in result.final_state


# ---------------------------------------------------------------------------
# Resume from paused checkpoint
# ---------------------------------------------------------------------------


class TestResumeFromPaused:
    """Test that resuming from a paused checkpoint works correctly."""

    async def test_resume_completes_workflow(self, tmp_path: Path) -> None:
        """Resume from a paused checkpoint runs remaining steps to success."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        # First run — pauses at step_b
        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
            checkpointer=checkpointer,
            run_id="pause-resume-test",
        )
        first_result = await engine.run()
        assert first_result.status == WorkflowStatus.PAUSED

        # Resume with default registry — step_b and step_c should execute normally
        resume_provider = MockProvider()
        resume_gw = ProviderGateway()
        resume_gw.register(resume_provider, priority=0)

        checkpoint_data = await checkpointer.load("pause-resume-test")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=resume_gw,
        )
        second_result = await resumed.run()

        assert second_result.status == WorkflowStatus.SUCCESS
        assert "result_b" in second_result.final_state
        assert "result_c" in second_result.final_state

    async def test_resume_skips_completed_steps(self, tmp_path: Path) -> None:
        """On resume, only the remaining steps execute."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
            checkpointer=checkpointer,
            run_id="skip-test",
        )
        await engine.run()

        # Resume
        resume_provider = MockProvider()
        resume_gw = ProviderGateway()
        resume_gw.register(resume_provider, priority=0)

        checkpoint_data = await checkpointer.load("skip-test")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=resume_gw,
        )
        await resumed.run()

        # step_a was already done — provider called for step_b and step_c
        assert len(resume_provider.calls) == 2

    async def test_resume_updates_checkpoint_to_success(self, tmp_path: Path) -> None:
        """After successful resume, checkpoint status changes to 'success'."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_three_step_workflow(),
            provider_gateway=_mock_gateway(),
            step_registry=_registry_that_pauses_on("step_b"),
            checkpointer=checkpointer,
            run_id="update-test",
        )
        await engine.run()

        # Resume
        checkpoint_data = await checkpointer.load("update-test")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=_mock_gateway(),
        )
        await resumed.run()

        final = await checkpointer.load("update-test")
        assert final.status == "success"
        assert final.paused_step_id is None

    async def test_resume_from_manual_checkpoint(self, tmp_path: Path) -> None:
        """Resume from a manually-constructed paused checkpoint."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        workflow = _three_step_workflow()

        checkpoint_data = CheckpointData(
            workflow_name="pause-test",
            run_id="manual-pause",
            workflow_definition=workflow.model_dump(),
            state={
                "input": "hello",
                "result_a": "Mock response",
                "steps": {
                    "step_a": {"output": "Mock response", "status": "success"},
                    "step_b": {"output": None, "status": "paused"},
                },
            },
            step_results={
                "step_a": {
                    "step_id": "step_a",
                    "status": "success",
                    "output": "Mock response",
                    "duration_ms": 10.0,
                },
            },
            completed_steps=["step_a"],
            status="paused",
            paused_step_id="step_b",
            created_at="2026-04-12T10:00:00+00:00",
            updated_at="2026-04-12T10:00:01+00:00",
        )
        await checkpointer.save(checkpoint_data)

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)

        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=gw,
        )
        result = await resumed.run()

        assert result.status == WorkflowStatus.SUCCESS
        # step_b and step_c should run
        assert len(provider.calls) == 2
        assert result.step_results["step_b"].status == StepStatus.SUCCESS
        assert result.step_results["step_c"].status == StepStatus.SUCCESS


# ---------------------------------------------------------------------------
# Pause with router
# ---------------------------------------------------------------------------


class TestPauseWithRouter:
    """Test pause/resume with router branching."""

    async def test_pause_after_router_resumes_correctly(self, tmp_path: Path) -> None:
        """Pause after a router; resume respects the router's branch selection."""
        workflow = WorkflowDefinition(
            name="router-pause",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"input": "test", "classification": "billing"},
            steps=[
                StepDefinition(
                    id="classify",
                    type=StepType.LLM_CALL,
                    prompt="Classify: {state.input}",
                    output="classification",
                ),
                StepDefinition(
                    id="route",
                    type=StepType.ROUTER,
                    depends_on=["classify"],
                    conditions=[
                        Condition(
                            expression="state.classification == 'billing'",
                            target="billing",
                        ),
                    ],
                    default="general",
                ),
                StepDefinition(
                    id="billing",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="Handle billing: {state.input}",
                    output="response",
                ),
                StepDefinition(
                    id="general",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="Handle general: {state.input}",
                    output="response",
                ),
            ],
        )

        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        # Build a checkpoint where classify + route are done, paused before billing.
        checkpoint_data = CheckpointData(
            workflow_name="router-pause",
            run_id="router-pause-run",
            workflow_definition=workflow.model_dump(),
            state={
                "input": "test",
                "classification": "billing",
                "steps": {
                    "classify": {"output": "billing", "status": "success"},
                    "route": {"output": "billing", "status": "success"},
                },
            },
            step_results={
                "classify": {
                    "step_id": "classify",
                    "status": "success",
                    "output": "billing",
                    "duration_ms": 5.0,
                },
                "route": {
                    "step_id": "route",
                    "status": "success",
                    "output": "billing",
                    "duration_ms": 1.0,
                },
            },
            completed_steps=["classify", "route"],
            status="paused",
            paused_step_id="billing",
            created_at="2026-04-12T10:00:00+00:00",
            updated_at="2026-04-12T10:00:01+00:00",
        )
        await checkpointer.save(checkpoint_data)

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)

        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=gw,
        )
        result = await resumed.run()

        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["billing"].status == StepStatus.SUCCESS
        assert result.step_results["general"].status == StepStatus.SKIPPED
        assert len(provider.calls) == 1
