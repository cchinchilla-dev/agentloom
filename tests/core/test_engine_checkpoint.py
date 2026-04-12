"""Tests for checkpoint integration in WorkflowEngine."""

from __future__ import annotations

from agentloom.checkpointing.base import CheckpointData
from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepStatus, WorkflowStatus
from agentloom.providers.gateway import ProviderGateway
from tests.conftest import MockProvider


def _two_step_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="checkpoint-test",
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
        ],
    )


def _mock_gateway() -> ProviderGateway:
    gw = ProviderGateway()
    gw.register(MockProvider(), priority=0)
    return gw


class TestEngineCheckpoint:
    """Test that the engine saves checkpoints when a checkpointer is provided."""

    async def test_checkpoint_saved_on_success(self, tmp_path: object) -> None:
        from pathlib import Path

        cp_dir = Path(str(tmp_path))
        checkpointer = FileCheckpointer(checkpoint_dir=cp_dir)
        workflow = _two_step_workflow()

        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        assert engine.run_id  # auto-generated

        # Verify checkpoint was saved
        loaded = await checkpointer.load(engine.run_id)
        assert loaded.status == "success"
        assert "step_a" in loaded.completed_steps
        assert "step_b" in loaded.completed_steps
        assert loaded.workflow_name == "checkpoint-test"

    async def test_checkpoint_preserves_state(self, tmp_path: object) -> None:
        from pathlib import Path

        cp_dir = Path(str(tmp_path))
        checkpointer = FileCheckpointer(checkpoint_dir=cp_dir)
        workflow = _two_step_workflow()

        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
        )
        await engine.run()
        loaded = await checkpointer.load(engine.run_id)

        assert loaded.state["input"] == "hello"
        assert "result_a" in loaded.state
        assert "result_b" in loaded.state

    async def test_no_checkpoint_without_checkpointer(self, tmp_path: object) -> None:
        from pathlib import Path

        cp_dir = Path(str(tmp_path))
        workflow = _two_step_workflow()

        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_mock_gateway(),
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS

        # No files should be created
        checkpointer = FileCheckpointer(checkpoint_dir=cp_dir)
        runs = await checkpointer.list_runs()
        assert runs == []

    async def test_custom_run_id(self, tmp_path: object) -> None:
        from pathlib import Path

        cp_dir = Path(str(tmp_path))
        checkpointer = FileCheckpointer(checkpoint_dir=cp_dir)

        engine = WorkflowEngine(
            workflow=_two_step_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="my-custom-id",
        )
        await engine.run()

        loaded = await checkpointer.load("my-custom-id")
        assert loaded.run_id == "my-custom-id"


class TestEngineResumeFromCheckpoint:
    """Test that from_checkpoint reconstructs the engine and skips completed steps."""

    async def test_resume_skips_completed_steps(self, tmp_path: object) -> None:
        from pathlib import Path

        cp_dir = Path(str(tmp_path))
        checkpointer = FileCheckpointer(checkpoint_dir=cp_dir)

        # Run first, then checkpoint
        engine = WorkflowEngine(
            workflow=_two_step_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="resume-test",
        )
        first_result = await engine.run()
        assert first_result.status == WorkflowStatus.SUCCESS

        # Load checkpoint and resume — all steps already completed, should be a no-op
        checkpoint_data = await checkpointer.load("resume-test")

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
        # Provider should NOT have been called — all steps were already done
        assert len(provider.calls) == 0

    async def test_resume_continues_from_midpoint(self, tmp_path: object) -> None:
        """Simulate resuming when only step_a was completed."""
        from pathlib import Path

        cp_dir = Path(str(tmp_path))
        checkpointer = FileCheckpointer(checkpoint_dir=cp_dir)
        workflow = _two_step_workflow()

        # Manually create a checkpoint where only step_a is done
        checkpoint_data = CheckpointData(
            workflow_name="checkpoint-test",
            run_id="partial-run",
            workflow_definition=workflow.model_dump(),
            state={
                "input": "hello",
                "result_a": "Mock response",
                "steps": {
                    "step_a": {"output": "Mock response", "status": "success"},
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
            status="failed",
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
        # Only step_b should have been executed
        assert len(provider.calls) == 1
        assert "step_b" in result.step_results
        assert result.step_results["step_b"].status == StepStatus.SUCCESS
