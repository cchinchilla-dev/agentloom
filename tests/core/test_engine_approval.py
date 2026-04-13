"""Tests for the approval gate step integrated with the workflow engine."""

from __future__ import annotations

from pathlib import Path

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


def _mock_gateway(provider: MockProvider | None = None) -> ProviderGateway:
    gw = ProviderGateway()
    gw.register(provider or MockProvider(), priority=0)
    return gw


def _approval_workflow() -> WorkflowDefinition:
    """Linear workflow: draft → approve → send."""
    return WorkflowDefinition(
        name="approval-test",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"input": "hello"},
        steps=[
            StepDefinition(
                id="draft",
                type=StepType.LLM_CALL,
                prompt="Draft: {state.input}",
                output="draft_text",
            ),
            StepDefinition(
                id="approve",
                type=StepType.APPROVAL_GATE,
                depends_on=["draft"],
                output="decision",
            ),
            StepDefinition(
                id="send",
                type=StepType.LLM_CALL,
                depends_on=["approve"],
                prompt="Send (decision={state.decision}): {state.draft_text}",
                output="result",
            ),
        ],
    )


class TestApprovalGatePause:
    async def test_approval_gate_pauses_workflow(self) -> None:
        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.PAUSED
        assert result.step_results["draft"].status == StepStatus.SUCCESS
        assert result.step_results["approve"].status == StepStatus.PAUSED
        assert "send" not in result.step_results

    async def test_approval_gate_saves_checkpoint(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="approval-ckpt",
        )
        await engine.run()

        loaded = await checkpointer.load("approval-ckpt")
        assert loaded.status == "paused"
        assert loaded.paused_step_id == "approve"
        assert "draft" in loaded.completed_steps
        assert "approve" not in loaded.completed_steps

    async def test_approval_preserves_prior_outputs(self) -> None:
        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
        )
        result = await engine.run()

        assert "draft_text" in result.final_state
        assert "decision" not in result.final_state


class TestApprovalGateResume:
    async def test_resume_with_approve(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        # First run — pauses at approve
        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="approve-run",
        )
        first = await engine.run()
        assert first.status == WorkflowStatus.PAUSED

        # Resume with approval
        checkpoint_data = await checkpointer.load("approve-run")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=_mock_gateway(),
            approval_decisions={"approve": "approved"},
        )
        second = await resumed.run()

        assert second.status == WorkflowStatus.SUCCESS
        assert second.step_results["approve"].status == StepStatus.SUCCESS
        assert second.step_results["approve"].output == "approved"
        assert second.step_results["send"].status == StepStatus.SUCCESS

    async def test_resume_with_reject(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="reject-run",
        )
        await engine.run()

        checkpoint_data = await checkpointer.load("reject-run")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=_mock_gateway(),
            approval_decisions={"approve": "rejected"},
        )
        second = await resumed.run()

        assert second.status == WorkflowStatus.SUCCESS
        assert second.step_results["approve"].output == "rejected"
        assert second.final_state["decision"] == "rejected"

    async def test_downstream_reads_decision(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="downstream-run",
        )
        await engine.run()

        checkpoint_data = await checkpointer.load("downstream-run")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=_mock_gateway(),
            approval_decisions={"approve": "approved"},
        )
        second = await resumed.run()

        assert second.final_state["decision"] == "approved"
        assert "result" in second.final_state

    async def test_resume_skips_completed_steps(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="skip-run",
        )
        await engine.run()

        resume_provider = MockProvider()
        resume_gw = ProviderGateway()
        resume_gw.register(resume_provider, priority=0)

        checkpoint_data = await checkpointer.load("skip-run")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=resume_gw,
            approval_decisions={"approve": "approved"},
        )
        await resumed.run()

        # draft was already done — only send should call the provider
        assert len(resume_provider.calls) == 1

    async def test_resume_without_decision_pauses_again(self, tmp_path: Path) -> None:
        """Resuming without --approve/--reject re-pauses at the gate."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="no-decision-run",
        )
        await engine.run()

        checkpoint_data = await checkpointer.load("no-decision-run")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=_mock_gateway(),
        )
        second = await resumed.run()

        assert second.status == WorkflowStatus.PAUSED

    async def test_final_checkpoint_updated(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        engine = WorkflowEngine(
            workflow=_approval_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="final-ckpt-run",
        )
        await engine.run()

        checkpoint_data = await checkpointer.load("final-ckpt-run")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=_mock_gateway(),
            approval_decisions={"approve": "approved"},
        )
        await resumed.run()

        final = await checkpointer.load("final-ckpt-run")
        assert final.status == "success"
        assert final.paused_step_id is None
