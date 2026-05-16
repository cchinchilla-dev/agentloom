"""Tests for checkpoint integration in WorkflowEngine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentloom.checkpointing.base import BaseCheckpointer, CheckpointData
from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    Condition,
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

    async def test_checkpoint_saved_on_success(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
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

    async def test_checkpoint_preserves_state(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
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

    async def test_no_checkpoint_without_checkpointer(self, tmp_path: Path) -> None:
        workflow = _two_step_workflow()

        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_mock_gateway(),
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS

        # No files should be created
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        runs = await checkpointer.list_runs()
        assert runs == []

    async def test_custom_run_id(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

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

    async def test_resume_skips_completed_steps(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

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

    async def test_resume_continues_from_midpoint(self, tmp_path: Path) -> None:
        """Simulate resuming when only step_a was completed."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
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


class TestCheckpointErrorHandling:
    """Test that checkpoint save failures are handled gracefully."""

    async def test_save_checkpoint_io_error_is_swallowed(self, tmp_path: Path) -> None:
        """Engine should continue even if checkpoint save raises an I/O error."""

        class FailingCheckpointer(BaseCheckpointer):
            async def save(self, data: CheckpointData) -> None:
                raise OSError("Disk full")

            async def load(self, run_id: str) -> CheckpointData:
                raise KeyError(run_id)

            async def list_runs(self) -> list[CheckpointData]:
                return []

            async def delete(self, run_id: str) -> None:
                pass

        engine = WorkflowEngine(
            workflow=_two_step_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=FailingCheckpointer(),
        )
        result = await engine.run()
        # Should still succeed even though checkpoint save failed
        assert result.status == WorkflowStatus.SUCCESS

    async def test_checkpoint_saved_on_budget_exceeded(self, tmp_path: Path) -> None:
        """Engine should save checkpoint when budget is exceeded."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        # MockProvider costs $0.001 per call — budget is $0.0001 so it overruns.
        workflow = WorkflowDefinition(
            name="budget-test",
            config=WorkflowConfig(provider="mock", model="mock-model", budget_usd=0.0001),
            state={"input": "hello"},
            steps=[
                StepDefinition(
                    id="step1",
                    type=StepType.LLM_CALL,
                    prompt="Process: {state.input}",
                    output="result",
                ),
            ],
        )
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
        )
        result = await engine.run()
        # Budget check is post-hoc so the step succeeds but the workflow
        # reports budget_exceeded. However, BudgetExceededError is raised
        # inside a task group and may be wrapped — accept either status.
        assert result.status in (WorkflowStatus.BUDGET_EXCEEDED, WorkflowStatus.FAILED)

        loaded = await checkpointer.load(engine.run_id)
        assert loaded.status in ("budget_exceeded", "failed")

    async def test_checkpoint_saved_on_failure(self, tmp_path: Path) -> None:
        """Engine should save checkpoint with 'failed' status on exception."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)

        class ErrorProvider(MockProvider):
            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Provider crash")

        gw = ProviderGateway()
        gw.register(ErrorProvider(), priority=0)

        engine = WorkflowEngine(
            workflow=_two_step_workflow(),
            provider_gateway=gw,
            checkpointer=checkpointer,
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.FAILED

        loaded = await checkpointer.load(engine.run_id)
        assert loaded.status == "failed"


class TestResumeWithRouter:
    """Test that resuming a workflow with routers restores branch activation."""

    async def test_resume_restores_router_activation(self, tmp_path: Path) -> None:
        """When a completed router is skipped on resume, its target must still activate."""
        workflow = WorkflowDefinition(
            name="router-resume",
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
                        Condition(expression="state.classification == 'billing'", target="billing"),
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

        # Build a checkpoint where classify + route are done, billing/general pending.
        # The router chose "billing".
        checkpoint_data = CheckpointData(
            workflow_name="router-resume",
            run_id="router-run",
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
        # Only billing should have run, general should be skipped
        assert result.step_results["billing"].status == StepStatus.SUCCESS
        assert result.step_results["general"].status == StepStatus.SKIPPED
        # Provider called once (for billing only)
        assert len(provider.calls) == 1


class TestCheckpointStateRedaction:
    """Checkpoint files must NOT carry plaintext for redacted keys.

    Approval-gate workflows always checkpoint on pause, so any state key
    flagged via ``state_schema: {key: {redact: true}}`` or
    ``AGENTLOOM_REDACT_STATE_KEYS`` has to be masked before the JSON
    lands on disk — both the runtime snapshot and the literal ``state:``
    block carried inside ``workflow_definition``.
    """

    async def test_state_keys_are_redacted_in_persisted_file(
        self, tmp_path: Path
    ) -> None:
        from agentloom.core.models import StateKeyConfig

        wf = WorkflowDefinition(
            name="redact-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={
                "api_key": "sk-secret-do-not-log-AAAAAAAA",
                "password": "P@ssw0rd!",
                "jwt": "eyJhbGciOi.payload.sig",
                "user": "alice",
            },
            state_schema={
                "api_key": StateKeyConfig(redact=True),
                "password": StateKeyConfig(redact=True),
                "jwt": StateKeyConfig(redact=True),
            },
            steps=[
                StepDefinition(
                    id="noop",
                    type=StepType.LLM_CALL,
                    prompt="Hello {state.user}",
                    output="ans",
                )
            ],
        )
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, checkpointer=cp)
        await engine.run()
        await engine._save_checkpoint("success")

        files = list(tmp_path.glob("*.json"))
        assert files, "expected a checkpoint file to be written"
        raw = files[0].read_text()
        for secret in (
            "sk-secret-do-not-log-AAAAAAAA",
            "P@ssw0rd!",
            "eyJhbGciOi.payload.sig",
        ):
            assert secret not in raw, f"{secret!r} leaked into {files[0]}"
        assert "alice" in raw

    async def test_redaction_applies_to_workflow_definition_state_block(
        self, tmp_path: Path
    ) -> None:
        from agentloom.core.models import StateKeyConfig

        wf = WorkflowDefinition(
            name="redact-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"api_key": "sk-yaml-seed"},
            state_schema={"api_key": StateKeyConfig(redact=True)},
            steps=[
                StepDefinition(
                    id="noop", type=StepType.LLM_CALL, prompt="hi", output="ans"
                )
            ],
        )
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, checkpointer=cp)
        await engine.run()
        await engine._save_checkpoint("success")

        import json

        doc = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert "sk-yaml-seed" not in json.dumps(doc)

    async def test_no_redaction_keeps_plaintext(self, tmp_path: Path) -> None:
        wf = WorkflowDefinition(
            name="no-redact",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"api_key": "plain"},
            steps=[
                StepDefinition(
                    id="noop", type=StepType.LLM_CALL, prompt="hi", output="ans"
                )
            ],
        )
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, checkpointer=cp)
        await engine.run()
        await engine._save_checkpoint("success")

        raw = next(tmp_path.glob("*.json")).read_text()
        assert "plain" in raw

    async def test_step_results_output_redacted(self, tmp_path: Path) -> None:
        from agentloom.core.models import StateKeyConfig
        from agentloom.core.results import StepResult, StepStatus

        wf = WorkflowDefinition(
            name="redact-step-output",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"user": "alice"},
            state_schema={"api_key": StateKeyConfig(redact=True)},
            steps=[
                StepDefinition(
                    id="extract", type=StepType.LLM_CALL, prompt="hi", output="data"
                )
            ],
        )
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, checkpointer=cp)

        leaky_result = StepResult(
            step_id="extract",
            status=StepStatus.SUCCESS,
            output={"api_key": "sk-leaked-via-step-output", "label": "ok"},
            duration_ms=1.0,
        )
        await engine.state.set_step_result("extract", leaky_result)
        await engine._save_checkpoint("success")

        raw = next(tmp_path.glob("*.json")).read_text()
        assert "sk-leaked-via-step-output" not in raw

    async def test_workflow_definition_fields_redacted(
        self, tmp_path: Path
    ) -> None:
        from agentloom.core.models import StateKeyConfig, WebhookConfig

        wf = WorkflowDefinition(
            name="redact-step-config",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"user": "alice"},
            state_schema={"api_key": StateKeyConfig(redact=True)},
            steps=[
                StepDefinition(
                    id="extract",
                    type=StepType.LLM_CALL,
                    prompt="hi",
                    tool_args={"api_key": "sk-tool-arg-leak"},
                ),
                StepDefinition(
                    id="approve",
                    type=StepType.APPROVAL_GATE,
                    depends_on=["extract"],
                    notify=WebhookConfig(
                        url="https://hooks.example.com/wh",
                        headers={"api_key": "sk-header-leak"},
                    ),
                    timeout_seconds=1,
                    on_timeout="reject",
                ),
            ],
        )
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, checkpointer=cp)
        await engine._save_checkpoint("success")

        raw = next(tmp_path.glob("*.json")).read_text()
        assert "sk-tool-arg-leak" not in raw
        assert "sk-header-leak" not in raw

    async def test_final_state_redacted_in_result(self, tmp_path: Path) -> None:
        from agentloom.core.models import StateKeyConfig

        wf = WorkflowDefinition(
            name="redact-final-state",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"api_key": "sk-final-state-leak", "user": "alice"},
            state_schema={"api_key": StateKeyConfig(redact=True)},
            steps=[
                StepDefinition(
                    id="noop", type=StepType.LLM_CALL, prompt="hi", output="ans"
                )
            ],
        )
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        dumped = result.model_dump_json()
        assert "sk-final-state-leak" not in dumped
        assert "alice" in dumped
