"""Tests for checkpoint integration in WorkflowEngine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest

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
from agentloom.core.redact import is_redacted
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

        engine = WorkflowEngine(
            workflow=_two_step_workflow(),
            provider_gateway=_mock_gateway(),
            checkpointer=checkpointer,
            run_id="resume-test",
        )
        first_result = await engine.run()
        assert first_result.status == WorkflowStatus.SUCCESS

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
        assert len(provider.calls) == 0

    async def test_resume_continues_from_midpoint(self, tmp_path: Path) -> None:
        """Simulate resuming when only step_a was completed."""
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        workflow = _two_step_workflow()

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
        assert len(provider.calls) == 1
        assert "step_b" in result.step_results
        assert result.step_results["step_b"].status == StepStatus.SUCCESS


class TestCheckpointErrorHandling:
    """Test that checkpoint save failures are handled gracefully."""

    async def test_save_checkpoint_io_error_is_swallowed(self, tmp_path: Path) -> None:
        """A checkpoint save failing with an I/O error does not fail the run."""

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
        assert result.status == WorkflowStatus.SUCCESS

    async def test_checkpoint_saved_on_budget_exceeded(self, tmp_path: Path) -> None:
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
        # The step succeeds; the charge after it is what breaks the budget.
        assert result.status in (WorkflowStatus.BUDGET_EXCEEDED, WorkflowStatus.FAILED)

        loaded = await checkpointer.load(engine.run_id)
        assert loaded.status in ("budget_exceeded", "failed")

    async def test_checkpoint_saved_on_failure(self, tmp_path: Path) -> None:
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
        """A completed router skipped on resume still activates its target."""
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
        assert result.step_results["billing"].status == StepStatus.SUCCESS
        assert result.step_results["general"].status == StepStatus.SKIPPED
        assert len(provider.calls) == 1


class TestCheckpointStateRedaction:
    """Checkpoint files carry no plaintext for redacted keys.

    Approval gates checkpoint on every pause, so keys flagged via ``state_schema`` or
    ``AGENTLOOM_REDACT_STATE_KEYS`` are masked before the JSON lands on disk: in the runtime
    snapshot and in the ``state:`` block inside ``workflow_definition``.
    """

    async def test_state_keys_are_redacted_in_persisted_file(self, tmp_path: Path) -> None:
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
            steps=[StepDefinition(id="noop", type=StepType.LLM_CALL, prompt="hi", output="ans")],
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
            steps=[StepDefinition(id="noop", type=StepType.LLM_CALL, prompt="hi", output="ans")],
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
                StepDefinition(id="extract", type=StepType.LLM_CALL, prompt="hi", output="data")
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

    async def test_workflow_definition_fields_redacted(self, tmp_path: Path) -> None:
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
            steps=[StepDefinition(id="noop", type=StepType.LLM_CALL, prompt="hi", output="ans")],
        )
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        dumped = result.model_dump_json()
        assert "sk-final-state-leak" not in dumped
        assert "alice" in dumped


class TestRedactionSparesOwnFields:
    """A pattern written for state keys never touches AgentLoom's own fields."""

    @staticmethod
    def _workflow(pattern: str, **config: Any) -> WorkflowDefinition:
        from agentloom.core.models import StateKeyConfig

        return WorkflowDefinition(
            name="redact-own-fields",
            config=WorkflowConfig(**{"provider": "mock", "model": "mock-model", **config}),
            state={"api_token": "dummy-token", "api_key": "dummy-key"},
            state_schema={pattern: StateKeyConfig(redact=True)},
            steps=[
                StepDefinition(
                    id="ask", type=StepType.LLM_CALL, prompt="hi", max_tokens=50, output="answer"
                ),
                StepDefinition(id="approve", type=StepType.APPROVAL_GATE, depends_on=["ask"]),
                StepDefinition(
                    id="leak",
                    type=StepType.LLM_CALL,
                    depends_on=["approve"],
                    prompt="p",
                    output="api_token",
                ),
            ],
        )

    async def test_a_token_pattern_leaves_the_run_and_its_resume_working(
        self, tmp_path: Path
    ) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(
            workflow=self._workflow("*token*"),
            provider_gateway=_mock_gateway(),
            checkpointer=cp,
            run_id="tok",
        )
        first = await engine.run()
        assert first.status == WorkflowStatus.PAUSED
        assert first.step_results["ask"].token_usage.prompt_tokens > 0
        assert is_redacted(first.final_state["api_token"])

        data = await cp.load("tok")
        assert data.workflow_definition["steps"][0]["max_tokens"] == 50
        resumed = await WorkflowEngine.from_checkpoint(
            data, cp, provider_gateway=_mock_gateway(), approval_decisions={"approve": "approved"}
        )
        second = await resumed.run()
        assert second.status == WorkflowStatus.SUCCESS
        second.model_dump_json()

    async def test_a_key_pattern_keeps_the_endpoint_variable_name(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        workflow = self._workflow(
            "*key*",
            provider="generic_openai",
            base_url="http://localhost:8000",
            api_key_env="VLLM_API_KEY",
        )
        engine = WorkflowEngine(
            workflow=workflow, provider_gateway=_mock_gateway(), checkpointer=cp
        )
        await engine.run()
        data = await cp.load(engine.run_id)
        assert data.workflow_definition["config"]["api_key_env"] == "VLLM_API_KEY"
        assert is_redacted(data.workflow_definition["state"]["api_key"])

    async def test_schemas_and_inline_subworkflows_stay_loadable(self, tmp_path: Path) -> None:
        from agentloom.core.models import StateKeyConfig

        workflow = WorkflowDefinition(
            name="redact-schemas",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state_schema={
                "*key*": StateKeyConfig(redact=True),
                "*token*": StateKeyConfig(redact=True),
            },
            steps=[
                StepDefinition(
                    id="ask",
                    type=StepType.LLM_CALL,
                    prompt="hi",
                    tools=[
                        {
                            "name": "lookup",
                            "parameters": {"properties": {"api_key": {"type": "string"}}},
                        }
                    ],
                ),
                StepDefinition(
                    id="sub",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline={
                        "name": "child",
                        "state": {"api_key": "dummy-child"},
                        "steps": [{"id": "a", "type": "llm_call", "prompt": "p", "max_tokens": 9}],
                    },
                ),
            ],
        )
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(workflow=workflow, checkpointer=cp, run_id="schemas")
        await engine._save_checkpoint("success")

        saved = (await cp.load("schemas")).workflow_definition
        tool = saved["steps"][0]["tools"][0]
        assert tool["parameters"]["properties"]["api_key"] == {"type": "string"}
        child = saved["steps"][1]["workflow_inline"]
        assert child["steps"][0]["max_tokens"] == 9
        assert is_redacted(child["state"]["api_key"])
        WorkflowDefinition.model_validate(saved)

    async def test_a_step_writing_a_secret_key_has_its_output_redacted(
        self, tmp_path: Path
    ) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(
            workflow=self._workflow("api_token"),
            provider_gateway=_mock_gateway(),
            checkpointer=cp,
            run_id="out",
        )
        await engine.run()
        data = await cp.load("out")
        resumed = await WorkflowEngine.from_checkpoint(
            data, cp, provider_gateway=_mock_gateway(), approval_decisions={"approve": "approved"}
        )
        result = await resumed.run()
        assert is_redacted(result.step_results["leak"].output)
        assert result.step_results["ask"].output == result.final_state["answer"]
        saved = await cp.load("out")
        assert is_redacted(saved.step_results["leak"]["output"])


class _Blocking(MockProvider):
    """Never answers the prompt ``block``: a run waiting on it can only be cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = anyio.Event()

    async def complete(self, messages: list[dict[str, Any]], model: str, **kwargs: Any) -> Any:
        if messages[-1].get("content") == "block":
            self.waiting.set()
            await anyio.sleep_forever()
        return await super().complete(messages, model, **kwargs)


class TestInterruptedRun:
    """A run cancelled from outside, as by Ctrl-C, saves what it finished."""

    @staticmethod
    async def _interrupt(tmp_path: Path, last: StepDefinition) -> FileCheckpointer:
        provider = _Blocking()
        gateway = ProviderGateway()
        gateway.register(provider, priority=0)
        workflow = WorkflowDefinition(
            name="interrupted",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[StepDefinition(id="one", type=StepType.LLM_CALL, prompt="a"), last],
        )
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        engine = WorkflowEngine(
            workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer, run_id="cut"
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(engine.run)
            await provider.waiting.wait()
            tg.cancel_scope.cancel()
        return checkpointer

    async def test_it_saves_the_finished_steps_and_resumes_after_them(self, tmp_path: Path) -> None:
        last = StepDefinition(id="two", type=StepType.LLM_CALL, depends_on=["one"], prompt="block")
        checkpointer = await self._interrupt(tmp_path, last)
        data = await checkpointer.load("cut")
        assert data.status == "cancelled"
        assert data.completed_steps == ["one"]

        provider = MockProvider()
        gateway = ProviderGateway()
        gateway.register(provider, priority=0)
        resumed = await WorkflowEngine.from_checkpoint(data, checkpointer, provider_gateway=gateway)
        result = await resumed.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert [call["messages"][-1]["content"] for call in provider.calls] == ["block"]

    async def test_a_subworkflow_leaves_the_checkpoint_to_its_parent(self, tmp_path: Path) -> None:
        last = StepDefinition(
            id="sub",
            type=StepType.SUBWORKFLOW,
            depends_on=["one"],
            workflow_inline={
                "name": "child",
                "steps": [{"id": "wait", "type": "llm_call", "prompt": "block"}],
            },
        )
        checkpointer = await self._interrupt(tmp_path, last)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["cut.json"]
        assert (await checkpointer.load("cut")).status == "cancelled"


class TestCheckpointSchemaStrictness:
    """A checkpoint from a newer runtime is refused at resume with a migration hint.

    A future ``schema_version`` or an unknown step status fails loudly instead of mis-resuming.
    """

    async def test_future_schema_version_rejected(self, tmp_path: Path) -> None:
        from agentloom.checkpointing.base import CURRENT_CHECKPOINT_SCHEMA_VERSION
        from agentloom.exceptions import CheckpointSchemaError

        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        workflow = _two_step_workflow()
        checkpoint_data = CheckpointData(
            workflow_name="future",
            run_id="future-run",
            schema_version=CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
            workflow_definition=workflow.model_dump(),
            state={},
            status="paused",
        )
        with pytest.raises(CheckpointSchemaError, match="newer than this runtime"):
            await WorkflowEngine.from_checkpoint(
                checkpoint_data=checkpoint_data, checkpointer=checkpointer
            )

    async def test_unknown_step_status_enum_rejected(self, tmp_path: Path) -> None:
        from agentloom.exceptions import CheckpointSchemaError

        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        workflow = _two_step_workflow()
        checkpoint_data = CheckpointData(
            workflow_name="enum",
            run_id="enum-run",
            workflow_definition=workflow.model_dump(),
            state={},
            step_results={
                # ``future_unknown`` is not a StepStatus this runtime defines.
                "step_a": {"step_id": "step_a", "status": "future_unknown"},
            },
            status="running",
        )
        with pytest.raises(CheckpointSchemaError, match="cannot parse"):
            await WorkflowEngine.from_checkpoint(
                checkpoint_data=checkpoint_data, checkpointer=checkpointer
            )

    async def test_current_schema_version_resumes(self, tmp_path: Path) -> None:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        workflow = _two_step_workflow()
        checkpoint_data = CheckpointData(
            workflow_name="ok",
            run_id="ok-run",
            workflow_definition=workflow.model_dump(),
            state={"input": "hi"},
            status="running",
        )
        engine = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data, checkpointer=checkpointer
        )
        assert engine.workflow.name == "checkpoint-test"
