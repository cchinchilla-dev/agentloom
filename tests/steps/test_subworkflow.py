"""Tests for subworkflow step executor."""

from __future__ import annotations

from typing import Any

import pytest

from agentloom.core.models import StepDefinition, StepType, WorkflowConfig
from agentloom.core.results import StepStatus
from agentloom.core.state import StateManager
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.base import StepContext
from agentloom.steps.subworkflow import SubworkflowStep
from tests.conftest import MockProvider


class TestSubworkflowStep:
    @pytest.fixture
    def step(self) -> SubworkflowStep:
        return SubworkflowStep()

    @pytest.fixture
    def gateway(self) -> ProviderGateway:
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        return gw

    async def test_no_path_or_inline_raises(self, step: SubworkflowStep) -> None:
        ctx = StepContext(
            step_definition=StepDefinition(id="sub", type=StepType.SUBWORKFLOW),
            state_manager=StateManager(),
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        with pytest.raises(Exception, match="requires 'workflow_path' or 'workflow_inline'"):
            await step.execute(ctx)

    async def test_inline_subworkflow_executes(
        self, step: SubworkflowStep, gateway: ProviderGateway
    ) -> None:
        inline = {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [
                {
                    "id": "child_step",
                    "type": "llm_call",
                    "prompt": "hello",
                    "output": "child_out",
                }
            ],
        }
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=inline,
                output="sub_result",
            ),
            state_manager=StateManager(initial_state={"parent_data": "yes"}),
            provider_gateway=gateway,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert result.cost_usd > 0

    async def test_token_aggregation(self, step: SubworkflowStep, gateway: ProviderGateway) -> None:
        """Verify token usage sums all child steps, not just the last."""
        inline = {
            "name": "multi-step-child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [
                {"id": "a", "type": "llm_call", "prompt": "step a", "output": "out_a"},
                {
                    "id": "b",
                    "type": "llm_call",
                    "prompt": "step b",
                    "output": "out_b",
                    "depends_on": ["a"],
                },
            ],
        }
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=inline,
                output="sub_result",
            ),
            state_manager=StateManager(),
            provider_gateway=gateway,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        # MockProvider returns 30 tokens per call, 2 steps = 60
        assert result.token_usage.total_tokens == 60

    async def test_parent_state_passed_to_child(
        self, step: SubworkflowStep, gateway: ProviderGateway
    ) -> None:
        inline = {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [
                {
                    "id": "echo",
                    "type": "llm_call",
                    "prompt": "{state.parent_val}",
                    "output": "out",
                }
            ],
        }
        state_mgr = StateManager(initial_state={"parent_val": "inherited"})
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=inline,
                output="sub_out",
            ),
            state_manager=state_mgr,
            provider_gateway=gateway,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS

    async def test_invalid_inline_returns_error(self, step: SubworkflowStep) -> None:
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline={"name": "bad", "steps": "not-a-list"},
            ),
            state_manager=StateManager(),
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        with pytest.raises(Exception, match="Invalid inline subworkflow"):
            await step.execute(ctx)


class TestSubworkflowObservability:
    """Parent observer must receive events from the child engine."""

    async def test_child_observer_receives_events(self, mock_gateway) -> None:
        gateway = mock_gateway
        from agentloom.core.models import StepDefinition, StepType
        from agentloom.core.state import StateManager
        from agentloom.steps.base import StepContext
        from agentloom.steps.subworkflow import SubworkflowStep

        events: list[tuple[str, tuple]] = []

        class Recorder:
            def __getattr__(self, name):
                def _hook(*args, **_kwargs):
                    events.append((name, args))

                return _hook

        observer = Recorder()
        inline = {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [
                {"id": "c", "type": "llm_call", "prompt": "hi", "output": "out"},
            ],
        }
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub", type=StepType.SUBWORKFLOW, workflow_inline=inline
            ),
            state_manager=StateManager(),
            provider_gateway=gateway,
            workflow_model="mock-model",
            observer=observer,
            run_id="parent-run",
        )
        await SubworkflowStep().execute(ctx)

        seen = {name for name, _ in events}
        assert "on_workflow_start" in seen
        assert "on_workflow_end" in seen
        # Child step events surface to the parent observer.
        assert "on_step_start" in seen
        assert "on_step_end" in seen


class TestSubworkflowFailurePaths:
    """Failure branches inside ``SubworkflowStep.execute()``."""

    async def test_bad_workflow_path_raises_step_error(self) -> None:
        from agentloom.exceptions import StepError

        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_path="/does/not/exist.yaml",
            ),
            state_manager=StateManager(),
        )
        with pytest.raises(StepError, match="Failed to load subworkflow"):
            await SubworkflowStep().execute(ctx)

    async def test_invalid_inline_raises_step_error(self) -> None:
        from agentloom.exceptions import StepError

        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline={"steps": []},  # malformed: no name
            ),
            state_manager=StateManager(),
        )
        with pytest.raises(StepError, match="Invalid inline subworkflow"):
            await SubworkflowStep().execute(ctx)

    async def test_child_engine_failure_returns_failed_result(self) -> None:
        gateway = ProviderGateway()
        gateway.register(MockProvider(), priority=0)
        inline = {
            "name": "broken-child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [
                {"id": "r", "type": "router"},  # invalid: no conditions, no default
            ],
        }
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=inline,
            ),
            state_manager=StateManager(),
            provider_gateway=gateway,
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.FAILED


class TestSubworkflowInheritsSecurityPosture:
    """Subworkflow must inherit the parent's redaction policy AND sandbox."""

    async def test_child_inherits_parent_redaction_policy(self, tmp_path: Any) -> None:
        from agentloom.checkpointing.file import FileCheckpointer
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.models import (
            StateKeyConfig,
            WorkflowDefinition,
        )

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, priority=0)
        cp = FileCheckpointer(checkpoint_dir=tmp_path)

        parent = WorkflowDefinition(
            name="parent",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"api_key": "sk-parent-secret"},
            state_schema={"api_key": StateKeyConfig(redact=True)},
            steps=[
                StepDefinition(
                    id="sub",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline={
                        "name": "child",
                        "config": {"provider": "mock", "model": "x"},
                        "steps": [
                            {
                                "id": "noop",
                                "type": "llm_call",
                                "prompt": "hi",
                                "output": "ans",
                            }
                        ],
                    },
                ),
            ],
        )
        engine = WorkflowEngine(workflow=parent, provider_gateway=gw, checkpointer=cp)
        await engine.run()

        for f in tmp_path.glob("*.json"):
            assert "sk-parent-secret" not in f.read_text(), f.name

    async def test_child_inherits_parent_sandbox_config(self) -> None:
        from agentloom.core.models import SandboxConfig

        gateway = ProviderGateway()
        gateway.register(MockProvider(), priority=0)
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline={
                    "name": "child",
                    "config": {"provider": "mock", "model": "x"},
                    "steps": [
                        {
                            "id": "noop",
                            "type": "llm_call",
                            "prompt": "hi",
                        }
                    ],
                },
            ),
            state_manager=StateManager(),
            provider_gateway=gateway,
            sandbox_config=SandboxConfig(
                enabled=True,
                allowed_commands=["echo"],
                allowed_domains=["api.openai.com"],
                allowed_schemes=["https"],
            ),
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS


class TestSubworkflowStateIsolation:
    """#057 regression — opt-in state isolation must keep parent state hidden.

    The default ``isolated_state: False`` keeps the pre-0.5.0 leaky
    behaviour for backwards compatibility (every existing workflow keeps
    running unchanged). Setting it to ``True`` seeds the child only from
    its own declared ``state:`` plus the explicit ``input:`` mapping —
    parent state is invisible. ``return_keys`` filters what travels back
    up so the parent's named ``output:`` key holds only the agreed slice.
    """

    @pytest.fixture
    def gateway(self) -> ProviderGateway:
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        return gw

    async def test_isolated_state_hides_parent_keys(self, gateway: ProviderGateway) -> None:
        inline = {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "state": {"child_only": True},
            "steps": [
                {
                    "id": "leaf",
                    "type": "llm_call",
                    "prompt": "ok",
                    "output": "leaf_out",
                }
            ],
        }
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=inline,
                isolated_state=True,
                input={"forwarded": "ok"},
                output="result",
            ),
            state_manager=StateManager(initial_state={"parent_secret": "do-not-leak"}),
            provider_gateway=gateway,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert isinstance(result.output, dict)
        # Parent state did not propagate into the child's final state.
        assert "parent_secret" not in result.output
        # Child saw its own declared state + the explicit ``input:`` seed.
        assert result.output["child_only"] is True
        assert result.output["forwarded"] == "ok"
        assert result.output["leaf_out"] == "Mock response"

    async def test_return_keys_filters_child_state_at_boundary(
        self, gateway: ProviderGateway
    ) -> None:
        inline = {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "state": {"keep_me": "yes", "drop_me": "secret"},
            "steps": [
                {
                    "id": "leaf",
                    "type": "llm_call",
                    "prompt": "ok",
                    "output": "leaf_out",
                }
            ],
        }
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=inline,
                isolated_state=True,
                return_keys=["leaf_out", "keep_me"],
                output="result",
            ),
            state_manager=StateManager(),
            provider_gateway=gateway,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert isinstance(result.output, dict)
        assert set(result.output.keys()) == {"leaf_out", "keep_me"}
        assert "drop_me" not in result.output

    async def test_default_isolated_state_false_keeps_legacy_behaviour(
        self, gateway: ProviderGateway
    ) -> None:
        """Default behaviour must continue to leak parent state both ways."""
        inline = {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [
                {
                    "id": "leaf",
                    "type": "llm_call",
                    "prompt": "{state.parent_val}",
                    "output": "leaf_out",
                }
            ],
        }
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=inline,
                output="result",
            ),
            state_manager=StateManager(initial_state={"parent_val": "leaked"}),
            provider_gateway=gateway,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        # Pre-0.5.0 contract: child sees parent keys.
        assert result.output["parent_val"] == "leaked"


class TestSubworkflowPausePropagation:
    """#057 regression — approval-gate pauses must cross subworkflow boundaries.

    Pre-0.5.0 the parent treated a child pause as a generic exception,
    marked the subworkflow FAILED, and left no resume path. Now the
    subworkflow re-raises ``PauseRequestedError`` with a qualified
    ``parent.child`` step id; the parent checkpoint stores ``sub.gate``
    so ``agentloom resume <parent_run_id> --approve`` lands on the gate
    inside the child.
    """

    async def test_subworkflow_with_approval_gate_pauses_parent(self) -> None:
        import tempfile

        from agentloom.checkpointing.file import FileCheckpointer
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.parser import WorkflowParser
        from agentloom.core.results import WorkflowStatus

        yaml_text = """
name: parent
config: {provider: mock, model: x}
steps:
  - id: sub
    type: subworkflow
    workflow_inline:
      name: gated
      config: {provider: mock, model: x}
      steps:
        - id: gate
          type: approval_gate
"""
        chk = FileCheckpointer(checkpoint_dir=tempfile.mkdtemp())
        wf = WorkflowParser.from_yaml(yaml_text)
        eng = WorkflowEngine(workflow=wf, checkpointer=chk)
        result = await eng.run()

        assert result.status == WorkflowStatus.PAUSED
        # Parent reports the fully-qualified path so the operator sees
        # ``sub.gate`` not the bare ``sub``.
        assert "sub.gate" in (result.error or "")

        # Checkpoint must persist the qualified id so resume can dispatch
        # the decision to the right child step.
        chk_data = await chk.load(eng.run_id)
        assert chk_data.paused_step_id == "sub.gate"

    async def test_subworkflow_pause_resume_round_trip(self) -> None:
        """End-to-end: pause at sub.gate, resume --approve, child gate finishes."""
        import tempfile

        from agentloom.checkpointing.file import FileCheckpointer
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.parser import WorkflowParser
        from agentloom.core.results import WorkflowStatus

        yaml_text = """
name: parent
config: {provider: mock, model: x}
steps:
  - id: sub
    type: subworkflow
    workflow_inline:
      name: gated
      config: {provider: mock, model: x}
      steps:
        - id: gate
          type: approval_gate
"""
        chk = FileCheckpointer(checkpoint_dir=tempfile.mkdtemp())
        wf = WorkflowParser.from_yaml(yaml_text)
        eng = WorkflowEngine(workflow=wf, checkpointer=chk)
        first = await eng.run()
        assert first.status == WorkflowStatus.PAUSED
        run_id = eng.run_id

        chk_data = await chk.load(run_id)
        eng2 = await WorkflowEngine.from_checkpoint(
            chk_data,
            checkpointer=chk,
            approval_decisions={chk_data.paused_step_id: "approved"},
        )
        second = await eng2.run()
        assert second.status == WorkflowStatus.SUCCESS
        assert second.step_results["sub"].status == StepStatus.SUCCESS
