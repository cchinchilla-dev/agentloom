"""Tests for subworkflow step executor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import respx

from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.models import SandboxConfig, StepDefinition, StepType, WorkflowConfig
from agentloom.core.redact import RedactionPolicy, is_redacted
from agentloom.core.results import StepResult, StepStatus
from agentloom.core.state import StateManager
from agentloom.exceptions import PauseRequestedError, SandboxViolationError, StepError
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.base import StepContext
from agentloom.steps.subworkflow import SubworkflowStep, _child_registry, _effective_sandbox
from agentloom.tools.base import BaseTool
from agentloom.tools.builtins import ShellCommandTool, register_builtins
from agentloom.tools.registry import ToolRegistry
from agentloom.tools.sandbox import ToolSandbox
from tests.conftest import MockProvider, MockTool


class TestSubworkflowStep:
    @pytest.fixture
    def step(self) -> SubworkflowStep:
        return SubworkflowStep()

    @pytest.fixture
    def gateway(self) -> ProviderGateway:
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        return gw

    def test_no_path_or_inline_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 'workflow_path' or 'workflow_inline'"):
            StepDefinition(id="sub", type=StepType.SUBWORKFLOW)

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
        # Two steps at MockProvider's 30 tokens per call.
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
                # Loads, and fails when it runs: nothing matches and there is no default.
                {
                    "id": "r",
                    "type": "router",
                    "conditions": [{"expression": "False", "target": "r"}],
                },
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
    """A subworkflow inherits the parent's redaction policy and sandbox."""

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
    """Opt-in state isolation keeps parent state hidden from the child.

    By default (``isolated_state: False``) the child shares parent state. With ``True`` it is
    seeded only from its own ``state:`` plus the ``input:`` mapping, and ``return_keys`` filters
    what travels back to the parent's ``output:`` key.
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
        assert "parent_secret" not in result.output
        assert result.output["child_only"] is True
        assert result.output["forwarded"] == "ok"
        assert result.output["leaf_out"] == "Mock response"

    @staticmethod
    def _inline(**fields: Any) -> dict[str, Any]:
        return {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [{"id": "leaf", "type": "llm_call", "prompt": "ok", "output": "leaf_out"}],
            **fields,
        }

    async def test_input_is_rendered_with_the_parents_state(self, gateway: ProviderGateway) -> None:
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=self._inline(),
                isolated_state=True,
                input={
                    "greeting": "hi {state.who}",
                    "items": "state.items",
                    "nested": {"names": ["{state.who}"]},
                    "literal": {"value": "{state.who}", "template": False},
                },
                output="result",
            ),
            state_manager=StateManager(initial_state={"who": "ada", "items": [1, 2]}),
            provider_gateway=gateway,
            workflow_model="mock-model",
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert result.output["greeting"] == "hi ada"
        assert result.output["items"] == [1, 2]
        assert result.output["nested"] == {"names": ["ada"]}
        assert result.output["literal"] == "{state.who}"

    async def test_input_naming_a_missing_key_fails_for_good(
        self, gateway: ProviderGateway
    ) -> None:
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=self._inline(),
                isolated_state=True,
                input={"topic": "state.absent"},
            ),
            state_manager=StateManager(initial_state={}),
            provider_gateway=gateway,
            workflow_model="mock-model",
        )
        with pytest.raises(StepError, match="input references 'state.absent'") as info:
            await SubworkflowStep().execute(ctx)
        assert info.value.is_retryable is False

    async def test_child_state_seeds_defaults_under_the_parents_values(
        self, gateway: ProviderGateway
    ) -> None:
        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline=self._inline(state={"own": "child", "shared": "child"}),
                output="result",
            ),
            state_manager=StateManager(initial_state={"shared": "parent", "theirs": 1}),
            provider_gateway=gateway,
            workflow_model="mock-model",
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert result.output["own"] == "child"
        assert result.output["shared"] == "parent"
        assert result.output["theirs"] == 1

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
        """By default the child sees parent state, and its final state carries it back."""
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
        assert result.output["parent_val"] == "leaked"


class TestSubworkflowPausePropagation:
    """Approval-gate pauses cross subworkflow boundaries.

    The subworkflow raises ``PauseRequestedError`` with a qualified ``parent.child`` step id; the
    parent checkpoint stores ``sub.gate``, so ``agentloom resume <parent_run_id> --approve`` lands
    on the gate inside the child.
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
        # The operator sees the qualified ``sub.gate``, not the bare ``sub``.
        assert "sub.gate" in (result.error or "")

        # Resume dispatches the decision to the child step by this qualified id.
        chk_data = await chk.load(eng.run_id)
        assert chk_data.paused_step_id == "sub.gate"

    async def test_failed_child_workflow_surfaces_error_on_parent_step(self) -> None:
        """A failed child's error is surfaced on the parent's subworkflow step result.

        Dependents' ``skipped due to upstream failure: sub`` then has a cause visible from the
        parent result alone, without opening the child's checkpoint.
        """
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.models import RetryConfig
        from agentloom.core.parser import WorkflowParser
        from agentloom.core.results import WorkflowStatus

        # ``from_yaml`` calls ``Path.exists()`` on its input: ENAMETOOLONG on a long YAML string.
        wf = WorkflowParser.from_dict(
            {
                "name": "parent",
                "config": {"provider": "mock", "model": "x"},
                "steps": [
                    {
                        "id": "sub",
                        "type": "subworkflow",
                        "workflow_inline": {
                            "name": "gated",
                            "config": {"provider": "mock", "model": "x"},
                            "steps": [
                                {
                                    "id": "doomed",
                                    "type": "tool",
                                    "tool_name": "does_not_exist",
                                }
                            ],
                        },
                    }
                ],
            }
        )
        # Reduce retries to keep this test fast.
        wf.steps[0].retry = RetryConfig(max_retries=0)
        eng = WorkflowEngine(workflow=wf)
        result = await eng.run()

        assert result.status == WorkflowStatus.FAILED
        sub_result = result.step_results["sub"]
        assert sub_result.status == StepStatus.FAILED
        # Loose: the child's own error and the fallback naming its step both carry the cause.
        assert sub_result.error is not None
        assert "doomed" in sub_result.error or "does_not_exist" in sub_result.error

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


class TestSubworkflowApprovalRewriteWithDottedStepId:
    """Subworkflow step ids that contain a ``.`` still resume.

    ``WorkflowEngine.from_checkpoint`` sets decisions at ``_approval.{step_id}``, which nests at
    every dot, so the hand-down to the child walks the path segment by segment: a literal
    ``"my.sub"`` lookup would miss and block the child gate again.
    """

    async def test_dotted_step_id_resolves_nested_approval(self) -> None:
        from agentloom.steps.subworkflow import SubworkflowStep

        # As ``from_checkpoint`` nests the decision for gate ``gate`` under step id ``parent.sub``.
        parent_state = StateManager(
            initial_state={"_approval": {"parent": {"sub": {"gate": "approved"}}}}
        )
        captured_child_state: dict[str, Any] = {}

        async def _capture_run(self: Any) -> Any:
            from agentloom.core.results import WorkflowResult, WorkflowStatus

            snap = await self.state.get_state_snapshot()
            captured_child_state.update(snap)
            return WorkflowResult(
                workflow_name="child",
                status=WorkflowStatus.SUCCESS,
                step_results={},
                final_state=snap,
                total_duration_ms=0.0,
            )

        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            from agentloom.core.engine import WorkflowEngine

            mp.setattr(WorkflowEngine, "run", _capture_run)
            ctx = StepContext(
                step_definition=StepDefinition(
                    id="parent.sub",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline={
                        "name": "child",
                        "config": {"provider": "mock", "model": "x"},
                        "steps": [{"id": "gate", "type": "approval_gate"}],
                    },
                ),
                state_manager=parent_state,
                workflow_config=WorkflowConfig(),
                workflow_model="mock-model",
            )
            await SubworkflowStep().execute(ctx)

        assert captured_child_state.get("_approval", {}).get("gate") == "approved"


class TestSubworkflowDefensiveBranches:
    """Defensive paths in ``SubworkflowStep.execute`` that the child engine does not reach.

    * ``run()`` returns a paused result rather than raising ``PauseRequestedError``; a raised
      pause still surfaces the qualified ``parent.child`` step id.
    * The engine always fills ``result.error`` on failure; a hand-built ``WorkflowResult``
      without one falls back to the first failed child step's error.
    """

    @pytest.mark.parametrize("budget", [None, 1.0], ids=["shared", "own"])
    async def test_defensive_pause_reraise_propagates_qualified_id(
        self, monkeypatch: Any, budget: float | None
    ) -> None:
        from agentloom.core.attempts import attempt_spend, enter_attempt
        from agentloom.core.engine import WorkflowEngine
        from agentloom.exceptions import PauseRequestedError

        async def fake_run(self: Any) -> Any:
            await self._budget.charge(0.1)  # paid before its gate
            raise PauseRequestedError("inner-gate")

        monkeypatch.setattr(WorkflowEngine, "run", fake_run)

        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline={
                    "name": "child",
                    "config": {"provider": "mock", "model": "x", "budget_usd": budget},
                    "steps": [{"id": "noop", "type": "llm_call", "prompt": "hi"}],
                },
            ),
            state_manager=StateManager(),
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        enter_attempt(())
        with pytest.raises(PauseRequestedError) as exc_info:
            await SubworkflowStep().execute(ctx)
        assert exc_info.value.step_id == "sub.inner-gate"
        # A budget of its own: the calling attempt pays what the child spent before the gate
        assert attempt_spend().cost_usd == pytest.approx(0.1 if budget else 0.0)

    async def test_falls_back_to_inner_step_error_when_workflow_error_is_empty(
        self, monkeypatch: Any
    ) -> None:
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.results import (
            WorkflowResult,
            WorkflowStatus,
        )

        async def fake_run(self: Any) -> Any:
            return WorkflowResult(
                workflow_name="child",
                status=WorkflowStatus.FAILED,
                step_results={
                    "inner": StepResult(step_id="inner", status=StepStatus.FAILED, error="boom"),
                },
                final_state={},
                total_duration_ms=0.0,
                error=None,
            )

        monkeypatch.setattr(WorkflowEngine, "run", fake_run)

        ctx = StepContext(
            step_definition=StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                workflow_inline={
                    "name": "child",
                    "config": {"provider": "mock", "model": "x"},
                    "steps": [{"id": "inner", "type": "llm_call", "prompt": "x"}],
                },
            ),
            state_manager=StateManager(),
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await SubworkflowStep().execute(ctx)
        assert result.status == StepStatus.FAILED
        assert result.error is not None
        assert "inner" in result.error and "boom" in result.error


class TestSubworkflowFailedAttempt:
    """A failed attempt leaves no output for the retry to inherit.

    The child's ``inner`` step is answered from a recording whose turns can
    be faults: a 503 fails the child, and the parent retries ``sub``.
    """

    FAULT = {"fault": {"type": "http_error", "status_code": 503}}
    ANSWER = {"content": "done"}

    @staticmethod
    async def _run(
        recording: Any, isolated: bool, turns: list[dict[str, Any]], first: bool = False
    ) -> Any:
        """``first`` adds a child step before ``inner`` that answers, so the child leaves state."""
        import json

        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.parser import WorkflowParser
        from agentloom.providers.mock import MockProvider as RecordedMock

        config = {"provider": "mock", "model": "m"}
        inner = {"id": "inner", "type": "llm_call", "prompt": "hi", "output": "answer"}
        inner["retry"] = {"max_retries": 0}
        child = {"name": "child", "config": config, "steps": [inner]}
        if first:
            inner["depends_on"] = ["first"]
            first_step = {"id": "first", "type": "llm_call", "prompt": "one", "output": "first"}
            child["steps"] = [first_step, inner]
        sub = {"id": "sub", "type": "subworkflow", "workflow_inline": child, "output": "sub"}
        sub.update(isolated_state=isolated, retry={"max_retries": 1, "backoff_max": 0})
        workflow = WorkflowParser.from_dict({"name": "parent", "config": config, "steps": [sub]})
        recording.write_text(json.dumps({"inner": turns, "first": {"content": "kept"}}))
        gateway = ProviderGateway()
        gateway.register(RecordedMock(responses_file=recording), priority=0)
        return await WorkflowEngine(workflow=workflow, provider_gateway=gateway).run()

    @pytest.mark.parametrize("isolated", [False, True], ids=["shared", "isolated"])
    async def test_a_retried_subworkflow_ends_as_if_it_had_not_failed(
        self, tmp_path: Any, isolated: bool
    ) -> None:
        from agentloom.core.results import WorkflowStatus

        clean = await self._run(tmp_path / "clean.json", isolated, [self.ANSWER])
        retried = await self._run(tmp_path / "retried.json", isolated, [self.FAULT, self.ANSWER])
        assert retried.status == WorkflowStatus.SUCCESS
        assert retried.final_state["sub"]["answer"] == "done"
        assert retried.final_state == clean.final_state

    @pytest.mark.parametrize("isolated", [False, True], ids=["shared", "isolated"])
    async def test_a_failed_subworkflow_writes_no_output(
        self, tmp_path: Any, isolated: bool
    ) -> None:
        from agentloom.core.results import WorkflowStatus

        turns = [self.FAULT, self.FAULT]
        failed = await self._run(tmp_path / "failed.json", isolated, turns, first=True)
        assert failed.status == WorkflowStatus.FAILED
        assert "sub" not in failed.final_state
        assert "sub" not in failed.final_state.get("steps", {})
        # The child's final state, kept on the result only
        assert failed.step_results["sub"].output["first"] == "kept"


def _shell_child(command: str, sandbox: dict[str, Any], cwd: str) -> dict[str, Any]:
    return {
        "name": "child",
        "config": {"provider": "mock", "model": "x", "sandbox": sandbox},
        "steps": [
            {
                "id": "run",
                "type": "tool",
                "tool_name": "shell_command",
                "tool_args": {"command": command, "cwd": cwd},
                "retry": {"max_retries": 0},
                "output": "ran",
            }
        ],
    }


def _sandboxed_context(
    inline: dict[str, Any], registry: ToolRegistry, parent: SandboxConfig | None = None
) -> StepContext:
    return StepContext(
        step_definition=StepDefinition(id="sub", type=StepType.SUBWORKFLOW, workflow_inline=inline),
        state_manager=StateManager(),
        tool_registry=registry,
        sandbox_config=parent or SandboxConfig(),
    )


def _builtins(sandbox: ToolSandbox | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    register_builtins(registry, sandbox=sandbox)
    return registry


@pytest.mark.skipif(sys.platform == "win32", reason="runs sh")
class TestChildSandboxReachesItsBuiltins:
    async def test_child_sandbox_binds_under_a_parent_with_none(self, tmp_path: Path) -> None:
        inline = _shell_child(
            "echo SUBPROBE",
            {"enabled": True, "allowed_commands": [], "allowed_paths": [str(tmp_path)]},
            str(tmp_path),
        )
        result = await SubworkflowStep().execute(_sandboxed_context(inline, _builtins()))
        assert result.status == StepStatus.FAILED, result.output
        assert "SUBPROBE" not in str(result.output)
        assert "not in allowlist" in (result.error or "")

    async def test_child_cannot_loosen_an_enabled_parent(self, tmp_path: Path) -> None:
        parent = SandboxConfig(
            enabled=True, allowed_commands=["echo"], allowed_paths=[str(tmp_path)]
        )
        inline = _shell_child(
            "ls .",
            {"enabled": True, "allowed_commands": ["echo", "ls"], "allowed_paths": [str(tmp_path)]},
            str(tmp_path),
        )
        registry = _builtins(ToolSandbox(**parent.model_dump()))
        result = await SubworkflowStep().execute(_sandboxed_context(inline, registry, parent))
        assert result.status == StepStatus.FAILED, result.output
        assert "not in allowlist" in (result.error or "")

    async def test_child_tightens_an_enabled_parent(self, tmp_path: Path) -> None:
        parent = SandboxConfig(
            enabled=True, allowed_commands=["echo", "ls"], allowed_paths=[str(tmp_path)]
        )
        inline = _shell_child(
            "ls .",
            {"enabled": True, "allowed_commands": ["echo"], "allowed_paths": [str(tmp_path)]},
            str(tmp_path),
        )
        registry = _builtins(ToolSandbox(**parent.model_dump()))
        result = await SubworkflowStep().execute(_sandboxed_context(inline, registry, parent))
        assert result.status == StepStatus.FAILED, result.output
        assert "not in allowlist" in (result.error or "")

    async def test_child_without_a_sandbox_runs_under_the_parents(self, tmp_path: Path) -> None:
        parent = SandboxConfig(
            enabled=True, allowed_commands=["echo"], allowed_paths=[str(tmp_path)]
        )
        registry = _builtins(ToolSandbox(**parent.model_dump()))
        allowed = await SubworkflowStep().execute(
            _sandboxed_context(_shell_child("echo ok", {}, str(tmp_path)), registry, parent)
        )
        assert allowed.status == StepStatus.SUCCESS, allowed.error
        refused = await SubworkflowStep().execute(
            _sandboxed_context(_shell_child("ls .", {}, str(tmp_path)), registry, parent)
        )
        assert refused.status == StepStatus.FAILED

    async def test_child_cannot_list_an_internal_host_the_parent_refuses(self) -> None:
        parent = SandboxConfig(enabled=True)
        url = "http://169.254.169.254/latest/meta-data/"
        inline = {
            "name": "child",
            "config": {
                "provider": "mock",
                "model": "x",
                "sandbox": {"enabled": True, "allowed_domains": ["169.254.169.254"]},
            },
            "steps": [
                {
                    "id": "fetch",
                    "type": "tool",
                    "tool_name": "http_request",
                    "tool_args": {"url": url},
                    "retry": {"max_retries": 0},
                }
            ],
        }
        registry = _builtins(ToolSandbox(**parent.model_dump()))
        with respx.mock(assert_all_called=False) as router:
            route = router.get(url).respond(200, text="dummy-credentials")
            result = await SubworkflowStep().execute(_sandboxed_context(inline, registry, parent))
        assert not route.called
        assert result.status == StepStatus.FAILED
        assert "internal" in (result.error or "")

    async def test_the_parents_tools_stay_as_they_were(self, tmp_path: Path) -> None:
        registry = _builtins()
        inline = _shell_child(
            "echo SUBPROBE",
            {"enabled": True, "allowed_commands": [], "allowed_paths": [str(tmp_path)]},
            str(tmp_path),
        )
        await SubworkflowStep().execute(_sandboxed_context(inline, registry))
        outcome = await registry.get("shell_command").execute(command="echo still-free")
        assert outcome["stdout"].strip() == "still-free"

    async def test_a_resumed_child_gets_the_same_tools(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        inline = _shell_child(
            "echo SUBPROBE",
            {"enabled": True, "allowed_commands": [], "allowed_paths": [str(tmp_path)]},
            str(tmp_path),
        )
        ctx = _sandboxed_context(inline, _builtins())
        ctx.checkpointer = FileCheckpointer(checkpoint_dir=tmp_path / "checkpoints")
        await ctx.state_manager.set_step_result(
            "sub", StepResult(step_id="sub", status=StepStatus.PAUSED)
        )

        class StopError(Exception):
            pass

        seen: dict[str, Any] = {}

        async def from_checkpoint(cls: Any, paused: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            raise StopError

        async def paused_child(*args: Any) -> Any:
            return SimpleNamespace(spent_usd=0.0)

        monkeypatch.setattr("agentloom.steps.subworkflow._paused_child", paused_child)
        monkeypatch.setattr(
            "agentloom.core.engine.WorkflowEngine.from_checkpoint",
            classmethod(from_checkpoint),
        )
        with pytest.raises(StopError):
            await SubworkflowStep().execute(ctx)
        with pytest.raises(SandboxViolationError, match="not in allowlist"):
            await (
                seen["tool_registry"]
                .get("shell_command")
                .execute(command="echo SUBPROBE", cwd=str(tmp_path))
            )


class TestChildRegistry:
    def _child(self, registry: ToolRegistry) -> ToolRegistry:
        return _child_registry(
            registry, SandboxConfig(), SandboxConfig(enabled=True, allowed_commands=[])
        )

    async def test_own_tools_and_tape_carry_over(self) -> None:
        registry = _builtins()
        own = MockTool()
        registry.register(own)
        registry.tape = object()  # type: ignore[assignment]
        child = self._child(registry)
        assert child.get("mock_tool") is own
        assert child.tape is registry.tape

    async def test_a_tool_of_your_own_under_a_builtin_name_is_left_alone(self) -> None:
        class OwnShell(BaseTool):
            name = "shell_command"

            async def execute(self, **kwargs: Any) -> Any:
                return "own"

        registry = ToolRegistry()
        own = OwnShell()
        registry.register(own)
        assert self._child(registry).get("shell_command") is own

    async def test_a_subclass_of_a_builtin_is_rebound(self) -> None:
        class Loud(ShellCommandTool):
            pass

        registry = ToolRegistry()
        registry.register(Loud())
        child = self._child(registry).get("shell_command")
        assert isinstance(child, Loud)
        with pytest.raises(SandboxViolationError):
            await child.execute(command="echo hi")

    def test_same_policy_keeps_the_parents_registry(self) -> None:
        registry = _builtins()
        policy = SandboxConfig(enabled=True, allowed_commands=["echo"])
        assert _child_registry(registry, policy, policy) is registry

    def test_a_disabled_policy_keeps_the_parents_registry(self) -> None:
        registry = _builtins()
        assert _child_registry(registry, SandboxConfig(), SandboxConfig(allow_network=False)) is (
            registry
        )

    def test_a_registry_stand_in_passes_through(self) -> None:
        stand_in = object()
        result = _child_registry(stand_in, SandboxConfig(), SandboxConfig(enabled=True))
        assert result is stand_in


class TestEffectiveSandbox:
    def _on(self, **fields: Any) -> SandboxConfig:
        return SandboxConfig(enabled=True, **fields)

    def test_neither_enabled_keeps_the_childs(self) -> None:
        child = SandboxConfig(allow_network=False)
        assert _effective_sandbox(SandboxConfig(), child) is child

    def test_only_the_parent_enabled(self) -> None:
        parent = self._on(allowed_commands=["echo"])
        assert _effective_sandbox(parent, SandboxConfig()) is parent

    def test_only_the_child_enabled(self) -> None:
        child = self._on(allowed_commands=["echo"])
        assert _effective_sandbox(SandboxConfig(), child) is child

    def test_commands_and_opt_ins_intersect(self) -> None:
        merged = _effective_sandbox(
            self._on(allowed_commands=["echo", "ls"], danger_opt_in=["bash", "python"]),
            self._on(allowed_commands=["ls", "cat"], danger_opt_in=["python"]),
        )
        assert merged.enabled is True
        assert merged.allowed_commands == ["ls"]
        assert merged.danger_opt_in == ["python"]

    def test_an_empty_command_list_allows_none(self) -> None:
        merged = _effective_sandbox(self._on(allowed_commands=["echo"]), self._on())
        assert merged.allowed_commands == []

    def test_permissive_booleans_need_both(self) -> None:
        merged = _effective_sandbox(
            self._on(allow_network=False, allow_internal_webhook_targets=True),
            self._on(allow_network=True, allow_internal_webhook_targets=False),
        )
        assert merged.allow_network is False
        assert merged.allow_internal_webhook_targets is False

    def test_smaller_write_limit(self) -> None:
        merged = _effective_sandbox(self._on(max_write_bytes=100), self._on(max_write_bytes=10))
        assert merged.max_write_bytes == 10
        one_sided = _effective_sandbox(self._on(), self._on(max_write_bytes=10))
        assert one_sided.max_write_bytes == 10
        assert _effective_sandbox(self._on(), self._on()).max_write_bytes is None

    def test_a_domain_list_on_one_side_applies(self) -> None:
        merged = _effective_sandbox(self._on(allowed_domains=["a.example"]), self._on())
        assert merged.allowed_domains == ["a.example"]
        assert merged.allow_network is True

    def test_domains_intersect_ignoring_case(self) -> None:
        merged = _effective_sandbox(
            self._on(allowed_domains=["A.example", "b.example"]),
            self._on(allowed_domains=["a.EXAMPLE"]),
        )
        assert merged.allowed_domains == ["a.example"]

    def test_disjoint_domains_refuse_the_network(self) -> None:
        # An empty allowed_domains means any domain: it must not stand for "none in common"
        merged = _effective_sandbox(
            self._on(allowed_domains=["a.example"]), self._on(allowed_domains=["b.example"])
        )
        assert merged.allow_network is False

    def test_disjoint_schemes_refuse_the_network(self) -> None:
        # An empty allowed_schemes falls back to http and https
        merged = _effective_sandbox(
            self._on(allowed_schemes=["http"]), self._on(allowed_schemes=["https"])
        )
        assert merged.allow_network is False

    def test_schemes_intersect(self) -> None:
        merged = _effective_sandbox(self._on(), self._on(allowed_schemes=["https"]))
        assert merged.allowed_schemes == ["https"]

    def test_of_nested_directories_the_deeper_one_applies(self, tmp_path: Path) -> None:
        inner = tmp_path / "inner"
        merged = _effective_sandbox(
            self._on(allowed_paths=[str(tmp_path)]), self._on(allowed_paths=[str(inner)])
        )
        assert merged.readable_paths == [str(inner.resolve())]
        assert merged.writable_paths == [str(inner.resolve())]
        assert merged.allowed_paths == []

    def test_read_only_on_one_side_stays_read_only(self, tmp_path: Path) -> None:
        merged = _effective_sandbox(
            self._on(allowed_paths=[str(tmp_path)]), self._on(readable_paths=[str(tmp_path)])
        )
        assert merged.readable_paths == [str(tmp_path.resolve())]
        assert merged.writable_paths == []

    def test_a_childs_domains_cannot_open_internal_hosts_the_parent_refuses(self) -> None:
        merged = _effective_sandbox(self._on(), self._on(allowed_domains=["169.254.169.254"]))
        assert merged.deny_internal_hosts is True
        with pytest.raises(SandboxViolationError, match="internal"):
            ToolSandbox(**merged.model_dump()).validate_network("http://169.254.169.254/latest/")

    def test_a_host_both_list_stays_reachable(self) -> None:
        merged = _effective_sandbox(
            self._on(allowed_domains=["localhost"]), self._on(allowed_domains=["localhost"])
        )
        assert merged.deny_internal_hosts is False
        ToolSandbox(**merged.model_dump()).validate_network("http://localhost/x")

    def test_a_host_the_parent_lists_stays_reachable_from_a_child_with_no_list(self) -> None:
        merged = _effective_sandbox(self._on(allowed_domains=["localhost"]), self._on())
        assert merged.deny_internal_hosts is False

    def test_the_refusal_reaches_a_grandchild(self) -> None:
        child = _effective_sandbox(self._on(), self._on(allowed_domains=["localhost"]))
        grandchild = _effective_sandbox(child, self._on())
        assert grandchild.deny_internal_hosts is True

    def test_no_domains_on_either_side_sets_nothing(self) -> None:
        assert _effective_sandbox(self._on(), self._on()).deny_internal_hosts is False

    def test_unrelated_directories_leave_nothing(self, tmp_path: Path) -> None:
        merged = _effective_sandbox(
            self._on(allowed_paths=[str(tmp_path / "a")]),
            self._on(allowed_paths=[str(tmp_path / "b")]),
        )
        assert merged.readable_paths == []
        assert merged.writable_paths == []


class TestInputThatCarriesASecret:
    """A redacted key handed over under another name stays redacted."""

    SECRET = "dummy-secret-value"

    def _context(
        self, tmp_path: Path, inline: dict[str, Any], **step: Any
    ) -> tuple[StepContext, MockTool]:
        tool = MockTool()
        registry = ToolRegistry()
        registry.register(tool)
        definition = StepDefinition(
            id="sub",
            type=StepType.SUBWORKFLOW,
            workflow_inline=inline,
            isolated_state=True,
            input={"credential": "state.api_key", "topic": "state.topic"},
            **step,
        )
        context = StepContext(
            step_definition=definition,
            state_manager=StateManager({"api_key": self.SECRET, "topic": "billing"}),
            tool_registry=registry,
            run_id="root",
            redaction_policy=RedactionPolicy(["api_key"]),
            checkpointer=FileCheckpointer(checkpoint_dir=tmp_path),
        )
        return context, tool

    @staticmethod
    def _child(*steps: dict[str, Any]) -> dict[str, Any]:
        return {"name": "child", "config": {"provider": "mock", "model": "x"}, "steps": list(steps)}

    async def test_the_childs_checkpoint_never_holds_it(self, tmp_path: Path) -> None:
        inline = self._child({"id": "gate", "type": "approval_gate"})
        context, _ = self._context(tmp_path, inline)
        with pytest.raises(PauseRequestedError):
            await SubworkflowStep().execute(context)
        saved = (tmp_path / "root.sub.json").read_text(encoding="utf-8")
        assert self.SECRET not in saved
        assert "billing" in saved

    async def test_a_template_counts_too(self, tmp_path: Path) -> None:
        inline = self._child({"id": "gate", "type": "approval_gate"})
        context, _ = self._context(tmp_path, inline)
        context.step_definition.input["credential"] = "Bearer {state.api_key}"
        with pytest.raises(PauseRequestedError):
            await SubworkflowStep().execute(context)
        assert self.SECRET not in (tmp_path / "root.sub.json").read_text(encoding="utf-8")

    async def test_the_child_reads_it_and_the_parent_gets_it_back_masked(
        self, tmp_path: Path
    ) -> None:
        inline = self._child(
            {
                "id": "use",
                "type": "tool",
                "tool_name": "mock_tool",
                "tool_args": {"input": "state.credential"},
            }
        )
        context, tool = self._context(tmp_path, inline, output="sub_out")
        result = await SubworkflowStep().execute(context)
        assert result.status == StepStatus.SUCCESS, result.error
        assert tool.calls == [{"input": self.SECRET}]
        assert is_redacted(result.output["credential"])
        assert result.output["topic"] == "billing"
        back = await context.state_manager.get("sub_out")
        assert is_redacted(back["credential"])


class TestRecordingKeys:
    """A subworkflow's step records under its path, and older recordings still replay."""

    WORKFLOW: ClassVar[dict[str, Any]] = {
        "name": "parent",
        "config": {"provider": "mock", "model": "m"},
        "steps": [
            {"id": "x", "type": "llm_call", "prompt": "parent", "output": "own"},
            {
                "id": "sub",
                "type": "subworkflow",
                "depends_on": ["x"],
                "output": "sub",
                "workflow_inline": {
                    "name": "child",
                    "config": {"provider": "mock", "model": "m"},
                    "steps": [{"id": "x", "type": "llm_call", "prompt": "child", "output": "kid"}],
                },
            },
        ],
    }

    async def _run(self, provider: Any) -> Any:
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.parser import WorkflowParser

        gateway = ProviderGateway()
        gateway.register(provider, priority=0)
        workflow = WorkflowParser.from_dict(self.WORKFLOW)
        result = await WorkflowEngine(workflow=workflow, provider_gateway=gateway).run()
        await gateway.close()
        return result

    async def _record(self, tmp_path: Any) -> Any:
        from agentloom.providers.mock import MockProvider as RecordedMock
        from agentloom.providers.recorder import RecordingProvider

        answers = tmp_path / "answers.json"
        answers.write_text(json.dumps({"x": [{"content": "FIRST"}, {"content": "SECOND"}]}))
        recording = tmp_path / "rec.json"
        await self._run(RecordingProvider(RecordedMock(responses_file=answers), recording))
        return recording

    async def test_parent_and_child_with_one_id_get_their_own_entries(self, tmp_path: Any) -> None:
        data = json.loads((await self._record(tmp_path)).read_text())
        assert (data["x"]["content"], data["sub/x"]["content"]) == ("FIRST", "SECOND")
        assert data["_version"] == 2

    async def test_strict_replay_gives_each_its_own_answer(self, tmp_path: Any) -> None:
        from agentloom.providers.mock import MockProvider as RecordedMock

        recording = await self._record(tmp_path)
        result = await self._run(RecordedMock(responses_file=recording, strict=True))
        assert result.final_state["own"] == "FIRST"
        assert result.final_state["sub"]["kid"] == "SECOND"

    async def test_a_recording_with_the_bare_id_still_replays(self, tmp_path: Any) -> None:
        from agentloom.providers.mock import MockProvider as RecordedMock

        recording = tmp_path / "old.json"
        recording.write_text(json.dumps({"x": [{"content": "FIRST"}, {"content": "SECOND"}]}))
        result = await self._run(RecordedMock(responses_file=recording, strict=True))
        assert result.final_state["own"] == "FIRST"
        assert result.final_state["sub"]["kid"] == "SECOND"
