"""Tests for subworkflow step executor."""

from __future__ import annotations

from typing import Any

import pytest

from agentloom.core.models import StepDefinition, StepType, WorkflowConfig
from agentloom.core.results import StepResult, StepStatus
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
