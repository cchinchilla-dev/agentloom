"""Engine-level tests for budgets, pause-over-budget precedence and error classification."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepResult, StepStatus, WorkflowStatus
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse
from agentloom.providers.gateway import ProviderGateway


class _CostedProvider(BaseProvider):
    """Provider with a configurable cost per call, so the expected overshoot is exact."""

    name = "costed-mock"

    def __init__(self, cost_per_call: float = 0.001) -> None:
        super().__init__(api_key="", base_url="")
        self.cost_per_call = cost_per_call
        self.calls = 0

    def supports_model(self, model: str) -> bool:
        return True

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        from agentloom.core.results import TokenUsage

        self.calls += 1
        return ProviderResponse(
            content="ok",
            model=model,
            provider="costed-mock",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_usd=self.cost_per_call,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        raise NotImplementedError


def _gateway_for(provider: BaseProvider) -> ProviderGateway:
    gateway = ProviderGateway()
    gateway.register(provider)
    return gateway


class TestBudgetEnforcerWiredIntoEngine:
    """The engine budgets through the shared ``BudgetEnforcer`` class."""

    async def test_engine_constructs_budget_enforcer(self) -> None:
        from agentloom.resilience.budget import BudgetEnforcer

        workflow = WorkflowDefinition(
            name="wired",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.10),
            state={},
            steps=[StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi")],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(_CostedProvider()))
        assert isinstance(engine._budget, BudgetEnforcer)
        assert engine._budget.limit_usd == 0.10

    async def test_engine_accepts_external_enforcer(self) -> None:
        """A subworkflow shares its parent's budget through the pre-built enforcer passed in."""
        from agentloom.resilience.budget import BudgetEnforcer

        shared = BudgetEnforcer(limit_usd=0.50)
        workflow = WorkflowDefinition(
            name="shared",
            config=WorkflowConfig(provider="costed-mock", model="x"),
            state={},
            steps=[StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi")],
        )
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_gateway_for(_CostedProvider()),
            budget_enforcer=shared,
        )
        assert engine._budget is shared


class TestParallelLayerBudget:
    """The pre-dispatch gate bounds the overshoot to the steps in flight, never across layers."""

    async def test_parallel_layer_pre_dispatch_gate_bounds_overshoot(self) -> None:
        # Layer 0: 3 parallel steps ($0.03 if all run) against $0.005; layer 1 must not run.
        provider = _CostedProvider(cost_per_call=0.01)
        workflow = WorkflowDefinition(
            name="parallel-overrun",
            config=WorkflowConfig(
                provider="costed-mock",
                model="x",
                budget_usd=0.005,
                max_concurrent_steps=10,
            ),
            state={},
            steps=[
                StepDefinition(id="p1", type=StepType.LLM_CALL, prompt="a"),
                StepDefinition(id="p2", type=StepType.LLM_CALL, prompt="b"),
                StepDefinition(id="p3", type=StepType.LLM_CALL, prompt="c"),
                StepDefinition(
                    id="downstream",
                    type=StepType.LLM_CALL,
                    prompt="d",
                    depends_on=["p1", "p2", "p3"],
                ),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()

        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        # Never dispatched, downstream is either absent or not SUCCESS.
        downstream = result.step_results.get("downstream")
        assert downstream is None or downstream.status != StepStatus.SUCCESS

    async def test_sequential_overrun_stops_at_breach_step(self) -> None:
        provider = _CostedProvider(cost_per_call=0.001)
        workflow = WorkflowDefinition(
            name="seq-overrun",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.0005),
            state={},
            steps=[
                StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi"),
                StepDefinition(id="s2", type=StepType.LLM_CALL, prompt="hi", depends_on=["s1"]),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        # s1 alone overruns the $0.0005 limit, so s2 never reaches the provider.
        assert provider.calls == 1

    async def test_pre_dispatch_gate_blocks_next_layer_when_budget_exhausted(self) -> None:
        # s1 spends the budget exactly (``charge`` raises only on ``>``): the gate must refuse s2.
        provider = _CostedProvider(cost_per_call=0.0005)
        workflow = WorkflowDefinition(
            name="exact-exhaust",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.0005),
            state={},
            steps=[
                StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi"),
                StepDefinition(id="s2", type=StepType.LLM_CALL, prompt="hi", depends_on=["s1"]),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        assert provider.calls == 1
        assert engine._budget.remaining == 0.0


class TestSubworkflowSharedBudget:
    """A child subworkflow with no budget of its own charges the parent's counter."""

    async def test_child_charges_count_against_parent_budget(self) -> None:
        provider = _CostedProvider(cost_per_call=0.001)
        # Two $0.001 child steps against the parent's $0.0015: only a shared counter breaks it.
        child_yaml = {
            "name": "child",
            "config": {"provider": "costed-mock", "model": "x"},
            "steps": [
                {"id": "c1", "type": "llm_call", "prompt": "hi"},
                {"id": "c2", "type": "llm_call", "prompt": "hi", "depends_on": ["c1"]},
            ],
        }
        workflow = WorkflowDefinition(
            name="parent",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.0015),
            state={},
            steps=[
                StepDefinition(
                    id="sub",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline=child_yaml,
                ),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()

        assert result.status == WorkflowStatus.BUDGET_EXCEEDED

    async def test_shared_enforcer_does_not_double_charge_parent(self) -> None:
        """A child sharing the parent's enforcer is not charged twice.

        Each child step charges the parent counter as it runs, so the parent does not re-charge
        the subworkflow's rolled-up ``cost_usd``.
        """
        provider = _CostedProvider(cost_per_call=0.10)
        child_yaml = {
            "name": "child",
            "config": {"provider": "costed-mock", "model": "x"},
            "steps": [
                {"id": "c1", "type": "llm_call", "prompt": "hi"},
                {"id": "c2", "type": "llm_call", "prompt": "hi", "depends_on": ["c1"]},
            ],
        }
        workflow = WorkflowDefinition(
            name="parent",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.25),
            state={},
            steps=[
                StepDefinition(
                    id="sub",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline=child_yaml,
                ),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()
        # Real spend is $0.20, within the $0.25 budget; a double charge would make it $0.40.
        assert result.status == WorkflowStatus.SUCCESS
        assert engine._budget.spent == pytest.approx(0.20)
        # The total comes from the enforcer, as the shared-budget sub step reports ``cost_usd=0``.
        assert result.total_cost_usd == pytest.approx(0.20)

    async def test_child_with_own_budget_uses_own_enforcer(self) -> None:
        """A child with its own ``budget_usd`` owns that budget: no shared enforcer.

        Its own gate ends it. The parent sees a FAILED subworkflow, not its own breach, and
        does not retry it: the non-retryable ``StepError`` keeps the child from being billed again.
        """
        provider = _CostedProvider(cost_per_call=0.001)
        # The child's $0.0005 budget cannot cover one $0.001 call.
        child_yaml = {
            "name": "child-with-own-budget",
            "config": {"provider": "costed-mock", "model": "x", "budget_usd": 0.0005},
            "steps": [
                {"id": "c1", "type": "llm_call", "prompt": "hi"},
            ],
        }
        workflow = WorkflowDefinition(
            name="parent-generous",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=1.0),
            state={},
            steps=[
                StepDefinition(
                    id="sub",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline=child_yaml,
                ),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()
        assert result.status != WorkflowStatus.BUDGET_EXCEEDED
        sub = result.step_results["sub"]
        assert sub.status == StepStatus.FAILED
        assert sub.error_classification == "permanent"
        assert provider.calls == 1


class TestPauseOverBudgetPrecedence:
    """When an approval gate and a budget-blowing step share a layer, pause wins."""

    async def test_pause_in_same_layer_as_budget_breach_surfaces_pause(self) -> None:
        # No deps between them: the LLM call ($0.001 vs $0.0005) and the gate share layer 0.
        provider = _CostedProvider(cost_per_call=0.001)
        workflow = WorkflowDefinition(
            name="pause-vs-budget",
            config=WorkflowConfig(
                provider="costed-mock",
                model="x",
                budget_usd=0.0005,
                max_concurrent_steps=10,
            ),
            state={},
            steps=[
                StepDefinition(id="llm", type=StepType.LLM_CALL, prompt="warm"),
                StepDefinition(
                    id="gate",
                    type=StepType.APPROVAL_GATE,
                    prompt="approve?",
                ),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()

        # PAUSED keeps the resumable checkpoint at the gate, and with it the human decision.
        assert result.status == WorkflowStatus.PAUSED
        gate_result = result.step_results.get("gate")
        assert gate_result is not None
        assert gate_result.status == StepStatus.PAUSED

    async def test_a_pause_from_the_checkpoint_does_not_hide_a_budget_breach(self) -> None:
        provider = _CostedProvider(cost_per_call=0.001)
        workflow = WorkflowDefinition(
            name="resumed",
            config=WorkflowConfig(
                provider="costed-mock", model="x", budget_usd=0.0005, on_step_failure="continue"
            ),
            state={},
            steps=[
                StepDefinition(id="llm", type=StepType.LLM_CALL, prompt="warm"),
                StepDefinition(
                    id="gate", type=StepType.APPROVAL_GATE, prompt="approve?", depends_on=["llm"]
                ),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        # What a checkpoint restores after the run paused at the gate, approved since
        await engine.state.set_step_result(
            "llm", StepResult(step_id="llm", status=StepStatus.FAILED)
        )
        await engine.state.set_step_result(
            "gate", StepResult(step_id="gate", status=StepStatus.PAUSED)
        )
        await engine.state.set("_approval.gate", "approved")
        result = await engine.run()

        assert result.status == WorkflowStatus.BUDGET_EXCEEDED

    async def test_budget_breach_alone_still_ends_budget_exceeded(self) -> None:
        provider = _CostedProvider(cost_per_call=0.001)
        workflow = WorkflowDefinition(
            name="budget-only",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.0005),
            state={},
            steps=[
                StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi"),
                StepDefinition(id="s2", type=StepType.LLM_CALL, depends_on=["s1"], prompt="hi"),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider))
        result = await engine.run()
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED


class TestErrorClassificationField:
    """``StepResult.error_classification`` tells permanent failures from transient ones.

    ``"permanent"`` marks a non-retryable failure and ``"transient"`` one that ran out of
    retries, for dashboards to tell them apart.
    """

    async def test_permanent_classification_for_tool_not_found(self) -> None:
        from agentloom.tools.registry import ToolRegistry

        workflow = WorkflowDefinition(
            name="permanent-classification",
            config=WorkflowConfig(provider="costed-mock", model="x"),
            state={},
            steps=[
                StepDefinition(
                    id="s1",
                    type=StepType.TOOL,
                    tool_name="does_not_exist",
                    tool_args={},
                ),
            ],
        )
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_gateway_for(_CostedProvider()),
            tool_registry=ToolRegistry(),
        )
        result = await engine.run()
        s1 = result.step_results["s1"]
        assert s1.status == StepStatus.FAILED
        assert s1.error_classification == "permanent"

    async def test_transient_classification_for_generic_exception(self) -> None:
        """A status-less exception runs out of retries and is classified ``"transient"``."""

        class _AlwaysFailingProvider(_CostedProvider):
            name = "always-failing"

            async def complete(self, *args: Any, **kwargs: Any) -> ProviderResponse:
                raise RuntimeError("transient hiccup")

        workflow = WorkflowDefinition(
            name="transient-classification",
            config=WorkflowConfig(provider="always-failing", model="x"),
            state={},
            steps=[
                StepDefinition(
                    id="s1",
                    type=StepType.LLM_CALL,
                    prompt="hi",
                    retry={"max_retries": 1, "backoff_base": 0.01, "jitter": False},
                ),
            ],
        )
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_gateway_for(_AlwaysFailingProvider()),
        )
        result = await engine.run()
        s1 = result.step_results["s1"]
        assert s1.status == StepStatus.FAILED
        assert s1.error_classification == "transient"


class TestBudgetObserverNotification:
    """A budget overrun reaches the observer's ``on_workflow_end`` as ``"budget_exceeded"``."""

    async def test_budget_exceeded_notifies_observer(self) -> None:
        provider = _CostedProvider(cost_per_call=0.001)
        workflow = WorkflowDefinition(
            name="budget-observer",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.0005),
            state={},
            steps=[StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi")],
        )
        observer = MagicMock()
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=_gateway_for(provider),
            observer=observer,
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        observer.on_workflow_end.assert_called_once()
        # Positional args: (workflow_name, status, duration, tokens, cost).
        call_args = observer.on_workflow_end.call_args[0]
        assert call_args[0] == "budget-observer"
        assert call_args[1] == "budget_exceeded"
        assert call_args[4] == pytest.approx(engine._budget.spent)


class _FailingProvider(_CostedProvider):
    """Fails every call with a status-less error, which the engine retries."""

    name = "failing-mock"

    async def complete(self, *args: Any, **kwargs: Any) -> ProviderResponse:
        self.calls += 1
        raise RuntimeError("transient hiccup")


class TestWorkflowMaxRetries:
    """A step without its own ``retry.max_retries`` takes its workflow's ``config.max_retries``."""

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("agentloom.core.engine.compute_backoff", lambda *_args: 0.0)

    async def _calls(self, workflow: WorkflowDefinition) -> int:
        provider = _FailingProvider()
        await WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(provider)).run()
        return provider.calls

    @pytest.mark.parametrize(("max_retries", "calls"), [(0, 1), (2, 3)])
    async def test_step_without_retry_block_takes_the_workflow_value(
        self, max_retries: int, calls: int
    ) -> None:
        workflow = WorkflowDefinition(
            name="wf",
            config=WorkflowConfig(provider="failing-mock", model="x", max_retries=max_retries),
            steps=[StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi")],
        )
        assert await self._calls(workflow) == calls

    async def test_step_max_retries_wins(self) -> None:
        workflow = WorkflowDefinition(
            name="wf",
            config=WorkflowConfig(provider="failing-mock", model="x", max_retries=0),
            steps=[
                StepDefinition(
                    id="s1", type=StepType.LLM_CALL, prompt="hi", retry={"max_retries": 1}
                )
            ],
        )
        assert await self._calls(workflow) == 2

    async def test_retry_block_without_max_retries_takes_the_workflow_value(self) -> None:
        workflow = WorkflowDefinition(
            name="wf",
            config=WorkflowConfig(provider="failing-mock", model="x", max_retries=0),
            steps=[
                StepDefinition(
                    id="s1", type=StepType.LLM_CALL, prompt="hi", retry={"backoff_base": 1.5}
                )
            ],
        )
        assert workflow.steps[0].retry.backoff_base == 1.5
        assert await self._calls(workflow) == 1

    async def test_subworkflow_steps_take_the_child_value(self) -> None:
        """The child's ``config.max_retries`` (1) applies, not the parent's (0)."""
        child_yaml = {
            "name": "child",
            "config": {"provider": "failing-mock", "model": "x", "max_retries": 1},
            "steps": [{"id": "c1", "type": "llm_call", "prompt": "hi"}],
        }
        workflow = WorkflowDefinition(
            name="parent",
            config=WorkflowConfig(provider="failing-mock", model="x", max_retries=0),
            steps=[StepDefinition(id="sub", type=StepType.SUBWORKFLOW, workflow_inline=child_yaml)],
        )
        assert await self._calls(workflow) == 2
