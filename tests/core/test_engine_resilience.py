"""Engine-level integration tests for the resilience changes.

Covers:

* Parallel-layer pre-dispatch budget gate.
* Subworkflow budget aggregation against the parent enforcer.
* ``BudgetEnforcer`` is wired into the engine and no longer dead code.
* Pause outranks budget in a same-layer conflict.
* ``error_classification`` field on step results.
"""

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
from agentloom.core.results import StepStatus, WorkflowStatus
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse
from agentloom.providers.gateway import ProviderGateway


class _CostedProvider(BaseProvider):
    """Provider that returns a configurable cost per call.

    Tests that need fine-grained budget arithmetic use this instead of
    the shared ``MockProvider`` so the expected overshoot is exact.
    """

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
    """The ``BudgetEnforcer`` is no longer dead code; the engine must
    use the shared class instead of the bare-float counter."""

    async def test_engine_constructs_budget_enforcer(self) -> None:
        from agentloom.resilience.budget import BudgetEnforcer

        workflow = WorkflowDefinition(
            name="wired",
            config=WorkflowConfig(provider="costed-mock", model="x", budget_usd=0.10),
            state={},
            steps=[StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi")],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=_gateway_for(_CostedProvider()))
        # The internal enforcer is a real ``BudgetEnforcer`` instance,
        # not a bare float — regression net so a future refactor that
        # re-orphans the class can't silently land.
        assert isinstance(engine._budget, BudgetEnforcer)
        assert engine._budget.limit_usd == 0.10

    async def test_engine_accepts_external_enforcer(self) -> None:
        """The subworkflow hand-off relies on the engine accepting a
        pre-built enforcer; without this argument the child-budget path
        can't share state with the parent."""
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
    """The pre-dispatch gate must prevent a parallel layer from
    overrunning the budget by every step's full cost. Pre-0.5.0 a
    5-step parallel layer with a tight budget ran all 5 to completion
    before the engine noticed; the gate bounds the overshoot to the
    in-flight worst case rather than letting it compound across layers."""

    async def test_parallel_layer_pre_dispatch_gate_bounds_overshoot(self) -> None:
        # Two-layer workflow: layer 0 is 3 parallel steps; if all run,
        # spend = $0.03. Budget = $0.005, so the pre-dispatch gate must
        # refuse later layers entirely — layer 1 must never dispatch.
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

        # Budget breached, workflow ends BUDGET_EXCEEDED.
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        # Layer 1 must never have dispatched — that's the whole point
        # of the pre-dispatch gate. The downstream step either never
        # appears in step_results or appears with non-success status.
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
        # s1 charged $0.001 against a $0.0005 limit — overrun is on s1.
        # s2 must never have dispatched (provider call count is 1).
        assert provider.calls == 1

    async def test_pre_dispatch_gate_blocks_next_layer_when_budget_exhausted(self) -> None:
        # Layer 0's single step charges exactly the budget — ``charge``
        # does not raise (the overrun test is strict ``>``), so the
        # workflow does not terminate inside layer 0. The pre-dispatch
        # gate must then refuse layer 1 because ``remaining`` is zero,
        # rather than dispatching a step that would only burn its cost
        # before ``charge`` raised.
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
        # s1 spent the budget exactly; s2 was refused by the gate before
        # reaching the provider — exactly one provider call.
        assert provider.calls == 1
        assert engine._budget.remaining == 0.0


class TestSubworkflowSharedBudget:
    """Child subworkflows must charge against the parent counter when
    the child has no budget of its own; pre-0.5.0 the child had a
    separate counter and the parent only learned of the overrun after
    the child completed."""

    async def test_child_charges_count_against_parent_budget(self) -> None:
        provider = _CostedProvider(cost_per_call=0.001)
        # Inline child with 2 steps × $0.001 = $0.002 against parent $0.0015.
        # Without the hand-off, both child steps run; with it, the parent
        # gate trips after the first.
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

        # Parent ends BUDGET_EXCEEDED — proof the child's charges
        # propagated to the parent counter.
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED

    async def test_shared_enforcer_does_not_double_charge_parent(self) -> None:
        """B1 regression net: when the child shares the parent's
        enforcer, each child step already charges the parent counter
        during execution. The parent's per-step accounting must NOT
        re-charge the subworkflow's rolled-up ``cost_usd`` on top —
        otherwise a parent with ``budget_usd: 0.25`` and a child that
        actually spent ``$0.20`` would surface ``budget exceeded:
        spent $0.40`` even though the real spend fit in budget."""
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
        # Real spend is 2 × $0.10 = $0.20, well within the $0.25 budget.
        # If the bug regresses, ``_spent`` reaches $0.40 (charged once
        # from inside the child engine, once from the parent's
        # ``_execute_step`` rolling up ``result.total_cost_usd``), and
        # the workflow ends BUDGET_EXCEEDED instead of SUCCESS.
        assert result.status == WorkflowStatus.SUCCESS
        # The parent enforcer reflects the true spend, not 2× it.
        assert engine._budget.spent == pytest.approx(0.20)

    async def test_child_with_own_budget_uses_own_enforcer(self) -> None:
        """When the child declares its own ``budget_usd``, it owns the
        budget — the shared-enforcer hand-off does NOT happen. The
        child's BudgetEnforcer is fresh, and its own per-step gating is
        what terminates it. The parent's per-step accounting still rolls
        up the subworkflow step's cost (so the parent's budget cap still
        applies to the aggregate), but child charges do not bypass the
        parent's own gate via the shared-enforcer path."""
        provider = _CostedProvider(cost_per_call=0.001)
        # Child budget = $0.0005, which can't even cover one $0.001 call —
        # child terminates BUDGET_EXCEEDED inside its own enforcer. The
        # parent must see this as a generic FAILED subworkflow (not
        # propagated as the parent's own budget breach).
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
        # The parent did NOT terminate BUDGET_EXCEEDED — the child's
        # own enforcer absorbed the breach, so the parent only sees a
        # generic FAILED subworkflow. Pre-0.5.0 behaviour preserved.
        assert result.status != WorkflowStatus.BUDGET_EXCEEDED


class TestPauseOverBudgetPrecedence:
    """When an approval gate and a budget-blowing step land in the
    same layer, pause must win. Pre-0.5.0 the order was reversed,
    silently spending the money AND dropping the pause."""

    async def test_pause_in_same_layer_as_budget_breach_surfaces_pause(self) -> None:
        # Layer 0 has both an LLM call (budget-blowing — cost $0.001 vs
        # limit $0.0005) AND an approval gate. With no deps between
        # them they run in the same parallel layer; the LLM completes
        # and trips the budget, the gate raises PauseRequestedError.
        # The precedence contract: pause wins so the human decision is
        # preserved over the budget overrun.
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

        # PAUSED wins, surfacing the resumable checkpoint at the gate.
        # If BUDGET_EXCEEDED wins, the user loses the human-decision
        # opportunity entirely.
        assert result.status == WorkflowStatus.PAUSED
        gate_result = result.step_results.get("gate")
        assert gate_result is not None
        assert gate_result.status == StepStatus.PAUSED

    async def test_budget_breach_alone_still_ends_budget_exceeded(self) -> None:
        """Regression net: without a pause in the same layer, the
        original BUDGET_EXCEEDED termination is unchanged."""
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
    """``StepResult.error_classification`` carries ``"permanent"`` for
    non-retryable failures and ``"transient"`` for failures that
    exhausted the retry budget. Surface for observability dashboards to
    distinguish "we wasted 30 s retrying nothing" from "we actually
    retried a transient one"."""

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
        """A status-less exception with no permanent marker exhausts
        the retry budget — the engine still records the classification
        so dashboards can see "retried 4× and gave up"."""

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
    """A budget overrun must reach the observer's ``on_workflow_end``
    hook with the ``"budget_exceeded"`` status so dashboards and trace
    consumers see the terminal state, not just a silent stop."""

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
        # Cost carried to the observer is the enforcer's recorded spend.
        assert call_args[4] == pytest.approx(engine._budget.spent)
