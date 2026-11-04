"""What every attempt of a step spent is charged and reported, however it ends (#120)."""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import WorkflowDefinition
from agentloom.core.results import StepStatus, TokenUsage, WorkflowResult, WorkflowStatus
from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse, ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry


class _Paid(BaseProvider):
    """Answers each call with the next turn of a script, or raises it, or hangs."""

    name = "paid"

    def __init__(self, *turns: ProviderResponse | Exception | None) -> None:
        super().__init__()
        self._turns = list(turns)

    async def complete(self, **kw: Any) -> ProviderResponse:
        turn = self._turns.pop(0)
        if turn is None:
            await anyio.sleep(60)  # cut by the step's timeout
        if isinstance(turn, Exception):
            raise turn
        assert turn is not None
        return turn

    async def stream(self, **kw: Any) -> StreamResponse:
        raise AssertionError("no step here streams")

    def supports_model(self, _m: str) -> bool:
        return True


class _Add(BaseTool):
    name = "add"
    description = "Add two integers."
    parameters_schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    }

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs["a"] + kwargs["b"]


class _Tokens:
    """Keeps the token observations of a run, and how its steps ended."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []
        self.providers: list[str] = []
        self.ends: list[tuple[str, str, float]] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def on_tokens(self, provider: str, model: str, prompt: int, completion: int, **_: Any) -> None:
        self.seen.append((model, prompt))
        self.providers.append(provider)

    def on_step_end(
        self,
        step_id: str,
        step_type: str,
        status: str,
        duration_ms: float,
        cost_usd: float = 0.0,
        **_: Any,
    ) -> None:
        self.ends.append((step_id, status, cost_usd))


def _paid(content: str = "", *calls: ToolCall, model: str = "m") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        model=model,
        provider="paid",
        tool_calls=list(calls),
        cost_usd=0.1,
        usage=TokenUsage(
            prompt_tokens=10, completion_tokens=1, total_tokens=13, reasoning_tokens=2
        ),
    )


def _unavailable() -> ProviderError:
    return ProviderError("paid", "unavailable", status_code=503)


async def _run(
    provider: _Paid,
    *,
    retries: int = 1,
    budget: float | None = None,
    workflow_timeout: float | None = None,
    **step: Any,
) -> tuple[WorkflowResult, WorkflowEngine]:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {
                "provider": "paid",
                "model": "m",
                "budget_usd": budget,
                "timeout": workflow_timeout,
            },
            "steps": [
                {
                    "id": "ask",
                    "type": "llm_call",
                    "prompt": "2+3?",
                    "tools": [{"name": "add", "parameters": _Add.parameters_schema}],
                    "output": "answer",
                    "retry": {"max_retries": retries, "backoff_base": 0, "jitter": False},
                    **step,
                }
            ],
        }
    )
    gateway = ProviderGateway()
    gateway.register(provider)
    registry = ToolRegistry()
    registry.register(_Add())
    engine = WorkflowEngine(
        workflow=workflow, provider_gateway=gateway, tool_registry=registry, observer=_Tokens()
    )
    return await engine.run(), engine


async def test_a_retried_step_reports_what_every_attempt_spent() -> None:
    provider = _Paid(
        _paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})),
        _unavailable(),
        _paid("", ToolCall(id="c2", name="add", arguments={"a": 2, "b": 3})),
        _paid("5"),
    )
    result, engine = await _run(provider)

    ask = result.step_results["ask"]
    assert ask.status == StepStatus.SUCCESS
    assert ask.cost_usd == pytest.approx(0.3)
    assert ask.token_usage == TokenUsage(
        prompt_tokens=30, completion_tokens=3, total_tokens=39, reasoning_tokens=6
    )
    assert result.total_cost_usd == pytest.approx(0.3)
    assert engine._budget.spent == pytest.approx(0.3)


async def test_a_step_that_fails_for_good_reports_and_is_charged_what_it_spent() -> None:
    provider = _Paid(
        _paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})),
        ProviderError("paid", "bad request", status_code=400),
    )
    result, engine = await _run(provider)

    ask = result.step_results["ask"]
    assert (ask.status, ask.error_classification) == (StepStatus.FAILED, "permanent")
    assert ask.cost_usd == pytest.approx(0.1)
    assert ask.token_usage.prompt_tokens == 10
    assert engine._budget.spent == pytest.approx(0.1)


async def test_the_budget_stops_a_step_that_keeps_failing_after_paid_calls() -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    provider = _Paid(_paid("", call), _unavailable(), _paid("", call), _unavailable(), _paid("5"))
    result, _ = await _run(provider, retries=3, budget=0.15)

    assert result.status == WorkflowStatus.BUDGET_EXCEEDED
    assert result.step_results["ask"].cost_usd == pytest.approx(0.2)
    assert len(provider._turns) == 1  # the third attempt never ran


async def test_a_step_cut_by_its_timeout_keeps_what_its_answered_calls_cost() -> None:
    provider = _Paid(_paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})), None)
    result, engine = await _run(provider, retries=0, timeout=0.2)

    ask = result.step_results["ask"]
    assert ask.status == StepStatus.TIMEOUT
    assert ask.cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)


async def test_the_tokens_of_a_failed_attempt_reach_the_metrics_once() -> None:
    provider = _Paid(
        _paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}), model="first"),
        _unavailable(),
        _paid("5"),
    )
    result, engine = await _run(provider)

    assert result.step_results["ask"].token_usage.prompt_tokens == 20
    assert sorted(engine.observer.seen) == [("first", 10), ("m", 10)]


async def test_an_agent_turn_that_fails_is_charged() -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m"},
            "steps": [
                {
                    "id": "ask",
                    "type": "agent",
                    "agent": {"name": "a", "constraints": {"forbidden_patterns": ["5"]}},
                    "prompt": "2+3?",
                }
            ],
        }
    )
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("5")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway)
    result = await engine.run()

    assert result.step_results["ask"].status == StepStatus.FAILED
    assert result.step_results["ask"].cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)


async def test_a_streamed_call_of_a_failed_attempt_is_charged(tmp_path: Any) -> None:
    from agentloom.providers.mock import MockProvider

    recording = tmp_path / "rec.json"
    recording.write_text('{"ask": {"content": "not JSON", "cost_usd": 0.1}}', encoding="utf-8")
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "mock", "model": "m"},
            "steps": [
                {
                    "id": "ask",
                    "type": "llm_call",
                    "prompt": "2+3?",
                    "stream": True,
                    "response_schema": {"type": "json_object"},
                    "retry": {"max_retries": 0},
                }
            ],
        }
    )
    gateway = ProviderGateway()
    gateway.register(MockProvider(responses_file=recording))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway)
    result = await engine.run()

    assert result.step_results["ask"].status == StepStatus.FAILED
    assert result.step_results["ask"].cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)


async def test_a_call_counts_under_the_provider_that_answered_it() -> None:
    # A replay answers as the provider it recorded, not as the mock that serves it
    provider = _Paid(_paid("5").model_copy(update={"provider": "recorded"}))
    _, engine = await _run(provider)

    assert engine.observer.providers == ["recorded"]


async def test_a_summary_counts_when_its_step_succeeds() -> None:
    long = "x" * 400
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m"},
            "state": {
                "chat": {
                    "messages": [
                        {"role": "user", "content": long},
                        {"role": "assistant", "content": long},
                    ],
                    "token_budget": 60,
                    "trim_policy": "summarize_oldest",
                }
            },
            "steps": [
                {
                    "id": "ask",
                    "type": "llm_call",
                    "prompt": "hi",
                    "conversation": "state.chat",
                    "output": "answer",
                }
            ],
        }
    )
    provider = _Paid(_paid("summary"), _paid("reply"))
    gateway = ProviderGateway()
    gateway.register(provider)
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, observer=_Tokens())
    result = await engine.run()

    ask = result.step_results["ask"]
    assert ask.status == StepStatus.SUCCESS
    assert not provider._turns  # the summary ran
    assert ask.cost_usd == pytest.approx(0.2)
    assert ask.token_usage.prompt_tokens == 20
    assert engine._budget.spent == pytest.approx(0.2)
    assert engine.observer.seen == [("m", 20)]


class _OwnGateway:
    """A gateway of its own: the engine never sees its calls."""

    def __init__(self, *turns: ProviderResponse) -> None:
        self._turns = list(turns)

    async def complete(self, **kw: Any) -> ProviderResponse:
        return self._turns.pop(0)


async def test_a_failed_step_keeps_what_a_gateway_of_its_own_spent() -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m"},
            "steps": [
                {
                    "id": "ask",
                    "type": "agent",
                    "agent": {"name": "a", "constraints": {"forbidden_patterns": ["5"]}},
                    "prompt": "2+3?",
                }
            ],
        }
    )
    engine = WorkflowEngine(workflow=workflow, provider_gateway=_OwnGateway(_paid("5")))
    result = await engine.run()

    assert result.step_results["ask"].status == StepStatus.FAILED
    assert result.step_results["ask"].cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)


async def _run_subworkflow(
    provider: _Paid,
    child_budget: float | None,
    *child_steps: dict[str, Any],
    workflow_timeout: float | None = None,
    **sub: Any,
) -> tuple[WorkflowResult, WorkflowEngine]:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {
                "provider": "paid",
                "model": "m",
                "budget_usd": 1.0,
                "timeout": workflow_timeout,
            },
            "steps": [
                {
                    "id": "sub",
                    "type": "subworkflow",
                    "retry": {"max_retries": 1, "backoff_base": 0, "jitter": False},
                    "workflow_inline": {
                        "name": "child",
                        "config": {"provider": "paid", "model": "m", "budget_usd": child_budget},
                        "steps": list(child_steps),
                    },
                    **sub,
                }
            ],
        }
    )
    gateway = ProviderGateway()
    gateway.register(provider)
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, observer=_Tokens())
    return await engine.run(), engine


_PAYS = {"id": "a", "type": "llm_call", "prompt": "hi", "retry": {"max_retries": 0}}
# A template that never renders: the child fails for good after "a" paid
_BREAKS = {"id": "b", "type": "llm_call", "prompt": "{", "depends_on": ["a"]}


@pytest.mark.parametrize("child_budget", [None, 10.0], ids=["shared", "own"])
async def test_a_failed_subworkflow_is_charged_what_its_steps_spent_once(
    child_budget: float | None,
) -> None:
    result, engine = await _run_subworkflow(_Paid(_paid("ok")), child_budget, _PAYS, _BREAKS)

    assert result.step_results["sub"].status == StepStatus.FAILED
    assert result.total_cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)
    assert engine.observer.seen == [("m", 10)]  # by the child's step, not again by the parent


async def test_a_subworkflow_over_its_own_budget_charges_the_parent_and_is_not_retried() -> None:
    provider = _Paid(_paid("ok"), _paid("again"))
    result, engine = await _run_subworkflow(provider, 0.05, _PAYS)

    sub = result.step_results["sub"]
    assert (sub.status, sub.error_classification) == (StepStatus.FAILED, "permanent")
    assert sub.cost_usd == pytest.approx(0.1)
    assert sub.token_usage.prompt_tokens == 10
    assert engine._budget.spent == pytest.approx(0.1)
    assert len(provider._turns) == 1  # no second run of the child


@pytest.mark.parametrize("budget", [None, 0.05], ids=["no-budget", "over-budget"])
async def test_a_step_cut_by_the_workflow_timeout_keeps_what_it_spent(
    budget: float | None,
) -> None:
    provider = _Paid(_paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})), None)
    result, engine = await _run(provider, retries=0, budget=budget, workflow_timeout=0.2)

    ask = result.step_results["ask"]
    assert (result.status, ask.status) == (WorkflowStatus.TIMEOUT, StepStatus.TIMEOUT)
    assert ask.error == "Workflow timed out after 0.2s"
    assert ask.cost_usd == pytest.approx(0.1)
    assert ask.token_usage.prompt_tokens == 10
    assert result.total_cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)
    assert engine.observer.seen == [("m", 10)]
    assert engine.observer.ends == [("ask", "timeout", pytest.approx(0.1))]


async def test_a_step_cut_while_it_waits_to_retry_is_charged_once() -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    provider = _Paid(_paid("", call), _unavailable())
    # The first backoff waits 1s, past the workflow's timeout
    result, engine = await _run(
        provider, workflow_timeout=0.2, retry={"max_retries": 1, "jitter": False}
    )

    ask = result.step_results["ask"]
    assert ask.status == StepStatus.TIMEOUT
    assert ask.cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)
    assert engine.observer.seen == [("m", 10)]


async def test_a_step_cut_while_it_is_charged_is_charged_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m", "timeout": 0.2},
            "steps": [{"id": "ask", "type": "llm_call", "prompt": "2+3?"}],
        }
    )
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("5")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway)
    charge = engine._budget.charge

    async def contended(cost_usd: float) -> None:
        await anyio.sleep(0.3)  # the budget's lock is held past the workflow's deadline
        await charge(cost_usd)

    monkeypatch.setattr(engine._budget, "charge", contended)
    result = await engine.run()

    assert result.step_results["ask"].status == StepStatus.SUCCESS
    assert engine._budget.spent == pytest.approx(0.1)


class _Opens(BaseTool):
    """Opens a gate once its step's first call is paid."""

    name = "open"
    description = "Open the gate."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.gate = anyio.Event()

    async def execute(self, **kwargs: Any) -> Any:
        self.gate.set()
        return "open"


class _TwoSteps(BaseProvider):
    """Step "slow" pays a call that runs the gate tool, then hangs; "fast" waits for the gate."""

    name = "paid"

    def __init__(self, gate: anyio.Event) -> None:
        super().__init__()
        self._gate = gate

    async def complete(self, **kw: Any) -> ProviderResponse:
        if kw["model"] == "slow":
            if not self._gate.is_set():
                return _paid("", ToolCall(id="c1", name="open", arguments={}), model="slow")
            await anyio.sleep_forever()
        await self._gate.wait()
        return _paid("done", model="fast")

    async def stream(self, **kw: Any) -> StreamResponse:
        raise AssertionError("no step here streams")

    def supports_model(self, _m: str) -> bool:
        return True


async def test_a_step_cut_by_a_siblings_budget_breach_is_charged() -> None:
    tool = _Opens()
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m", "budget_usd": 0.05},
            "steps": [
                {
                    "id": "slow",
                    "type": "llm_call",
                    "model": "slow",
                    "prompt": "open it",
                    "tools": [{"name": "open", "parameters": tool.parameters_schema}],
                },
                {"id": "fast", "type": "llm_call", "model": "fast", "prompt": "hi"},
            ],
        }
    )
    gateway = ProviderGateway()
    gateway.register(_TwoSteps(tool.gate))
    registry = ToolRegistry()
    registry.register(tool)
    engine = WorkflowEngine(
        workflow=workflow, provider_gateway=gateway, tool_registry=registry, observer=_Tokens()
    )
    result = await engine.run()

    assert result.status == WorkflowStatus.BUDGET_EXCEEDED
    assert result.total_cost_usd == pytest.approx(0.2)
    assert engine._budget.spent == pytest.approx(0.2)
    assert "slow" not in result.step_results  # no status says cancelled: its span does
    assert ("slow", "cancelled", pytest.approx(0.1)) in engine.observer.ends


_HANGS = {"id": "b", "type": "llm_call", "prompt": "hi", "depends_on": ["a"]}


@pytest.mark.parametrize(
    "cut",
    [{"timeout": 0.2, "retry": {"max_retries": 0}}, {"workflow_timeout": 0.2}],
    ids=["its-timeout", "the-workflows"],
)
async def test_a_subworkflow_with_a_budget_of_its_own_cut_short_is_charged_what_it_spent(
    cut: dict[str, Any],
) -> None:
    result, engine = await _run_subworkflow(_Paid(_paid("ok"), None), 10.0, _PAYS, _HANGS, **cut)

    sub = result.step_results["sub"]
    assert sub.status == StepStatus.TIMEOUT
    assert sub.cost_usd == pytest.approx(0.1)
    assert engine._budget.spent == pytest.approx(0.1)
