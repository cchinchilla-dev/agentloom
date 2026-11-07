"""What every attempt of a step spent is charged and reported, however it ends (#120)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import _RUNS, WorkflowEngine
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
        self.runs: list[tuple[str, float]] = []
        self.run_tokens: dict[str, int] = {}

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

    def on_workflow_end(
        self, name: str, status: str, duration_ms: float, tokens: int, cost_usd: float, **_: Any
    ) -> None:
        self.runs.append((name, cost_usd))
        self.run_tokens[name] = tokens


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


def _bad_request() -> ProviderError:
    return ProviderError("paid", "bad request", status_code=400)


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

    def __init__(self, *turns: ProviderResponse | Exception) -> None:
        self._turns = list(turns)

    async def complete(self, **kw: Any) -> ProviderResponse:
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


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


async def test_a_structured_reply_asked_again_reports_every_call() -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m"},
            "steps": [
                {
                    "id": "ask",
                    "type": "llm_call",
                    "prompt": "2+3?",
                    "response_schema": {"type": "json_object"},
                    "retry": {"max_retries": 1},
                }
            ],
        }
    )
    gateway = _OwnGateway(_paid("not JSON"), _paid('{"sum": 5}'))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway)
    result = await engine.run()

    ask = result.step_results["ask"]
    assert ask.status == StepStatus.SUCCESS
    assert ask.cost_usd == pytest.approx(0.2)
    assert ask.token_usage.prompt_tokens == 20
    assert engine._budget.spent == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("shared", "older"),
    [(False, False), (False, True), (True, False)],
    ids=["its-checkpoint", "an-older-checkpoint", "a-subworkflow-sharing-the-budget"],
)
async def test_a_resumed_run_goes_on_from_what_it_had_spent(
    tmp_path: Any, shared: bool, older: bool
) -> None:
    draft: dict[str, Any] = {"id": "draft", "type": "llm_call", "prompt": "draft"}
    if shared:  # its step reports nothing: only the checkpoint knows what its child spent
        child = {"name": "child", "config": {"provider": "paid", "model": "m"}, "steps": [_PAYS]}
        draft = {"id": "draft", "type": "subworkflow", "workflow_inline": child}
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m", "budget_usd": 0.15},
            "steps": [
                draft,
                {"id": "gate", "type": "approval_gate", "prompt": "ok?", "depends_on": ["draft"]},
                {"id": "send", "type": "llm_call", "prompt": "send", "depends_on": ["gate"]},
            ],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("drafted")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer)
    assert (await engine.run()).status == WorkflowStatus.PAUSED
    saved = await checkpointer.load(engine.run_id)
    assert saved.spent_usd == pytest.approx(0.1)
    if older:  # written before the checkpoint kept it: the step results' cost stands in
        saved.spent_usd = 0.0

    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("sent")))
    resumed = await WorkflowEngine.from_checkpoint(
        saved,
        checkpointer=checkpointer,
        provider_gateway=gateway,
        approval_decisions={"gate": "approved"},
    )
    result = await resumed.run()

    assert result.status == WorkflowStatus.BUDGET_EXCEEDED
    assert result.total_cost_usd == pytest.approx(0.2)
    assert resumed._budget.spent == pytest.approx(0.2)


async def _run_subworkflow(
    provider: _Paid,
    child_budget: float | None,
    *child_steps: dict[str, Any],
    workflow_timeout: float | None = None,
    budget: float = 1.0,
    observer: _Tokens | None = None,
    **sub: Any,
) -> tuple[WorkflowResult, WorkflowEngine]:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {
                "provider": "paid",
                "model": "m",
                "budget_usd": budget,
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
    registry = ToolRegistry()
    registry.register(_Add())
    engine = WorkflowEngine(
        workflow=workflow,
        provider_gateway=gateway,
        tool_registry=registry,
        observer=observer or _Tokens(),
    )
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


async def test_a_subworkflow_reports_its_steps_tokens_whole() -> None:
    result, _ = await _run_subworkflow(_Paid(_paid("ok")), None, _PAYS)

    assert result.step_results["sub"].token_usage == TokenUsage(
        prompt_tokens=10, completion_tokens=1, total_tokens=13, reasoning_tokens=2
    )
    assert result.total_tokens == 13


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
            "config": {"provider": "paid", "model": "m", "timeout": 0.5},
            "steps": [{"id": "ask", "type": "llm_call", "prompt": "2+3?"}],
        }
    )
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("5")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway)
    charge = engine._budget.charge

    async def contended(cost_usd: float) -> None:
        # The budget's lock is held past the workflow's deadline, however long the call took
        await anyio.sleep(engine._deadline - anyio.current_time() + 0.1)
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
    assert (result.total_cost_usd, result.total_tokens) == (pytest.approx(0.2), 26)
    assert engine._budget.spent == pytest.approx(0.2)
    assert "slow" not in result.step_results  # no status says cancelled: its span does
    assert ("slow", "cancelled", pytest.approx(0.1)) in engine.observer.ends


async def test_a_subworkflow_sharing_the_budget_reports_what_its_own_steps_spent() -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m", "budget_usd": 1.0},
            "steps": [
                {"id": "first", "type": "llm_call", "prompt": "hi"},
                {
                    "id": "sub",
                    "type": "subworkflow",
                    "depends_on": ["first"],
                    "workflow_inline": {
                        "name": "child",
                        "config": {"provider": "paid", "model": "m"},
                        "steps": [_PAYS],
                    },
                },
            ],
        }
    )
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("first"), _paid("child")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, observer=_Tokens())
    result = await engine.run()

    assert result.total_cost_usd == pytest.approx(0.2)
    assert engine.observer.runs == [("child", pytest.approx(0.1)), ("parent", pytest.approx(0.2))]
    assert _RUNS.get() == ()  # a task that runs workflows one after another holds none of them


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


_GATE = {"id": "gate", "type": "approval_gate", "prompt": "ok?", "depends_on": ["a"]}


@pytest.mark.parametrize("parent_budget", [1.0, 0.05], ids=["within-budget", "over-budget"])
async def test_a_subworkflow_with_a_budget_of_its_own_that_pauses_is_charged(
    tmp_path: Any, parent_budget: float
) -> None:
    child = {
        "name": "child",
        "config": {"provider": "paid", "model": "m", "budget_usd": 10.0},
        "steps": [_PAYS, _GATE],
    }
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m", "budget_usd": parent_budget},
            "steps": [
                {"id": "sub", "type": "subworkflow", "workflow_inline": child},
                {"id": "send", "type": "llm_call", "prompt": "send", "depends_on": ["sub"]},
            ],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("drafted")))
    engine = WorkflowEngine(
        workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer, observer=_Tokens()
    )
    result = await engine.run()

    assert result.status == WorkflowStatus.PAUSED  # a breach gives way to the pause
    assert result.step_results["sub"].cost_usd == pytest.approx(0.1)
    assert result.total_cost_usd == pytest.approx(0.1)
    assert result.total_tokens == 13
    assert engine._budget.spent == pytest.approx(0.1)
    assert (await checkpointer.load(engine.run_id)).spent_usd == pytest.approx(0.1)
    assert engine.observer.seen == [("m", 10)]  # by the child's step, not again by the parent
    assert engine.observer.runs == [("child", pytest.approx(0.1)), ("parent", pytest.approx(0.1))]


class _BreaksAt(_Tokens):
    """An observer that raises when step ``b`` starts: the run ends ``failed`` from outside."""

    def on_step_start(self, step_id: str, *_: Any, **__: Any) -> None:
        if step_id == "b":
            raise RuntimeError("observer down")


_CHILD = {"name": "child", "config": {"provider": "paid", "model": "m"}, "steps": [_PAYS]}


@pytest.mark.parametrize(
    ("status", "steps", "budget", "observer"),
    [
        # its child charged the shared budget, and its step reports cost 0
        (
            WorkflowStatus.PAUSED,
            [
                {"id": "sub", "type": "subworkflow", "workflow_inline": _CHILD},
                {**_GATE, "depends_on": ["sub"]},
            ],
            1.0,
            _Tokens,
        ),
        (WorkflowStatus.BUDGET_EXCEEDED, [_PAYS], 0.05, _Tokens),
        (WorkflowStatus.FAILED, [_PAYS, _HANGS], None, _BreaksAt),
    ],
    ids=["paused", "budget-exceeded", "failed-from-outside"],
)
async def test_a_run_reports_what_it_spent_however_it_ends(
    tmp_path: Any,
    status: WorkflowStatus,
    steps: list[dict[str, Any]],
    budget: float | None,
    observer: type[_Tokens],
) -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m", "budget_usd": budget},
            "steps": steps,
        }
    )
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("ok")))
    engine = WorkflowEngine(
        workflow=workflow,
        provider_gateway=gateway,
        observer=observer(),
        checkpointer=FileCheckpointer(checkpoint_dir=tmp_path),
    )
    result = await engine.run()

    assert result.status == status
    assert (result.total_cost_usd, result.total_tokens) == (pytest.approx(0.1), 13)
    assert engine.observer.runs[-1] == ("paid", pytest.approx(0.1))


async def test_a_step_that_breaks_the_budget_reaches_the_token_metrics() -> None:
    result, engine = await _run(_Paid(_paid("5")), budget=0.05)

    assert result.status == WorkflowStatus.BUDGET_EXCEEDED
    assert engine.observer.seen == [("m", 10)]


@pytest.mark.parametrize("child_budget", [None, 10.0], ids=["shared", "own"])
@pytest.mark.parametrize(
    "cut",
    [{"timeout": 0.2, "retry": {"max_retries": 0}}, {"workflow_timeout": 0.2}],
    ids=["its-timeout", "the-workflows"],
)
async def test_a_subworkflow_cut_short_reports_its_steps_tokens(
    cut: dict[str, Any], child_budget: float | None
) -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    asks_then_hangs = {**_HANGS, "tools": [{"name": "add", "parameters": _Add.parameters_schema}]}
    result, engine = await _run_subworkflow(
        _Paid(_paid("ok"), _paid("", call), None), child_budget, _PAYS, asks_then_hangs, **cut
    )

    sub = result.step_results["sub"]
    assert sub.status == StepStatus.TIMEOUT
    assert sub.token_usage.prompt_tokens == 20  # "b" was cut after one answered call
    assert (result.total_tokens, result.total_cost_usd) == (26, pytest.approx(0.2))
    assert engine._budget.spent == pytest.approx(0.2)
    assert engine.observer.seen == [("m", 10), ("m", 10)]  # by the child's steps only
    assert engine.observer.run_tokens == {"child": 26, "parent": 26}


_LONG_CHAT = {
    "messages": [
        {"role": "user", "content": "x" * 400},
        {"role": "assistant", "content": "x" * 400},
    ],
    "token_budget": 60,
    "trim_policy": "summarize_oldest",
}


@pytest.mark.parametrize(
    ("step", "turns", "status"),
    [
        (  # a tool call answered, then the provider fails for good
            {"tools": [{"name": "add", "parameters": _Add.parameters_schema}]},
            [_paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})), _bad_request()],
            StepStatus.FAILED,
        ),
        (  # a structured reply never valid, asked again once
            {"response_schema": {"type": "json_object"}, "retry": {"max_retries": 1}},
            [_paid("not JSON"), _paid("not JSON either")],
            StepStatus.FAILED,
        ),
        (  # a summary paid before the reply fails
            {"conversation": "state.chat"},
            [_paid("summary"), _bad_request()],
            StepStatus.FAILED,
        ),
        (  # or before it succeeds
            {"conversation": "state.chat"},
            [_paid("summary"), _paid("reply")],
            StepStatus.SUCCESS,
        ),
    ],
    ids=["tool-loop", "structured-reply", "summary", "summary-then-reply"],
)
async def test_an_llm_call_reports_what_a_gateway_of_its_own_spent(
    step: dict[str, Any], turns: list[ProviderResponse | Exception], status: StepStatus
) -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m"},
            "state": {"chat": _LONG_CHAT},
            "steps": [
                {
                    "id": "ask",
                    "type": "llm_call",
                    "prompt": "2+3?",
                    "retry": {"max_retries": 0},
                    **step,
                }
            ],
        }
    )
    registry = ToolRegistry()
    registry.register(_Add())
    paid = sum(isinstance(turn, ProviderResponse) for turn in turns)
    engine = WorkflowEngine(
        workflow=workflow, provider_gateway=_OwnGateway(*turns), tool_registry=registry
    )
    result = await engine.run()

    ask = result.step_results["ask"]
    assert ask.status == status
    assert ask.cost_usd == pytest.approx(0.1 * paid)
    assert ask.token_usage.prompt_tokens == 10 * paid
    assert engine._budget.spent == pytest.approx(0.1 * paid)


@pytest.mark.parametrize("child_budget", [None, 10.0], ids=["shared", "own"])
async def test_a_subworkflow_resumed_at_its_gate_goes_on_from_its_checkpoint(
    tmp_path: Any, child_budget: float | None
) -> None:
    child = {
        "name": "child",
        "config": {"provider": "paid", "model": "m", "budget_usd": child_budget},
        "steps": [
            {**_PAYS, "output": "draft"},
            _GATE,
            {"id": "c", "type": "llm_call", "prompt": "send {draft}", "depends_on": ["gate"]},
        ],
    }
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m", "budget_usd": 1.0},
            "steps": [
                {"id": "sub", "type": "subworkflow", "workflow_inline": child},
                {"id": "send", "type": "llm_call", "prompt": "send", "depends_on": ["sub"]},
            ],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("drafted")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer)
    assert (await engine.run()).status == WorkflowStatus.PAUSED

    second = _Paid(_paid("sent by the child"), _paid("sent"))  # nothing for "a" again
    gateway = ProviderGateway()
    gateway.register(second)
    resumed = await WorkflowEngine.from_checkpoint(
        await checkpointer.load(engine.run_id),
        checkpointer=checkpointer,
        provider_gateway=gateway,
        observer=_Tokens(),
        approval_decisions={"sub.gate": "approved"},
    )
    result = await resumed.run()

    assert result.status == WorkflowStatus.SUCCESS
    assert not second._turns
    sub = result.step_results["sub"]
    assert sub.cost_usd == pytest.approx(0.0 if child_budget is None else 0.2)
    assert sub.token_usage.prompt_tokens == 20  # "a" before the pause, "c" after it
    assert (result.total_cost_usd, result.total_tokens) == (pytest.approx(0.3), 39)
    assert resumed._budget.spent == pytest.approx(0.3)
    assert resumed.observer.seen == [("m", 10), ("m", 10)]


async def test_a_subworkflow_resumed_at_its_gate_goes_on_from_its_own_budget(
    tmp_path: Any,
) -> None:
    child = {
        "name": "child",
        "config": {"provider": "paid", "model": "m", "budget_usd": 0.15},
        "steps": [
            _PAYS,
            _GATE,
            {"id": "c", "type": "llm_call", "prompt": "c", "depends_on": ["gate"]},
        ],
    }
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m"},
            "steps": [{"id": "sub", "type": "subworkflow", "workflow_inline": child}],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("drafted")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer)
    assert (await engine.run()).status == WorkflowStatus.PAUSED

    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("sent")))
    resumed = await WorkflowEngine.from_checkpoint(
        await checkpointer.load(engine.run_id),
        checkpointer=checkpointer,
        provider_gateway=gateway,
        approval_decisions={"sub.gate": "approved"},
    )
    result = await resumed.run()

    sub = result.step_results["sub"]  # its budget held what "a" had spent before the gate
    assert (sub.status, sub.error_classification) == (StepStatus.FAILED, "permanent")
    assert "exceeded its own budget" in (sub.error or "")
    assert sub.cost_usd == pytest.approx(0.2)
    assert result.total_cost_usd == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("first", "timeout", "decisions"),
    [
        (  # it failed next to the gate the run paused at
            [_paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})), _bad_request()],
            None,
            {"gate": "approved"},
        ),
        (  # the workflow's timeout cut it
            [_paid("", ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})), None],
            0.2,
            None,
        ),
    ],
    ids=["failed", "timed-out"],
)
async def test_a_step_run_again_on_a_resume_keeps_what_it_spent_before(
    tmp_path: Any,
    first: list[ProviderResponse | Exception | None],
    timeout: float | None,
    decisions: dict[str, str] | None,
) -> None:
    steps: list[dict[str, Any]] = [
        {
            "id": "ask",
            "type": "llm_call",
            "prompt": "2+3?",
            "retry": {"max_retries": 0},
            "tools": [{"name": "add", "parameters": _Add.parameters_schema}],
        }
    ]
    if decisions:
        steps.append({"id": "gate", "type": "approval_gate", "prompt": "ok?"})
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m", "timeout": timeout},
            "steps": steps,
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(_Add())
    gateway = ProviderGateway()
    gateway.register(_Paid(*first))
    engine = WorkflowEngine(
        workflow=workflow,
        provider_gateway=gateway,
        tool_registry=registry,
        checkpointer=checkpointer,
    )
    assert (await engine.run()).step_results["ask"].cost_usd == pytest.approx(0.1)

    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("5")))
    resumed = await WorkflowEngine.from_checkpoint(
        await checkpointer.load(engine.run_id),
        checkpointer=checkpointer,
        provider_gateway=gateway,
        tool_registry=registry,
        approval_decisions=decisions,
    )
    result = await resumed.run()

    ask = result.step_results["ask"]
    assert (result.status, ask.status) == (WorkflowStatus.SUCCESS, StepStatus.SUCCESS)
    assert (ask.cost_usd, ask.token_usage.prompt_tokens) == (pytest.approx(0.2), 20)
    assert (result.total_cost_usd, result.total_tokens) == (pytest.approx(0.2), 26)
    assert resumed._budget.spent == pytest.approx(0.2)


@pytest.mark.parametrize("child_budget", [None, 10.0], ids=["shared", "own"])
async def test_a_resumed_subworkflow_counts_a_step_its_child_runs_again(
    tmp_path: Any, child_budget: float | None
) -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    flaky = {
        "id": "flaky",
        "type": "llm_call",
        "prompt": "2+3?",
        "depends_on": ["a"],
        "retry": {"max_retries": 0},
        "tools": [{"name": "add", "parameters": _Add.parameters_schema}],
    }
    child = {
        "name": "child",
        "config": {"provider": "paid", "model": "m", "budget_usd": child_budget},
        "steps": [_PAYS, flaky, _GATE],
    }
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m"},
            "steps": [{"id": "sub", "type": "subworkflow", "workflow_inline": child}],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(_Add())
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("a"), _paid("", call), _bad_request()))
    engine = WorkflowEngine(
        workflow=workflow,
        provider_gateway=gateway,
        tool_registry=registry,
        checkpointer=checkpointer,
    )
    assert (await engine.run()).status == WorkflowStatus.PAUSED

    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("", call), _paid("5")))  # "flaky" again
    resumed = await WorkflowEngine.from_checkpoint(
        await checkpointer.load(engine.run_id),
        checkpointer=checkpointer,
        provider_gateway=gateway,
        tool_registry=registry,
        approval_decisions={"sub.gate": "approved"},
    )
    result = await resumed.run()

    sub = result.step_results["sub"]
    assert (result.status, sub.status) == (WorkflowStatus.SUCCESS, StepStatus.SUCCESS)
    assert sub.token_usage.prompt_tokens == 40  # four paid calls, two of them before the pause
    assert (result.total_cost_usd, result.total_tokens) == (pytest.approx(0.4), 52)
    assert resumed._budget.spent == pytest.approx(0.4)


async def test_a_new_run_that_reuses_a_run_id_runs_its_child_afresh(tmp_path: Any) -> None:
    child = {
        "name": "child",
        "config": {"provider": "paid", "model": "m"},
        "steps": [{**_PAYS, "output": "draft"}, _GATE],
    }
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m"},
            "steps": [{"id": "sub", "type": "subworkflow", "workflow_inline": child}],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    for _ in range(2):  # the second run finds the checkpoint the first one's child left
        provider = _Paid(_paid("draft"))
        gateway = ProviderGateway()
        gateway.register(provider)
        engine = WorkflowEngine(
            workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer, run_id="nightly"
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.PAUSED
        assert not provider._turns  # the child's first step ran, and was paid, again
        assert engine._budget.spent == pytest.approx(0.1)


@pytest.mark.parametrize(
    "checkpointed", [True, False], ids=["its-checkpoint-gone", "no-checkpointer"]
)
async def test_a_subworkflow_paused_with_no_checkpoint_of_its_child_runs_it_afresh(
    tmp_path: Any, checkpointed: bool
) -> None:
    from agentloom.core.results import StepResult
    from agentloom.core.state import StateManager

    child = {"name": "child", "config": {"provider": "paid", "model": "m"}, "steps": [_PAYS, _GATE]}
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m"},
            "steps": [{"id": "sub", "type": "subworkflow", "workflow_inline": child}],
        }
    )
    provider = _Paid(_paid("a"), _paid("a again")) if checkpointed else _Paid(_paid("a"))
    gateway = ProviderGateway()
    gateway.register(provider)
    if checkpointed:
        checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
        first = WorkflowEngine(
            workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer
        )
        assert (await first.run()).status == WorkflowStatus.PAUSED
        await checkpointer.delete(f"{first.run_id}.sub")
        engine = await WorkflowEngine.from_checkpoint(
            await checkpointer.load(first.run_id),
            checkpointer=checkpointer,
            provider_gateway=gateway,
            approval_decisions={"sub.gate": "approved"},
        )
    else:  # a step paused in the state, and no checkpointer to resume its child from
        state = StateManager(initial_state={"_approval": {"sub": {"gate": "approved"}}})
        await state.set_step_result("sub", StepResult(step_id="sub", status=StepStatus.PAUSED))
        engine = WorkflowEngine(workflow=workflow, state_manager=state, provider_gateway=gateway)
    result = await engine.run()

    assert result.status == WorkflowStatus.SUCCESS
    assert not provider._turns  # the child ran from its first step


async def test_a_subworkflow_sharing_the_budget_that_breaks_it_reports_its_tokens() -> None:
    result, engine = await _run_subworkflow(
        _Paid(_paid("a"), _paid("b")),
        None,
        _PAYS,
        {**_PAYS, "id": "b", "depends_on": ["a"]},
        budget=0.15,
    )

    assert result.status == WorkflowStatus.BUDGET_EXCEEDED
    sub = result.step_results["sub"]
    assert (sub.status, sub.cost_usd) == (StepStatus.FAILED, 0.0)  # its child charged it
    assert sub.token_usage.prompt_tokens == 20
    assert (result.total_cost_usd, result.total_tokens) == (pytest.approx(0.2), 26)
    assert engine._budget.spent == pytest.approx(0.2)
    assert engine.observer.seen == [("m", 10), ("m", 10)]  # by the child's steps only
    assert ("sub", "budget_exceeded", 0.0) in engine.observer.ends


class _ChildEndFails(_Tokens):
    """An observer that raises when the child run ends: its step fails outside its steps."""

    def on_workflow_end(self, name: str, *args: Any, **kwargs: Any) -> None:
        super().on_workflow_end(name, *args, **kwargs)
        if name == "child":
            raise RuntimeError("observer down")


@pytest.mark.parametrize("child_budget", [None, 10.0], ids=["shared", "own"])
async def test_a_subworkflow_that_fails_outside_its_steps_is_charged_what_they_spent(
    child_budget: float | None,
) -> None:
    result, engine = await _run_subworkflow(
        _Paid(_paid("a")),
        child_budget,
        _PAYS,
        observer=_ChildEndFails(),
        retry={"max_retries": 0},
    )

    sub = result.step_results["sub"]
    assert sub.status == StepStatus.FAILED
    assert sub.cost_usd == pytest.approx(0.0 if child_budget is None else 0.1)
    assert sub.token_usage.prompt_tokens == 10
    assert engine._budget.spent == pytest.approx(0.1)
    assert engine.observer.seen == [("m", 10)]


class _StreamsThenFails(BaseProvider):
    """Streams an answer; the first stream fails once the provider has billed it."""

    name = "paid"

    def __init__(self) -> None:
        super().__init__()
        self.streams = 0

    async def complete(self, **kw: Any) -> ProviderResponse:
        raise AssertionError("the step streams")

    async def stream(self, **kw: Any) -> StreamResponse:
        self.streams += 1
        fails = self.streams == 1
        response = StreamResponse(model="m", provider="paid")

        async def chunks() -> AsyncIterator[str]:
            yield "5"
            response.usage = _paid().usage
            response.cost_usd = 0.1
            if fails:
                raise ProviderError("paid", "reset after its usage", status_code=503)

        response._set_iterator(chunks())
        return response

    def supports_model(self, _m: str) -> bool:
        return True


async def test_a_stream_that_fails_after_its_usage_reaches_the_token_metrics() -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m"},
            "steps": [
                {
                    "id": "ask",
                    "type": "llm_call",
                    "prompt": "2+3?",
                    "stream": True,
                    "retry": {"max_retries": 1, "backoff_base": 0, "jitter": False},
                }
            ],
        }
    )
    gateway = ProviderGateway()
    gateway.register(_StreamsThenFails())
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, observer=_Tokens())
    result = await engine.run()

    ask = result.step_results["ask"]
    assert ask.status == StepStatus.SUCCESS
    assert (ask.cost_usd, ask.token_usage.prompt_tokens) == (pytest.approx(0.2), 20)
    assert engine._budget.spent == pytest.approx(0.2)
    assert engine.observer.seen == [("m", 10), ("m", 10)]


async def test_a_subworkflow_whose_step_id_no_checkpoint_takes_resumes_its_child_afresh(
    tmp_path: Any,
) -> None:
    child = {"name": "child", "config": {"provider": "paid", "model": "m"}, "steps": [_PAYS, _GATE]}
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "parent",
            "config": {"provider": "paid", "model": "m"},
            "steps": [{"id": "my sub", "type": "subworkflow", "workflow_inline": child}],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    gateway = ProviderGateway()
    gateway.register(_Paid(_paid("a")))
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, checkpointer=checkpointer)
    # The file checkpointer refuses the child's run_id: the parent pauses all the same
    assert (await engine.run()).status == WorkflowStatus.PAUSED

    provider = _Paid(_paid("a again"))
    gateway = ProviderGateway()
    gateway.register(provider)
    resumed = await WorkflowEngine.from_checkpoint(
        await checkpointer.load(engine.run_id),
        checkpointer=checkpointer,
        provider_gateway=gateway,
        approval_decisions={"my sub.gate": "approved"},
    )
    result = await resumed.run()

    assert result.status == WorkflowStatus.SUCCESS
    assert not provider._turns  # the child ran from its first step


async def test_a_step_the_deadline_cuts_before_it_runs_again_keeps_what_it_spent(
    tmp_path: Any,
) -> None:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m", "timeout": 0.3, "max_concurrent_steps": 1},
            "steps": [
                # One step at a time: "aaa" runs before "ask"
                {"id": "aaa", "type": "llm_call", "prompt": "x", "retry": {"max_retries": 0}},
                {
                    "id": "ask",
                    "type": "llm_call",
                    "prompt": "2+3?",
                    "retry": {"max_retries": 0},
                    "tools": [{"name": "add", "parameters": _Add.parameters_schema}],
                },
            ],
        }
    )
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(_Add())
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    gateway = ProviderGateway()
    gateway.register(_Paid(_bad_request(), _paid("", call), None))  # "ask" pays, then hangs
    engine = WorkflowEngine(
        workflow=workflow,
        provider_gateway=gateway,
        tool_registry=registry,
        checkpointer=checkpointer,
    )
    assert (await engine.run()).step_results["ask"].cost_usd == pytest.approx(0.1)

    gateway = ProviderGateway()
    gateway.register(_Paid(None))  # "aaa" hangs: the deadline comes before "ask" starts
    resumed = await WorkflowEngine.from_checkpoint(
        await checkpointer.load(engine.run_id),
        checkpointer=checkpointer,
        provider_gateway=gateway,
        tool_registry=registry,
    )
    result = await resumed.run()

    ask = result.step_results["ask"]
    assert (result.status, ask.status) == (WorkflowStatus.TIMEOUT, StepStatus.TIMEOUT)
    assert (ask.cost_usd, ask.token_usage.prompt_tokens) == (pytest.approx(0.1), 10)
    assert (result.total_cost_usd, result.total_tokens) == (pytest.approx(0.1), 13)
