"""What the attempts of a step that failed spent is charged and reported (#120)."""

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
    """Keeps the token observations of a run."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def on_tokens(self, provider: str, model: str, prompt: int, completion: int, **_: Any) -> None:
        self.seen.append((model, prompt))


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
    provider: _Paid, *, retries: int = 1, budget: float | None = None, **step: Any
) -> tuple[WorkflowResult, WorkflowEngine]:
    workflow = WorkflowDefinition.model_validate(
        {
            "name": "paid",
            "config": {"provider": "paid", "model": "m", "budget_usd": budget},
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
