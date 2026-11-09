"""Tests for failures that cannot pass on a later attempt: they skip the step's retries."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import StepDefinition, StepType, WorkflowConfig, WorkflowDefinition
from agentloom.core.results import StepResult, StepStatus, WorkflowResult
from agentloom.core.state import StateManager
from agentloom.exceptions import (
    ProviderError,
    SandboxViolationError,
    StepError,
    UnsupportedRequestError,
)
from agentloom.providers.base import ProviderResponse, ToolCall
from agentloom.providers.gateway import ProviderGateway, _Failures
from agentloom.resilience.circuit_breaker import CircuitState
from agentloom.resilience.retry import classify_failure
from agentloom.steps.agent import AgentStep
from agentloom.steps.base import StepContext
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry
from tests.conftest import MockProvider

MESSAGES = [{"role": "user", "content": "hi"}]
CONDITION = {"expression": "state.x", "target": "s"}
UNKNOWN_DEPENDENCY = {
    "name": "child",
    "steps": [{"id": "c", "type": "llm_call", "prompt": "p", "depends_on": ["nope"]}],
}


@pytest.fixture
def retries(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """One entry per retry the engine is about to make, each without a wait."""
    made: list[float] = []

    def _backoff(*_: object) -> float:
        made.append(0.0)
        return 0.0

    monkeypatch.setattr("agentloom.core.engine.compute_backoff", _backoff)
    return made


class _Raises(BaseTool):
    name = "raises"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("flaky")

    async def execute(self, **kwargs: Any) -> Any:
        raise self.error


class _Refuses(MockProvider):
    name = "refuses"

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise UnsupportedRequestError(self.name, "takes no PDFs")

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise UnsupportedRequestError(self.name, "takes no PDFs")


class _Down(MockProvider):
    name = "down"

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise ProviderError(self.name, "boom", status_code=500)


class _CallsATool(MockProvider):
    name = "calls"

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        call = ToolCall(id="c", name="raises", arguments={})
        return ProviderResponse(content="", model="m", provider=self.name, tool_calls=[call])


class _NoEmbeddings(MockProvider):
    name = "no-embeddings"

    async def embed(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("no embeddings")


def _fetch_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com/a.png")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("fetch failed", request=request, response=response)


def _gateway(*providers: MockProvider, **options: Any) -> ProviderGateway:
    gateway = ProviderGateway()
    for priority, provider in enumerate(providers or (MockProvider(),)):
        gateway.register(provider, priority=priority, **options)
    return gateway


def _tools(error: Exception | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_Raises(error))
    return registry


def _step(step_type: StepType, **fields: Any) -> StepDefinition:
    return StepDefinition(id="s", type=step_type, retry={"max_retries": 2}, **fields)


async def _run(
    step: StepDefinition,
    state: dict[str, Any] | None = None,
    gateway: ProviderGateway | None = None,
    tools: ToolRegistry | None = None,
) -> StepResult:
    workflow = WorkflowDefinition(
        name="permanent",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state=state or {},
        steps=[step],
    )
    result = await WorkflowEngine(
        workflow=workflow, provider_gateway=gateway, tool_registry=tools
    ).run()
    return result.step_results["s"]


def _at_once(result: StepResult, retries: list[float]) -> None:
    assert result.status == StepStatus.FAILED
    assert result.error_classification == "permanent"
    assert retries == []


def _all_retries(result: StepResult, retries: list[float]) -> None:
    assert result.status == StepStatus.FAILED
    assert result.error_classification == "transient"
    assert len(retries) == 2


class TestFailsAtOnce:
    @pytest.mark.parametrize(
        ("step_type", "fields", "state"),
        [
            (StepType.ROUTER, {"conditions": [{**CONDITION, "expression": "1 >"}]}, {}),
            (
                StepType.ROUTER,
                {"conditions": [{**CONDITION, "expression": "state.x > 1"}]},
                {"x": "a"},
            ),
            (StepType.ROUTER, {"conditions": [CONDITION]}, {"x": False}),
            (StepType.ROUTER, {}, {}),
            (StepType.TOOL, {}, {}),
            (StepType.TOOL, {"tool_name": "raises", "tool_args": {"a": "{x:d}"}}, {"x": "a"}),
            (StepType.EMBED, {}, {}),
            (StepType.EMBED, {"inputs": "state.nope"}, {}),
            (StepType.EMBED, {"inputs": "state.x"}, {"x": 1}),
            (StepType.APPROVAL_GATE, {}, {"_approval": {"s": "maybe"}}),
            (
                StepType.LLM_CALL,
                {"prompt": "p", "attachments": [{"type": "image", "source": "{x:d}"}]},
                {"x": "a"},
            ),
            (StepType.LLM_CALL, {"prompt": "p", "conversation": "state..chat"}, {}),
            (StepType.LLM_CALL, {"prompt": "p", "conversation": "state.chat"}, {"chat": 5}),
            (StepType.LLM_CALL, {"conversation": "state.chat"}, {"chat": {"messages": []}}),
            (StepType.SUBWORKFLOW, {"workflow_path": "/no/such/workflow.yaml"}, {}),
            (StepType.SUBWORKFLOW, {"workflow_inline": UNKNOWN_DEPENDENCY}, {}),
            (StepType.SUBWORKFLOW, {}, {}),
        ],
        ids=[
            "router-syntax",
            "router-raises",
            "router-no-match",
            "router-empty",
            "tool-no-name",
            "tool-args",
            "embed-no-inputs",
            "embed-missing-input",
            "embed-wrong-type",
            "gate-decision",
            "llm-attachment-template",
            "llm-conversation-unreadable",
            "llm-conversation-malformed",
            "llm-conversation-last-turn",
            "subworkflow-load",
            "subworkflow-inline",
            "subworkflow-empty",
        ],
    )
    async def test_a_failure_that_cannot_pass_is_not_retried(
        self,
        retries: list[float],
        step_type: StepType,
        fields: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        result = await _run(_step(step_type, **fields), state, _gateway(), _tools())
        _at_once(result, retries)

    @pytest.mark.parametrize(
        ("step_type", "fields", "state"),
        [
            (StepType.TOOL, {"tool_name": "raises"}, {}),
            (StepType.EMBED, {"inputs": "state.x"}, {"x": "a"}),
            (StepType.LLM_CALL, {"prompt": "p"}, {}),
        ],
        ids=["tool-registry", "embed-gateway", "llm-gateway"],
    )
    async def test_a_step_missing_its_wiring_is_not_retried(
        self,
        retries: list[float],
        step_type: StepType,
        fields: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        _at_once(await _run(_step(step_type, **fields), state), retries)

    async def test_a_tool_call_without_a_registry_is_not_retried(
        self, retries: list[float]
    ) -> None:
        tools = [{"name": "raises", "description": "d", "parameters": {}}]
        step = _step(StepType.LLM_CALL, prompt="p", tools=tools)
        _at_once(await _run(step, gateway=_gateway(_CallsATool())), retries)

    async def test_an_agent_step_needs_a_gateway(self) -> None:
        step = StepDefinition(id="s", type=StepType.AGENT, agent={"name": "a"})
        context = StepContext(step_definition=step, state_manager=StateManager())
        with pytest.raises(StepError) as raised:
            await AgentStep().execute(context)
        assert raised.value.is_retryable is False

    @pytest.mark.parametrize(
        ("step_type", "fields", "state"),
        [
            (StepType.LLM_CALL, {"prompt": "p"}, {}),
            (StepType.EMBED, {"inputs": "state.x"}, {"x": "a"}),
        ],
        ids=["llm", "embed"],
    )
    async def test_a_model_no_provider_serves_is_not_retried(
        self,
        retries: list[float],
        step_type: StepType,
        fields: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        _at_once(await _run(_step(step_type, **fields), state, ProviderGateway()), retries)

    async def test_an_attachment_no_provider_takes_is_not_retried(
        self, retries: list[float]
    ) -> None:
        result = await _run(_step(StepType.LLM_CALL, prompt="p"), gateway=_gateway(_Refuses()))
        _at_once(result, retries)


class TestClassifiedByTheError:
    @pytest.mark.parametrize(
        "error",
        [SandboxViolationError("raises", "refused"), ProviderError("p", "gone", status_code=404)],
        ids=["sandbox", "status-not-listed"],
    )
    async def test_a_tool_error_that_cannot_pass_is_not_retried(
        self, retries: list[float], error: Exception
    ) -> None:
        result = await _run(_step(StepType.TOOL, tool_name="raises"), tools=_tools(error))
        _at_once(result, retries)
        assert "failed:" in (result.error or "")

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("flaky"), ProviderError("p", "busy", status_code=503)],
        ids=["own-error", "status-listed"],
    )
    async def test_any_other_tool_error_is_retried(
        self, retries: list[float], error: Exception
    ) -> None:
        result = await _run(_step(StepType.TOOL, tool_name="raises"), tools=_tools(error))
        _all_retries(result, retries)

    @staticmethod
    async def _attachment(error: Exception) -> StepResult:
        step = _step(
            StepType.LLM_CALL,
            prompt="p",
            attachments=[{"type": "image", "source": "https://example.com/a.png"}],
        )
        with patch("agentloom.steps.llm_call.resolve_attachments", AsyncMock(side_effect=error)):
            return await _run(step, gateway=_gateway())

    @pytest.mark.parametrize(
        "error", [PermissionError("private address"), _fetch_error(404)], ids=["refused", "404"]
    )
    async def test_an_attachment_that_cannot_pass_is_not_retried(
        self, retries: list[float], error: Exception
    ) -> None:
        _at_once(await self._attachment(error), retries)

    @pytest.mark.parametrize(
        "error", [httpx.ConnectError("down"), _fetch_error(503)], ids=["no-answer", "503"]
    )
    async def test_an_attachment_that_may_pass_is_retried(
        self, retries: list[float], error: Exception
    ) -> None:
        _all_retries(await self._attachment(error), retries)

    @staticmethod
    async def _child_fails(monkeypatch: pytest.MonkeyPatch, error: Exception) -> StepResult:
        real = WorkflowEngine.run

        async def _run_or_raise(self: WorkflowEngine) -> WorkflowResult:
            if self.workflow.name == "child":
                raise error
            return await real(self)

        monkeypatch.setattr(WorkflowEngine, "run", _run_or_raise)
        inline = {
            "name": "child",
            "config": {"provider": "mock", "model": "mock-model"},
            "steps": [{"id": "c", "type": "llm_call", "prompt": "p"}],
        }
        return await _run(_step(StepType.SUBWORKFLOW, workflow_inline=inline), gateway=_gateway())

    async def test_a_child_that_cannot_run_is_not_retried(
        self, retries: list[float], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = SandboxViolationError("c", "refused")
        _at_once(await self._child_fails(monkeypatch, error), retries)

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("flaky"), ProviderError("p", "busy", status_code=503)],
        ids=["own-error", "status-listed"],
    )
    async def test_a_child_cut_by_anything_else_is_retried(
        self, retries: list[float], monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        _all_retries(await self._child_fails(monkeypatch, error), retries)


class TestGateway:
    async def test_no_provider_for_the_model_is_unsupported(self) -> None:
        gateway = ProviderGateway()
        for call in (
            gateway.complete(MESSAGES, "m"),
            gateway.stream(MESSAGES, "m"),
            gateway.embed(["a"], "m"),
        ):
            with pytest.raises(UnsupportedRequestError, match="No provider registered"):
                await call

    @pytest.mark.parametrize("method", ["complete", "stream"])
    async def test_every_provider_refusing_is_unsupported(self, method: str) -> None:
        gateway = _gateway(_Refuses(), _Refuses())
        with pytest.raises(UnsupportedRequestError, match="All providers failed") as raised:
            await getattr(gateway, method)(MESSAGES, "m")
        assert "takes no PDFs" in str(raised.value)

    async def test_a_refusal_moves_on_to_the_next_provider(self) -> None:
        response = await _gateway(_Refuses(), MockProvider()).complete(MESSAGES, "m")
        assert response.content == "Mock response"

    async def test_a_refusal_beside_a_server_error_is_retried(self) -> None:
        with pytest.raises(ProviderError) as raised:
            await _gateway(_Refuses(), _Down()).complete(MESSAGES, "m")
        assert not isinstance(raised.value, UnsupportedRequestError)
        assert classify_failure(raised.value, [500]) == "transient"

    @pytest.mark.parametrize("refused_first", [True, False])
    def test_a_refusal_beside_an_open_circuit_is_retried(self, refused_first: bool) -> None:
        failures = _Failures()
        if refused_first:
            failures.add("refused", UnsupportedRequestError("p", "no"))
        failures.add("circuit is open")
        if not refused_first:
            failures.add("refused", UnsupportedRequestError("p", "no"))
        with pytest.raises(ProviderError) as raised:
            failures.raise_all("All failed")
        assert not isinstance(raised.value, UnsupportedRequestError)

    async def test_no_provider_with_embeddings_is_unsupported(self) -> None:
        with pytest.raises(UnsupportedRequestError, match="has no embeddings"):
            await _gateway(_NoEmbeddings()).embed(["a"], "m")

    @pytest.mark.parametrize("method", ["complete", "stream"])
    async def test_a_refusal_does_not_count_against_the_circuit(self, method: str) -> None:
        gateway = _gateway(_Refuses(), circuit_fail_threshold=1)
        for _ in range(2):
            with pytest.raises(UnsupportedRequestError):
                await getattr(gateway, method)(MESSAGES, "m")
        assert gateway._providers[0].circuit_breaker.state == CircuitState.CLOSED

    async def test_a_refused_stream_frees_the_half_open_slot(self) -> None:
        gateway = _gateway(_Refuses(), circuit_fail_threshold=1, circuit_reset_timeout=0.01)
        breaker = gateway._providers[0].circuit_breaker
        breaker._set_state(CircuitState.OPEN)
        breaker._last_failure_time = time.monotonic() - 1.0
        with pytest.raises(UnsupportedRequestError):
            await gateway.stream(MESSAGES, "m")
        assert breaker._half_open_calls == 0
