"""An ``llm_call`` step with ``tools`` runs its tool loop when it streams."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    ToolDefinition,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepStatus, WorkflowResult, WorkflowStatus
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse, ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry

_CLOCK = ToolDefinition(
    name="clock", description="Current time.", parameters={"type": "object", "properties": {}}
)
_CALL = ToolCall(id="c1", name="clock", arguments={})


class _Clock(BaseTool):
    name = "clock"
    description = "Current time."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: Any) -> Any:
        self.calls += 1
        return "12:00"


class _Scripted(BaseProvider):
    """Answers ``complete()`` from a script of turns and counts ``stream()`` calls."""

    name = "scripted"

    def __init__(self, *turns: ProviderResponse) -> None:
        super().__init__()
        self._turns = list(turns)
        self.streams = 0

    async def complete(self, **_kw: Any) -> ProviderResponse:
        return self._turns.pop(0)

    async def stream(self, **kw: Any) -> StreamResponse:
        self.streams += 1
        return await super().stream(**kw)

    def supports_model(self, _m: str) -> bool:
        return True


def _turn(content: str, *calls: ToolCall) -> ProviderResponse:
    return ProviderResponse(content=content, model="m", provider="scripted", tool_calls=list(calls))


async def _run(
    provider: _Scripted,
    chunks: list[tuple[str, str]],
    *,
    step_stream: bool | None = True,
    config_stream: bool = False,
    broken_callback: bool = False,
) -> tuple[WorkflowResult, _Clock]:
    def _on_chunk(step_id: str, chunk: str) -> None:
        chunks.append((step_id, chunk))
        if broken_callback:
            raise RuntimeError("callback broke")

    workflow = WorkflowDefinition(
        name="stream-tools",
        config=WorkflowConfig(provider="scripted", model="m", stream=config_stream),
        steps=[
            StepDefinition(
                id="ask",
                type=StepType.LLM_CALL,
                prompt="What time is it?",
                tools=[_CLOCK],
                stream=step_stream,
                output="answer",
                retry={"max_retries": 0},
            )
        ],
    )
    clock = _Clock()
    registry = ToolRegistry()
    registry.register(clock)
    gateway = ProviderGateway()
    gateway.register(provider, models=["m"])
    engine = WorkflowEngine(
        workflow=workflow,
        provider_gateway=gateway,
        tool_registry=registry,
        on_stream_chunk=_on_chunk,
    )
    result = await engine.run()
    await gateway.close()
    return result, clock


@pytest.mark.parametrize(("step_stream", "config_stream"), [(True, False), (None, True)])
async def test_streamed_step_with_tools_runs_its_tool_loop(
    step_stream: bool | None, config_stream: bool, caplog: pytest.LogCaptureFixture
) -> None:
    provider = _Scripted(_turn("", _CALL), _turn("It is 12:00."))
    chunks: list[tuple[str, str]] = []
    with caplog.at_level(logging.INFO, logger="agentloom.steps"):
        result, clock = await _run(
            provider, chunks, step_stream=step_stream, config_stream=config_stream
        )

    assert result.status == WorkflowStatus.SUCCESS
    assert clock.calls == 1
    assert provider.streams == 0
    assert result.final_state["answer"] == "It is 12:00."
    # The answer reaches the callback once, whole, after the loop.
    assert chunks == [("ask", "It is 12:00.")]
    assert "Step 'ask' has tools" in caplog.text


async def test_step_with_tools_that_does_not_stream_sends_no_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunks: list[tuple[str, str]] = []
    with caplog.at_level(logging.INFO, logger="agentloom.steps"):
        result, clock = await _run(
            _Scripted(_turn("", _CALL), _turn("It is 12:00.")), chunks, step_stream=False
        )

    assert result.final_state["answer"] == "It is 12:00."
    assert clock.calls == 1
    assert chunks == []
    assert "has tools" not in caplog.text


async def test_empty_final_answer_sends_no_chunk() -> None:
    chunks: list[tuple[str, str]] = []
    result, clock = await _run(_Scripted(_turn("", _CALL), _turn("")), chunks)

    assert result.step_results["ask"].status == StepStatus.SUCCESS
    assert clock.calls == 1
    assert chunks == []


async def test_failing_stream_callback_leaves_the_step_successful(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunks: list[tuple[str, str]] = []
    with caplog.at_level(logging.WARNING, logger="agentloom.steps"):
        result, _ = await _run(
            _Scripted(_turn("", _CALL), _turn("It is 12:00.")), chunks, broken_callback=True
        )

    assert result.step_results["ask"].status == StepStatus.SUCCESS
    assert result.final_state["answer"] == "It is 12:00."
    assert chunks == [("ask", "It is 12:00.")]
    assert "Stream chunk callback failed" in caplog.text
