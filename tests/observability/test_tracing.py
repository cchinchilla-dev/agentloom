"""Tests for tracing manager."""

from __future__ import annotations

import logging
from typing import Any

import anyio
import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    RetryConfig,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepStatus, WorkflowStatus
from agentloom.observability.noop import NoopSpan
from agentloom.observability.observer import WorkflowObserver
from agentloom.observability.schema import SpanAttr, SpanName
from agentloom.observability.tracing import TracingManager
from agentloom.providers.base import ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from tests.conftest import MockProvider


class TestTracingManagerDisabled:
    def test_disabled_returns_noop(self) -> None:
        tm = TracingManager(enabled=False)
        span = tm.start_span("test")
        assert isinstance(span, NoopSpan)

    def test_disabled_get_tracer_returns_noop(self) -> None:
        from agentloom.observability.noop import NoopTracer

        tm = TracingManager(enabled=False)
        tracer = tm.get_tracer()
        assert isinstance(tracer, NoopTracer)

    def test_disabled_shutdown_no_error(self) -> None:
        tm = TracingManager(enabled=False)
        tm.shutdown()


class TestTracingManagerEnabled:
    def test_enabled_with_otel_creates_tracer(self) -> None:
        tm = TracingManager(enabled=True)
        # If OTel is installed, we get a real tracer; if not, noop
        tracer = tm.get_tracer()
        assert tracer is not None

    def test_start_span_with_attributes(self) -> None:
        tm = TracingManager(enabled=True)
        span = tm.start_span("test-span", attributes={"key": "value"})
        assert span is not None
        # Clean up
        if hasattr(tm, "end_span"):
            tm.end_span(span)
        elif hasattr(span, "end"):
            span.end()

    def test_end_span_restores_context(self) -> None:
        tm = TracingManager(enabled=True)
        span = tm.start_span("parent")
        child = tm.start_span("child")
        tm.end_span(child)
        tm.end_span(span)
        # No error = context tokens properly detached


class _BarrierProvider(MockProvider):
    """Holds every call until ``parties`` are in flight, so parallel step spans overlap."""

    def __init__(self, parties: int) -> None:
        super().__init__()
        self._parties = parties
        self._arrived = 0
        self._all_in = anyio.Event()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self._arrived += 1
        if self._arrived == self._parties:
            self._all_in.set()
        await self._all_in.wait()
        return await super().complete(messages, model, temperature, max_tokens, **kwargs)


class _HangingProvider(MockProvider):
    """Never answers, so only a timeout ends the call."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        await anyio.Event().wait()
        raise AssertionError("unreachable")


class _SpendAfterWait(MockProvider):
    """``spend`` answers once ``wait`` is in flight; ``wait`` never answers."""

    def __init__(self) -> None:
        super().__init__()
        self._waiting = anyio.Event()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if "wait" in str(messages):
            self._waiting.set()
            await anyio.Event().wait()
        await self._waiting.wait()
        return await super().complete(messages, model, temperature, max_tokens, **kwargs)


def _localize(lang: str) -> dict[str, Any]:
    return {
        "name": f"localize_{lang}",
        "config": {"provider": "mock", "model": "mock-model"},
        "steps": [{"id": "translate", "type": "llm_call", "prompt": f"to {lang}", "output": "t"}],
    }


def _in_memory_tracing() -> tuple[TracingManager, Any]:
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = TracingManager(enabled=False)
    # A local tracer provider instead of the global one with an OTLP exporter.
    tracing._enabled = True
    tracing._tracer = provider.get_tracer("test")
    return tracing, exporter


class TestParallelSubworkflows:
    """#147: parallel subworkflows that reuse a step id, under a live OTel tracer."""

    async def test_each_run_keeps_its_own_spans(self, caplog: pytest.LogCaptureFixture) -> None:
        tracing, exporter = _in_memory_tracing()
        langs = ("es", "de", "ja")
        workflow = WorkflowDefinition(
            name="localize",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(
                    id=f"localize_{lang}",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline=_localize(lang),
                )
                for lang in langs
            ],
        )
        gateway = ProviderGateway()
        gateway.register(_BarrierProvider(len(langs)), priority=0)
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=gateway,
            observer=WorkflowObserver(tracing=tracing),
            run_id="root",
        )
        with caplog.at_level(logging.ERROR, logger="opentelemetry.context"), anyio.fail_after(5):
            result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        assert "Failed to detach context" not in caplog.text
        spans = exporter.get_finished_spans()
        by_id = {s.context.span_id: s for s in spans}

        def parent_name(span: Any) -> str | None:
            up = by_id.get(span.parent.span_id) if span.parent else None
            return up.name if up else None

        def run_of(span: Any) -> Any:
            return span.attributes.get(SpanAttr.WORKFLOW_RUN_ID)

        roots = [s for s in spans if s.name == SpanName.WORKFLOW.format(workflow_name="localize")]
        assert len(roots) == 1
        inference = SpanName.GEN_AI_INFERENCE.format(operation_name="chat", model="mock-model")
        for lang in langs:
            run_id = f"root.localize_{lang}"
            child = SpanName.WORKFLOW.format(workflow_name=f"localize_{lang}")
            step = SpanName.STEP.format(step_id="translate")
            mine = [s for s in spans if run_of(s) == run_id]
            assert sorted(s.name for s in mine) == sorted([child, step, inference])
            parents = {s.name: parent_name(s) for s in mine}
            assert parents[child] == SpanName.STEP.format(step_id=f"localize_{lang}")
            assert parents[step] == child
            assert parents[inference] == step

    async def test_sequential_child_hands_the_run_back(self) -> None:
        tracing, exporter = _in_memory_tracing()
        workflow = WorkflowDefinition(
            name="parent",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(
                    id="child", type=StepType.SUBWORKFLOW, workflow_inline=_localize("es")
                ),
                StepDefinition(
                    id="after",
                    type=StepType.LLM_CALL,
                    prompt="done",
                    output="after",
                    depends_on=["child"],
                ),
            ],
        )
        gateway = ProviderGateway()
        gateway.register(MockProvider(), priority=0)
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=gateway,
            observer=WorkflowObserver(tracing=tracing),
            run_id="root",
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        spans = {s.name: s for s in exporter.get_finished_spans()}
        # The child's end hook used to drop the parent's workflow span and run id.
        root = spans[SpanName.WORKFLOW.format(workflow_name="parent")]
        after = spans[SpanName.STEP.format(step_id="after")]
        assert after.attributes[SpanAttr.WORKFLOW_RUN_ID] == "root"
        assert after.parent is not None and after.parent.span_id == root.context.span_id

    async def test_timed_out_child_hands_the_run_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracing, exporter = _in_memory_tracing()
        workflow = WorkflowDefinition(
            name="parent",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(
                    id="child",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline=_localize("es"),
                    timeout=0.05,
                    retry=RetryConfig(max_retries=0),
                ),
            ],
        )
        gateway = ProviderGateway()
        gateway.register(_HangingProvider(), priority=0)
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=gateway,
            observer=WorkflowObserver(tracing=tracing),
            run_id="root",
        )
        with caplog.at_level(logging.ERROR, logger="opentelemetry.context"), anyio.fail_after(5):
            result = await engine.run()

        assert result.step_results["child"].status == StepStatus.TIMEOUT
        assert "Failed to detach context" not in caplog.text
        spans = {s.name: s for s in exporter.get_finished_spans()}
        # The cancelled child never ran its end hook, so the parent's step span was lost.
        step = spans[SpanName.STEP.format(step_id="child")]
        assert step.attributes[SpanAttr.WORKFLOW_RUN_ID] == "root"
        child = spans[SpanName.WORKFLOW.format(workflow_name="localize_es")]
        assert child.attributes[SpanAttr.WORKFLOW_STATUS] == "cancelled"
        translate = spans[SpanName.STEP.format(step_id="translate")]
        assert translate.attributes[SpanAttr.STEP_STATUS] == "cancelled"
        assert child.parent is not None and child.parent.span_id == step.context.span_id


class TestStoppedSteps:
    """#169: steps stopped from outside still end their span."""

    async def test_sibling_failure_ends_the_cancelled_step_span(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracing, exporter = _in_memory_tracing()
        # MockProvider charges 0.001 per call, so the first answer breaches the budget.
        workflow = WorkflowDefinition(
            name="layer",
            config=WorkflowConfig(provider="mock", model="mock-model", budget_usd=0.0005),
            steps=[
                StepDefinition(id="spend", type=StepType.LLM_CALL, prompt="spend", output="s"),
                StepDefinition(id="wait", type=StepType.LLM_CALL, prompt="wait", output="w"),
            ],
        )
        gateway = ProviderGateway()
        gateway.register(_SpendAfterWait(), priority=0)
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=gateway,
            observer=WorkflowObserver(tracing=tracing),
            run_id="root",
        )
        with caplog.at_level(logging.ERROR, logger="opentelemetry.context"), anyio.fail_after(5):
            result = await engine.run()

        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        assert "Failed to detach context" not in caplog.text
        spans = {s.name: s for s in exporter.get_finished_spans()}
        spend = spans[SpanName.STEP.format(step_id="spend")]
        assert spend.attributes[SpanAttr.STEP_STATUS] == "budget_exceeded"
        # It answered and was charged before the breach: the span reports what it spent.
        assert spend.attributes[SpanAttr.STEP_COST_USD] == pytest.approx(0.001)
        assert spend.attributes[SpanAttr.GEN_AI_USAGE_INPUT_TOKENS] == 10
        wait = spans[SpanName.STEP.format(step_id="wait")]
        assert wait.attributes[SpanAttr.STEP_STATUS] == "cancelled"

    async def test_an_error_escaping_a_step_ends_its_span_as_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracing, exporter = _in_memory_tracing()
        workflow = WorkflowDefinition(
            name="broken",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[StepDefinition(id="ask", type=StepType.LLM_CALL, prompt="hi", output="a")],
        )
        gateway = ProviderGateway()
        gateway.register(MockProvider(), priority=0)
        engine = WorkflowEngine(
            workflow=workflow, provider_gateway=gateway, observer=WorkflowObserver(tracing=tracing)
        )

        async def broken(*_: Any) -> None:
            raise RuntimeError("engine bug")

        monkeypatch.setattr(engine, "_run_step", broken)
        assert (await engine.run()).status == WorkflowStatus.FAILED
        step = {s.name: s for s in exporter.get_finished_spans()}[
            SpanName.STEP.format(step_id="ask")
        ]
        assert step.attributes[SpanAttr.STEP_STATUS] == "failed"
        assert step.attributes[SpanAttr.STEP_ERROR] == "engine bug"
