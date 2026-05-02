"""Integration tests: engine → gateway → provider end-to-end."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    Condition,
    RetryConfig,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepStatus, WorkflowStatus
from agentloom.core.state import StateManager
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.tools.registry import ToolRegistry
from tests.conftest import MockProvider, MockTool


# Helpers
class FailingProvider(BaseProvider):
    """Provider that always fails."""

    name = "failing"

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        raise RuntimeError("Primary down")

    def supports_model(self, model: str) -> bool:
        return True


class CountingProvider(MockProvider):
    """MockProvider that counts calls."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        self.call_count += 1
        return await super().complete(**kwargs)


# Basic flows
class TestSingleStep:
    async def test_completes_with_output(self) -> None:
        provider = MockProvider(responses={"Answer: test": "42"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="e2e-simple",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"question": "test"},
            steps=[
                StepDefinition(
                    id="answer",
                    type=StepType.LLM_CALL,
                    prompt="Answer: {state.question}",
                    output="answer",
                )
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert result.total_tokens > 0
        assert result.total_cost_usd > 0
        assert "answer" in result.final_state

    async def test_state_override(self) -> None:
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="state-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"name": "default"},
            steps=[
                StepDefinition(
                    id="greet",
                    type=StepType.LLM_CALL,
                    prompt="Hello {state.name}",
                    output="greeting",
                )
            ],
        )
        state = StateManager(initial_state={"name": "Alice"})
        engine = WorkflowEngine(workflow=wf, state_manager=state, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        # MockProvider receives the rendered prompt with "Alice" (no system prompt → index 0)
        assert provider.calls[0]["messages"][0]["content"] == "Hello Alice"


# Parallel execution
class TestParallelExecution:
    async def test_parallel_steps_all_complete(self) -> None:
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="e2e-parallel",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(id="a", type=StepType.LLM_CALL, prompt="a", output="out_a"),
                StepDefinition(id="b", type=StepType.LLM_CALL, prompt="b", output="out_b"),
                StepDefinition(
                    id="merge",
                    type=StepType.LLM_CALL,
                    prompt="merge",
                    output="merged",
                    depends_on=["a", "b"],
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert len(result.step_results) == 3
        assert result.total_tokens == 90  # 3 × 30 tokens

    async def test_parallel_steps_are_concurrent(self) -> None:
        """Verify both parallel steps actually execute (not just one)."""
        counter = CountingProvider()
        gw = ProviderGateway()
        gw.register(counter, models=["mock-model"])

        wf = WorkflowDefinition(
            name="concurrency-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(id="a", type=StepType.LLM_CALL, prompt="a", output="oa"),
                StepDefinition(id="b", type=StepType.LLM_CALL, prompt="b", output="ob"),
                StepDefinition(id="c", type=StepType.LLM_CALL, prompt="c", output="oc"),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert counter.call_count == 3


# Router / conditional branching
class TestRouterIntegration:
    async def test_router_activates_correct_branch(self) -> None:
        provider = MockProvider(responses={"Classify: billing issue": "billing"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="router-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"input": "billing issue"},
            steps=[
                StepDefinition(
                    id="classify",
                    type=StepType.LLM_CALL,
                    prompt="Classify: {state.input}",
                    output="classification",
                ),
                StepDefinition(
                    id="route",
                    type=StepType.ROUTER,
                    depends_on=["classify"],
                    conditions=[
                        Condition(expression="state.classification == 'billing'", target="billing"),
                    ],
                    default="general",
                ),
                StepDefinition(
                    id="billing",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="Handle billing",
                    output="response",
                ),
                StepDefinition(
                    id="general",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="Handle general",
                    output="response",
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["billing"].status == StepStatus.SUCCESS
        assert result.step_results["general"].status == StepStatus.SKIPPED

    async def test_router_default_branch(self) -> None:
        provider = MockProvider(responses={"Classify: weird": "unknown"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="default-route",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"input": "weird"},
            steps=[
                StepDefinition(
                    id="classify",
                    type=StepType.LLM_CALL,
                    prompt="Classify: {state.input}",
                    output="classification",
                ),
                StepDefinition(
                    id="route",
                    type=StepType.ROUTER,
                    depends_on=["classify"],
                    conditions=[
                        Condition(expression="state.classification == 'billing'", target="billing"),
                    ],
                    default="fallback",
                ),
                StepDefinition(
                    id="billing",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="billing",
                    output="r",
                ),
                StepDefinition(
                    id="fallback",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="fallback",
                    output="r",
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["fallback"].status == StepStatus.SUCCESS
        assert result.step_results["billing"].status == StepStatus.SKIPPED


# Tool integration
class TestToolIntegration:
    async def test_tool_step_executes(self) -> None:
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        mock_tool = MockTool(result={"data": "from_tool"})
        registry = ToolRegistry()
        registry.register(mock_tool)

        wf = WorkflowDefinition(
            name="tool-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(
                    id="fetch",
                    type=StepType.TOOL,
                    tool_name="mock_tool",
                    tool_args={"input": "test"},
                    output="fetched",
                ),
                StepDefinition(
                    id="analyze",
                    type=StepType.LLM_CALL,
                    depends_on=["fetch"],
                    prompt="Analyze: {state.fetched}",
                    output="analysis",
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, tool_registry=registry)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["fetch"].status == StepStatus.SUCCESS
        assert result.step_results["analyze"].status == StepStatus.SUCCESS
        assert mock_tool.calls == [{"input": "test"}]


# Resilience
class TestProviderFallback:
    async def test_fallback_on_primary_failure(self) -> None:
        backup = MockProvider()
        gw = ProviderGateway()
        gw.register(FailingProvider(), priority=0, models=["mock-model"])
        gw.register(backup, priority=10, models=["mock-model"])

        wf = WorkflowDefinition(
            name="e2e-fallback",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[StepDefinition(id="s", type=StepType.LLM_CALL, prompt="test", output="out")],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert backup.calls


class TestBudgetEnforcement:
    async def test_budget_exceeded_stops_workflow(self) -> None:
        provider = MockProvider()  # costs 0.001 per call
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="e2e-budget",
            config=WorkflowConfig(provider="mock", model="mock-model", budget_usd=0.0001),
            steps=[
                StepDefinition(id="a", type=StepType.LLM_CALL, prompt="a", output="oa"),
                StepDefinition(
                    id="b",
                    type=StepType.LLM_CALL,
                    prompt="b",
                    output="ob",
                    depends_on=["a"],
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status in (WorkflowStatus.FAILED, WorkflowStatus.BUDGET_EXCEEDED)

    async def test_budget_remaining_emitted_to_observer(self) -> None:
        provider = MockProvider()  # costs 0.001 per call
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="budget-obs",
            config=WorkflowConfig(provider="mock", model="mock-model", budget_usd=1.0),
            steps=[
                StepDefinition(id="a", type=StepType.LLM_CALL, prompt="a", output="oa"),
            ],
        )
        observer = MagicMock()
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, observer=observer)
        await engine.run()
        observer.on_budget_remaining.assert_called_once()
        remaining = observer.on_budget_remaining.call_args[0][1]
        assert remaining >= 0.0

    async def test_within_budget_completes(self) -> None:
        provider = MockProvider()  # costs 0.001 per call
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="budget-ok",
            config=WorkflowConfig(provider="mock", model="mock-model", budget_usd=10.0),
            steps=[
                StepDefinition(id="a", type=StepType.LLM_CALL, prompt="a", output="oa"),
                StepDefinition(
                    id="b",
                    type=StepType.LLM_CALL,
                    prompt="b",
                    output="ob",
                    depends_on=["a"],
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert result.total_cost_usd <= 10.0


class TestRetryBehavior:
    async def test_step_retries_on_failure(self) -> None:
        """Step with max_retries=1 should try twice before failing."""
        gw = ProviderGateway()
        gw.register(FailingProvider(), models=["mock-model"])

        wf = WorkflowDefinition(
            name="retry-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(
                    id="fail_step",
                    type=StepType.LLM_CALL,
                    prompt="will fail",
                    retry=RetryConfig(max_retries=1, backoff_base=0.01, backoff_max=0.01),
                )
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.FAILED


# Observer integration
class TestObserverIntegration:
    async def test_observer_receives_lifecycle_events(self) -> None:
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        observer = MagicMock()
        wf = WorkflowDefinition(
            name="obs-test",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[StepDefinition(id="s", type=StepType.LLM_CALL, prompt="hi", output="out")],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, observer=observer)
        result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        observer.on_workflow_start.assert_called_once_with("obs-test", run_id="")
        observer.on_step_start.assert_called_once_with("s", "llm_call", stream=False)
        observer.on_step_end.assert_called_once()
        observer.on_workflow_end.assert_called_once()
        # Token events also fired (provider-level spans use start/end pair).
        observer.on_tokens.assert_called_once()

    async def test_observer_on_failure(self) -> None:
        gw = ProviderGateway()
        gw.register(FailingProvider(), models=["mock-model"])

        observer = MagicMock()
        wf = WorkflowDefinition(
            name="obs-fail",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            steps=[
                StepDefinition(
                    id="s",
                    type=StepType.LLM_CALL,
                    prompt="fail",
                    retry=RetryConfig(max_retries=0),
                )
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, observer=observer)
        await engine.run()

        observer.on_workflow_end.assert_called_once()
        call_args = observer.on_workflow_end.call_args
        assert call_args[0][1] == "failed"  # status arg

    async def test_streaming_wiring(self) -> None:
        """Verify stream=True propagates through engine to observer and callback."""
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        observer = MagicMock()
        chunks: list[str] = []

        def _on_chunk(step_id: str, text: str) -> None:
            chunks.append(text)

        wf = WorkflowDefinition(
            name="stream-test",
            config=WorkflowConfig(provider="mock", model="mock-model", stream=True),
            steps=[
                StepDefinition(
                    id="s",
                    type=StepType.LLM_CALL,
                    prompt="hi",
                    output="out",
                ),
            ],
        )
        engine = WorkflowEngine(
            workflow=wf,
            provider_gateway=gw,
            observer=observer,
            on_stream_chunk=_on_chunk,
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        # Observer received stream=True
        observer.on_step_start.assert_called_once_with("s", "llm_call", stream=True)
        # Callback was invoked
        assert len(chunks) > 0
        assert "".join(chunks) == "Mock response"
        # TTFT was reported
        step_end_kwargs = observer.on_step_end.call_args[1]
        assert step_end_kwargs.get("stream") is True
        assert step_end_kwargs.get("time_to_first_token_ms") is not None

    async def test_step_level_stream_override(self) -> None:
        """Per-step stream: false overrides config.stream: true."""
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        observer = MagicMock()
        wf = WorkflowDefinition(
            name="override-test",
            config=WorkflowConfig(provider="mock", model="mock-model", stream=True),
            steps=[
                StepDefinition(
                    id="s",
                    type=StepType.LLM_CALL,
                    prompt="hi",
                    output="out",
                    stream=False,
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, observer=observer)
        result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        # Step-level stream=False wins
        observer.on_step_start.assert_called_once_with("s", "llm_call", stream=False)
        # No TTFT for non-streaming
        sr = result.step_results["s"]
        assert sr.time_to_first_token_ms is None


# Subworkflow integration
class TestSubworkflowIntegration:
    async def test_inline_subworkflow(self) -> None:
        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        wf = WorkflowDefinition(
            name="parent",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"topic": "AI"},
            steps=[
                StepDefinition(
                    id="sub",
                    type=StepType.SUBWORKFLOW,
                    workflow_inline={
                        "name": "child",
                        "config": {"provider": "mock", "model": "mock-model"},
                        "steps": [
                            {
                                "id": "child_step",
                                "type": "llm_call",
                                "prompt": "Research {state.topic}",
                                "output": "research",
                            }
                        ],
                    },
                    output="sub_result",
                ),
                StepDefinition(
                    id="summarize",
                    type=StepType.LLM_CALL,
                    depends_on=["sub"],
                    prompt="Summarize: {state.sub_result}",
                    output="summary",
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["sub"].status == StepStatus.SUCCESS
        assert result.step_results["summarize"].status == StepStatus.SUCCESS
        # 1 child step + 1 parent step = 2 total LLM calls
        assert len(provider.calls) == 2


# Multi-layer complex workflow
class TestComplexWorkflow:
    async def test_five_layer_workflow(self) -> None:
        """Simulates a realistic multi-layer workflow:
        fetch(tool) → classify(llm) → route(router) → handle(llm) || skip
        """
        provider = MockProvider(responses={"Classify: data": "critical"})
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        mock_tool = MockTool(result="raw data")
        registry = ToolRegistry()
        registry.register(mock_tool)

        wf = WorkflowDefinition(
            name="complex-e2e",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"input": "data"},
            steps=[
                StepDefinition(
                    id="fetch",
                    type=StepType.TOOL,
                    tool_name="mock_tool",
                    tool_args={"input": "state.input"},
                    output="raw",
                ),
                StepDefinition(
                    id="classify",
                    type=StepType.LLM_CALL,
                    depends_on=["fetch"],
                    prompt="Classify: {state.input}",
                    output="classification",
                ),
                StepDefinition(
                    id="route",
                    type=StepType.ROUTER,
                    depends_on=["classify"],
                    conditions=[
                        Condition(
                            expression="state.classification == 'critical'",
                            target="escalate",
                        ),
                    ],
                    default="log_only",
                ),
                StepDefinition(
                    id="escalate",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="Escalate: {state.raw}",
                    output="action",
                ),
                StepDefinition(
                    id="log_only",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="Log: {state.raw}",
                    output="action",
                ),
            ],
        )
        engine = WorkflowEngine(workflow=wf, provider_gateway=gw, tool_registry=registry)
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["fetch"].status == StepStatus.SUCCESS
        assert result.step_results["classify"].status == StepStatus.SUCCESS
        assert result.step_results["route"].status == StepStatus.SUCCESS
        assert result.step_results["escalate"].status == StepStatus.SUCCESS
        assert result.step_results["log_only"].status == StepStatus.SKIPPED
