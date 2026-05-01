"""Tests for the WorkflowEngine module."""

from __future__ import annotations

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import WorkflowDefinition
from agentloom.core.results import StepResult, StepStatus, WorkflowStatus
from agentloom.providers.gateway import ProviderGateway
from tests.conftest import MockProvider


class TestSimpleWorkflow:
    """Test WorkflowEngine with a simple single-step workflow."""

    async def test_simple_workflow_succeeds(
        self,
        simple_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=simple_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS

    async def test_simple_workflow_has_step_result(
        self,
        simple_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=simple_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert "answer" in result.step_results
        answer_result = result.step_results["answer"]
        assert answer_result.status == StepStatus.SUCCESS
        assert answer_result.output is not None

    async def test_simple_workflow_tracks_cost(
        self,
        simple_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=simple_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert result.total_cost_usd > 0

    async def test_simple_workflow_tracks_tokens(
        self,
        simple_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=simple_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert result.total_tokens > 0

    async def test_simple_workflow_has_duration(
        self,
        simple_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=simple_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert result.total_duration_ms > 0

    async def test_simple_workflow_final_state(
        self,
        simple_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=simple_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert "question" in result.final_state


class TestParallelWorkflow:
    """Test WorkflowEngine with parallel steps."""

    async def test_parallel_workflow_succeeds(
        self,
        parallel_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=parallel_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS

    async def test_parallel_workflow_all_steps_complete(
        self,
        parallel_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=parallel_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert "step_a" in result.step_results
        assert "step_b" in result.step_results
        assert "merge" in result.step_results
        assert result.step_results["step_a"].status == StepStatus.SUCCESS
        assert result.step_results["step_b"].status == StepStatus.SUCCESS
        assert result.step_results["merge"].status == StepStatus.SUCCESS

    async def test_parallel_workflow_merge_depends_on_both(
        self,
        parallel_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
        mock_provider: MockProvider,
    ) -> None:
        engine = WorkflowEngine(
            workflow=parallel_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        # The merge step should have executed after step_a and step_b
        assert result.step_results["merge"].status == StepStatus.SUCCESS
        # Provider should have been called 3 times (step_a, step_b, merge)
        assert len(mock_provider.calls) == 3


class TestRouterWorkflow:
    """Test WorkflowEngine with conditional routing."""

    async def test_router_workflow_succeeds(
        self,
        router_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=router_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS

    async def test_router_workflow_has_classify_step(
        self,
        router_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=router_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert "classify" in result.step_results
        assert result.step_results["classify"].status == StepStatus.SUCCESS

    async def test_router_workflow_has_route_step(
        self,
        router_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=router_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        assert "route" in result.step_results
        assert result.step_results["route"].status == StepStatus.SUCCESS

    async def test_router_workflow_skips_unmatched_branches(
        self,
        router_workflow: WorkflowDefinition,
        mock_gateway: ProviderGateway,
    ) -> None:
        engine = WorkflowEngine(
            workflow=router_workflow,
            provider_gateway=mock_gateway,
        )
        result = await engine.run()
        # The router should route to the default ("general") since
        # MockProvider returns "Mock response" which doesn't match billing or technical.
        # At least one of the three branches should be skipped.
        statuses = [
            result.step_results.get("billing", None),
            result.step_results.get("technical", None),
            result.step_results.get("general", None),
        ]
        # Verify that some branches were skipped
        skipped = [s for s in statuses if s is not None and s.status == StepStatus.SKIPPED]
        executed = [s for s in statuses if s is not None and s.status == StepStatus.SUCCESS]
        assert len(skipped) >= 1, "At least one branch should be skipped"
        assert len(executed) >= 1, "At least one branch should execute"


class TestJitteredBackoff:
    """Verify the retry backoff helper applies jitter.

    Lives now in ``resilience.retry.compute_backoff`` and is consumed by
    both ``WorkflowEngine._execute_step`` and ``retry_with_policy``.
    """

    def test_jitter_varies_across_calls(self) -> None:
        from agentloom.resilience.retry import compute_backoff

        values = {compute_backoff(2.0, 3, 60.0, jitter=True) for _ in range(50)}
        # 50 samples over a +/-25% window must yield more than one unique value.
        assert len(values) > 1

    def test_jitter_bounded_within_range(self) -> None:
        from agentloom.resilience.retry import compute_backoff

        # base=2.0, attempt=3 => raw=8.0, jitter window = [6.0, 10.0]
        for _ in range(100):
            d = compute_backoff(2.0, 3, 60.0, jitter=True)
            assert 6.0 <= d <= 10.0

    def test_no_jitter_is_deterministic(self) -> None:
        from agentloom.resilience.retry import compute_backoff

        assert compute_backoff(2.0, 3, 60.0, jitter=False) == 8.0

    def test_backoff_capped_at_maximum(self) -> None:
        from agentloom.resilience.retry import compute_backoff

        # raw = 2**10 = 1024, capped at 10.
        d = compute_backoff(2.0, 10, 10.0, jitter=False)
        assert d == 10.0


class TestRetryableStatusCodes:
    """Verify ``RetryConfig.retryable_status_codes`` is consulted on the
    engine retry path. Status-less exceptions retry by default (transient
    network errors); status-coded exceptions retry only when the code is
    in the list."""

    def test_is_retryable_with_status_in_list(self) -> None:
        from agentloom.exceptions import RateLimitError
        from agentloom.resilience.retry import is_retryable_exception

        exc = RateLimitError("openai", retry_after_s=1.0)
        assert exc.status_code == 429
        assert is_retryable_exception(exc, [429, 500, 502, 503, 504]) is True

    def test_is_retryable_with_status_not_in_list(self) -> None:
        from agentloom.exceptions import ProviderError
        from agentloom.resilience.retry import is_retryable_exception

        exc = ProviderError("openai", "bad request", status_code=400)
        assert is_retryable_exception(exc, [429, 500, 502, 503, 504]) is False

    def test_is_retryable_without_status_defaults_true(self) -> None:
        from agentloom.resilience.retry import is_retryable_exception

        # Generic exception with no status_code attribute — treated as
        # transient (network error, parser hiccup) and retried.
        assert is_retryable_exception(RuntimeError("boom"), [429, 500]) is True

    async def test_engine_does_not_retry_non_retryable_status(
        self, mock_gateway: ProviderGateway
    ) -> None:
        """When a step raises a ProviderError with a status not in
        retryable_status_codes, the engine must surface FAILED on the
        first attempt — no retries consumed."""
        from agentloom.core.models import (
            RetryConfig,
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )
        from agentloom.exceptions import ProviderError
        from agentloom.steps.base import BaseStep

        attempts: list[int] = []

        class FailingStep(BaseStep):
            async def execute(self, ctx) -> StepResult:  # type: ignore[no-untyped-def]
                attempts.append(1)
                raise ProviderError("mock", "permanent failure", status_code=400)

        engine = WorkflowEngine(
            workflow=WorkflowDefinition(
                name="non-retryable",
                config=WorkflowConfig(provider="mock", model="mock-model"),
                state={},
                steps=[
                    StepDefinition(
                        id="s",
                        type=StepType.LLM_CALL,
                        prompt="hi",
                        retry=RetryConfig(
                            max_retries=3,
                            backoff_base=1.0,
                            backoff_max=0.0,
                            jitter=False,
                            retryable_status_codes=[429, 500, 502, 503, 504],
                        ),
                    ),
                ],
            ),
            provider_gateway=mock_gateway,
        )
        engine.step_registry.register(StepType.LLM_CALL, FailingStep)
        result = await engine.run()
        assert result.status == WorkflowStatus.FAILED
        assert len(attempts) == 1, (
            f"non-retryable status must not be retried, got {len(attempts)} attempts"
        )

    async def test_engine_retries_retryable_status(self, mock_gateway: ProviderGateway) -> None:
        """When a step raises a 429, the engine retries up to max_retries."""
        from agentloom.core.models import (
            RetryConfig,
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )
        from agentloom.exceptions import RateLimitError
        from agentloom.steps.base import BaseStep

        attempts: list[int] = []

        class FlakyStep(BaseStep):
            async def execute(self, ctx) -> StepResult:  # type: ignore[no-untyped-def]
                attempts.append(1)
                raise RateLimitError("mock", retry_after_s=0.0)

        engine = WorkflowEngine(
            workflow=WorkflowDefinition(
                name="retryable",
                config=WorkflowConfig(provider="mock", model="mock-model"),
                state={},
                steps=[
                    StepDefinition(
                        id="s",
                        type=StepType.LLM_CALL,
                        prompt="hi",
                        retry=RetryConfig(
                            max_retries=2,
                            backoff_base=1.0,
                            backoff_max=0.0,
                            jitter=False,
                            retryable_status_codes=[429, 500],
                        ),
                    ),
                ],
            ),
            provider_gateway=mock_gateway,
        )
        engine.step_registry.register(StepType.LLM_CALL, FlakyStep)
        result = await engine.run()
        assert result.status == WorkflowStatus.FAILED
        # max_retries=2 means up to 3 attempts total (initial + 2 retries).
        assert len(attempts) == 3, f"429 should be retried, got {len(attempts)} attempts"


class TestRouterSkipCascade:
    """Regression: router skip must propagate through the full DAG closure."""

    async def test_router_skip_cascades_to_grandchildren(
        self, mock_gateway: ProviderGateway
    ) -> None:
        """Router -> {path_a, path_b}, path_a -> review_a, path_b -> review_b.
        Activating only path_a must leave review_b SKIPPED, not SUCCESS."""
        from agentloom.core.models import (
            Condition,
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )

        workflow = WorkflowDefinition(
            name="cascade",
            config=WorkflowConfig(provider="mock", model="mock-model"),
            state={"choice": "a"},
            steps=[
                StepDefinition(
                    id="route",
                    type=StepType.ROUTER,
                    conditions=[
                        Condition(expression="state.choice == 'a'", target="path_a"),
                        Condition(expression="state.choice == 'b'", target="path_b"),
                    ],
                ),
                StepDefinition(
                    id="path_a",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="A",
                ),
                StepDefinition(
                    id="path_b",
                    type=StepType.LLM_CALL,
                    depends_on=["route"],
                    prompt="B",
                ),
                StepDefinition(
                    id="review_a",
                    type=StepType.LLM_CALL,
                    depends_on=["path_a"],
                    prompt="review A",
                ),
                StepDefinition(
                    id="review_b",
                    type=StepType.LLM_CALL,
                    depends_on=["path_b"],
                    prompt="review B",
                ),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=mock_gateway)
        result = await engine.run()

        assert result.step_results["path_a"].status == StepStatus.SUCCESS
        assert result.step_results["path_b"].status == StepStatus.SKIPPED
        assert result.step_results["review_a"].status == StepStatus.SUCCESS
        # The critical regression: review_b used to execute with empty state.
        assert result.step_results["review_b"].status == StepStatus.SKIPPED


class TestBudgetPreDispatchGate:
    """Prior-exhausted budget must stop further step dispatch before the call."""

    async def test_budget_pre_dispatch_gate_blocks_after_exhaustion(
        self, mock_gateway: ProviderGateway
    ) -> None:
        from agentloom.core.models import (
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )

        workflow = WorkflowDefinition(
            name="budget",
            config=WorkflowConfig(provider="mock", model="mock-model", budget_usd=0.0005),
            state={},
            steps=[
                StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi"),
                StepDefinition(id="s2", type=StepType.LLM_CALL, depends_on=["s1"], prompt="hi"),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=mock_gateway)
        # MockProvider returns cost_usd=0.001 per call — the first step alone
        # overshoots the 0.0005 budget, so s2 must never dispatch.
        result = await engine.run()
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        # s2 never reached the provider (no SUCCESS).
        s2 = result.step_results.get("s2")
        assert s2 is None or s2.status != StepStatus.SUCCESS

    async def test_budget_enforcement_accounts_for_reasoning_tokens(self) -> None:
        """Reasoning tokens billed at output rate must count toward the
        workflow budget. A model returning many thinking tokens should hit
        BUDGET_EXCEEDED on the first step even when prompt+completion
        tokens alone fit comfortably."""
        from typing import Any

        from agentloom.core.models import (
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )
        from agentloom.core.results import TokenUsage
        from agentloom.providers.base import (
            BaseProvider,
            ProviderResponse,
            StreamResponse,
        )
        from agentloom.providers.gateway import ProviderGateway

        class ThinkingProvider(BaseProvider):
            name = "thinking-mock"

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
                # Cost dominated by reasoning tokens; the prompt+completion
                # portion is intentionally tiny so the test fails on the
                # reasoning component if accounting drops it.
                return ProviderResponse(
                    content="42",
                    model=model,
                    provider="thinking-mock",
                    usage=TokenUsage(
                        prompt_tokens=5,
                        completion_tokens=2,
                        total_tokens=507,
                        reasoning_tokens=500,
                    ),
                    cost_usd=0.0008,  # incl. reasoning component
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

        gateway = ProviderGateway()
        gateway.register(ThinkingProvider())

        workflow = WorkflowDefinition(
            name="budget-with-thinking",
            config=WorkflowConfig(
                provider="thinking-mock",
                model="claude-opus-4",
                budget_usd=0.0005,
            ),
            state={},
            steps=[
                StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi"),
                StepDefinition(id="s2", type=StepType.LLM_CALL, depends_on=["s1"], prompt="hi"),
            ],
        )
        engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway)
        result = await engine.run()

        # First step alone (0.0008) overshoots 0.0005 because the cost
        # includes the reasoning portion. If reasoning were dropped from
        # accounting the cost would be much smaller and s2 would dispatch.
        assert result.status == WorkflowStatus.BUDGET_EXCEEDED
        s1 = result.step_results.get("s1")
        assert s1 is not None and s1.status == StepStatus.SUCCESS
        assert s1.token_usage.reasoning_tokens == 500
        s2 = result.step_results.get("s2")
        assert s2 is None or s2.status != StepStatus.SUCCESS
