"""Tests for the WorkflowEngine module."""

from __future__ import annotations

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import WorkflowDefinition
from agentloom.core.results import StepStatus, WorkflowStatus
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
