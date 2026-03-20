"""Workflow execution engine — the core of AgentLoom."""

from __future__ import annotations

import logging
import time
from typing import Any

import anyio

from agentloom.core.models import StepType, WorkflowDefinition
from agentloom.core.parser import WorkflowParser
from agentloom.core.results import (
    StepResult,
    StepStatus,
    WorkflowResult,
    WorkflowStatus,
)
from agentloom.core.state import StateManager
from agentloom.exceptions import (
    BudgetExceededError,
    WorkflowError,
    WorkflowTimeoutError,
)
from agentloom.steps.base import BaseStep, StepContext
from agentloom.steps.registry import StepRegistry, create_default_registry

logger = logging.getLogger("agentloom.engine")


class WorkflowEngine:
    """Executes a workflow by traversing its DAG and running steps.

    Uses layer-based parallel execution: steps in the same DAG layer
    (no dependencies between them) execute concurrently via anyio task groups.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        state_manager: StateManager | None = None,
        provider_gateway: Any | None = None,
        tool_registry: Any | None = None,
        step_registry: StepRegistry | None = None,
        observer: Any | None = None,
    ) -> None:
        self.workflow = workflow
        self.state = state_manager or StateManager(initial_state=dict(workflow.state))
        self.provider_gateway = provider_gateway
        self.tool_registry = tool_registry
        self.step_registry = step_registry or create_default_registry()
        self.observer = observer
        self._budget_spent: float = 0.0

    async def run(self) -> WorkflowResult:
        """Execute the workflow end-to-end.

        Returns:
            WorkflowResult with all step results and final state.
        """
        start = time.monotonic()
        workflow_name = self.workflow.name

        logger.info("Starting workflow: %s", workflow_name)

        if self.observer:
            self.observer.on_workflow_start(workflow_name)

        # Build and validate DAG
        dag = WorkflowParser.build_dag(self.workflow)
        layers = dag.execution_layers()

        logger.info(
            "Workflow '%s' has %d steps in %d layers",
            workflow_name,
            len(self.workflow.steps),
            len(layers),
        )

        try:
            # Track which steps are skipped (not activated by router)
            skipped_steps: set[str] = set()
            activated_targets: set[str] | None = None

            for layer_idx, layer in enumerate(layers):
                logger.debug("Executing layer %d: %s", layer_idx, layer)

                # Filter layer: skip steps not activated by a router
                active_steps = []
                for step_id in layer:
                    if step_id in skipped_steps:
                        await self.state.set_step_result(
                            step_id,
                            StepResult(step_id=step_id, status=StepStatus.SKIPPED),
                        )
                        continue
                    # If a router has been evaluated, only activate its target
                    if activated_targets is not None:
                        step_def = self.workflow.get_step(step_id)
                        if step_def and step_def.depends_on:
                            # Check if any dependency was a router
                            has_router_dep = False
                            for dep_id in step_def.depends_on:
                                dep_step = self.workflow.get_step(dep_id)
                                if dep_step and dep_step.type == StepType.ROUTER:
                                    has_router_dep = True
                                    break
                            if has_router_dep and step_id not in activated_targets:
                                skipped_steps.add(step_id)
                                await self.state.set_step_result(
                                    step_id,
                                    StepResult(step_id=step_id, status=StepStatus.SKIPPED),
                                )
                                continue
                    active_steps.append(step_id)

                if not active_steps:
                    continue

                # Execute steps in this layer concurrently
                max_concurrent = self.workflow.config.max_concurrent_steps
                limiter = anyio.CapacityLimiter(max_concurrent)

                async with anyio.create_task_group() as tg:
                    for step_id in active_steps:
                        tg.start_soon(self._execute_step_with_limit, step_id, limiter)

                # After layer execution, check for router results
                activated_targets_for_next = set()
                for step_id in active_steps:
                    step_def = self.workflow.get_step(step_id)
                    if step_def and step_def.type == StepType.ROUTER:
                        step_res = await self.state.get_step_result(step_id)
                        if step_res and step_res.status == StepStatus.SUCCESS and step_res.output:
                            activated_targets_for_next.add(step_res.output)

                if activated_targets_for_next:
                    activated_targets = activated_targets_for_next
                    # Mark all non-activated downstream steps as skipped
                    for step in self.workflow.steps:
                        for dep in step.depends_on:
                            dep_step = self.workflow.get_step(dep)
                            if (
                                dep_step
                                and dep_step.type == StepType.ROUTER
                                and step.id not in activated_targets
                            ):
                                skipped_steps.add(step.id)
                else:
                    # Reset activation for non-router layers
                    if activated_targets is not None:
                        activated_targets = None

            # Gather results
            duration = (time.monotonic() - start) * 1000
            step_results = await self.state.all_step_results()
            final_state = await self.state.get_state_snapshot()

            total_tokens = sum(r.token_usage.total_tokens for r in step_results.values())
            total_cost = sum(r.cost_usd for r in step_results.values())

            # Determine workflow status
            failed_steps = [r for r in step_results.values() if r.status == StepStatus.FAILED]
            status = WorkflowStatus.FAILED if failed_steps else WorkflowStatus.SUCCESS

            result = WorkflowResult(
                workflow_name=workflow_name,
                status=status,
                step_results=step_results,
                final_state=final_state,
                total_duration_ms=duration,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                error=failed_steps[0].error if failed_steps else None,
            )

            if self.observer:
                self.observer.on_workflow_end(
                    workflow_name, status.value, duration, total_tokens, total_cost
                )

            logger.info(
                "Workflow '%s' completed: status=%s, duration=%.1fms, cost=$%.4f",
                workflow_name,
                status.value,
                duration,
                total_cost,
            )

            return result

        except BudgetExceededError as e:
            duration = (time.monotonic() - start) * 1000
            if self.observer:
                self.observer.on_workflow_end(
                    workflow_name, "budget_exceeded", duration, 0, self._budget_spent
                )
            return WorkflowResult(
                workflow_name=workflow_name,
                status=WorkflowStatus.BUDGET_EXCEEDED,
                step_results=await self.state.all_step_results(),
                final_state=await self.state.get_state_snapshot(),
                total_duration_ms=duration,
                total_cost_usd=self._budget_spent,
                error=str(e),
            )

        except WorkflowTimeoutError as e:
            duration = (time.monotonic() - start) * 1000
            if self.observer:
                self.observer.on_workflow_end(workflow_name, "timeout", duration, 0, 0.0)
            return WorkflowResult(
                workflow_name=workflow_name,
                status=WorkflowStatus.TIMEOUT,
                step_results=await self.state.all_step_results(),
                final_state=await self.state.get_state_snapshot(),
                total_duration_ms=duration,
                error=str(e),
            )

        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            if self.observer:
                self.observer.on_workflow_end(
                    workflow_name, "failed", duration, 0, self._budget_spent
                )
            logger.error("Workflow '%s' failed: %s", workflow_name, e)
            return WorkflowResult(
                workflow_name=workflow_name,
                status=WorkflowStatus.FAILED,
                step_results=await self.state.all_step_results(),
                final_state=await self.state.get_state_snapshot(),
                total_duration_ms=duration,
                error=str(e),
            )

    async def _execute_step_with_limit(self, step_id: str, limiter: anyio.CapacityLimiter) -> None:
        """Execute a step with concurrency limiting."""
        async with limiter:
            await self._execute_step(step_id)

    async def _execute_step(self, step_id: str) -> None:
        """Execute a single step with retry and timeout support."""
        step_def = self.workflow.get_step(step_id)
        if step_def is None:
            raise WorkflowError(f"Step '{step_id}' not found in workflow")

        logger.debug("Executing step: %s (type=%s)", step_id, step_def.type.value)

        if self.observer:
            self.observer.on_step_start(step_id, step_def.type.value)

        # Get the executor
        executor_cls = self.step_registry.get(step_def.type)
        executor: BaseStep = executor_cls()

        # Build context
        context = StepContext(
            step_definition=step_def,
            state_manager=self.state,
            provider_gateway=self.provider_gateway,
            tool_registry=self.tool_registry,
            workflow_model=self.workflow.config.model,
            workflow_provider=self.workflow.config.provider,
        )

        # Execute with retry
        max_retries = step_def.retry.max_retries
        last_result: StepResult | None = None

        for attempt in range(max_retries + 1):
            try:
                if step_def.timeout:
                    with anyio.fail_after(step_def.timeout):
                        result = await executor.execute(context)
                else:
                    result = await executor.execute(context)

                last_result = result

                if result.status == StepStatus.SUCCESS:
                    await self.state.set_step_result(step_id, result)

                    # Budget tracking
                    self._budget_spent += result.cost_usd
                    if (
                        self.workflow.config.budget_usd is not None
                        and self._budget_spent > self.workflow.config.budget_usd
                    ):
                        raise BudgetExceededError(
                            self.workflow.config.budget_usd, self._budget_spent
                        )

                    if self.observer:
                        self.observer.on_step_end(
                            step_id,
                            step_def.type.value,
                            "success",
                            result.duration_ms,
                            result.cost_usd,
                            result.token_usage.total_tokens,
                        )
                        if result.provider and result.model:
                            self.observer.on_provider_call(
                                result.provider,
                                result.model,
                                result.duration_ms / 1000.0,
                            )
                            if result.token_usage.total_tokens > 0:
                                self.observer.on_tokens(
                                    result.provider,
                                    result.model,
                                    result.token_usage.prompt_tokens,
                                    result.token_usage.completion_tokens,
                                )

                    logger.debug(
                        "Step '%s' succeeded (attempt %d): %.1fms",
                        step_id,
                        attempt + 1,
                        result.duration_ms,
                    )
                    return

                # Failed but might retry
                if attempt < max_retries:
                    # FIXME: jitter not applied here, only in resilience/retry.py
                    backoff = min(
                        step_def.retry.backoff_base**attempt,
                        step_def.retry.backoff_max,
                    )
                    logger.warning(
                        "Step '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                        step_id,
                        attempt + 1,
                        max_retries + 1,
                        backoff,
                        result.error,
                    )
                    await anyio.sleep(backoff)

            except TimeoutError:
                last_result = StepResult(
                    step_id=step_id,
                    status=StepStatus.TIMEOUT,
                    error=f"Step timed out after {step_def.timeout}s",
                )
                if attempt < max_retries:
                    backoff = min(
                        step_def.retry.backoff_base**attempt,
                        step_def.retry.backoff_max,
                    )
                    await anyio.sleep(backoff)
                    continue
                break

            except BudgetExceededError:
                raise

            except Exception as e:
                last_result = StepResult(
                    step_id=step_id,
                    status=StepStatus.FAILED,
                    error=str(e),
                )
                if attempt < max_retries:
                    backoff = min(
                        step_def.retry.backoff_base**attempt,
                        step_def.retry.backoff_max,
                    )
                    await anyio.sleep(backoff)
                    continue
                break

        # All retries exhausted
        if last_result:
            await self.state.set_step_result(step_id, last_result)

            if self.observer:
                self.observer.on_step_end(
                    step_id,
                    step_def.type.value,
                    last_result.status.value,
                    last_result.duration_ms,
                    last_result.cost_usd,
                    last_result.token_usage.total_tokens,
                    error=last_result.error,
                )

            logger.error(
                "Step '%s' failed after %d attempts: %s",
                step_id,
                max_retries + 1,
                last_result.error,
            )
