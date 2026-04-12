"""Workflow execution engine — the core of AgentLoom."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import anyio

from agentloom.checkpointing.base import BaseCheckpointer, CheckpointData
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
    PauseRequestedError,
    WorkflowError,
    WorkflowTimeoutError,
)
from agentloom.steps.base import BaseStep, StepContext
from agentloom.steps.registry import StepRegistry, create_default_registry

logger = logging.getLogger("agentloom.engine")


def _extract_pause_error(exc: BaseException) -> PauseRequestedError | None:
    """Unwrap a ``PauseRequestedError`` from an ``ExceptionGroup``.

    anyio task groups always wrap step exceptions in an ``ExceptionGroup``,
    so we need to inspect the group to find pause requests.  Returns
    ``None`` if *exc* does not contain a ``PauseRequestedError``.
    """
    if isinstance(exc, PauseRequestedError):
        return exc
    if isinstance(exc, ExceptionGroup):
        for inner in exc.exceptions:
            found = _extract_pause_error(inner)
            if found is not None:
                return found
    return None


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
        on_stream_chunk: Callable[[str, str], None] | None = None,
        checkpointer: BaseCheckpointer | None = None,
        run_id: str | None = None,
    ) -> None:
        self.workflow = workflow
        self.state = state_manager or StateManager(initial_state=dict(workflow.state))
        self.provider_gateway = provider_gateway
        self.tool_registry = tool_registry
        self.step_registry = step_registry or create_default_registry()
        self.observer = observer
        self._stream_callback = on_stream_chunk
        self._budget_spent: float = 0.0

        # Checkpointing
        self._checkpointer = checkpointer
        self.run_id = run_id or (uuid.uuid4().hex[:12] if checkpointer else "")
        self._completed_steps: set[str] = set()
        self._checkpoint_created_at = datetime.now(UTC).isoformat()

        # Wire observer into gateway for circuit breaker events
        if observer and provider_gateway:
            set_obs = getattr(provider_gateway, "set_observer", None)
            if set_obs:
                set_obs(observer)

    async def _save_checkpoint(
        self,
        status: str,
        paused_step_id: str | None = None,
    ) -> None:
        """Persist current execution state via the configured checkpointer."""
        if self._checkpointer is None:
            return

        step_results = await self.state.all_step_results()
        state_snapshot = await self.state.get_state_snapshot()

        # Derive completed_steps from step_results so mid-layer aborts are captured
        completed_steps = sorted(
            step_id
            for step_id, result in step_results.items()
            if result.status == StepStatus.SUCCESS
        )

        data = CheckpointData(
            workflow_name=self.workflow.name,
            run_id=self.run_id,
            workflow_definition=self.workflow.model_dump(),
            state=state_snapshot,
            step_results={k: v.model_dump() for k, v in step_results.items()},
            completed_steps=completed_steps,
            status=status,
            paused_step_id=paused_step_id,
            created_at=self._checkpoint_created_at,
            updated_at=datetime.now(UTC).isoformat(),
        )
        try:
            await self._checkpointer.save(data)
        except Exception:
            logger.warning(
                "Failed to save checkpoint for run '%s' — continuing without checkpoint",
                self.run_id,
                exc_info=True,
            )
            return
        logger.debug("Checkpoint saved: run_id=%s status=%s", self.run_id, status)

    @classmethod
    async def from_checkpoint(
        cls,
        checkpoint_data: CheckpointData,
        checkpointer: BaseCheckpointer,
        provider_gateway: Any | None = None,
        tool_registry: Any | None = None,
        step_registry: StepRegistry | None = None,
        observer: Any | None = None,
        on_stream_chunk: Callable[[str, str], None] | None = None,
    ) -> WorkflowEngine:
        """Reconstruct an engine from a checkpoint, ready to resume.

        The returned engine's :meth:`run` will skip already-completed steps
        and continue execution from where it left off.
        """
        workflow = WorkflowDefinition.model_validate(checkpoint_data.workflow_definition)

        # Restore state manager with completed step results via the public API
        # so internal state, snapshots, and locking remain consistent.
        state_manager = StateManager(initial_state=checkpoint_data.state)
        for step_id, result_data in checkpoint_data.step_results.items():
            result = StepResult.model_validate(result_data)
            await state_manager.set_step_result(step_id, result)

        engine = cls(
            workflow=workflow,
            state_manager=state_manager,
            provider_gateway=provider_gateway,
            tool_registry=tool_registry,
            step_registry=step_registry,
            observer=observer,
            on_stream_chunk=on_stream_chunk,
            checkpointer=checkpointer,
            run_id=checkpoint_data.run_id,
        )
        engine._completed_steps = set(checkpoint_data.completed_steps)
        engine._checkpoint_created_at = checkpoint_data.created_at
        return engine

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
                    # Skip steps already completed in a previous run (resume)
                    if step_id in self._completed_steps:
                        logger.debug("Skipping already-completed step: %s", step_id)
                        # If this was a router, restore its activation so
                        # downstream branch filtering works correctly.
                        step_def = self.workflow.get_step(step_id)
                        if step_def and step_def.type == StepType.ROUTER:
                            step_res = await self.state.get_step_result(step_id)
                            if (
                                step_res
                                and step_res.status == StepStatus.SUCCESS
                                and step_res.output
                            ):
                                activated_targets = activated_targets or set()
                                activated_targets.add(step_res.output)
                        continue

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

                # Track completed steps for checkpoint/resume
                for step_id in active_steps:
                    step_result = await self.state.get_step_result(step_id)
                    if step_result and step_result.status == StepStatus.SUCCESS:
                        self._completed_steps.add(step_id)

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

            await self._save_checkpoint(status.value)

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

        except BudgetExceededError as e:  # pragma: no cover — anyio wraps in ExceptionGroup
            duration = (time.monotonic() - start) * 1000
            await self._save_checkpoint("budget_exceeded")
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

        except WorkflowTimeoutError as e:  # pragma: no cover — anyio wraps in ExceptionGroup
            duration = (time.monotonic() - start) * 1000
            await self._save_checkpoint("timeout")
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

            # Check for PauseRequestedError wrapped in ExceptionGroup
            # (anyio task groups always wrap step exceptions).
            pause_err = _extract_pause_error(e)
            if pause_err is not None:
                await self._save_checkpoint("paused", paused_step_id=pause_err.step_id)
                step_results = await self.state.all_step_results()
                final_state = await self.state.get_state_snapshot()
                if self.observer:
                    self.observer.on_workflow_end(
                        workflow_name,
                        "paused",
                        duration,
                        sum(r.token_usage.total_tokens for r in step_results.values()),
                        sum(r.cost_usd for r in step_results.values()),
                    )
                logger.info(
                    "Workflow '%s' paused at step '%s'",
                    workflow_name,
                    pause_err.step_id,
                )
                return WorkflowResult(
                    workflow_name=workflow_name,
                    status=WorkflowStatus.PAUSED,
                    step_results=step_results,
                    final_state=final_state,
                    total_duration_ms=duration,
                    total_tokens=sum(r.token_usage.total_tokens for r in step_results.values()),
                    total_cost_usd=sum(r.cost_usd for r in step_results.values()),
                    error=str(pause_err),
                )

            await self._save_checkpoint("failed")
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

        should_stream = (
            step_def.stream if step_def.stream is not None else self.workflow.config.stream
        )

        if self.observer:
            self.observer.on_step_start(step_id, step_def.type.value, stream=should_stream)

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
            sandbox_config=self.workflow.config.sandbox,
            stream=should_stream,
            on_stream_chunk=self._stream_callback,
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

                    # Budget tracking — post-hoc check, no async lock on _budget_spent.
                    # Cooperative concurrency makes this safe in practice (no preemption
                    # between read and write), but a proper BudgetEnforcer refactor is planned.
                    self._budget_spent += result.cost_usd
                    if self.observer and self.workflow.config.budget_usd is not None:
                        hook = getattr(self.observer, "on_budget_remaining", None)
                        if hook:
                            hook(
                                self.workflow.name,
                                max(0.0, self.workflow.config.budget_usd - self._budget_spent),
                            )
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
                            attachment_count=result.attachment_count,
                            time_to_first_token_ms=result.time_to_first_token_ms,
                            stream=should_stream,
                        )
                        if result.provider and result.model:
                            self.observer.on_provider_call(
                                result.provider,
                                result.model,
                                result.duration_ms / 1000.0,
                                stream=should_stream,
                            )
                            if result.token_usage.total_tokens > 0:
                                self.observer.on_tokens(
                                    result.provider,
                                    result.model,
                                    result.token_usage.prompt_tokens,
                                    result.token_usage.completion_tokens,
                                )
                            if result.time_to_first_token_ms is not None:
                                self.observer.on_stream_response(
                                    result.provider,
                                    result.model,
                                    result.time_to_first_token_ms / 1000.0,
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

            except PauseRequestedError:
                paused_result = StepResult(step_id=step_id, status=StepStatus.PAUSED)
                await self.state.set_step_result(step_id, paused_result)
                if self.observer:
                    self.observer.on_step_end(
                        step_id,
                        step_def.type.value,
                        "paused",
                        paused_result.duration_ms,
                        paused_result.cost_usd,
                        paused_result.token_usage.total_tokens,
                        stream=should_stream,
                    )
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
                    attachment_count=last_result.attachment_count,
                    stream=should_stream,
                )

            logger.error(
                "Step '%s' failed after %d attempts: %s",
                step_id,
                max_retries + 1,
                last_result.error,
            )
