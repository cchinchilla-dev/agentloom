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
)
from agentloom.resilience.retry import (
    compute_backoff,
    extract_status_code,
    is_retryable_exception,
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


def _extract_budget_error(exc: BaseException) -> BudgetExceededError | None:
    """Unwrap a ``BudgetExceededError`` from an ``ExceptionGroup``.

    Mirrors ``_extract_pause_error``; used so pre-dispatch budget gates
    raised inside a task group surface as ``WorkflowStatus.BUDGET_EXCEEDED``
    rather than a generic ``FAILED``.
    """
    if isinstance(exc, BudgetExceededError):
        return exc
    if isinstance(exc, ExceptionGroup):
        for inner in exc.exceptions:
            found = _extract_budget_error(inner)
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
        approval_decisions: dict[str, str] | None = None,
    ) -> WorkflowEngine:
        """Reconstruct an engine from a checkpoint, ready to resume.

        The returned engine's :meth:`run` will skip already-completed steps
        and continue execution from where it left off.

        Args:
            approval_decisions: Optional map of ``step_id → "approved"|"rejected"``
                injected into state so that paused approval gate steps can read
                the human decision on re-execution.
        """
        workflow = WorkflowDefinition.model_validate(checkpoint_data.workflow_definition)

        # Restore state manager with completed step results via the public API
        # so internal state, snapshots, and locking remain consistent.
        state_manager = StateManager(initial_state=checkpoint_data.state)
        for step_id, result_data in checkpoint_data.step_results.items():
            result = StepResult.model_validate(result_data)
            await state_manager.set_step_result(step_id, result)

        # Inject approval decisions so gate steps find them on re-execution
        if approval_decisions:
            for step_id, decision in approval_decisions.items():
                await state_manager.set(f"_approval.{step_id}", decision)

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
            self.observer.on_workflow_start(workflow_name, run_id=self.run_id)

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

                max_concurrent = self.workflow.config.max_concurrent_steps
                limiter = anyio.CapacityLimiter(max_concurrent)

                async with anyio.create_task_group() as tg:
                    for step_id in active_steps:
                        tg.start_soon(self._execute_step_with_limit, step_id, limiter)

                # After the layer finishes, check whether any step paused.
                # Pauses are NOT re-raised inside _execute_step so siblings
                # can complete their in-flight provider calls without being
                # cancelled. We halt the outer layer loop here instead.
                paused_step_id: str | None = None
                for step_id in active_steps:
                    step_result = await self.state.get_step_result(step_id)
                    if step_result and step_result.status == StepStatus.SUCCESS:
                        self._completed_steps.add(step_id)
                    elif step_result and step_result.status == StepStatus.PAUSED:
                        paused_step_id = paused_step_id or step_id

                if paused_step_id is not None:
                    raise PauseRequestedError(paused_step_id)

                activated_targets_for_next = set()
                for step_id in active_steps:
                    step_def = self.workflow.get_step(step_id)
                    if step_def and step_def.type == StepType.ROUTER:
                        step_res = await self.state.get_step_result(step_id)
                        if step_res and step_res.status == StepStatus.SUCCESS and step_res.output:
                            activated_targets_for_next.add(step_res.output)

                if activated_targets_for_next:
                    activated_targets = activated_targets_for_next
                    # Skip non-activated direct children of any router that
                    # just fired, then propagate the skip forward through the
                    # DAG's transitive closure. Any descendant of a skipped
                    # branch is unreachable, including join-nodes whose other
                    # predecessor is on an activated branch — they would
                    # otherwise execute with one upstream output missing.
                    non_activated_children: set[str] = set()
                    for step_id in active_steps:
                        step_def = self.workflow.get_step(step_id)
                        if step_def and step_def.type == StepType.ROUTER:
                            for child in dag.successors(step_id):
                                if child not in activated_targets:
                                    non_activated_children.add(child)
                    skipped_steps.update(dag.transitive_successors(non_activated_children))
                else:
                    # Reset activation for non-router layers
                    if activated_targets is not None:
                        activated_targets = None

            duration = (time.monotonic() - start) * 1000
            step_results = await self.state.all_step_results()
            final_state = await self.state.get_state_snapshot()

            total_tokens = sum(r.token_usage.total_tokens for r in step_results.values())
            total_cost = sum(r.cost_usd for r in step_results.values())

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

        except Exception as e:
            duration = (time.monotonic() - start) * 1000

            # Unwrap BudgetExceededError from the task group before pauses —
            # a budget overrun supersedes any pause that may have been
            # pending in the same layer.
            budget_err = _extract_budget_error(e)
            if budget_err is not None:
                await self._save_checkpoint("budget_exceeded")
                if self.observer:
                    self.observer.on_workflow_end(
                        workflow_name,
                        "budget_exceeded",
                        duration,
                        0,
                        self._budget_spent,
                    )
                return WorkflowResult(
                    workflow_name=workflow_name,
                    status=WorkflowStatus.BUDGET_EXCEEDED,
                    step_results=await self.state.all_step_results(),
                    final_state=await self.state.get_state_snapshot(),
                    total_duration_ms=duration,
                    total_cost_usd=self._budget_spent,
                    error=str(budget_err),
                )

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

        # Pre-dispatch budget gate: if prior completions already exhausted
        # the budget, refuse to start this step. Post-hoc enforcement lets
        # in-flight sibling calls overshoot by their cost; a pre-check here
        # at least bounds the overshoot to the worst-case single-layer
        # in-flight set instead of compounding across layers.
        budget = self.workflow.config.budget_usd
        if budget is not None and self._budget_spent >= budget:
            raise BudgetExceededError(budget, self._budget_spent)

        logger.debug("Executing step: %s (type=%s)", step_id, step_def.type.value)

        should_stream = (
            step_def.stream if step_def.stream is not None else self.workflow.config.stream
        )

        if self.observer:
            self.observer.on_step_start(step_id, step_def.type.value, stream=should_stream)

        executor_cls = self.step_registry.get(step_def.type)
        executor: BaseStep = executor_cls()

        context = StepContext(
            step_definition=step_def,
            state_manager=self.state,
            provider_gateway=self.provider_gateway,
            tool_registry=self.tool_registry,
            run_id=self.run_id,
            workflow_name=self.workflow.name,
            workflow_model=self.workflow.config.model,
            workflow_provider=self.workflow.config.provider,
            sandbox_config=self.workflow.config.sandbox,
            observer=self.observer,
            stream=should_stream,
            on_stream_chunk=self._stream_callback,
            checkpointer=self._checkpointer,
            capture_prompts=self.workflow.config.capture_prompts,
        )

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
                        pmeta = result.prompt_metadata
                        self.observer.on_step_end(
                            step_id,
                            step_def.type.value,
                            "success",
                            result.duration_ms,
                            result.cost_usd,
                            attachment_count=result.attachment_count,
                            time_to_first_token_ms=result.time_to_first_token_ms,
                            stream=should_stream,
                            prompt_tokens=result.token_usage.prompt_tokens,
                            completion_tokens=result.token_usage.completion_tokens,
                            reasoning_tokens=result.token_usage.reasoning_tokens,
                            model=result.model,
                            provider=result.provider,
                            finish_reason=(pmeta.finish_reason if pmeta else None),
                            prompt_hash=(pmeta.hash if pmeta else None),
                            prompt_length_chars=(pmeta.length_chars if pmeta else None),
                            prompt_template_id=(pmeta.template_id if pmeta else None),
                            prompt_template_vars=(",".join(pmeta.template_vars) if pmeta else None),
                        )
                        if result.provider and result.model:
                            if result.token_usage.total_tokens > 0:
                                self.observer.on_tokens(
                                    result.provider,
                                    result.model,
                                    result.token_usage.prompt_tokens,
                                    result.token_usage.completion_tokens,
                                    reasoning_tokens=result.token_usage.reasoning_tokens,
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

                # Soft-failure path: ``executor.execute`` returned a
                # non-success result without raising. ``result.error`` is a
                # string with no ``status_code`` to inspect, so the
                # retryable-status-codes filter is intentionally skipped —
                # treat the returned failure as transient and retry.
                if attempt < max_retries:
                    backoff = compute_backoff(
                        step_def.retry.backoff_base,
                        attempt,
                        step_def.retry.backoff_max,
                        step_def.retry.jitter,
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
                # Timeouts are transient by definition — always retry until
                # the budget is exhausted, no status-code filter applies.
                if attempt < max_retries:
                    backoff = compute_backoff(
                        step_def.retry.backoff_base,
                        attempt,
                        step_def.retry.backoff_max,
                        step_def.retry.jitter,
                    )
                    await anyio.sleep(backoff)
                    continue
                break

            except BudgetExceededError:
                raise

            except PauseRequestedError:
                # Record the pause and return normally instead of re-raising.
                # Re-raising would cancel sibling tasks that are already
                # mid-flight against their providers, leading to double-billing
                # on resume. The engine's post-layer pass detects PAUSED results
                # and halts further layers without cancelling in-flight work.
                paused_result = StepResult(step_id=step_id, status=StepStatus.PAUSED)
                await self.state.set_step_result(step_id, paused_result)
                if self.observer:
                    self.observer.on_step_end(
                        step_id,
                        step_def.type.value,
                        "paused",
                        paused_result.duration_ms,
                        paused_result.cost_usd,
                        stream=should_stream,
                    )
                return

            except Exception as e:
                last_result = StepResult(
                    step_id=step_id,
                    status=StepStatus.FAILED,
                    error=str(e),
                )
                # Bail out on permanent failures (4xx that's not 429, etc.)
                # before consuming the retry budget. Status-less exceptions
                # are treated as transient and retried — see
                # ``is_retryable_exception`` for the rule.
                if not is_retryable_exception(e, step_def.retry.retryable_status_codes):
                    logger.warning(
                        "Step '%s' failed with non-retryable status %s; not retrying: %s",
                        step_id,
                        extract_status_code(e),
                        e,
                    )
                    break
                if attempt < max_retries:
                    backoff = compute_backoff(
                        step_def.retry.backoff_base,
                        attempt,
                        step_def.retry.backoff_max,
                        step_def.retry.jitter,
                    )
                    await anyio.sleep(backoff)
                    continue
                break

        if last_result:
            await self.state.set_step_result(step_id, last_result)

            if self.observer:
                self.observer.on_step_end(
                    step_id,
                    step_def.type.value,
                    last_result.status.value,
                    last_result.duration_ms,
                    last_result.cost_usd,
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
