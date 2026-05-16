"""Workflow execution engine — the core of AgentLoom."""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import anyio

from agentloom.checkpointing.base import BaseCheckpointer, CheckpointData
from agentloom.core.dag import DAG
from agentloom.core.models import StepType, WorkflowDefinition
from agentloom.core.parser import WorkflowParser
from agentloom.core.redact import RedactionPolicy, is_redacted, redact_state
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


def _closest_failed_ancestor(
    target: str, failed: set[str], dag: DAG
) -> str | None:
    """Find the closest ancestor of *target* present in *failed*.

    BFS backward through the DAG's predecessor edges. Returns the first
    failed step encountered, or ``None`` if no failed ancestor is reachable
    (defensive: the caller only invokes this when *target* is a transitive
    successor of *failed*, so a hit is expected). Used by the cascade-skip
    block to attribute each SKIPPED dependent to the nearest failure it
    can name, instead of a generic "upstream failure" without context.
    """
    visited: set[str] = {target}
    queue: deque[str] = deque([target])
    while queue:
        node = queue.popleft()
        for pred in dag.predecessors(node):
            if pred in failed:
                return pred
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)
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
        # Always generate a run_id so ``workflow.run_id`` propagates through
        # the span tree on every workflow execution, not only checkpointed
        # ones — external trace consumers (AgentTest, Jaeger search) rely on
        # it to correlate a Jaeger trace with a workflow run.
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._completed_steps: set[str] = set()
        self._checkpoint_created_at = datetime.now(UTC).isoformat()

        # Build the redaction policy once at startup — combines the
        # workflow-level ``state_schema:`` declarations with the
        # ``AGENTLOOM_REDACT_STATE_KEYS`` env var so a deployment-wide
        # baseline (env) can coexist with per-workflow overrides (YAML).
        self._redaction_policy = RedactionPolicy(workflow.redaction_patterns()).merge(
            RedactionPolicy.from_env()
        )

        # Wire observer into gateway for circuit breaker events
        if observer and provider_gateway:
            set_obs = getattr(provider_gateway, "set_observer", None)
            if set_obs:
                set_obs(observer)

    async def _write_run_history(self, result: WorkflowResult) -> None:
        """Persist a per-run JSON record.

        Silent on failure so a broken history directory cannot prevent a
        workflow from returning its result to the caller.
        """
        # Lazy import to keep the engine's import graph minimal and avoid
        # paying the cost for callers that never look at run history.
        from agentloom.history.writer import RunHistoryWriter

        try:
            writer = RunHistoryWriter()
            await writer.record(result, self.workflow, run_id=self.run_id)
        except Exception:
            logger.debug("Run history write skipped", exc_info=True)

    def _attach_quality_emitter(self, result: WorkflowResult, workflow_name: str) -> None:
        """Wire a quality-span emitter onto *result* if a tracer is available.

        Builds a closure capturing the live tracing manager + run_id +
        workflow name so subsequent ``result.annotate(...)`` calls publish
        a ``quality:<target>`` OTel span without the caller threading any
        infrastructure through. Keeps ``core/results.py`` free of any
        ``observability/`` import (per ``CLAUDE.md`` layering rule) — the
        engine is the only module that knits the two together.
        """
        if self.observer is None:
            return
        tracing_ctx = getattr(self.observer, "tracing", None)
        if tracing_ctx is None:
            return
        # Lazy import: ``agentloom.observability.quality`` carries no
        # opentelemetry dependency itself (only schema constants), so the
        # import is always safe; deferring it keeps the engine import
        # graph minimal for callers that never annotate.
        from agentloom.observability.quality import emit_quality_annotation

        def _emitter(annotation: Any) -> None:
            emit_quality_annotation(
                annotation,
                tracing_ctx,
                run_id=self.run_id,
                workflow_name=workflow_name,
            )

        result.attach_quality_emitter(_emitter)

    async def _finalize_result(self, result: WorkflowResult, workflow_name: str) -> None:
        """Common post-construction wiring for every terminal ``WorkflowResult``.

        Centralises four concerns that previously only ran on the success
        path:
          * stamp ``run_id`` so trace-correlation works across all outcomes
            (failure / budget_exceeded / paused traces are otherwise
            orphaned)
          * attach the quality-span emitter so ``result.annotate(...)``
            keeps working on non-success terminations too
          * write the per-run history record so ``agentloom history``
            shows every execution, not only successful ones
          * apply the redaction policy to ``final_state`` and
            ``step_results`` so callers that dump the result (``agentloom
            run --json``, programmatic ``result.model_dump_json()``) see
            sentinels for flagged keys — the in-memory state stays
            plaintext for step composition, but the moment the result
            crosses the process boundary the same persistence-gate the
            checkpoint enjoys applies here too.
        """
        result.run_id = self.run_id
        if self._redaction_policy:
            result.final_state = redact_state(result.final_state, self._redaction_policy)
            redacted_step_results: dict[str, StepResult] = {}
            for step_id, step_result in result.step_results.items():
                dumped = step_result.model_dump()
                dumped = redact_state(dumped, self._redaction_policy)
                redacted_step_results[step_id] = StepResult.model_validate(dumped)
            result.step_results = redacted_step_results
        self._attach_quality_emitter(result, workflow_name)
        await self._write_run_history(result)

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

        # Apply the redaction policy on the way to disk. The in-memory
        # ``self.state`` stays plaintext so a subsequent step that
        # interpolates ``{state.api_key}`` keeps working; only the
        # serialized snapshot carries the sentinel. Redaction is applied
        # uniformly to every surface that lands in the checkpoint JSON:
        # the runtime snapshot, the literal ``state:`` seed inside the
        # workflow definition, the step result outputs (an LLM call may
        # return a flagged key in structured output), and any other
        # workflow-definition field whose key name matches the policy
        # (e.g. a step-level ``notify.headers.api_key``).
        #
        # ``state_schema`` is intentionally lifted out before the walk:
        # its leaves are the policy *metadata* itself (``redact: true``),
        # not user state. Redacting them would rewrite the bool flag into
        # a sentinel string that breaks ``WorkflowDefinition.model_validate``
        # on resume.
        persisted_state = redact_state(state_snapshot, self._redaction_policy)
        workflow_dump = self.workflow.model_dump()
        if self._redaction_policy:
            preserved_schema = workflow_dump.pop("state_schema", None)
            workflow_dump = redact_state(workflow_dump, self._redaction_policy)
            if preserved_schema is not None:
                workflow_dump["state_schema"] = preserved_schema
        step_results_dump = {k: v.model_dump() for k, v in step_results.items()}
        if self._redaction_policy:
            step_results_dump = redact_state(step_results_dump, self._redaction_policy)

        data = CheckpointData(
            workflow_name=self.workflow.name,
            run_id=self.run_id,
            workflow_definition=workflow_dump,
            state=persisted_state,
            step_results=step_results_dump,
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

        # Detect redacted state keys carried in the checkpoint. Sentinels were
        # written by ``_save_checkpoint`` on the way to disk, so on the way
        # back any downstream step that references such a key will receive
        # the sentinel literal — not the original plaintext. Surface a single
        # warning that lists the affected keys so the operator notices.
        redacted_keys = [k for k, v in checkpoint_data.state.items() if is_redacted(v)]
        if redacted_keys:
            logger.warning(
                "Resuming run '%s' with redacted state keys %s: subsequent steps "
                "that reference these will receive the redaction sentinel, not "
                "the original value. Re-inject the plaintext before resuming, "
                "or remove ``redact: true`` from the workflow's state_schema.",
                checkpoint_data.run_id,
                redacted_keys,
            )

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
            # Track which steps are skipped (not activated by router OR
            # cascaded behind a failed upstream). ``skip_reasons`` carries the
            # explanatory message for steps cascaded behind a failure so the
            # operator can audit the chain — success-cascade entries stay
            # silent because "branch not activated" is normal flow control.
            skipped_steps: set[str] = set()
            skip_reasons: dict[str, str] = {}
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
                            StepResult(
                                step_id=step_id,
                                status=StepStatus.SKIPPED,
                                error=skip_reasons.get(step_id),
                            ),
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
                        # ``error`` carries the qualified ``parent.child`` path
                        # when the pause came from inside a subworkflow; fall
                        # back to the local id for ordinary approval gates.
                        qualified = step_result.error or step_id
                        paused_step_id = paused_step_id or qualified

                if paused_step_id is not None:
                    raise PauseRequestedError(paused_step_id)

                # Cascade-skip dependents of any step that ended FAILED in this
                # layer. Applies uniformly to routers (AST error, no-match-no-
                # default, evaluator exception) and regular steps that exhausted
                # retries — both leave downstream branches unable to make
                # meaningful progress, and letting them run wastes tokens and
                # can fire side-effect tools / webhooks against partial state.
                # Opt out per-workflow via ``config.on_step_failure: continue``
                # for best-effort fan-outs that explicitly want today's
                # swallow-and-continue behaviour.
                if self.workflow.config.on_step_failure == "skip_downstream":
                    failed_in_layer: set[str] = set()
                    for step_id in active_steps:
                        step_result = await self.state.get_step_result(step_id)
                        if step_result and step_result.status == StepStatus.FAILED:
                            failed_in_layer.add(step_id)
                    if failed_in_layer:
                        direct_children: set[str] = set()
                        for fid in failed_in_layer:
                            direct_children.update(dag.successors(fid))
                        for descendant in dag.transitive_successors(direct_children):
                            if descendant in skipped_steps:
                                continue
                            skipped_steps.add(descendant)
                            # Attribute the skip to the closest known failed
                            # ancestor (BFS backward). Transitive descendants
                            # may sit several hops below the failure; naming
                            # the nearest is the most actionable for the
                            # operator.
                            upstream = _closest_failed_ancestor(
                                descendant, failed_in_layer, dag
                            )
                            skip_reasons.setdefault(
                                descendant,
                                f"skipped due to upstream failure: "
                                f"{upstream or ', '.join(sorted(failed_in_layer))}",
                            )

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
            await self._finalize_result(result, workflow_name)

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
                result = WorkflowResult(
                    workflow_name=workflow_name,
                    status=WorkflowStatus.BUDGET_EXCEEDED,
                    step_results=await self.state.all_step_results(),
                    final_state=await self.state.get_state_snapshot(),
                    total_duration_ms=duration,
                    total_cost_usd=self._budget_spent,
                    error=str(budget_err),
                )
                await self._finalize_result(result, workflow_name)
                return result

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
                result = WorkflowResult(
                    workflow_name=workflow_name,
                    status=WorkflowStatus.PAUSED,
                    step_results=step_results,
                    final_state=final_state,
                    total_duration_ms=duration,
                    total_tokens=sum(r.token_usage.total_tokens for r in step_results.values()),
                    total_cost_usd=sum(r.cost_usd for r in step_results.values()),
                    error=str(pause_err),
                )
                await self._finalize_result(result, workflow_name)
                return result

            await self._save_checkpoint("failed")
            if self.observer:
                self.observer.on_workflow_end(
                    workflow_name, "failed", duration, 0, self._budget_spent
                )
            logger.error("Workflow '%s' failed: %s", workflow_name, e)
            result = WorkflowResult(
                workflow_name=workflow_name,
                status=WorkflowStatus.FAILED,
                step_results=await self.state.all_step_results(),
                final_state=await self.state.get_state_snapshot(),
                total_duration_ms=duration,
                error=str(e),
            )
            await self._finalize_result(result, workflow_name)
            return result

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
            redaction_policy=self._redaction_policy,
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

            except PauseRequestedError as pause_err:
                # Record the pause and return normally instead of re-raising.
                # Re-raising would cancel sibling tasks that are already
                # mid-flight against their providers, leading to double-billing
                # on resume. The engine's post-layer pass detects PAUSED results
                # and halts further layers without cancelling in-flight work.
                # ``pause_err.step_id`` may be a qualified path (``sub.gate``)
                # when the pause originated inside a subworkflow — stash it on
                # the result so the post-layer pass can re-emit the full path
                # to the parent's checkpoint hint instead of just the local
                # subworkflow step id.
                qualified_path = (
                    pause_err.step_id
                    if pause_err.step_id and pause_err.step_id != step_id
                    else None
                )
                paused_result = StepResult(
                    step_id=step_id,
                    status=StepStatus.PAUSED,
                    error=qualified_path,
                )
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
