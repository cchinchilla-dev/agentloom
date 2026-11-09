"""Workflow execution engine — the core of AgentLoom."""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

import anyio
from pydantic import ValidationError as PydanticValidationError

from agentloom.checkpointing.base import (
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    BaseCheckpointer,
    CheckpointData,
)
from agentloom.core.attempts import (
    Attempt,
    Spend,
    attempt_spend,
    enter_attempt,
    enter_step,
    restart_spend,
    running_attempt,
    running_steps,
)
from agentloom.core.dag import DAG
from agentloom.core.inter_agent import active_log
from agentloom.core.models import RetryConfig, StepDefinition, StepType, WorkflowDefinition
from agentloom.core.parser import WorkflowParser
from agentloom.core.protocols import call_hook
from agentloom.core.redact import RedactionPolicy, is_redacted, redact_model, redact_state
from agentloom.core.results import (
    StepResult,
    StepStatus,
    WorkflowResult,
    WorkflowStatus,
)
from agentloom.core.state import LayerStateView, StateManager
from agentloom.exceptions import (
    BudgetExceededError,
    CheckpointSchemaError,
    PauseRequestedError,
    WorkflowError,
    WorkflowTimeoutError,
)
from agentloom.observability.schema import GenAIOperationName
from agentloom.resilience.budget import BudgetEnforcer
from agentloom.resilience.retry import (
    compute_backoff,
    is_retryable_exception,
)
from agentloom.steps.base import BaseStep, StepContext
from agentloom.steps.registry import StepRegistry, create_default_registry

logger = logging.getLogger("agentloom.engine")

# The runs the running code belongs to, outermost first: a subworkflow's inside its parent's
_RUNS: ContextVar[tuple[WorkflowEngine, ...]] = ContextVar("agentloom_runs", default=())

# A step that ran out of time failed, as one that raised did
_FAILED = (StepStatus.FAILED, StepStatus.TIMEOUT)
# JSON schemas a checkpoint keeps as written: their property names are not the workflow's data
_SCHEMA_FIELDS = frozenset({"parameters", "schema_"})


def _extract_pause_error(exc: BaseException) -> PauseRequestedError | None:
    """Unwrap a ``PauseRequestedError`` from an ``ExceptionGroup``.

    anyio task groups wrap step exceptions in an ``ExceptionGroup``. Return ``None`` when
    *exc* contains no ``PauseRequestedError``.
    """
    if isinstance(exc, PauseRequestedError):
        return exc
    if isinstance(exc, ExceptionGroup):
        for inner in exc.exceptions:
            found = _extract_pause_error(inner)
            if found is not None:
                return found
    return None


def _closest_failed_ancestor(target: str, failed: set[str], dag: DAG) -> str | None:
    """Find the closest ancestor of *target* present in *failed*.

    Breadth-first backward through predecessor edges, so each cascade-skipped dependent
    names its nearest failure. Return ``None`` when none is reachable (defensive: the
    caller only asks about transitive successors of *failed*).
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

    Mirrors ``_extract_pause_error``, so a breach raised inside a task group ends the
    run ``WorkflowStatus.BUDGET_EXCEEDED`` rather than a generic ``FAILED``.
    """
    if isinstance(exc, BudgetExceededError):
        return exc
    if isinstance(exc, ExceptionGroup):
        for inner in exc.exceptions:
            found = _extract_budget_error(inner)
            if found is not None:
                return found
    return None


def _retry_wait(retry: RetryConfig, attempt: int) -> float:
    """The wait before a retry: the backoff, or the provider's ``Retry-After`` when longer."""
    backoff = compute_backoff(retry.backoff_base, attempt, retry.backoff_max, retry.jitter)
    asked = attempt_spend().retry_after_s
    return backoff if asked is None else min(max(backoff, asked), retry.backoff_max)


def _spend_of(result: StepResult) -> Spend:
    """What the running attempt spent: the calls the gateway saw, or ``result``'s when more.

    A step can report calls the gateway never saw: those of a gateway of its own, or of
    a subworkflow with a budget of its own. Cost and tokens are compared separately.
    """
    seen = attempt_spend()
    reported = result.token_usage.total_tokens
    named = bool(result.provider and result.model)
    by_model = seen.by_model
    # On a tie the step's own names win: a replay answers as the provider it recorded
    if reported > seen.usage.total_tokens or (named and reported == seen.usage.total_tokens):
        if named:
            by_model = {(u.provider, u.model): u.token_usage for u in result.usage_by_model} or {
                (result.provider or "", result.model or ""): result.token_usage
            }
        else:
            by_model = {("", ""): result.token_usage}  # emitted by the steps that made them
    return Spend(max(seen.cost_usd, result.cost_usd), dict(by_model))


class WorkflowEngine:
    """Workflow executor that traverses the DAG and runs its steps.

    Steps in the same DAG layer (no dependencies between them) run concurrently in an
    anyio task group.
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
        budget_enforcer: BudgetEnforcer | None = None,
    ) -> None:
        self.workflow = workflow
        self.state = state_manager or StateManager(initial_state=dict(workflow.state))
        self.provider_gateway = provider_gateway
        self.tool_registry = tool_registry
        self.step_registry = step_registry or create_default_registry()
        self.observer = observer
        self._stream_callback = on_stream_chunk
        # A subworkflow shares its parent's enforcer; an unlimited run gets one too (no None checks)
        self._budget = budget_enforcer or BudgetEnforcer(workflow.config.budget_usd)
        # What this run charged, its subworkflows that share its budget included, and
        # what it had charged before a resume: a shared enforcer counts its parent's too
        self._charged = 0.0
        self._spent_before = 0.0
        self._deadline = math.inf
        # The steps cut short with no result: what they spent counts in the run's totals
        self._cut: dict[str, StepResult] = {}
        self._layer_start: dict[str, Any] = {}

        self._checkpointer = checkpointer
        # Always set, checkpointed or not: trace consumers match traces to runs by its value
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._completed_steps: set[str] = set()
        self._checkpoint_created_at = datetime.now(UTC).isoformat()
        self._gate_deadlines: dict[str, str] = {}

        # The env var is a deployment-wide baseline; ``state_schema`` adds the workflow's own keys
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
        # Deferred so that importing the engine stays light
        from agentloom.history.writer import RunHistoryWriter

        try:
            writer = RunHistoryWriter()
            await writer.record(result, self.workflow, run_id=self.run_id)
        except Exception:
            logger.debug("Run history write skipped", exc_info=True)

    def _attach_quality_emitter(self, result: WorkflowResult, workflow_name: str) -> None:
        """Wire a quality-span emitter onto *result* if a tracer is available.

        The closure captures the tracing manager, run id and workflow name, so
        ``result.annotate(...)`` publishes a ``quality:<target>`` OTel span with no
        infrastructure from the caller, and ``core/results.py`` imports nothing from
        ``observability/``.
        """
        if self.observer is None:
            return
        tracing_ctx = getattr(self.observer, "tracing", None)
        if tracing_ctx is None:
            return
        # Deferred, though dependency-free: a run without a tracer never imports it
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
        """Wire every terminal ``WorkflowResult``, whatever the outcome.

        Stamp ``run_id`` for trace correlation, attach the quality-span emitter so
        ``result.annotate(...)`` works, write the run history record, and redact
        ``final_state`` and ``step_results``. In-memory state stays plaintext for step
        composition; a result that leaves the process (``agentloom run --json``,
        ``result.model_dump_json()``) carries sentinels, as the checkpoint does.
        """
        result.run_id = self.run_id
        if self._redaction_policy:
            result.final_state = redact_state(result.final_state, self._redaction_policy)
            result.step_results = {
                step_id: StepResult.model_validate(self._redacted_result(step_id, step_result))
                for step_id, step_result in result.step_results.items()
            }
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

        # Redact everything written to disk; in-memory state stays plaintext for later templates
        persisted_state = redact_state(state_snapshot, self._redaction_policy)
        workflow_dump = redact_model(
            self.workflow.model_dump(),
            WorkflowDefinition,
            self._redaction_policy,
            parse_as={"workflow_inline": WorkflowDefinition},
            opaque=_SCHEMA_FIELDS,
        )
        if "model" not in self.workflow.config.model_fields_set:
            # Derived from the provider: left out, so ``resume --provider`` can replace it
            workflow_dump["config"].pop("model", None)
        step_results_dump = {k: self._redacted_result(k, v) for k, v in step_results.items()}

        data = CheckpointData(
            workflow_name=self.workflow.name,
            run_id=self.run_id,
            workflow_definition=workflow_dump,
            state=persisted_state,
            step_results=step_results_dump,
            completed_steps=completed_steps,
            status=status,
            paused_step_id=paused_step_id,
            gate_deadlines=dict(self._gate_deadlines),
            spent_usd=self._spent,
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

    def _redacted_result(self, step_id: str, result: StepResult) -> dict[str, Any]:
        """*result*, dumped with its data redacted: its output as the state redacts its key."""
        dumped = redact_model(result.model_dump(), StepResult, self._redaction_policy)
        # Not a router's: its output is the id of the step a resume has to activate
        key = next(
            (
                s.output
                for s in self.workflow.steps
                if s.id == step_id and s.output and s.type != StepType.ROUTER
            ),
            None,
        )
        if key is not None and self._redaction_policy:
            dumped["output"] = redact_state({key: result.output}, self._redaction_policy)[key]
            if result.reasoning_content is not None:  # it can say what the answer says
                text = {key: result.reasoning_content}
                dumped["reasoning_content"] = redact_state(text, self._redaction_policy)[key]
        return dumped

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
        budget_enforcer: BudgetEnforcer | None = None,
    ) -> WorkflowEngine:
        """Reconstruct an engine from a checkpoint, ready to resume.

        The returned engine's :meth:`run` will skip already-completed steps
        and continue execution from where it left off.

        Args:
            approval_decisions: Optional map of ``step_id → "approved"|"rejected"``
                injected into state so that paused approval gate steps can read
                the human decision on re-execution.
            budget_enforcer: The budget to charge, as a subworkflow sharing its
                parent's passes it; by default, one that goes on from what the run
                had spent.
        """
        # Refuse a checkpoint from a newer runtime up front rather than mis-resume it
        if checkpoint_data.schema_version > CURRENT_CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointSchemaError(
                f"Checkpoint for run '{checkpoint_data.run_id}' has "
                f"schema_version={checkpoint_data.schema_version}, newer than this "
                f"runtime supports (v{CURRENT_CHECKPOINT_SCHEMA_VERSION}). It was "
                f"written by a newer AgentLoom — upgrade to resume it."
            )

        workflow = WorkflowDefinition.model_validate(checkpoint_data.workflow_definition)

        # Redacted keys come back as sentinel literals: warn once so the operator re-injects them
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

        # Through the public API, so snapshots and locking stay consistent
        state_manager = StateManager(initial_state=checkpoint_data.state)
        restored_cost = 0.0
        for step_id, result_data in checkpoint_data.step_results.items():
            try:
                result = StepResult.model_validate(result_data)
            except PydanticValidationError as e:
                # An enum value this runtime lacks marks a checkpoint from a newer AgentLoom
                raise CheckpointSchemaError(
                    f"Checkpoint for run '{checkpoint_data.run_id}' has a step "
                    f"result ({step_id!r}) this runtime cannot parse — likely an "
                    f"enum value added in a newer AgentLoom. Upgrade to resume it."
                ) from e
            await state_manager.set_step_result(step_id, result)
            restored_cost += result.cost_usd

        # Inject approval decisions so gate steps find them on re-execution
        if approval_decisions:
            for step_id, decision in approval_decisions.items():
                await state_manager.set(f"_approval.{step_id}", decision)

        spent = max(checkpoint_data.spent_usd, restored_cost)
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
            # The budget goes on from what the run had spent, not from zero
            budget_enforcer=budget_enforcer
            or BudgetEnforcer(workflow.config.budget_usd, spent_usd=spent),
        )
        engine._spent_before = spent
        engine._completed_steps = set(checkpoint_data.completed_steps)
        engine._checkpoint_created_at = checkpoint_data.created_at
        engine._gate_deadlines = dict(checkpoint_data.gate_deadlines)
        return engine

    @property
    def _spent(self) -> float:
        return self._spent_before + self._charged

    async def run(self) -> WorkflowResult:
        """Execute the workflow end-to-end and return its result."""
        runs = _RUNS.set((*_RUNS.get(), self))
        try:
            return await self._run()
        finally:
            _RUNS.reset(runs)

    async def _run(self) -> WorkflowResult:
        start = time.monotonic()
        workflow_name = self.workflow.name
        timeout = self.workflow.config.timeout
        deadline = anyio.current_time() + timeout if timeout else math.inf  # 0: no limit
        self._deadline = deadline
        # A resumed run's results from its checkpoint: a pause among them is no pause of this run
        restored = await self.state.all_step_results()

        logger.info("Starting workflow: %s", workflow_name)

        if self.observer:
            call_hook(self.observer, "on_workflow_start", workflow_name, run_id=self.run_id)

        dag = WorkflowParser.build_dag(self.workflow)
        layers = dag.execution_layers()

        logger.info(
            "Workflow '%s' has %d steps in %d layers",
            workflow_name,
            len(self.workflow.steps),
            len(layers),
        )

        try:
            # ``skip_reasons`` explains failure cascades; a branch not taken is normal, so silent
            skipped_steps: set[str] = set()
            skip_reasons: dict[str, str] = {}
            activated_targets: set[str] | None = None

            for layer_idx, layer in enumerate(layers):
                logger.debug("Executing layer %d: %s", layer_idx, layer)

                active_steps = []
                restored_routers: dict[str, str] = {}
                for step_id in layer:
                    # Skip steps already completed in a previous run (resume)
                    if step_id in self._completed_steps:
                        logger.debug("Skipping already-completed step: %s", step_id)
                        # A completed router restores its activation for downstream filtering
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
                                restored_routers[step_id] = step_res.output
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
                    # As when a router runs: the subtree of a branch not taken stays out, joins too
                    skipped_steps.update(
                        dag.transitive_successors(
                            {
                                child
                                for router in restored_routers
                                for child in dag.successors(router)
                                if child not in restored_routers.values()
                            }
                        )
                    )
                    continue

                max_concurrent = self.workflow.config.max_concurrent_steps
                limiter = anyio.CapacityLimiter(max_concurrent)
                # Here, or a step the limiter holds back would see what its siblings wrote
                self._layer_start = await self.state.get_state_snapshot()

                # A resumed step keeps its previous run's result until it records a new one
                previous = {sid: await self.state.get_step_result(sid) for sid in active_steps}
                with anyio.CancelScope(deadline=deadline) as layer_scope:
                    async with anyio.create_task_group() as tg:
                        for step_id in active_steps:
                            tg.start_soon(self._execute_step_with_limit, step_id, limiter)
                if layer_scope.cancelled_caught:
                    message = f"Workflow timed out after {timeout}s"
                    for step_id in active_steps:  # the steps the deadline cut short
                        before = previous[step_id]
                        if await self.state.get_step_result(step_id) is before:
                            cut = StepResult(
                                step_id=step_id,
                                status=StepStatus.TIMEOUT,
                                error=message,
                                error_classification="transient",
                            )
                            if before is not None:  # never started: keeps what it spent before
                                cut.cost_usd, cut.token_usage = before.cost_usd, before.token_usage
                            await self.state.set_step_result(step_id, cut)
                    raise WorkflowTimeoutError(message)

                # Pauses surface after the layer so siblings finish in-flight provider calls
                paused_step_id: str | None = None
                for step_id in active_steps:
                    step_result = await self.state.get_step_result(step_id)
                    if step_result and step_result.status == StepStatus.SUCCESS:
                        self._completed_steps.add(step_id)
                    elif step_result and step_result.status == StepStatus.PAUSED:
                        # A subworkflow's pause puts its ``parent.child`` path in ``error``
                        qualified = step_result.error or step_id
                        paused_step_id = paused_step_id or qualified

                if paused_step_id is not None:
                    raise PauseRequestedError(paused_step_id)

                # A failed step's dependents would burn tokens, firing side effects on partial state
                failed_in_layer: set[str] = set()
                for step_id in active_steps:
                    step_result = await self.state.get_step_result(step_id)
                    if step_result and step_result.status in _FAILED:
                        step_def = self.workflow.get_step(step_id)
                        # A router that failed chose no branch, so ``continue`` runs none of them
                        if self.workflow.config.on_step_failure == "skip_downstream" or (
                            step_def and step_def.type == StepType.ROUTER
                        ):
                            failed_in_layer.add(step_id)
                if failed_in_layer:
                    direct_children: set[str] = set()
                    for fid in failed_in_layer:
                        direct_children.update(dag.successors(fid))
                    for descendant in dag.transitive_successors(direct_children):
                        if descendant in skipped_steps:
                            continue
                        skipped_steps.add(descendant)
                        # Name the nearest failure: a descendant can sit several hops below it
                        upstream = _closest_failed_ancestor(descendant, failed_in_layer, dag)
                        skip_reasons.setdefault(
                            descendant,
                            f"skipped due to upstream failure: "
                            f"{upstream or ', '.join(sorted(failed_in_layer))}",
                        )

                # A finished router's target stays active next to a router that ran again
                activated_targets_for_next = set(restored_routers.values())
                for step_id in active_steps:
                    step_def = self.workflow.get_step(step_id)
                    if step_def and step_def.type == StepType.ROUTER:
                        step_res = await self.state.get_step_result(step_id)
                        if step_res and step_res.status == StepStatus.SUCCESS and step_res.output:
                            activated_targets_for_next.add(step_res.output)

                if activated_targets_for_next:
                    activated_targets = activated_targets_for_next
                    # Skip the whole subtree of a branch not taken, joins too: they'd miss an input
                    non_activated_children: set[str] = set()
                    for step_id in (*active_steps, *restored_routers):
                        step_def = self.workflow.get_step(step_id)
                        if step_def and step_def.type == StepType.ROUTER:
                            for child in dag.successors(step_id):
                                if child not in activated_targets:
                                    non_activated_children.add(child)
                    skipped_steps.update(dag.transitive_successors(non_activated_children))
                else:
                    if activated_targets is not None:
                        activated_targets = None

            duration = (time.monotonic() - start) * 1000
            step_results = await self.state.all_step_results()
            final_state = await self.state.get_state_snapshot()

            total_tokens, total_cost = self._totals(step_results)

            failed_steps = [r for r in step_results.values() if r.status in _FAILED]
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
                call_hook(
                    self.observer,
                    "on_workflow_end",
                    workflow_name,
                    status.value,
                    duration,
                    total_tokens,
                    total_cost,
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

            # Pause outranks budget: the human decides first, and the resume re-checks the budget.
            # Check state too: the group may raise a sibling's breach while a pause sits in state
            pause_err = _extract_pause_error(e)
            if pause_err is None:
                step_results_for_pause = await self.state.all_step_results()
                for sid, sresult in step_results_for_pause.items():
                    if sresult.status == StepStatus.PAUSED and sresult is not restored.get(sid):
                        pause_err = PauseRequestedError(sresult.error or sid)
                        break
            if pause_err is not None:
                await self._save_checkpoint("paused", paused_step_id=pause_err.step_id)
                step_results = await self.state.all_step_results()
                final_state = await self.state.get_state_snapshot()
                total_tokens, total_cost = self._totals(step_results)
                if self.observer:
                    call_hook(
                        self.observer,
                        "on_workflow_end",
                        workflow_name,
                        "paused",
                        duration,
                        total_tokens,
                        total_cost,
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
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    error=str(pause_err),
                )
                await self._finalize_result(result, workflow_name)
                return result

            budget_err = _extract_budget_error(e)
            if budget_err is not None:
                await self._save_checkpoint("budget_exceeded")
                step_results = await self.state.all_step_results()
                total_tokens, total_cost = self._totals(step_results)
                if self.observer:
                    call_hook(
                        self.observer,
                        "on_workflow_end",
                        workflow_name,
                        "budget_exceeded",
                        duration,
                        total_tokens,
                        total_cost,
                    )
                result = WorkflowResult(
                    workflow_name=workflow_name,
                    status=WorkflowStatus.BUDGET_EXCEEDED,
                    step_results=step_results,
                    final_state=await self.state.get_state_snapshot(),
                    total_duration_ms=duration,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    error=str(budget_err),
                )
                await self._finalize_result(result, workflow_name)
                return result

            if isinstance(e, WorkflowTimeoutError):
                await self._save_checkpoint("timeout")
                step_results = await self.state.all_step_results()
                total_tokens, total_cost = self._totals(step_results)
                if self.observer:
                    call_hook(
                        self.observer,
                        "on_workflow_end",
                        workflow_name,
                        "timeout",
                        duration,
                        total_tokens,
                        total_cost,
                    )
                logger.error("Workflow '%s' timed out: %s", workflow_name, e)
                result = WorkflowResult(
                    workflow_name=workflow_name,
                    status=WorkflowStatus.TIMEOUT,
                    step_results=step_results,
                    final_state=await self.state.get_state_snapshot(),
                    total_duration_ms=duration,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    error=str(e),
                )
                await self._finalize_result(result, workflow_name)
                return result

            await self._save_checkpoint("failed")
            step_results = await self.state.all_step_results()
            total_tokens, total_cost = self._totals(step_results)
            if self.observer:
                call_hook(
                    self.observer,
                    "on_workflow_end",
                    workflow_name,
                    "failed",
                    duration,
                    total_tokens,
                    total_cost,
                )
            logger.error("Workflow '%s' failed: %s", workflow_name, e)
            result = WorkflowResult(
                workflow_name=workflow_name,
                status=WorkflowStatus.FAILED,
                step_results=step_results,
                final_state=await self.state.get_state_snapshot(),
                total_duration_ms=duration,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                error=str(e),
            )
            await self._finalize_result(result, workflow_name)
            return result

        except BaseException:
            # Cancelled from outside, as by Ctrl-C: the steps that finished can be resumed. A
            # subworkflow leaves it to its parent, or it would overwrite the pause it resumes from
            if _RUNS.get()[0] is self:
                with anyio.CancelScope(shield=True):
                    await self._save_checkpoint("cancelled")
            # Close this run's span so the observer returns to the parent
            if self.observer:
                with anyio.CancelScope(shield=True):
                    total_tokens, total_cost = self._totals(await self.state.all_step_results())
                call_hook(
                    self.observer,
                    "on_workflow_end",
                    workflow_name,
                    "cancelled",
                    (time.monotonic() - start) * 1000,
                    total_tokens,
                    total_cost,
                )
            raise

    def _totals(self, step_results: dict[str, StepResult]) -> tuple[int, float]:
        """The run's tokens and cost, however it ends: what its steps report.

        The cost is no less than what this run charged: a subworkflow that shares its
        enforcer reports ``cost_usd=0`` (its steps charged it already), and a step cut
        by a breach gets no result. A shared enforcer counts its parent's charges too,
        so ``self._spent`` is this run's only.
        """
        ran = {**step_results, **self._cut}  # a step cut short spent with no result
        return (
            sum(r.token_usage.total_tokens for r in ran.values()),
            max(sum(r.cost_usd for r in ran.values()), self._spent),
        )

    async def _execute_step_with_limit(self, step_id: str, limiter: anyio.CapacityLimiter) -> None:
        """Execute a step with concurrency limiting."""
        async with limiter:
            await self._execute_step(step_id)

    async def _begin_attempt(self, step_id: str, failed: Attempt | None) -> None:
        """Tell the recorders and the tool tape that a step starts an attempt.

        On a retry, ``failed`` is the attempt it replaces: they forget what
        it did, the steps of a subworkflow run inside it included, so a
        recording holds the attempt that finished and a replay starts the
        step over.
        """
        for listener in (self.provider_gateway, getattr(self.tool_registry, "tape", None)):
            begin = getattr(listener, "begin_attempt", None)
            if callable(begin):
                pending = begin(step_id, failed)
                if inspect.isawaitable(pending):  # a mocked gateway answers with a plain value
                    await pending

    async def _execute_step(self, step_id: str) -> None:
        """Execute a single step with retry and timeout support."""
        step_def = self.workflow.get_step(step_id)
        if step_def is None:
            raise WorkflowError(f"Step '{step_id}' not found in workflow")

        # Refuse to start once the budget is spent; ``charge`` raises a breach inside a layer
        if self._budget.has_limit() and self._budget.remaining == 0.0:
            assert self._budget.limit_usd is not None
            raise BudgetExceededError(self._budget.limit_usd, self._budget.spent)

        logger.debug("Executing step: %s (type=%s)", step_id, step_def.type.value)

        should_stream = (
            step_def.stream if step_def.stream is not None else self.workflow.config.stream
        )

        if self.observer:
            call_hook(
                self.observer, "on_step_start", step_id, step_def.type.value, stream=should_stream
            )
        started = time.monotonic()
        failed = Spend()  # what the attempts that failed spent: charged as they fail
        before = await self.state.get_step_result(step_id)
        if before is not None and before.status != StepStatus.SUCCESS:
            # Run again on a resume: its result goes on from what it spent before, charged then
            failed.add(before.cost_usd, "", "", before.token_usage)
        try:
            await self._run_step(step_id, step_def, should_stream, failed)
        except BaseException as exc:
            cancelled = isinstance(exc, anyio.get_cancelled_exc_class())
            if cancelled:
                with anyio.CancelScope(shield=True):
                    cut = await self._charge_cut(step_id, failed)
            # A step cancelled with its layer (a failing sibling, a parent's timeout) or stopped
            # by a budget breach never reaches on_step_end: close its span here, in its own task.
            if self.observer:
                error: str | None = str(exc)
                spent: dict[str, Any] = {}
                if cancelled:
                    status, error = "cancelled", None
                    if cut.error is not None:  # its result says timeout too
                        status, error = "timeout", cut.error
                    spent = {
                        "cost_usd": cut.cost_usd,
                        "prompt_tokens": cut.token_usage.prompt_tokens,
                        "completion_tokens": cut.token_usage.completion_tokens,
                        "reasoning_tokens": cut.token_usage.reasoning_tokens,
                    }
                elif isinstance(exc, BudgetExceededError):
                    status = "budget_exceeded"
                    # The breach is this step's own charge, made after its result was recorded.
                    with anyio.CancelScope(shield=True):
                        done = await self.state.get_step_result(step_id)
                    if done is not None:
                        spent = {
                            "cost_usd": done.cost_usd,
                            "prompt_tokens": done.token_usage.prompt_tokens,
                            "completion_tokens": done.token_usage.completion_tokens,
                            "reasoning_tokens": done.token_usage.reasoning_tokens,
                            "model": self._requested_model(step_def) if done.model else None,
                            "provider": done.provider,
                        }
                else:
                    status = "failed"
                call_hook(
                    self.observer,
                    "on_step_end",
                    step_id,
                    step_def.type.value,
                    status,
                    (time.monotonic() - started) * 1000,
                    error=error,
                    stream=should_stream,
                    **spent,
                )
            raise

    async def _charge(self, cost_usd: float) -> None:
        """Charge what the running attempt spent, once, even when its step is being cut."""
        with anyio.CancelScope(shield=True):
            restart_spend()
            for run in _RUNS.get():
                if run._budget is self._budget:  # this run, and the runs it shares its budget with
                    run._charged += cost_usd
            try:
                await self._budget.charge(cost_usd)
            finally:  # a breach raises from the charge, and the gauge has to show it
                if self._budget.has_limit():
                    call_hook(
                        self.observer,
                        "on_budget_remaining",
                        self.workflow.name,
                        self._budget.remaining or 0.0,
                    )

    def _requested_model(self, step_def: StepDefinition) -> str:
        """The model *step_def* asks for, resolved as its executor does."""
        own = step_def.agent.model if step_def.agent is not None else step_def.model
        return own or self.workflow.config.model

    def _emit_tokens(self, step_id: str, spend: Spend) -> None:
        """Each model's tokens under it; a subworkflow's steps emit their own."""
        if self.observer:
            step_def = self.workflow.get_step(step_id)
            embeds = step_def is not None and step_def.type == StepType.EMBED
            operation = GenAIOperationName.EMBEDDINGS if embeds else GenAIOperationName.CHAT
            for (provider, model), usage in spend.by_model.items():
                if provider and model and usage.total_tokens > 0:
                    call_hook(
                        self.observer,
                        "on_tokens",
                        provider,
                        model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        reasoning_tokens=usage.reasoning_tokens,
                        operation=operation.value,
                    )

    async def _charge_failure(self, step_id: str, result: StepResult, failed: Spend) -> None:
        """Charge what a failed attempt spent and keep the failed attempts' total on ``result``.

        Its calls were paid however it ended: a provider error after paid calls, its
        timeout, a structured reply never valid.
        """
        spend = _spend_of(result)
        failed.absorb(spend)
        result.cost_usd, result.token_usage = failed.cost_usd, failed.usage
        self._emit_tokens(step_id, spend)
        try:
            await self._charge(spend.cost_usd)
        except BudgetExceededError:
            # The breach reports what the step spent, from its result
            await self.state.set_step_result(step_id, result)
            raise

    async def _charge_cut(self, step_id: str, failed: Spend) -> StepResult:
        """Charge what a step cut short spent, its answered calls paid all the same.

        The budget of a step cut by a breach is broken already, so its charge raises
        nothing. The workflow's timeout leaves its result: timed out, with what it spent;
        any other cut leaves none, and the run's totals count what it spent.
        """
        spend = attempt_spend()
        failed.absorb(spend)
        self._emit_tokens(step_id, spend)
        with contextlib.suppress(BudgetExceededError):
            await self._charge(spend.cost_usd)
        cut = StepResult(
            step_id=step_id,
            status=StepStatus.TIMEOUT,
            cost_usd=failed.cost_usd,
            token_usage=failed.usage,
            error_classification="transient",
        )
        if anyio.current_time() >= self._deadline:
            cut.error = f"Workflow timed out after {self.workflow.config.timeout}s"
            await self.state.set_step_result(step_id, cut)
        else:
            self._cut[step_id] = cut
        return cut

    async def _run_step(
        self, step_id: str, step_def: StepDefinition, should_stream: bool, failed: Spend
    ) -> None:
        """Run a started step through its attempts; ``_execute_step`` covers escaping errors."""
        executor_cls = self.step_registry.get(step_def.type)
        executor: BaseStep = executor_cls()

        context = StepContext(
            step_definition=step_def,
            state_manager=LayerStateView(self.state, self._layer_start),
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
            budget_enforcer=self._budget,
            gate_deadlines=self._gate_deadlines,
        )

        max_retries = step_def.retry.max_retries
        last_result: StepResult | None = None
        outer = running_attempt()  # a subworkflow's step runs inside its parent's attempt
        enter_step(step_id)
        previous: Attempt | None = None
        executing = False  # a breach while the step runs, not one its charge raised

        for attempt in range(max_retries + 1):
            try:
                running = enter_attempt(outer, attempt + 1)
                await self._begin_attempt(step_id, previous)
                previous = running
                if (log := active_log()) is not None:
                    agent = step_def.speaker or (
                        step_def.agent.name if step_def.agent is not None else step_id
                    )
                    log.start_step(running_steps(), agent, step_def.depends_on)
                executing = True
                if step_def.timeout:
                    with anyio.fail_after(step_def.timeout):
                        result = await executor.execute(context)
                else:
                    result = await executor.execute(context)
                executing = False

                last_result = result

                if result.status == StepStatus.SUCCESS:
                    spend = _spend_of(result)
                    # The step's result tells what it cost, the attempts that failed included
                    result.cost_usd = spend.cost_usd + failed.cost_usd
                    result.token_usage = spend.usage + failed.usage
                    await self.state.set_step_result(step_id, result)
                    if (log := active_log()) is not None:
                        log.finish_step(running_steps(), result.output)
                    self._emit_tokens(step_id, spend)  # before a charge that may break the budget

                    # The enforcer's lock serializes parallel charges; the breach raises from it
                    await self._charge(spend.cost_usd)

                    if self.observer:
                        pmeta = result.prompt_metadata
                        call_hook(
                            self.observer,
                            "on_step_end",
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
                            model=self._requested_model(step_def) if result.model else None,
                            provider=result.provider,
                            finish_reason=(pmeta.finish_reason if pmeta else None),
                            prompt_hash=(pmeta.hash if pmeta else None),
                            prompt_length_chars=(pmeta.length_chars if pmeta else None),
                            prompt_template_id=(pmeta.template_id if pmeta else None),
                            prompt_template_vars=(",".join(pmeta.template_vars) if pmeta else None),
                        )
                        if result.provider and result.model:
                            if result.time_to_first_token_ms is not None:
                                call_hook(
                                    self.observer,
                                    "on_stream_response",
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

                # A returned failure has no status code to filter: retry it, stamped transient
                if result.error_classification is None:
                    result.error_classification = "transient"
                await self._charge_failure(step_id, result, failed)
                if result.error_classification == "permanent":
                    logger.warning(
                        "Step '%s' failed with non-retryable error; not retrying: %s",
                        step_id,
                        result.error,
                    )
                    break
                if attempt < max_retries:
                    backoff = _retry_wait(step_def.retry, attempt)
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
                    error_classification="transient",
                )
                await self._charge_failure(step_id, last_result, failed)
                # Timeouts are transient: retry while attempts remain, no status-code filter
                if attempt < max_retries:
                    backoff = _retry_wait(step_def.retry, attempt)
                    await anyio.sleep(backoff)
                    continue
                break

            except BudgetExceededError as breach:
                if executing:
                    # A subworkflow sharing the budget broke it: record what the attempts spent
                    spend = attempt_spend()
                    failed.absorb(spend)
                    self._emit_tokens(step_id, spend)
                    with contextlib.suppress(BudgetExceededError):
                        await self._charge(spend.cost_usd)
                    breached = StepResult(
                        step_id=step_id,
                        status=StepStatus.FAILED,
                        error=str(breach),
                        cost_usd=failed.cost_usd,
                        token_usage=failed.usage,
                    )
                    await self.state.set_step_result(step_id, breached)
                raise

            except PauseRequestedError as pause_err:
                # Return, not raise: cancelling in-flight siblings would double-bill them on resume.
                # A subworkflow's gate keeps its qualified path (``sub.gate``) for the parent.
                qualified_path = (
                    pause_err.step_id
                    if pause_err.step_id and pause_err.step_id != step_id
                    else None
                )
                if pause_err.expires_at is not None:
                    self._gate_deadlines[qualified_path or step_id] = pause_err.expires_at
                # What the step spent before its gate is charged, a breach giving way to the pause
                spend = attempt_spend()
                failed.absorb(spend)
                self._emit_tokens(step_id, spend)
                with contextlib.suppress(BudgetExceededError):
                    await self._charge(spend.cost_usd)
                paused_result = StepResult(
                    step_id=step_id,
                    status=StepStatus.PAUSED,
                    error=qualified_path,
                    cost_usd=failed.cost_usd,
                    token_usage=failed.usage,
                )
                await self.state.set_step_result(step_id, paused_result)
                if self.observer:
                    call_hook(
                        self.observer,
                        "on_step_end",
                        step_id,
                        step_def.type.value,
                        "paused",
                        paused_result.duration_ms,
                        paused_result.cost_usd,
                        stream=should_stream,
                    )
                return

            except Exception as e:
                # A permanent failure (non-429 4xx, sandbox, validation…) skips the retries left
                retryable = is_retryable_exception(e, step_def.retry.retryable_status_codes)
                last_result = StepResult(
                    step_id=step_id,
                    status=StepStatus.FAILED,
                    error=str(e),
                    error_classification=("transient" if retryable else "permanent"),
                )
                await self._charge_failure(step_id, last_result, failed)
                if not retryable:
                    logger.warning(
                        "Step '%s' failed with non-retryable error; not retrying: %s",
                        step_id,
                        e,
                    )
                    break
                if attempt < max_retries:
                    backoff = _retry_wait(step_def.retry, attempt)
                    await anyio.sleep(backoff)
                    continue
                break

        if last_result:
            await self.state.set_step_result(step_id, last_result)

            if self.observer:
                call_hook(
                    self.observer,
                    "on_step_end",
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
                "Step '%s' failed after %d attempt(s): %s",
                step_id,
                attempt + 1,  # a non-retryable failure stops before max_retries
                last_result.error,
            )
