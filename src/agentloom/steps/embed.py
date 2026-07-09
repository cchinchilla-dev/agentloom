"""Embedding step — resolves state inputs and calls ``gateway.embed()``.

``inputs`` is a dotted state path (e.g. ``state.documents``). Optional
``{...}`` wrapper braces are stripped so ``inputs: {state.docs}`` reads
the same as a template placeholder even though the step is not a
template — subscripts like ``state.docs[0]`` are NOT supported.
"""

from __future__ import annotations

import contextlib
import time

from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import StepError
from agentloom.steps.base import BaseStep, StepContext


def _resolve_state_ref(ref: str, state: dict[str, object]) -> object:
    """Resolve a ``state.foo.bar`` dotted reference against the snapshot."""
    ref = ref.strip()
    if ref.startswith("{") and ref.endswith("}"):
        ref = ref[1:-1]
    if not ref.startswith("state.") or ref == "state.":
        raise StepError(
            "embed",
            f"'inputs' must be a dotted state reference like 'state.documents'; "
            f"got {ref!r}. Subscript syntax ('state.docs[0]') is not supported.",
        )
    node: object = state
    segments = ref[len("state.") :].split(".")
    for segment in segments:
        if not segment:
            raise StepError(
                "embed",
                f"'inputs' has an empty path segment in {ref!r} — check for a "
                f"trailing or duplicated dot.",
            )
        if not isinstance(node, dict) or segment not in node:
            raise StepError("embed", f"State path {ref!r} not found or wrong shape.")
        node = node[segment]
    return node


class EmbedStep(BaseStep):
    """Reads a list of strings from state and writes a list of vectors back."""

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")
        if not step.inputs:
            raise StepError(step.id, "Embed step requires an 'inputs' state reference")

        state_snapshot = await context.state_manager.get_state_snapshot()
        try:
            raw_inputs = _resolve_state_ref(step.inputs, state_snapshot)
        except StepError as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_ms=duration,
            )

        if isinstance(raw_inputs, str):
            texts = [raw_inputs]
        elif isinstance(raw_inputs, list) and all(isinstance(x, str) for x in raw_inputs):
            texts = list(raw_inputs)
        else:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=(
                    f"'inputs' must resolve to a str or list[str]; got {type(raw_inputs).__name__}"
                ),
                duration_ms=duration,
            )

        model = step.model or context.workflow_model
        try:
            response = await context.provider_gateway.embed(
                inputs=texts,
                model=model,
                dimensions=step.dimensions,
                step_id=step.id,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_ms=duration,
            )

        duration = (time.monotonic() - start) * 1000
        if step.output:
            await context.state_manager.set(step.output, response.embeddings)

        if context.observer is not None:
            hook = getattr(context.observer, "on_embedding_call", None)
            if callable(hook):
                dim = len(response.embeddings[0]) if response.embeddings else 0
                with contextlib.suppress(Exception):
                    hook(
                        provider=response.provider,
                        model=response.model,
                        dimensions=dim,
                        input_count=len(texts),
                        prompt_tokens=response.usage.prompt_tokens,
                    )

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=(
                f"{len(response.embeddings)} vectors × "
                f"{len(response.embeddings[0]) if response.embeddings else 0}"
            ),
            duration_ms=duration,
            token_usage=response.usage,
            cost_usd=response.cost_usd,
            model=response.model,
            provider=response.provider,
        )
