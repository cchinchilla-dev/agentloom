"""Result models for step and workflow execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr


class StepStatus(StrEnum):
    """Execution status of a step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    PAUSED = "paused"


class TokenUsage(BaseModel):
    """Token usage for an LLM call.

    ``reasoning_tokens`` is populated only when the provider reports a
    separate reasoning / thinking token count — OpenAI o-series via
    ``completion_tokens_details.reasoning_tokens`` and Gemini 2.5+ via
    ``usageMetadata.thoughtsTokenCount``. Anthropic and Ollama do not
    expose a separate count today (Anthropic rolls thinking into
    ``output_tokens``; Ollama emits a single ``eval_count``), so the
    field stays ``0`` for those providers — the chain-of-thought trace
    is still surfaced via ``ProviderResponse.reasoning_content``. When
    populated, providers bill these tokens at the output rate, so
    cost calculation must include them.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def billable_completion_tokens(self) -> int:
        """Tokens billed at the provider's output rate.

        Always equal to ``completion_tokens + reasoning_tokens`` — exposed
        as a property so downstream cost code does not have to remember
        which field to add.
        """
        return self.completion_tokens + self.reasoning_tokens


class PromptMetadata(BaseModel):
    """Non-sensitive provenance metadata for a rendered LLM prompt.

    Captured by ``LLMCallStep`` and forwarded to the observer so traces
    can correlate a failed run to the prompt that produced it without
    storing the full prompt text (size, secrets). Full-prompt capture is
    a separate opt-in flag.
    """

    hash: str
    length_chars: int
    template_id: str
    template_vars: list[str] = Field(default_factory=list)
    finish_reason: str | None = None


class StepResult(BaseModel):
    """Result from executing a single step."""

    step_id: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    model: str | None = None
    provider: str | None = None
    attachment_count: int = 0
    time_to_first_token_ms: float | None = None
    prompt_metadata: PromptMetadata | None = None


class WorkflowStatus(StrEnum):
    """Execution status of a workflow."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    PAUSED = "paused"


class QualityAnnotation(BaseModel):
    """Post-hoc quality metadata attached to a :class:`WorkflowResult`.

    Produced by evaluators, human reviewers, or downstream scoring code
    *after* the run completes. Emitted to OTel as a standalone
    ``quality:<target>`` span that carries ``workflow.run_id`` so trace
    consumers can correlate annotations back to the original run.

    ``target`` is a free-form label (e.g. ``"answer"``, ``"summary"``,
    ``"step:review"``); ``source`` identifies who produced the score
    (``"human_feedback"``, ``"llm_judge"``, ``"regex"``).
    """

    target: str
    quality_score: float
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    """Result from executing a complete workflow."""

    workflow_name: str
    status: WorkflowStatus
    step_results: dict[str, StepResult] = Field(default_factory=dict)
    final_state: dict[str, Any] = Field(default_factory=dict)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None
    run_id: str = ""
    annotations: list[QualityAnnotation] = Field(default_factory=list)

    # Optional callable wired by the engine before the result is returned
    # to the caller. Signature: ``Callable[[QualityAnnotation], None]``.
    # ``PrivateAttr`` excludes it from serialization so ``model_dump()`` /
    # JSON output stays a pure data snapshot, and the indirection keeps
    # ``core/results.py`` free of any ``observability/`` import (per the
    # ``CLAUDE.md`` layering rule — "core code never imports from
    # observability/ directly"). When set, ``annotate()`` invokes the
    # emitter so the issue #59 contract — "metadata exported as OTel span
    # attributes, visible in Jaeger" — holds without the caller threading
    # any infrastructure through.
    _emit_quality: Any = PrivateAttr(default=None)

    def attach_quality_emitter(self, emitter: Any) -> None:
        """Wire a callback that publishes :class:`QualityAnnotation` spans.

        The engine builds and attaches the closure (capturing the live
        tracing manager + run_id + workflow name) so this module never
        imports from ``agentloom.observability`` directly. ``None`` is
        accepted and disables auto-emission for offline / replay paths.
        """
        self._emit_quality = emitter

    def annotate(
        self,
        target: str,
        *,
        quality_score: float,
        source: str = "unknown",
        **metadata: Any,
    ) -> QualityAnnotation:
        """Attach a quality annotation to this result.

        The annotation is always appended to :attr:`annotations`. When the
        engine has wired a quality-span emitter via
        :meth:`attach_quality_emitter` (the default for any workflow run
        with observability enabled), the annotation is also published
        immediately as a standalone ``quality:<target>`` OTel span carrying
        ``workflow.run_id`` so downstream consumers see it in Jaeger
        without additional code. Offline / replay scenarios where no
        emitter is wired keep working — the annotation is recorded on the
        result, and the OTel emission is simply a no-op.
        """
        annotation = QualityAnnotation(
            target=target,
            quality_score=quality_score,
            source=source,
            metadata=metadata,
        )
        self.annotations.append(annotation)
        if self._emit_quality is not None:
            self._emit_quality(annotation)
        return annotation
