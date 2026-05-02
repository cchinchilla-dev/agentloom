"""LLM call step executor."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

from agentloom.core.models import Attachment, StepDefinition
from agentloom.core.results import PromptMetadata, StepResult, StepStatus
from agentloom.core.templates import SafeFormatDict, build_template_vars
from agentloom.exceptions import StepError
from agentloom.observability.schema import SpanAttr
from agentloom.providers.multimodal import (
    ContentBlock,
    build_multimodal_content,
    resolve_attachments,
)
from agentloom.steps.base import BaseStep, StepContext

logger = logging.getLogger("agentloom.steps")

_TEMPLATE_VAR_RE = re.compile(r"\{([\w.]+)")


def _build_prompt_metadata(
    workflow_name: str,
    step_id: str,
    step_prompt_template: str | None,
    rendered: str,
) -> PromptMetadata:
    """Compute the non-sensitive bits of prompt provenance.

    Hash is truncated to 16 hex chars — plenty for correlating traces
    without the storage cost of a full SHA-256. Template-variable names
    are extracted from the *template* (not the rendered output) so we
    see ``state.user_input``, not the interpolated value.
    """
    h = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
    template_vars: list[str] = []
    if step_prompt_template:
        template_vars = sorted(set(_TEMPLATE_VAR_RE.findall(step_prompt_template)))
    return PromptMetadata(
        hash=h,
        length_chars=len(rendered),
        template_id=f"{workflow_name}:{step_id}",
        template_vars=template_vars,
    )


class LLMCallStep(BaseStep):
    """Executes an LLM call with prompt template rendering from state."""

    @staticmethod
    def _build_thinking_kwargs(step: StepDefinition) -> dict[str, Any]:
        """Forward ``StepDefinition.thinking`` to the gateway as a config object.

        The ``ThinkingConfig`` is passed through under the ``thinking_config``
        kwarg so each provider adapter can translate it to its own request
        shape (Anthropic ``thinking``, Gemini ``thinkingConfig``, Ollama
        ``think``). Disabled or absent configs return an empty dict so the
        request is unchanged.
        """
        cfg = step.thinking
        if cfg is None or not cfg.enabled:
            return {}
        return {"thinking_config": cfg}

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")

        if not step.prompt:
            raise StepError(step.id, "LLM call step requires a 'prompt' field")

        model = step.model or context.workflow_model
        state_snapshot = await context.state_manager.get_state_snapshot()

        template_vars = build_template_vars(state_snapshot)

        try:
            rendered_prompt = step.prompt.format_map(SafeFormatDict(template_vars))
            rendered_system = None
            if step.system_prompt:
                rendered_system = step.system_prompt.format_map(SafeFormatDict(template_vars))
        except (KeyError, ValueError) as e:
            raise StepError(step.id, f"Prompt template error: {e}") from e

        # Opt-in full-prompt capture as a span event so trusted environments
        # can debug from Jaeger without re-running. Off by default — see
        # ``WorkflowConfig.capture_prompts``.
        if context.capture_prompts and context.observer is not None:
            attach = getattr(context.observer, "attach_step_event", None)
            if callable(attach):
                attach(
                    step.id,
                    SpanAttr.PROMPT_CAPTURED_EVENT,
                    {
                        "prompt": rendered_prompt,
                        "system_prompt": rendered_system or "",
                    },
                )

        content_blocks: list[ContentBlock] = []
        if step.attachments:
            try:
                resolved_attachments = [
                    Attachment(
                        type=att.type,
                        source=att.source.format_map(SafeFormatDict(template_vars)),
                        media_type=att.media_type,
                        fetch=att.fetch,
                    )
                    for att in step.attachments
                ]
            except (KeyError, ValueError) as e:
                raise StepError(step.id, f"Attachment template error: {e}") from e
            try:
                content_blocks = await resolve_attachments(
                    resolved_attachments, sandbox=context.sandbox_config
                )
            except Exception as e:
                raise StepError(step.id, f"Attachment resolution error: {e}") from e

        messages: list[dict[str, Any]] = []
        if rendered_system:
            messages.append({"role": "system", "content": rendered_system})
        user_content = build_multimodal_content(rendered_prompt, content_blocks)
        messages.append({"role": "user", "content": user_content})

        if context.stream:
            return await self._execute_stream(
                context,
                messages,
                model,
                step,
                start,
                len(content_blocks),
                rendered_prompt=rendered_prompt,
            )

        provider_kwargs = self._build_thinking_kwargs(step)
        try:
            response = await context.provider_gateway.complete(
                messages=messages,
                model=model,
                temperature=step.temperature,
                max_tokens=step.max_tokens,
                step_id=step.id,
                **provider_kwargs,
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
            await context.state_manager.set(step.output, response.content)

        prompt_metadata = _build_prompt_metadata(
            context.workflow_name, step.id, step.prompt, rendered_prompt
        )
        prompt_metadata.finish_reason = response.finish_reason

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=response.content,
            duration_ms=duration,
            token_usage=response.usage,
            cost_usd=response.cost_usd,
            model=response.model,
            provider=response.provider,
            attachment_count=len(content_blocks),
            prompt_metadata=prompt_metadata,
        )

    async def _execute_stream(
        self,
        context: StepContext,
        messages: list[dict[str, Any]],
        model: str,
        step: StepDefinition,
        start: float,
        attachment_count: int,
        *,
        rendered_prompt: str = "",
    ) -> StepResult:
        """Execute the LLM call in streaming mode."""
        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")
        provider_kwargs = self._build_thinking_kwargs(step)
        try:
            sr = await context.provider_gateway.stream(
                messages=messages,
                model=model,
                temperature=step.temperature,
                max_tokens=step.max_tokens,
                step_id=step.id,
                **provider_kwargs,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_ms=duration,
            )

        # TTFT measures wall-clock time from just before the first stream
        # iteration to the first yielded chunk.  This *includes* HTTP
        # connection setup (the provider iterator is lazy), so it reflects
        # end-to-end latency to first token from the consumer's perspective.
        # Rate-limiter wait is excluded (happens before gateway.stream()
        # returns).
        ttft_ms: float | None = None
        stream_start = time.monotonic()
        first_chunk = True

        try:
            async for chunk in sr:
                if first_chunk:
                    ttft_ms = (time.monotonic() - stream_start) * 1000
                    first_chunk = False
                if context.on_stream_chunk:
                    try:
                        context.on_stream_chunk(step.id, chunk)
                    except Exception:
                        logger.warning("Stream chunk callback failed, disabling")
                        context.on_stream_chunk = None
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_ms=duration,
            )
        finally:
            # Ensure the underlying httpx stream is closed even on partial
            # consumption (e.g. MAX_ACCUMULATED_BYTES exceeded).
            if sr._iterator is not None:
                aclose = getattr(sr._iterator, "aclose", None)
                if aclose:
                    await aclose()

        response = sr.to_provider_response()
        duration = (time.monotonic() - start) * 1000

        if step.output:
            await context.state_manager.set(step.output, response.content)

        prompt_metadata = _build_prompt_metadata(
            context.workflow_name, step.id, step.prompt, rendered_prompt
        )
        prompt_metadata.finish_reason = response.finish_reason

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=response.content,
            duration_ms=duration,
            token_usage=response.usage,
            cost_usd=response.cost_usd,
            model=response.model,
            provider=response.provider,
            attachment_count=attachment_count,
            time_to_first_token_ms=ttft_ms,
            prompt_metadata=prompt_metadata,
        )

    @staticmethod
    def _build_template_vars(state: dict[str, object]) -> dict[str, object]:
        """Build a flat namespace for str.format_map().

        .. deprecated:: Use :func:`agentloom.core.templates.build_template_vars` instead.
        """
        return build_template_vars(state)
