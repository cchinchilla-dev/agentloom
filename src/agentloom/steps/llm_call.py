"""LLM call step executor."""

from __future__ import annotations

import logging
import time
from typing import Any

from agentloom.core.models import Attachment, StepDefinition
from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import StepError
from agentloom.providers.multimodal import (
    ContentBlock,
    build_multimodal_content,
    resolve_attachments,
)
from agentloom.steps.base import BaseStep, StepContext

logger = logging.getLogger("agentloom.steps")


class LLMCallStep(BaseStep):
    """Executes an LLM call with prompt template rendering from state."""

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")

        if not step.prompt:
            raise StepError(step.id, "LLM call step requires a 'prompt' field")

        # Resolve model: step-level override > workflow config
        model = step.model or context.workflow_model
        state_snapshot = await context.state_manager.get_state_snapshot()

        # Build a flat namespace for template rendering
        template_vars = self._build_template_vars(state_snapshot)

        try:
            rendered_prompt = step.prompt.format_map(SafeFormatDict(template_vars))
            rendered_system = None
            if step.system_prompt:
                rendered_system = step.system_prompt.format_map(SafeFormatDict(template_vars))
        except (KeyError, ValueError) as e:
            raise StepError(step.id, f"Prompt template error: {e}") from e

        # Resolve multimodal attachments
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

        # Build messages
        messages: list[dict[str, Any]] = []
        if rendered_system:
            messages.append({"role": "system", "content": rendered_system})
        user_content = build_multimodal_content(rendered_prompt, content_blocks)
        messages.append({"role": "user", "content": user_content})

        # Call provider (streaming or batch)
        if context.stream:
            return await self._execute_stream(
                context, messages, model, step, start, len(content_blocks)
            )

        try:
            response = await context.provider_gateway.complete(
                messages=messages,
                model=model,
                temperature=step.temperature,
                max_tokens=step.max_tokens,
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

        # Store output in state if output mapping is defined
        if step.output:
            await context.state_manager.set(step.output, response.content)

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
        )

    async def _execute_stream(
        self,
        context: StepContext,
        messages: list[dict[str, Any]],
        model: str,
        step: StepDefinition,
        start: float,
        attachment_count: int,
    ) -> StepResult:
        """Execute the LLM call in streaming mode."""
        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")
        try:
            sr = await context.provider_gateway.stream(
                messages=messages,
                model=model,
                temperature=step.temperature,
                max_tokens=step.max_tokens,
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
        )

    @staticmethod
    def _build_template_vars(state: dict[str, object]) -> dict[str, object]:
        """Build a flat namespace for str.format_map().

        Supports both {user_input} and {state.user_input} syntax.
        """
        flat: dict[str, object] = {}
        # Top-level state vars are directly accessible
        flat.update(state)
        # Also accessible via state.* prefix
        flat["state"] = DotAccessDict(state)
        return flat


class DotAccessDict:
    """Wrapper that allows attribute access on a dict for template rendering."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name not in self._data:
            logger.warning("Template variable 'state.%s' not found, rendering as empty", name)
            return ""
        value = self._data[name]
        if isinstance(value, dict):
            return DotAccessDict(value)
        return value

    def __str__(self) -> str:
        return str(self._data)

    def __format__(self, format_spec: str) -> str:
        return str(self._data)


class SafeFormatDict(dict[str, object]):
    """Dict that returns '{key}' for missing keys instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
