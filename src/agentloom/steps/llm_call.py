"""LLM call step executor."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agentloom.core.models import (
    Attachment,
    Conversation,
    Message,
    ResponseSchema,
    StepDefinition,
)
from agentloom.core.results import PromptMetadata, StepResult, StepStatus
from agentloom.core.templates import SafeFormatDict, build_template_vars
from agentloom.exceptions import StepError
from agentloom.observability.schema import SpanAttr
from agentloom.providers.base import ProviderResponse
from agentloom.providers.multimodal import (
    ContentBlock,
    build_multimodal_content,
    resolve_attachments,
)
from agentloom.steps._conversation import (
    TrimResult,
    apply_trim_policy_async,
    load_conversation,
    to_provider_messages,
)
from agentloom.steps.base import BaseStep, StepContext

logger = logging.getLogger("agentloom.steps")

# Captures the variable path inside a ``{path[!conv][:spec]}`` template
# placeholder, including ``state.items[0].name``-style indexed paths so the
# emitted ``agentloom.prompt.template_vars`` reflects the full reference.
_TEMPLATE_VAR_RE = re.compile(r"\{([\w][\w.\[\]]*?)(?:[!:][^}]*)?\}")


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
    async def _run_tool_loop(
        *,
        context: StepContext,
        step: StepDefinition,
        messages: list[dict[str, Any]],
        model: str,
        provider_kwargs: dict[str, Any],
    ) -> ProviderResponse:
        """Iterate complete() → dispatch tools → re-prompt until done.

        Cost and tokens accumulate across iterations. ``max_tool_iterations``
        bounds the loop; collapses to a single call when ``tools`` is empty.
        """
        from agentloom.core.results import TokenUsage
        from agentloom.steps._tools import (
            build_assistant_message_with_tool_calls,
            build_tool_result_messages,
            dispatch_tool_calls,
        )

        accumulated_prompt = 0
        accumulated_completion = 0
        accumulated_reasoning = 0
        accumulated_cost = 0.0

        gateway = context.provider_gateway
        if gateway is None:
            raise StepError(step.id, "No provider gateway configured")
        # ``max_tool_iterations`` is validated >= 1 on the model so the
        # loop runs at least once; ``response`` is therefore guaranteed
        # to be assigned when we exit.
        response: ProviderResponse | None = None
        for _ in range(step.max_tool_iterations):
            response = await gateway.complete(
                messages=messages,
                model=model,
                temperature=step.temperature,
                max_tokens=step.max_tokens,
                step_id=step.id,
                **provider_kwargs,
            )
            accumulated_prompt += response.usage.prompt_tokens
            accumulated_completion += response.usage.completion_tokens
            accumulated_reasoning += response.usage.reasoning_tokens
            accumulated_cost += response.cost_usd

            if not response.tool_calls or not step.tools:
                # Replace the response usage with the accumulated totals so
                # the caller sees the full conversation cost.
                response.usage = TokenUsage(
                    prompt_tokens=accumulated_prompt,
                    completion_tokens=accumulated_completion,
                    total_tokens=(
                        accumulated_prompt + accumulated_completion + accumulated_reasoning
                    ),
                    reasoning_tokens=accumulated_reasoning,
                )
                response.cost_usd = accumulated_cost
                return response

            if context.tool_registry is None:
                raise StepError(
                    step.id,
                    "Tool registry required for tools= declaration but not configured.",
                )

            results = await dispatch_tool_calls(
                response.tool_calls,
                context.tool_registry,
                observer=context.observer,
                step_id=step.id,
            )
            messages.append(
                build_assistant_message_with_tool_calls(
                    response.provider, response.content, response.tool_calls
                )
            )
            messages.extend(build_tool_result_messages(response.provider, results))

        # Loop exhausted; surface the last response with the cap noted as
        # finish_reason so callers can detect it. ``response`` is set
        # because ``max_tool_iterations`` is validated >= 1.
        assert response is not None
        response.usage = TokenUsage(
            prompt_tokens=accumulated_prompt,
            completion_tokens=accumulated_completion,
            total_tokens=(accumulated_prompt + accumulated_completion + accumulated_reasoning),
            reasoning_tokens=accumulated_reasoning,
        )
        response.cost_usd = accumulated_cost
        # ``tool_choice: required`` that hits the iteration cap with an
        # empty final message means the model never produced an answer —
        # it looped picking tools until the budget ran out. Pre-0.5.0
        # this surfaced as a bare ``success`` with empty output and no
        # signal. Tag it with a distinct ``finish_reason`` and a one-line
        # warning so callers and dashboards can tell it apart from a
        # normal cap hit (the model answered but kept calling tools).
        if step.tool_choice == "required" and not (response.content or "").strip():
            logger.warning(
                "Step %r: tool_choice='required' hit the %d-iteration cap with an "
                "empty final response — the model never produced an answer. Raise "
                "max_tool_iterations or relax tool_choice.",
                step.id,
                step.max_tool_iterations,
            )
            response.finish_reason = "max_tool_iterations_no_answer"
        else:
            response.finish_reason = "max_tool_iterations"
        return response

    @staticmethod
    def _resolve_path(state: dict[str, Any], path: str) -> Any:
        """Resolve a dotted state path, tolerating a leading ``state.``.

        Workflows write ``conversation: state.chat`` for symmetry with the
        rest of the DSL; the state manager stores under the un-prefixed
        key ``chat``. This helper accepts both so a workflow author using
        ``chat`` and one using ``state.chat`` both resolve to the same
        conversation.
        """
        from agentloom.core.state import StateManager

        key = path
        if key.startswith("state."):
            key = key[len("state.") :]
        return StateManager._resolve_key(state, key, None)

    @staticmethod
    def _strip_state_prefix(path: str) -> str:
        """Strip a leading ``state.`` from a dotted path for ``set()``."""
        return path[len("state.") :] if path.startswith("state.") else path

    @staticmethod
    def _append_assistant_reply(
        convo: Conversation, response: ProviderResponse, speaker: str | None = None
    ) -> None:
        """Persist the model's final text answer into the conversation.

        Only the semantic answer is stored — NOT the tool calls made during
        the tool loop, and NOT the intermediate ``tool_result`` turns. Two
        reasons:

        * **Replay safety.** An assistant turn carrying ``tool_calls`` is
          only valid on the wire when each call is immediately followed by
          its paired ``role="tool"`` result (OpenAI 400s otherwise). Since
          the conversation deliberately drops the mechanical tool-result
          turns, persisting the ``tool_calls`` alongside them would make the
          *next* ``llm_call`` render a malformed request. Storing the final
          answer only keeps every persisted history replayable.
        * **Semantic thread.** The conversation records the exchange the
          next turn actually needs — "user asked X, assistant answered Y" —
          not the wire-format back-and-forth of how Y was produced.

        A normal tool loop ends with a final response whose ``tool_calls``
        is already empty (the model stopped calling tools and answered), so
        this is also what happens in practice. ``speaker`` tags the reply
        with ``Message.name`` for multi-agent attribution.
        """
        convo.messages.append(
            Message(role="assistant", content=response.content or "", name=speaker)
        )

    async def _persist_conversation(
        self,
        context: StepContext,
        step: StepDefinition,
        conversation: Conversation,
        trim_result: TrimResult | None,
        response_model: str,
    ) -> None:
        """Serialize the conversation back to state and emit observability.

        The value is stored as a plain dict (Pydantic's ``model_dump``) so
        checkpoint round-trip stays JSON-friendly; the next ``llm_call``
        re-validates through :class:`Conversation` at load time. Observer
        hooks fire even when trimming was a no-op so dashboards can
        distinguish steady-state cost from trim-hot conversations.
        """
        assert step.conversation is not None  # narrowed by caller
        key = self._strip_state_prefix(step.conversation)
        await context.state_manager.set(key, conversation.model_dump())

        observer = context.observer
        if observer is None:
            return
        attach = getattr(observer, "attach_step_event", None)
        if callable(attach):
            attrs: dict[str, Any] = {
                SpanAttr.CONVERSATION_TURN_COUNT: len(conversation.messages),
                SpanAttr.CONVERSATION_TOKEN_COUNT: conversation.token_count(response_model),
            }
            if trim_result is not None:
                attrs[SpanAttr.CONVERSATION_TRIMMED_MESSAGES] = trim_result.trimmed_count
            attach(step.id, SpanAttr.CONVERSATION_EVENT, attrs)
        on_conv = getattr(observer, "on_conversation_turn", None)
        if callable(on_conv):
            try:
                on_conv(
                    step_id=step.id,
                    conversation_key=self._strip_state_prefix(step.conversation),
                    turn_count=len(conversation.messages),
                    token_count=conversation.token_count(response_model),
                    trim_policy=(trim_result.policy if trim_result else ""),
                    trimmed_count=(trim_result.trimmed_count if trim_result else 0),
                )
            except Exception:  # pragma: no cover — observability best-effort
                logger.debug("on_conversation_turn hook failed", exc_info=True)

    @staticmethod
    def _make_summarizer(
        context: StepContext, step: StepDefinition, model: str
    ) -> Callable[[list[Message]], Awaitable[str]]:
        """Build the async summariser the ``summarize_oldest`` trim policy uses.

        Returns a callable that renders the folded turns into a compact
        transcript and asks the gateway for a summary — the extra LLM call
        the issue calls for (#119). The call is deliberately cheap
        (``max_tokens=256``, temperature 0.3) and reuses the step's own
        model. When the gateway is missing or the call raises, the trim
        layer catches it and falls back to a deterministic concatenation
        so the budget contract still holds.
        """

        async def _summarize(folded: list[Message]) -> str:
            gateway = context.provider_gateway
            if gateway is None:
                return ""
            transcript_lines = []
            for m in folded:
                speaker = m.name or m.role
                body = (m.content or "").strip()
                if body:
                    transcript_lines.append(f"{speaker}: {body}")
            transcript = "\n".join(transcript_lines)
            if not transcript:
                return ""
            summarize_messages = [
                {
                    "role": "system",
                    "content": (
                        "You compress conversation history. Summarize the following "
                        "turns into a few concise sentences that preserve names, "
                        "decisions, and open questions. Reply with the summary only."
                    ),
                },
                {"role": "user", "content": transcript},
            ]
            resp = await gateway.complete(
                messages=summarize_messages,
                model=model,
                temperature=0.3,
                max_tokens=256,
                step_id=f"{step.id}::summarize",
            )
            return (resp.content or "").strip()

        return _summarize

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

    @staticmethod
    async def _validate_structured_output(
        *,
        response: ProviderResponse,
        response_schema: ResponseSchema,
        step: StepDefinition,
        messages: list[dict[str, Any]],
        context: StepContext,
        model: str,
        provider_kwargs: dict[str, Any],
    ) -> ProviderResponse:
        """Parse + validate; retry within ``step.retry.max_retries``."""
        import jsonschema  # type: ignore[import-untyped]
        from pydantic import ValidationError as PydanticValidationError

        from agentloom.core.models import ResponseSchemaConfigError
        from agentloom.steps._structured import (
            extract_parsed,
            format_validation_feedback,
            validate_parsed,
        )

        # Only "model emitted bad output" exceptions trigger retry;
        # everything else escapes to the gateway's own resilience layer.
        ValidationFailure = (
            json.JSONDecodeError,
            PydanticValidationError,
            jsonschema.ValidationError,
            TypeError,
        )

        attempts_remaining = max(0, step.retry.max_retries)
        current_response = response
        gateway = context.provider_gateway
        if gateway is None:
            raise StepError(step.id, "No provider gateway configured")
        while True:
            try:
                parsed = extract_parsed(current_response.content)
                coerced = validate_parsed(parsed, response_schema, step.id)
            except ResponseSchemaConfigError as exc:
                raise StepError(step.id, str(exc), is_retryable=False) from exc
            except ValidationFailure as exc:
                if attempts_remaining <= 0:
                    raise StepError(
                        step.id,
                        f"Structured output failed validation after "
                        f"{step.retry.max_retries + 1} attempts: {exc}",
                        is_retryable=False,
                    ) from exc
                attempts_remaining -= 1
                messages.append(
                    {
                        "role": "assistant",
                        "content": current_response.content or "",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": format_validation_feedback(exc, step.id),
                    }
                )
                logger.warning(
                    "Step %r: structured-output validation failed, retrying "
                    "(%d attempts left). Error: %s",
                    step.id,
                    attempts_remaining,
                    exc,
                )
                current_response = await gateway.complete(
                    messages=messages,
                    model=model,
                    temperature=step.temperature,
                    max_tokens=step.max_tokens,
                    step_id=step.id,
                    **provider_kwargs,
                )
                continue
            current_response.parsed = coerced
            return current_response

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")

        # A conversation-backed llm_call may omit ``prompt`` — the trailing
        # user turn already lives in the loaded messages. Free-form calls
        # still require it.
        if not step.prompt and not step.conversation:
            raise StepError(step.id, "LLM call step requires a 'prompt' field")

        model = step.model or context.workflow_model
        state_snapshot = await context.state_manager.get_state_snapshot()

        template_vars = build_template_vars(state_snapshot)

        try:
            rendered_prompt = (
                step.prompt.format_map(SafeFormatDict(template_vars)) if step.prompt else ""
            )
            rendered_system = None
            if step.system_prompt:
                rendered_system = step.system_prompt.format_map(SafeFormatDict(template_vars))
        except (KeyError, ValueError) as e:
            raise StepError(step.id, f"Prompt template error: {e}") from e

        # Opt-in full-prompt capture as a span event so trusted environments
        # can debug from Jaeger without re-running. Off by default — see
        # ``WorkflowConfig.capture_prompts``. When a redaction policy is in
        # effect the captured copy is re-rendered against the redacted state
        # so secret values never reach the trace backend.
        if context.capture_prompts and context.observer is not None:
            attach = getattr(context.observer, "attach_step_event", None)
            if callable(attach):
                captured_prompt = rendered_prompt
                captured_system = rendered_system or ""
                if context.redaction_policy:
                    from agentloom.core.redact import redact_state as _redact_state

                    safe_state = _redact_state(state_snapshot, context.redaction_policy)
                    safe_vars = build_template_vars(safe_state)
                    try:
                        if step.prompt:
                            captured_prompt = step.prompt.format_map(SafeFormatDict(safe_vars))
                        if step.system_prompt:
                            captured_system = step.system_prompt.format_map(
                                SafeFormatDict(safe_vars)
                            )
                    except (KeyError, ValueError):  # pragma: no cover — already validated
                        captured_prompt = rendered_prompt
                        captured_system = rendered_system or ""
                attach(
                    step.id,
                    SpanAttr.PROMPT_CAPTURED_EVENT,
                    {
                        "prompt": captured_prompt,
                        "system_prompt": captured_system,
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

        conversation: Conversation | None = None
        trim_result: TrimResult | None = None
        if step.conversation:
            try:
                raw_convo = self._resolve_path(state_snapshot, step.conversation)
            except (KeyError, ValueError) as e:
                raise StepError(
                    step.id, f"Conversation path {step.conversation!r} unreadable: {e}"
                ) from e
            try:
                conversation = load_conversation(raw_convo)
            except (TypeError, ValueError) as e:
                raise StepError(
                    step.id, f"Conversation at {step.conversation!r} malformed: {e}"
                ) from e

        messages: list[dict[str, Any]] = []
        if conversation is not None:
            # System-turn handling: a single-agent conversation gets the
            # system prompt inserted once at the head. A multi-agent
            # conversation (``speaker`` set) re-asserts the current
            # speaker's system prompt on THIS turn via a transient system
            # message that is NOT persisted — otherwise agent B would
            # inherit agent A's instructions from the shared history.
            transient_system: dict[str, Any] | None = None
            if rendered_system:
                if step.speaker:
                    transient_system = {"role": "system", "content": rendered_system}
                elif not any(m.role == "system" for m in conversation.messages):
                    conversation.messages.insert(0, Message(role="system", content=rendered_system))
            appended_user_turn = bool(rendered_prompt)
            if appended_user_turn:
                conversation.messages.append(
                    Message(role="user", content=rendered_prompt, name=step.speaker)
                )
            else:
                # No prompt this turn — the conversation must already end on
                # a turn the provider can answer (a ``user`` message, or a
                # ``tool`` result awaiting synthesis). Ending on ``system``
                # or ``assistant`` means the model would be asked to continue
                # after its own turn, which most providers reject. Surface a
                # clear StepError rather than shipping a malformed request.
                last_answerable = next(
                    (m for m in reversed(conversation.messages) if m.role != "system"),
                    None,
                )
                if last_answerable is None or last_answerable.role not in ("user", "tool"):
                    raise StepError(
                        step.id,
                        "conversation-backed llm_call has no 'prompt' and the "
                        "conversation does not end on a 'user' or 'tool' turn "
                        "(last non-system role: "
                        f"{last_answerable.role if last_answerable else 'none'}). "
                        "Set 'prompt' or ensure the conversation ends with a user turn.",
                    )
            trim_result = await apply_trim_policy_async(
                conversation,
                model,
                summarizer=self._make_summarizer(context, step, model),
            )
            messages = to_provider_messages(conversation)
            if transient_system is not None:
                # Re-assert the speaker's instruction at the head so the
                # model reads it as the operative system message for this
                # turn, ahead of the shared history + any summary prefix.
                messages.insert(0, transient_system)
            # Content blocks live only for THIS turn — they're not part of
            # the persisted conversation (base64 payloads would balloon the
            # checkpoint). Hoist them onto the user turn we just appended so
            # per-provider ``_format_messages`` picks up the
            # ``list[ContentBlock]`` shape via ``build_multimodal_content``.
            # Guarded on ``appended_user_turn`` so we never overwrite a
            # trailing assistant / tool message (which would drop its text
            # and attach blocks to a non-user role).
            if content_blocks and appended_user_turn and messages:
                messages[-1]["content"] = build_multimodal_content(rendered_prompt, content_blocks)
        else:
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
                conversation=conversation,
                trim_result=trim_result,
            )

        provider_kwargs = self._build_thinking_kwargs(step)
        if step.tools:
            from agentloom.core.models import ToolChoiceByName

            provider_kwargs["agentloom_tools"] = step.tools
            # Normalize the typed-union ``tool_choice`` to the dict form
            # the provider translators expect — keeps the wire layer
            # provider-shape-agnostic.
            choice: Any = step.tool_choice
            if isinstance(choice, ToolChoiceByName):
                choice = {"name": choice.name}
            provider_kwargs["agentloom_tool_choice"] = choice

        if step.response_schema is not None:
            provider_kwargs["agentloom_response_schema"] = step.response_schema
            provider_kwargs["agentloom_step_id"] = step.id

        # Tool-call loop: re-prompt with tool results until the model
        # stops requesting tools or we exhaust ``max_tool_iterations``.
        # Costs and tokens accumulate across iterations; only the final
        # response's content is exposed to the caller.
        try:
            response = await self._run_tool_loop(
                context=context,
                step=step,
                messages=messages,
                model=model,
                provider_kwargs=provider_kwargs,
            )
            if step.response_schema is not None:
                response = await self._validate_structured_output(
                    response=response,
                    response_schema=step.response_schema,
                    step=step,
                    messages=messages,
                    context=context,
                    model=model,
                    provider_kwargs=provider_kwargs,
                )
        except StepError as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=str(e),
                duration_ms=duration,
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
            # Structured steps store the parsed value; free-form ones the raw string.
            value: Any = response.parsed if step.response_schema is not None else response.content
            await context.state_manager.set(step.output, value)

        if conversation is not None:
            self._append_assistant_reply(conversation, response, step.speaker)
            await self._persist_conversation(
                context, step, conversation, trim_result, response.model
            )

        prompt_metadata = _build_prompt_metadata(
            context.workflow_name,
            step.id,
            step.prompt or "",
            rendered_prompt,
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
        conversation: Conversation | None = None,
        trim_result: TrimResult | None = None,
    ) -> StepResult:
        """Execute the LLM call in streaming mode."""
        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")
        provider_kwargs = self._build_thinking_kwargs(step)
        if step.response_schema is not None:
            provider_kwargs["agentloom_response_schema"] = step.response_schema
            provider_kwargs["agentloom_step_id"] = step.id
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

        if step.response_schema is not None:
            # Parse once at end-of-stream; no mid-stream retry.
            import jsonschema
            from pydantic import ValidationError as PydanticValidationError

            from agentloom.core.models import ResponseSchemaConfigError
            from agentloom.steps._structured import extract_parsed, validate_parsed

            try:
                parsed = extract_parsed(response.content)
                response.parsed = validate_parsed(parsed, step.response_schema, step.id)
            except ResponseSchemaConfigError as e:
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error=str(e),
                    duration_ms=duration,
                )
            except (
                json.JSONDecodeError,
                PydanticValidationError,
                jsonschema.ValidationError,
                TypeError,
            ) as e:
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error=f"Structured-output validation failed: {e}",
                    duration_ms=duration,
                )

        if step.output:
            value: Any = response.parsed if step.response_schema is not None else response.content
            await context.state_manager.set(step.output, value)

        if conversation is not None:
            self._append_assistant_reply(conversation, response, step.speaker)
            await self._persist_conversation(
                context, step, conversation, trim_result, response.model
            )

        prompt_metadata = _build_prompt_metadata(
            context.workflow_name,
            step.id,
            step.prompt or "",
            rendered_prompt,
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
