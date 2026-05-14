"""Tests for native tool calling across providers."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    ToolDefinition,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepStatus, WorkflowStatus
from agentloom.providers.anthropic import AnthropicProvider
from agentloom.providers.base import ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.openai import OpenAIProvider
from agentloom.steps._tools import (
    build_assistant_message_with_tool_calls,
    build_tool_result_messages,
    dispatch_tool_calls,
    parse_tool_calls_from_anthropic,
    parse_tool_calls_from_google,
    parse_tool_calls_from_openai,
    translate_tools_for_anthropic,
    translate_tools_for_google,
    translate_tools_for_openai,
)
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry


class _AddTool(BaseTool):
    name = "add"
    description = "Add two integers."
    parameters_schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs["a"] + kwargs["b"]


class _LookupTool(BaseTool):
    name = "lookup"
    description = "Look up a value."
    parameters_schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    async def execute(self, **kwargs: Any) -> Any:
        return f"value-for-{kwargs['key']}"


def _registry_with(*tools: BaseTool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


class TestToolTranslation:
    def test_openai_shape(self) -> None:
        out = translate_tools_for_openai(
            [ToolDefinition(name="add", description="d", parameters={"type": "object"})]
        )
        assert out == [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "d",
                    "parameters": {"type": "object"},
                },
            }
        ]

    def test_anthropic_shape(self) -> None:
        out = translate_tools_for_anthropic(
            [ToolDefinition(name="add", description="d", parameters={"type": "object"})]
        )
        assert out == [{"name": "add", "description": "d", "input_schema": {"type": "object"}}]

    def test_google_shape_groups_under_function_declarations(self) -> None:
        out = translate_tools_for_google(
            [
                ToolDefinition(name="add", description="d", parameters={"type": "object"}),
                ToolDefinition(name="lookup", description="l", parameters={"type": "object"}),
            ]
        )
        assert len(out) == 1
        decls = out[0]["function_declarations"]
        assert {d["name"] for d in decls} == {"add", "lookup"}


class TestToolParsing:
    def test_openai_parses_tool_calls(self) -> None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ],
        }
        calls = parse_tool_calls_from_openai(message)
        assert calls == [ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3})]

    def test_openai_skips_non_function_entries(self) -> None:
        message = {"tool_calls": [{"id": "x", "type": "code_interpreter"}, {"type": "function"}]}
        calls = parse_tool_calls_from_openai(message)
        # Both are dropped: the first is non-function, the second has no
        # ``id`` / ``name`` (would 400 on the follow-up tool message).
        assert calls == []

    def test_openai_skips_entries_with_empty_id_or_name(self) -> None:
        # Defense-in-depth: a malformed model response with blank id/name
        # would generate an invalid ``tool_call_id`` on the follow-up
        # message, which OpenAI rejects with 400. Skip + warn rather than
        # constructing a call we know can't dispatch.
        message = {
            "tool_calls": [
                {"id": "", "type": "function", "function": {"name": "ok"}},
                {"id": "good", "type": "function", "function": {"name": ""}},
                {"id": "good", "type": "function", "function": {"name": "ok"}},
            ]
        }
        calls = parse_tool_calls_from_openai(message)
        assert len(calls) == 1
        assert calls[0].id == "good"
        assert calls[0].name == "ok"

    def test_ollama_compat_response_with_dict_args_and_no_type(self) -> None:
        # Real Ollama 0.x ``/api/chat`` returns tool_calls in OpenAI-compatible
        # shape but (a) omits ``"type": "function"`` and (b) ships
        # ``arguments`` as an already-decoded dict, not a JSON string. Without
        # this regression, Ollama tool calling silently drops every call —
        # the parser's strict ``type == "function"`` check skips entries
        # missing the field, and ``json.loads(<dict>)`` would TypeError.
        # Captured from a live ``llama3.1:8b`` response on 2026-05-09.
        message = {
            "tool_calls": [
                {
                    "id": "call_eh7yrv0u",
                    "function": {
                        "index": 0,
                        "name": "add",
                        "arguments": {"a": 17, "b": 25},  # already a dict
                    },
                    # NOTE: no "type" key
                }
            ]
        }
        calls = parse_tool_calls_from_openai(message)
        assert calls == [ToolCall(id="call_eh7yrv0u", name="add", arguments={"a": 17, "b": 25})]

    def test_anthropic_parses_tool_use_blocks(self) -> None:
        blocks = [
            {"type": "text", "text": "I'll use a tool."},
            {"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {"key": "x"}},
        ]
        calls = parse_tool_calls_from_anthropic(blocks)
        assert calls == [ToolCall(id="tu_1", name="lookup", arguments={"key": "x"})]

    def test_google_parses_function_call_parts(self) -> None:
        parts = [
            {"text": "I'll call a function."},
            {"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}},
        ]
        calls = parse_tool_calls_from_google(parts)
        assert calls == [ToolCall(id="google-1", name="add", arguments={"a": 1, "b": 2})]


class TestDispatchToolCalls:
    async def test_dispatch_runs_in_parallel(self) -> None:
        registry = _registry_with(_AddTool(), _LookupTool())
        calls = [
            ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
            ToolCall(id="c2", name="lookup", arguments={"key": "alpha"}),
        ]
        results = await dispatch_tool_calls(calls, registry)
        assert len(results) == 2
        # Order preserved.
        assert results[0][0].id == "c1"
        assert results[0][1] == "5"
        assert results[0][2] is True
        assert results[1][1] == "value-for-alpha"

    async def test_unknown_tool_yields_failure(self) -> None:
        registry = _registry_with(_AddTool())
        calls = [ToolCall(id="c1", name="missing", arguments={})]
        results = await dispatch_tool_calls(calls, registry)
        assert results[0][2] is False
        assert "not registered" in results[0][1]

    async def test_tool_exception_yields_failure(self) -> None:
        class _Bad(BaseTool):
            name = "bad"

            async def execute(self, **kwargs: Any) -> Any:
                raise RuntimeError("kaboom")

        registry = _registry_with(_Bad())
        calls = [ToolCall(id="c1", name="bad", arguments={})]
        results = await dispatch_tool_calls(calls, registry)
        assert results[0][2] is False
        assert "kaboom" in results[0][1]


class TestResultMessageBuilders:
    def test_openai_role_tool_per_call(self) -> None:
        c = ToolCall(id="c1", name="add", arguments={})
        out = build_tool_result_messages("openai", [(c, "5", True)])
        assert out == [{"role": "tool", "tool_call_id": "c1", "content": "5"}]

    def test_anthropic_single_user_turn_with_tool_result_blocks(self) -> None:
        c = ToolCall(id="tu_1", name="lookup", arguments={})
        out = build_tool_result_messages("anthropic", [(c, "ok", True)])
        assert len(out) == 1
        assert out[0]["role"] == "user"
        block = out[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tu_1"
        assert block["content"] == "ok"
        assert block["is_error"] is False

    def test_assistant_message_for_anthropic_includes_tool_use_blocks(self) -> None:
        c = ToolCall(id="tu_1", name="lookup", arguments={"key": "x"})
        msg = build_assistant_message_with_tool_calls("anthropic", "Thinking...", [c])
        assert msg["role"] == "assistant"
        assert msg["content"][0] == {"type": "text", "text": "Thinking..."}
        assert msg["content"][1] == {
            "type": "tool_use",
            "id": "tu_1",
            "name": "lookup",
            "input": {"key": "x"},
        }


class TestOpenAIToolCallingWire:
    @respx.mock
    async def test_complete_returns_tool_calls_and_translates_tools(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"a": 2, "b": 3}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
            )
        )
        provider = OpenAIProvider(api_key="k")
        tools = [
            ToolDefinition(
                name="add",
                description="add two ints",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                },
            )
        ]
        r = await provider.complete(
            messages=[{"role": "user", "content": "what is 2+3?"}],
            model="gpt-4o-mini",
            agentloom_tools=tools,
        )
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["function"]["name"] == "add"
        assert r.tool_calls == [ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
        assert r.finish_reason == "tool_calls"
        await provider.close()


class TestAnthropicToolCallingWire:
    @respx.mock
    async def test_translates_tools_and_parses_tool_use_blocks(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "lookup",
                            "input": {"key": "alpha"},
                        }
                    ],
                    "model": "claude-haiku-4-5-20251001",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            )
        )
        provider = AnthropicProvider(api_key="k")
        tools = [ToolDefinition(name="lookup", description="d", parameters={"type": "object"})]
        r = await provider.complete(
            messages=[{"role": "user", "content": "x"}],
            model="claude-haiku-4-5-20251001",
            agentloom_tools=tools,
            agentloom_tool_choice="required",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["name"] == "lookup"
        assert body["tool_choice"] == {"type": "any"}
        assert r.tool_calls == [ToolCall(id="tu_1", name="lookup", arguments={"key": "alpha"})]
        await provider.close()


class TestGoogleToolChoiceDictSelectsSpecificFunction:
    """``tool_choice={"name": "fn"}`` must translate to Gemini's
    ANY-mode + ``allowedFunctionNames`` rather than silently falling
    through to AUTO."""

    @respx.mock
    async def test_specific_function_selection(self) -> None:
        from agentloom.providers.google import GoogleProvider

        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gemini-2.5-flash",
                    "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                },
            )
        )
        provider = GoogleProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "ping"}],
            model="gemini-2.5-flash",
            agentloom_tools=[
                ToolDefinition(name="lookup", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice={"name": "lookup"},
        )
        body = json.loads(route.calls[0].request.content)
        cfg = body["toolConfig"]["functionCallingConfig"]
        assert cfg["mode"] == "ANY"
        assert cfg["allowedFunctionNames"] == ["lookup"]
        await provider._client.aclose()


class TestDispatchObserverHook:
    """``dispatch_tool_calls`` calls ``observer.on_tool_call`` per call so
    each dispatch lands on the trace as a child span tagged by tool +
    success status."""

    async def test_observer_hook_fires_with_hashes_and_duration(self) -> None:
        from unittest.mock import MagicMock

        observer = MagicMock()
        registry = _registry_with(_AddTool())
        calls = [ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
        await dispatch_tool_calls(calls, registry, observer=observer, step_id="s1")

        observer.on_tool_call.assert_called_once()
        kwargs = observer.on_tool_call.call_args.kwargs
        assert kwargs["step_id"] == "s1"
        assert kwargs["call_id"] == "c1"
        assert kwargs["tool_name"] == "add"
        assert kwargs["success"] is True
        # Hashes are non-empty hex strings — args/result never logged raw.
        assert len(kwargs["args_hash"]) == 16
        assert len(kwargs["result_hash"]) == 16
        assert kwargs["duration_ms"] >= 0.0

    async def test_observer_hook_fires_with_failure_on_unknown_tool(self) -> None:
        from unittest.mock import MagicMock

        observer = MagicMock()
        registry = _registry_with(_AddTool())
        calls = [ToolCall(id="c1", name="missing", arguments={})]
        await dispatch_tool_calls(calls, registry, observer=observer, step_id="s1")

        observer.on_tool_call.assert_called_once()
        assert observer.on_tool_call.call_args.kwargs["success"] is False


class TestToolChoiceWireForwarding:
    """End-to-end checks that ``tool_choice`` reaches the provider's wire
    payload correctly — Copilot review surfaced silent drops on OpenAI
    and silent fall-through to AUTO on Google."""

    @respx.mock
    async def test_openai_forwards_none_to_disable_tools(self) -> None:
        # ``tool_choice="none"`` MUST reach the wire. With ``tools=`` set
        # OpenAI defaults to ``"auto"``, so dropping the field would mean
        # the user's explicit "no tools this turn" gets ignored.
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )
        provider = OpenAIProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "ping"}],
            model="gpt-4o-mini",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="none",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["tool_choice"] == "none"
        await provider.close()


class TestGoogleCompleteStringChoiceMapsToMode:
    """``complete`` with a string ``tool_choice`` hits the ``else`` branch
    that maps ``"auto"``/``"required"``/``"none"`` to Gemini's mode enum.
    The dict-form ``{"name": ...}`` test exercises the other branch."""

    @respx.mock
    async def test_required_maps_to_any_mode(self) -> None:
        from agentloom.providers.google import GoogleProvider

        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gemini-2.5-flash",
                    "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                },
            )
        )
        provider = GoogleProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "ping"}],
            model="gemini-2.5-flash",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="required",
        )
        body = json.loads(route.calls[0].request.content)
        cfg = body["toolConfig"]["functionCallingConfig"]
        assert cfg["mode"] == "ANY"
        assert "allowedFunctionNames" not in cfg
        await provider._client.aclose()

    @respx.mock
    async def test_none_maps_to_none_mode(self) -> None:
        from agentloom.providers.google import GoogleProvider

        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gemini-2.5-flash",
                    "candidates": [{"content": {"parts": [{"text": "5+5 is 10"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                },
            )
        )
        provider = GoogleProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "what is 5+5?"}],
            model="gemini-2.5-flash",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="none",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
        await provider._client.aclose()


class TestGoogleAssistantMessagePreservesText:
    """Gemini can return text + functionCall in the same turn. The
    assistant-message synthesizer must include the text part so iteration
    2+ replays the model's full prior turn (Copilot finding)."""

    def test_text_part_kept_alongside_function_calls(self) -> None:
        from agentloom.steps._tools import build_assistant_message_with_tool_calls

        msg = build_assistant_message_with_tool_calls(
            "google",
            "Let me check that.",
            [ToolCall(id="g_1", name="lookup", arguments={"key": "x"})],
        )
        # Text part comes first, then the functionCall part.
        assert msg["role"] == "model"
        assert msg["parts"][0] == {"text": "Let me check that."}
        assert msg["parts"][1] == {"functionCall": {"name": "lookup", "args": {"key": "x"}}}

    def test_no_text_part_when_content_empty(self) -> None:
        # Backward-compat: empty content yields the same shape as before.
        from agentloom.steps._tools import build_assistant_message_with_tool_calls

        msg = build_assistant_message_with_tool_calls(
            "google",
            "",
            [ToolCall(id="g_1", name="lookup", arguments={})],
        )
        assert msg["parts"] == [{"functionCall": {"name": "lookup", "args": {}}}]


class TestStreamFallbackPropagatesToolFields:
    """``BaseProvider.stream()`` default fallback wraps ``complete()``;
    the rolled-up ``StreamResponse.to_provider_response()`` must include
    ``tool_calls`` and ``reasoning_content`` so callers don't lose them
    when streaming through a provider that hasn't overridden ``stream()``."""

    async def test_fallback_copies_tool_calls_and_reasoning(self) -> None:
        from agentloom.providers.base import BaseProvider, ProviderResponse

        captured = ProviderResponse(
            content="hi",
            model="m",
            provider="p",
            reasoning_content="thinking...",
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1})],
        )

        class _Stub(BaseProvider):
            name = "p"

            async def complete(self, **_kw: Any) -> ProviderResponse:
                return captured

            def supports_model(self, model: str) -> bool:
                return True

        sr = await _Stub().stream(messages=[], model="m")
        async for _ in sr:
            pass
        out = sr.to_provider_response()
        assert out.reasoning_content == "thinking..."
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].id == "c1"


class TestToolChoiceValidation:
    """``StepDefinition.tool_choice`` is a constrained union — invalid
    YAML / Python values fail at model construction rather than silently
    coercing to AUTO at the wire layer."""

    def test_valid_string_choices(self) -> None:
        for value in ("auto", "required", "none"):
            step = StepDefinition(id="s", type=StepType.LLM_CALL, prompt="x", tool_choice=value)
            assert step.tool_choice == value

    def test_valid_dict_choice_coerces_to_typed_model(self) -> None:
        from agentloom.core.models import ToolChoiceByName

        step = StepDefinition(
            id="s", type=StepType.LLM_CALL, prompt="x", tool_choice={"name": "lookup"}
        )
        assert isinstance(step.tool_choice, ToolChoiceByName)
        assert step.tool_choice.name == "lookup"

    def test_invalid_string_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StepDefinition(
                id="s",
                type=StepType.LLM_CALL,
                prompt="x",
                tool_choice="anything-else",
            )

    def test_max_tool_iterations_zero_rejected(self) -> None:
        # ``max_tool_iterations >= 1`` invariant guards the loop body
        # against ``response`` being unset on exit.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StepDefinition(
                id="s",
                type=StepType.LLM_CALL,
                prompt="x",
                max_tool_iterations=0,
            )


class TestDispatchObserverHookFailure:
    """A flaky observability backend (exporter misconfig, transient OTel
    failure) must not abort the dispatch task group and break the step."""

    async def test_observer_hook_exception_swallowed(self) -> None:
        registry = _registry_with(_AddTool())

        class _BrokenObserver:
            def on_tool_call(self, **_kw: Any) -> None:
                raise RuntimeError("OTel exporter is down")

        calls = [ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})]
        # Must NOT raise — dispatch succeeds, hook failure is debug-logged.
        results = await dispatch_tool_calls(
            calls, registry, observer=_BrokenObserver(), step_id="s1"
        )
        assert results[0][2] is True  # tool succeeded despite hook failure
        assert results[0][1] == "3"


class TestStreamEvents:
    """Typed stream events surface. Adapters that haven't
    wired the typed iterator fall back to wrapping plain text chunks as
    ``TextDelta`` events, terminated with ``StreamDone``."""

    async def test_event_iterator_when_wired_takes_precedence(self) -> None:
        # Adapters that emit typed events natively register a separate
        # iterator; the default text-wrapping path is bypassed entirely.
        from agentloom.providers.base import (
            StreamDone,
            StreamEvent,
            StreamResponse,
            ToolCallComplete,
        )

        async def _events() -> Any:
            yield ToolCallComplete(tool_call=ToolCall(id="c1", name="add", arguments={"a": 1}))
            yield StreamDone(finish_reason="tool_calls")

        sr = StreamResponse(model="m", provider="p")
        sr._set_event_iterator(_events())

        events: list[StreamEvent] = [evt async for evt in sr.events()]
        assert len(events) == 2
        assert isinstance(events[0], ToolCallComplete)
        assert events[0].tool_call.name == "add"

    async def test_default_events_wrap_text_chunks(self) -> None:
        from agentloom.providers.base import (
            StreamDone,
            StreamResponse,
            TextDelta,
        )

        async def _chunks() -> Any:
            yield "hello "
            yield "world"

        sr = StreamResponse(model="m", provider="p")
        sr._set_iterator(_chunks())
        sr.finish_reason = "stop"

        events = [evt async for evt in sr.events()]
        # Two TextDelta events + one StreamDone — the default wrapper
        # provides the surface even when the adapter doesn't emit typed
        # events natively.
        assert [type(e).__name__ for e in events] == ["TextDelta", "TextDelta", "StreamDone"]
        assert isinstance(events[0], TextDelta)
        assert events[0].chunk == "hello "
        assert isinstance(events[-1], StreamDone)
        assert events[-1].finish_reason == "stop"


class TestProviderFormatPreservesToolMessages:
    """Regression for the critical bug where providers' ``_format_messages``
    silently dropped ``tool_calls`` / ``tool_call_id`` / ``parts`` keys on
    tool-loop iteration 2+, breaking real (non-mock) tool calling."""

    def test_openai_passes_through_assistant_with_tool_calls(self) -> None:
        msgs = [
            {"role": "user", "content": "what is 2+3?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "5"},
        ]
        out = OpenAIProvider._format_messages(msgs)
        # The two tool-loop messages must round-trip with their wire keys
        # intact — without this, OpenAI 400s on iteration 2 because the
        # assistant message has no tool_calls and the tool message has no
        # tool_call_id to anchor.
        assert out[1]["tool_calls"][0]["function"]["name"] == "add"
        assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "5"}

    def test_ollama_passes_through_tool_loop_messages(self) -> None:
        from agentloom.providers.ollama import OllamaProvider

        msgs = [
            {"role": "user", "content": "ping"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        out = OllamaProvider._format_messages(msgs)
        assert "tool_calls" in out[1]
        assert out[2]["tool_call_id"] == "c1"

    def test_google_passes_through_function_call_and_response_parts(self) -> None:
        from agentloom.providers.google import GoogleProvider

        msgs = [
            {"role": "user", "content": "ping"},
            # Assistant tool-call decision (Gemini wire shape).
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}}],
            },
            # Tool result (Gemini wire shape).
            {
                "role": "function",
                "parts": [{"functionResponse": {"name": "add", "response": {"result": "3"}}}],
            },
        ]
        _system, contents = GoogleProvider._format_messages(msgs)
        # Both ``parts``-based messages must round-trip unchanged so Gemini
        # sees the prior call + result on iteration 2; previously the
        # ``content``-based reformatter dropped ``parts`` entirely.
        assert contents[1]["parts"][0]["functionCall"]["name"] == "add"
        assert contents[2]["role"] == "function"
        assert contents[2]["parts"][0]["functionResponse"]["response"] == {"result": "3"}


class TestLLMStepToolLoop:
    @respx.mock
    async def test_step_dispatches_and_iterates(self) -> None:
        # Iteration 1: model asks to call `add`.
        # Iteration 2: model responds with the final answer.
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "c1",
                                            "type": "function",
                                            "function": {
                                                "name": "add",
                                                "arguments": '{"a": 2, "b": 3}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "The answer is 5."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
                    },
                ),
            ]
        )
        workflow = WorkflowDefinition(
            name="agent",
            config=WorkflowConfig(provider="openai", model="gpt-4o-mini"),
            state={},
            steps=[
                StepDefinition(
                    id="ask",
                    type=StepType.LLM_CALL,
                    prompt="What is 2+3?",
                    tools=[
                        ToolDefinition(
                            name="add",
                            description="add two integers",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                                "required": ["a", "b"],
                            },
                        )
                    ],
                    output="answer",
                )
            ],
        )
        gateway = ProviderGateway()
        gateway.register(OpenAIProvider(api_key="k"))
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=gateway,
            tool_registry=_registry_with(_AddTool()),
        )
        result = await engine.run()
        await gateway.close()

        assert result.status == WorkflowStatus.SUCCESS
        sr = result.step_results["ask"]
        assert sr.status == StepStatus.SUCCESS
        # Cost / tokens accumulated across both iterations.
        assert sr.token_usage.prompt_tokens == 5 + 12
        assert sr.token_usage.completion_tokens == 9 + 6
        assert result.final_state["answer"] == "The answer is 5."

    @respx.mock
    async def test_step_respects_max_tool_iterations(self) -> None:
        # Model never stops asking for the tool; loop must cap at 2.
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"a": 1, "b": 1}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            )
        )
        workflow = WorkflowDefinition(
            name="agent-loop",
            config=WorkflowConfig(provider="openai", model="gpt-4o-mini"),
            state={},
            steps=[
                StepDefinition(
                    id="ask",
                    type=StepType.LLM_CALL,
                    prompt="loop",
                    tools=[
                        ToolDefinition(
                            name="add",
                            description="d",
                            parameters={"type": "object"},
                        )
                    ],
                    max_tool_iterations=2,
                    output="answer",
                )
            ],
        )
        gateway = ProviderGateway()
        gateway.register(OpenAIProvider(api_key="k"))
        engine = WorkflowEngine(
            workflow=workflow,
            provider_gateway=gateway,
            tool_registry=_registry_with(_AddTool()),
        )
        result = await engine.run()
        await gateway.close()

        sr = result.step_results["ask"]
        assert sr.status == StepStatus.SUCCESS
        # Cap surfaced via finish_reason on the prompt metadata.
        assert sr.prompt_metadata is not None
        assert sr.prompt_metadata.finish_reason == "max_tool_iterations"

    async def test_tool_dispatch_without_registry_raises(self) -> None:
        # When a step declares tools but no registry is attached, the loop
        # raises StepError surfaced as FAILED.
        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "c1",
                                            "type": "function",
                                            "function": {
                                                "name": "add",
                                                "arguments": "{}",
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    },
                )
            )
            workflow = WorkflowDefinition(
                name="no-registry",
                config=WorkflowConfig(provider="openai", model="gpt-4o-mini"),
                state={},
                steps=[
                    StepDefinition(
                        id="ask",
                        type=StepType.LLM_CALL,
                        prompt="hi",
                        tools=[
                            ToolDefinition(
                                name="add", description="d", parameters={"type": "object"}
                            )
                        ],
                        max_tool_iterations=1,
                        retry={"max_retries": 0},
                    )
                ],
            )
            gateway = ProviderGateway()
            gateway.register(OpenAIProvider(api_key="k"))
            engine = WorkflowEngine(
                workflow=workflow,
                provider_gateway=gateway,
                # tool_registry intentionally omitted
            )
            result = await engine.run()
            await gateway.close()

            sr = result.step_results["ask"]
            assert sr.status == StepStatus.FAILED
            assert "tool registry" in (sr.error or "").lower()


class TestToolChoiceTranslators:
    """Direct coverage for the per-provider tool_choice translators —
    each branch (string modes, ``{"name": ...}``, fallback) needs to
    round-trip through the helper so silent regressions surface."""

    def test_openai_translates_name_dict_to_function_envelope(self) -> None:
        from agentloom.steps._tools import translate_tool_choice_for_openai

        out = translate_tool_choice_for_openai({"name": "lookup"})
        assert out == {"type": "function", "function": {"name": "lookup"}}

    def test_openai_passes_through_string_modes(self) -> None:
        from agentloom.steps._tools import translate_tool_choice_for_openai

        assert translate_tool_choice_for_openai("auto") == "auto"
        assert translate_tool_choice_for_openai("required") == "required"

    def test_anthropic_string_modes(self) -> None:
        from agentloom.steps._tools import translate_tool_choice_for_anthropic

        assert translate_tool_choice_for_anthropic("auto") == {"type": "auto"}
        assert translate_tool_choice_for_anthropic("required") == {"type": "any"}
        # ``"none"`` becomes ``None`` so the adapter omits the field
        # entirely — Anthropic has no explicit "disable tools" mode.
        assert translate_tool_choice_for_anthropic("none") is None

    def test_anthropic_name_dict_pins_specific_tool(self) -> None:
        from agentloom.steps._tools import translate_tool_choice_for_anthropic

        out = translate_tool_choice_for_anthropic({"name": "lookup"})
        assert out == {"type": "tool", "name": "lookup"}

    def test_anthropic_unknown_falls_back_to_auto(self) -> None:
        # An unexpected value (e.g. a stray dict without ``name``) must
        # not crash the request — fall back to AUTO so the call still
        # round-trips and the model decides whether to invoke a tool.
        from agentloom.steps._tools import translate_tool_choice_for_anthropic

        assert translate_tool_choice_for_anthropic({"unknown": "x"}) == {"type": "auto"}

    def test_ollama_translator_returns_openai_shape(self) -> None:
        # Ollama uses the OpenAI wire shape; the translator is a thin
        # delegate but the indirection needs at least one direct call.
        from agentloom.steps._tools import translate_tools_for_ollama

        out = translate_tools_for_ollama(
            [ToolDefinition(name="add", description="d", parameters={"type": "object"})]
        )
        assert out[0]["function"]["name"] == "add"


class TestParseToolCallsEdgeCases:
    def test_invalid_json_arguments_yield_empty_args(self) -> None:
        # A model that emits malformed JSON for ``arguments`` must not
        # crash the parser — fall back to ``{}`` and let dispatch
        # surface the missing-arg error to the model on the next turn.
        message = {
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{not json"},
                }
            ]
        }
        calls = parse_tool_calls_from_openai(message)
        assert calls == [ToolCall(id="c1", name="add", arguments={})]


class TestAssistantAndResultBuildersOllamaGoogle:
    """Cover the Ollama assistant-message branch and the Google /
    Ollama tool-result branches that the existing suite didn't reach."""

    def test_ollama_assistant_message_uses_dict_args(self) -> None:
        c = ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})
        msg = build_assistant_message_with_tool_calls("ollama", "thinking", [c])
        # Ollama wire shape: ``arguments`` is a dict (not a JSON string)
        # and the entry has no ``id`` / ``type`` keys.
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "add"
        assert tc["function"]["arguments"] == {"a": 1, "b": 2}
        assert "id" not in tc
        assert "type" not in tc

    def test_google_tool_result_message_shapes_function_response(self) -> None:
        c = ToolCall(id="g_1", name="lookup", arguments={"key": "x"})
        out = build_tool_result_messages("google", [(c, "ok", True)])
        assert out[0]["role"] == "function"
        fr = out[0]["parts"][0]["functionResponse"]
        assert fr["name"] == "lookup"
        assert fr["response"] == {"result": "ok"}

    def test_google_tool_result_message_surfaces_error_on_failure(self) -> None:
        # Gemini's ``functionResponse`` uses ``response: {error: ...}``
        # to signal failure; the wire branch must hit on ``success=False``.
        c = ToolCall(id="g_1", name="lookup", arguments={})
        out = build_tool_result_messages("google", [(c, "boom", False)])
        fr = out[0]["parts"][0]["functionResponse"]
        assert fr["response"] == {"error": "boom"}

    def test_ollama_tool_result_message_keyed_by_name(self) -> None:
        # Ollama rejects requests that include the OpenAI ``tool_call_id``
        # field — the result message must key by tool ``name`` instead.
        c = ToolCall(id="ignored", name="add", arguments={})
        out = build_tool_result_messages("ollama", [(c, "5", True)])
        assert out == [{"role": "tool", "name": "add", "content": "5"}]


class TestMetricsRecordToolCallDisabled:
    """When the metrics manager is disabled (no extras installed / no
    backend configured) ``record_tool_call`` must be a noop — guards the
    early-return that hot paths rely on."""

    def test_disabled_manager_short_circuits(self) -> None:
        from agentloom.observability.metrics import MetricsManager

        mgr = MetricsManager.__new__(MetricsManager)
        mgr._enabled = False  # type: ignore[attr-defined]
        # Must not touch ``_tool_call_counter`` / ``_tool_call_histogram``;
        # those attributes don't exist on the bare instance, so any code
        # path that forgets the guard would AttributeError here.
        mgr.record_tool_call("add", success=True, duration_s=0.01)


class TestMockProviderToolCallingReplay:
    """``MockProvider`` hydrates ``tool_calls`` from the recording and
    advances a per-step cursor so a single step replays a multi-turn
    tool-iteration loop."""

    async def test_list_form_replays_turns_in_order_and_clamps(self) -> None:
        from agentloom.providers.mock import MockProvider

        provider = MockProvider()
        # Two-turn loop: first call drives a tool dispatch, second emits
        # the final answer. A third call must replay the last turn
        # (clamped) rather than raise mid-step.
        provider._responses["ask"] = [  # type: ignore[assignment]
            {
                "content": "",
                "model": "mock",
                "tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}],
                "finish_reason": "tool_calls",
            },
            {
                "content": "the answer is 3",
                "model": "mock",
                "finish_reason": "stop",
            },
        ]
        r1 = await provider.complete(messages=[], model="mock", step_id="ask")
        assert r1.finish_reason == "tool_calls"
        assert r1.tool_calls[0].name == "add"
        assert r1.tool_calls[0].arguments == {"a": 1, "b": 2}

        r2 = await provider.complete(messages=[], model="mock", step_id="ask")
        assert r2.content == "the answer is 3"
        assert r2.tool_calls == []

        # Cursor clamps at the last turn — excess iterations replay the
        # final answer rather than raising or returning ``None``.
        r3 = await provider.complete(messages=[], model="mock", step_id="ask")
        assert r3.content == "the answer is 3"

    async def test_empty_list_falls_through_to_default_response(self) -> None:
        # Defensive: an empty turn list shouldn't crash — the lookup
        # returns ``None`` and ``complete`` falls back to the configured
        # default response rather than raising.
        from agentloom.providers.mock import MockProvider

        provider = MockProvider(default_response="fallback")
        provider._responses["ask"] = []  # type: ignore[assignment]
        r = await provider.complete(messages=[], model="mock", step_id="ask")
        assert r.content == "fallback"


class TestStreamForwardsToolKwargs:
    """Streaming + tool calling was silently broken: each provider's
    ``stream()`` called ``validate_extra_kwargs`` before popping the
    ``agentloom_tools`` / ``agentloom_tool_choice`` keys, so the request
    raised ``TypeError`` before the wire call. Regression for the bug
    surfaced by the real-provider validation matrix on 2026-05-14."""

    @respx.mock
    async def test_openai_stream_accepts_tool_kwargs_and_forwards(self) -> None:
        # Empty SSE response is fine — we only assert that the request
        # body carries ``tools`` + ``tool_choice`` and that the call
        # didn't crash on extras validation.
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, text="data: [DONE]\n\n")
        )
        provider = OpenAIProvider(api_key="k")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="required",
        )
        async for _ in sr:
            pass
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["function"]["name"] == "add"
        assert body["tool_choice"] == "required"
        await provider.close()

    @respx.mock
    async def test_openai_stream_forwards_none_choice_explicitly(self) -> None:
        # ``"none"`` must reach the wire on streaming too — otherwise the
        # default ``"auto"`` would let the model invoke tools.
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, text="data: [DONE]\n\n")
        )
        provider = OpenAIProvider(api_key="k")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="none",
        )
        async for _ in sr:
            pass
        body = json.loads(route.calls[0].request.content)
        assert body["tool_choice"] == "none"
        await provider.close()

    @respx.mock
    async def test_anthropic_stream_accepts_tool_kwargs(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, text="event: message_stop\ndata: {}\n\n")
        )
        provider = AnthropicProvider(api_key="k")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-haiku-4-5-20251001",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="required",
        )
        async for _ in sr:
            pass
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["name"] == "add"
        assert body["tool_choice"] == {"type": "any"}
        await provider.close()

    @respx.mock
    async def test_google_stream_accepts_tool_kwargs(self) -> None:
        from agentloom.providers.google import GoogleProvider

        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:streamGenerateContent?alt=sse&key=k"
        ).mock(return_value=httpx.Response(200, text="data: {}\n\n"))
        provider = GoogleProvider(api_key="k")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice={"name": "add"},
        )
        async for _ in sr:
            pass
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["function_declarations"][0]["name"] == "add"
        cfg = body["toolConfig"]["functionCallingConfig"]
        assert cfg["mode"] == "ANY"
        assert cfg["allowedFunctionNames"] == ["add"]
        await provider._client.aclose()

    @respx.mock
    async def test_google_stream_with_string_choice_hits_mode_lookup(self) -> None:
        # The ``{"name": "fn"}`` test exercises the dict branch; this
        # exercises the ``else`` branch that maps plain string choices
        # to Gemini's mode enum.
        from agentloom.providers.google import GoogleProvider

        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:streamGenerateContent?alt=sse&key=k"
        ).mock(return_value=httpx.Response(200, text="data: {}\n\n"))
        provider = GoogleProvider(api_key="k")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="required",
        )
        async for _ in sr:
            pass
        body = json.loads(route.calls[0].request.content)
        # ``"required"`` maps to Gemini's ANY mode (no allowedFunctionNames
        # pin since no specific function was named).
        cfg = body["toolConfig"]["functionCallingConfig"]
        assert cfg["mode"] == "ANY"
        assert "allowedFunctionNames" not in cfg
        await provider._client.aclose()

    @respx.mock
    async def test_ollama_stream_accepts_tool_kwargs(self) -> None:
        from agentloom.providers.ollama import OllamaProvider

        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text='{"done": true}\n')
        )
        provider = OllamaProvider()
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1:8b",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="required",  # silently ignored at wire level
        )
        async for _ in sr:
            pass
        body = json.loads(route.calls[0].request.content)
        # Ollama uses the OpenAI tool shape on the wire.
        assert body["tools"][0]["function"]["name"] == "add"
        await provider._client.aclose()


class TestOllamaCompleteForwardsToolKwargs:
    """``OllamaProvider.complete`` translates ``agentloom_tools`` to the
    OpenAI-compatible wire shape and logs a debug message when
    ``tool_choice`` is set (Ollama ignores it model-side). The bypass
    + log path was uncovered until now."""

    @respx.mock
    async def test_complete_forwards_tools_and_logs_choice(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from agentloom.providers.ollama import OllamaProvider

        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "ok"},
                    "model": "llama3.1:8b",
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                },
            )
        )
        provider = OllamaProvider()
        with caplog.at_level(logging.DEBUG, logger="agentloom.providers.ollama"):
            await provider.complete(
                messages=[{"role": "user", "content": "ping"}],
                model="llama3.1:8b",
                agentloom_tools=[
                    ToolDefinition(name="add", description="d", parameters={"type": "object"})
                ],
                # Non-``auto`` triggers the debug-log branch surfacing the
                # silent drop so users can debug "why doesn't my tool fire".
                agentloom_tool_choice="required",
            )
        body = json.loads(route.calls[0].request.content)
        assert body["tools"][0]["function"]["name"] == "add"
        assert any("Ollama ignores tool_choice" in rec.message for rec in caplog.records)
        await provider._client.aclose()


class TestAnthropicFormatPassesDictBlocksVerbatim:
    """When an assistant turn carries pre-formed Anthropic blocks
    (``tool_use`` / ``tool_result`` dicts), the formatter must pass them
    through verbatim — without this, iteration 2+ of the tool loop loses
    the prior tool-call context."""

    def test_dict_blocks_pass_through(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "lookup",
                        "input": {"key": "x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "value-x",
                    },
                ],
            },
        ]
        _system, formatted = AnthropicProvider._format_messages(msgs)
        # Both list-content messages must round-trip with their blocks
        # intact; verbatim passthrough is what makes iteration 2+ work.
        assert formatted[0]["content"][0] == {"type": "text", "text": "Let me check."}
        assert formatted[0]["content"][1]["type"] == "tool_use"
        assert formatted[0]["content"][1]["id"] == "tu_1"
        assert formatted[1]["content"][0]["type"] == "tool_result"
        assert formatted[1]["content"][0]["tool_use_id"] == "tu_1"


class TestLLMStepStreamFailurePaths:
    """``LLMCallStep`` must surface streaming failures as a FAILED
    ``StepResult`` rather than letting the exception escape — both the
    gateway crash before any chunk and an iteration error mid-stream
    were uncovered until now."""

    async def test_stream_raises_before_first_chunk_yields_failed_step(self) -> None:
        from agentloom.core.engine import WorkflowEngine
        from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse

        class _BoomProvider(BaseProvider):
            name = "boom"

            async def complete(self, **_kw: Any) -> ProviderResponse:  # pragma: no cover
                raise NotImplementedError

            async def stream(self, **_kw: Any) -> StreamResponse:
                raise RuntimeError("gateway is down")

            def supports_model(self, _m: str) -> bool:
                return True

        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        gw.register(_BoomProvider(), models=["boom-model"])
        wf = WorkflowDefinition(
            name="stream-boom",
            config=WorkflowConfig(provider="boom", model="boom-model"),
            steps=[
                StepDefinition(
                    id="ask",
                    type=StepType.LLM_CALL,
                    prompt="ping",
                    stream=True,
                    retry={"max_retries": 0},
                )
            ],
        )
        result = await WorkflowEngine(workflow=wf, provider_gateway=gw).run()
        await gw.close()
        sr = result.step_results["ask"]
        assert sr.status == StepStatus.FAILED
        assert "gateway is down" in (sr.error or "")

    async def test_stream_chunk_callback_failure_disables_callback(self) -> None:
        # When ``on_stream_chunk`` raises, the step must keep iterating
        # the stream (the callback is best-effort observability), then
        # disable the callback for subsequent chunks so we don't log on
        # every yield.
        from agentloom.core.engine import WorkflowEngine
        from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse

        class _TextStream(BaseProvider):
            name = "stream-test"

            async def complete(self, **_kw: Any) -> ProviderResponse:  # pragma: no cover
                raise NotImplementedError

            async def stream(self, **_kw: Any) -> StreamResponse:
                sr = StreamResponse(model="m", provider=self.name)

                async def _gen():  # type: ignore[no-untyped-def]
                    yield "hello "
                    yield "world"

                sr._set_iterator(_gen())
                sr.finish_reason = "stop"
                return sr

            def supports_model(self, _m: str) -> bool:
                return True

        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        gw.register(_TextStream(), models=["m"])
        seen: list[str] = []

        def _cb(step_id: str, chunk: str) -> None:
            seen.append(chunk)
            raise RuntimeError("callback broke")

        wf = WorkflowDefinition(
            name="stream-cb-boom",
            config=WorkflowConfig(provider="stream-test", model="m"),
            steps=[
                StepDefinition(
                    id="ask",
                    type=StepType.LLM_CALL,
                    prompt="ping",
                    stream=True,
                    retry={"max_retries": 0},
                )
            ],
        )
        result = await WorkflowEngine(workflow=wf, provider_gateway=gw, on_stream_chunk=_cb).run()
        await gw.close()
        sr = result.step_results["ask"]
        # The step still succeeds — the callback failure is swallowed
        # and the callback is disabled after the first throw, so only
        # the first chunk was observed by the caller.
        assert sr.status == StepStatus.SUCCESS
        assert seen == ["hello "]


class TestLLMStepToolChoiceByNameNormalisation:
    """``StepDefinition.tool_choice = {"name": "X"}`` is coerced to a
    ``ToolChoiceByName`` Pydantic model at validation time, and the
    step layer normalises it back to ``{"name": "X"}`` before forwarding
    to the provider. The normaliser branch was uncovered until now."""

    @respx.mock
    async def test_typed_choice_round_trips_to_provider_dict(self) -> None:
        from agentloom.core.models import ToolChoiceByName

        # Two-turn loop: iter 1 returns a tool call; iter 2 the final answer.
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "c1",
                                            "type": "function",
                                            "function": {
                                                "name": "lookup",
                                                "arguments": '{"key": "x"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 4,
                            "total_tokens": 9,
                        },
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "value-for-x"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 3,
                            "total_tokens": 10,
                        },
                    },
                ),
            ]
        )

        from agentloom.providers.gateway import ProviderGateway
        from agentloom.tools.base import BaseTool
        from agentloom.tools.registry import ToolRegistry

        class _Lookup(BaseTool):
            name = "lookup"
            description = "Look up a key."
            parameters_schema = {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            }

            async def execute(self, **kw: Any) -> Any:
                return f"value-for-{kw['key']}"

        wf = WorkflowDefinition(
            name="typed-choice",
            config=WorkflowConfig(provider="openai", model="gpt-4o-mini"),
            steps=[
                StepDefinition(
                    id="ask",
                    type=StepType.LLM_CALL,
                    prompt="pick the right one",
                    tools=[
                        ToolDefinition(
                            name="lookup",
                            description="d",
                            parameters={"type": "object"},
                        )
                    ],
                    # Pydantic coerces dict → ``ToolChoiceByName``; the
                    # step layer must convert back to a plain dict so
                    # the provider translator sees the expected shape.
                    tool_choice=ToolChoiceByName(name="lookup"),
                    max_tool_iterations=3,
                    retry={"max_retries": 0},
                )
            ],
        )
        gw = ProviderGateway()
        gw.register(OpenAIProvider(api_key="k"))
        reg = ToolRegistry()
        reg.register(_Lookup())
        result = await WorkflowEngine(workflow=wf, provider_gateway=gw, tool_registry=reg).run()
        await gw.close()
        sr = result.step_results["ask"]
        assert sr.status == StepStatus.SUCCESS
        assert sr.output == "value-for-x"


class TestLLMStepRunToolLoopDefensiveGuard:
    """``_run_tool_loop`` raises ``StepError`` when invoked with no
    gateway. The engine layer guards against this earlier so the
    branch is defensive — exercise it directly so the line stays
    covered if the guard order ever changes."""

    async def test_no_gateway_raises_step_error(self) -> None:
        from unittest.mock import MagicMock

        from agentloom.exceptions import StepError
        from agentloom.steps.base import StepContext
        from agentloom.steps.llm_call import LLMCallStep

        step_def = StepDefinition(id="ask", type=StepType.LLM_CALL, prompt="x")
        ctx = StepContext(
            step_definition=step_def,
            state_manager=MagicMock(),
            provider_gateway=None,
        )
        step = LLMCallStep()
        with pytest.raises(StepError, match="No provider gateway configured"):
            await step._run_tool_loop(
                context=ctx,
                step=step_def,
                messages=[{"role": "user", "content": "x"}],
                model="m",
                provider_kwargs={},
            )


class TestAnthropicNoneSuppressesTools:
    """Anthropic has no native ``tool_choice="none"`` mode — the translator
    returns ``None`` so the field is omitted, but historically ``tools``
    still went on the wire, letting the model invoke them anyway. The fix
    suppresses ``tools`` from the payload entirely when choice is ``"none"``
    so user intent is honored across providers."""

    @respx.mock
    async def test_complete_omits_tools_when_choice_is_none(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "10"}],
                    "model": "claude-haiku-4-5-20251001",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            )
        )
        provider = AnthropicProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "what is 5+5?"}],
            model="claude-haiku-4-5-20251001",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="none",
        )
        body = json.loads(route.calls[0].request.content)
        # Neither key should reach the wire — without this, Claude sees the
        # tool definition and may decide to call it despite the user's
        # explicit "no tools this turn".
        assert "tools" not in body
        assert "tool_choice" not in body
        await provider.close()

    @respx.mock
    async def test_stream_omits_tools_when_choice_is_none(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, text="event: message_stop\ndata: {}\n\n")
        )
        provider = AnthropicProvider(api_key="k")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-haiku-4-5-20251001",
            agentloom_tools=[
                ToolDefinition(name="add", description="d", parameters={"type": "object"})
            ],
            agentloom_tool_choice="none",
        )
        async for _ in sr:
            pass
        body = json.loads(route.calls[0].request.content)
        assert "tools" not in body
        assert "tool_choice" not in body
        await provider.close()
        # Defensive: an empty turn list shouldn't crash — the lookup
        # returns ``None`` and ``complete`` falls back to the configured
        # default response rather than raising.
        from agentloom.providers.mock import MockProvider

        provider = MockProvider(default_response="fallback")
        provider._responses["ask"] = []  # type: ignore[assignment]
        r = await provider.complete(messages=[], model="mock", step_id="ask")
        assert r.content == "fallback"
