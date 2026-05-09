"""Tests for native tool calling across providers."""

from __future__ import annotations

import json
from typing import Any

import httpx
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
        # Only the second entry is a function — its function dict is empty
        # so name="" and args={}; we accept it (degenerate case).
        assert len(calls) == 1
        assert calls[0].name == ""

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
    success status (#116 observability spec)."""

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


class TestStreamEvents:
    """Typed stream events surface (issue #116). Adapters that haven't
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
