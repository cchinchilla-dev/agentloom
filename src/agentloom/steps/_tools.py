"""Helpers for the tool-calling flow (#116).

Provider adapters call ``translate_tools_for_provider`` when building a
request payload and ``parse_tool_calls_from_*`` after receiving a response.
The LLM step uses ``dispatch_tool_calls`` to run them in parallel via
the tool registry and ``build_tool_result_messages`` to feed the results
back to the model on the next turn.

Kept in one module so the wire-format translation lives in a single
place per provider — adding Cohere or Bedrock later is one section
here, not a hunt across four adapters.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anyio

from agentloom.core.models import ToolDefinition
from agentloom.providers.base import ToolCall

logger = logging.getLogger("agentloom.steps")


# ---------------------------------------------------------------------------
# Outbound: ToolDefinition -> provider wire format
# ---------------------------------------------------------------------------


def translate_tools_for_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """OpenAI shape: ``[{"type": "function", "function": {name, description, parameters}}]``."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def translate_tools_for_anthropic(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Anthropic shape: ``[{name, description, input_schema}]``."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


def translate_tools_for_google(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Google shape: ``[{"function_declarations": [{name, description, parameters}, ...]}]``.

    Google groups all functions under one ``Tool`` object; we put them all
    in a single declaration list since AgentLoom doesn't yet expose tool
    grouping (everything is a function).
    """
    return [
        {
            "function_declarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                }
                for t in tools
            ]
        }
    ]


def translate_tools_for_ollama(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Ollama uses the OpenAI shape on supported models."""
    return translate_tools_for_openai(tools)


def translate_tool_choice_for_openai(choice: Any) -> Any:
    """OpenAI ``tool_choice`` accepts ``"auto"`` / ``"required"`` / ``"none"``
    or ``{"type": "function", "function": {"name": "..."}}``."""
    if isinstance(choice, dict) and "name" in choice:
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice


def translate_tool_choice_for_anthropic(choice: Any) -> Any:
    """Anthropic uses ``{"type": "auto"}`` / ``"any"`` / ``"tool"``."""
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return None  # omit the field entirely
    if isinstance(choice, dict) and "name" in choice:
        return {"type": "tool", "name": choice["name"]}
    return {"type": "auto"}


# ---------------------------------------------------------------------------
# Inbound: provider response -> ToolCall list
# ---------------------------------------------------------------------------


def parse_tool_calls_from_openai(message: dict[str, Any]) -> list[ToolCall]:
    """Parse ``message.tool_calls`` from OpenAI-shaped responses.

    Handles both wire variants seen in the wild:

    * **OpenAI canonical**: ``[{id, type:"function", function:{name, arguments:"<json>"}}]``
      — ``arguments`` is a JSON-encoded string.
    * **Ollama / OpenAI-compatible relays**: ``[{id, function:{name, arguments:{...}}}]``
      — ``type`` may be omitted entirely, and ``arguments`` may already be a
      decoded dict. Treat the absence of ``type`` as ``"function"`` (the
      only call kind we currently dispatch); a non-function explicit
      ``type`` (e.g. ``"code_interpreter"``) is skipped.
    """
    raw_calls = message.get("tool_calls") or []
    calls: list[ToolCall] = []
    for entry in raw_calls:
        entry_type = entry.get("type", "function")
        if entry_type != "function":
            continue
        fn = entry.get("function", {})
        raw_args = fn.get("arguments")
        if isinstance(raw_args, dict):
            args: dict[str, Any] = raw_args
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}
        else:
            args = {}
        calls.append(ToolCall(id=entry.get("id", ""), name=fn.get("name", ""), arguments=args))
    return calls


def parse_tool_calls_from_anthropic(content_blocks: list[dict[str, Any]]) -> list[ToolCall]:
    """Anthropic returns ``content`` blocks; ``type=tool_use`` carries calls."""
    calls: list[ToolCall] = []
    for block in content_blocks:
        if block.get("type") != "tool_use":
            continue
        calls.append(
            ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input", {}) or {},
            )
        )
    return calls


def parse_tool_calls_from_google(content_parts: list[dict[str, Any]]) -> list[ToolCall]:
    """Google returns parts with ``functionCall: {name, args}``. Ids aren't
    provider-assigned so we synthesize them — the result message echoes
    the same name (no id round-trip needed)."""
    calls: list[ToolCall] = []
    for idx, part in enumerate(content_parts):
        fc = part.get("functionCall")
        if not fc:
            continue
        calls.append(
            ToolCall(
                id=f"google-{idx}",
                name=fc.get("name", ""),
                arguments=fc.get("args", {}) or {},
            )
        )
    return calls


# ---------------------------------------------------------------------------
# Tool dispatch (engine-side) and result-message synthesis
# ---------------------------------------------------------------------------


async def dispatch_tool_calls(
    calls: list[ToolCall],
    tool_registry: Any,
) -> list[tuple[ToolCall, str, bool]]:
    """Execute each call via the registry, in parallel.

    Returns a list of ``(call, result_text, success)`` tuples in the
    same order as ``calls``. Failed calls have ``success=False`` and a
    stringified exception in ``result_text`` so the model can recover
    on the next turn (the spec calls this "feed the failure back").
    """
    results: list[tuple[ToolCall, str, bool]] = [None] * len(calls)  # type: ignore[list-item]

    async def _run(idx: int, call: ToolCall) -> None:
        try:
            tool = tool_registry.get(call.name)
        except KeyError as e:
            results[idx] = (call, f"tool '{call.name}' not registered: {e}", False)
            return
        try:
            outcome = await tool.execute(**call.arguments)
            text = outcome if isinstance(outcome, str) else json.dumps(outcome, default=str)
            results[idx] = (call, text, True)
        except Exception as e:  # noqa: BLE001 — reported back to the model
            logger.warning("Tool '%s' failed: %s", call.name, e)
            results[idx] = (call, f"tool execution failed: {e}", False)

    async with anyio.create_task_group() as tg:
        for idx, call in enumerate(calls):
            tg.start_soon(_run, idx, call)

    return results


def build_assistant_message_with_tool_calls(
    provider: str, content: str, calls: list[ToolCall]
) -> dict[str, Any]:
    """Replay the assistant's tool-call decision as a message on the next turn.

    Required so the model sees its own previous decision — without this,
    the conversation looks like the model spontaneously received tool
    results without ever asking for them.
    """
    if provider == "anthropic":
        blocks: list[dict[str, Any]] = []
        if content:
            blocks.append({"type": "text", "text": content})
        for c in calls:
            blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
        return {"role": "assistant", "content": blocks}
    if provider == "google":
        return {
            "role": "model",
            "parts": [{"functionCall": {"name": c.name, "args": c.arguments}} for c in calls],
        }
    # OpenAI / Ollama. ``content`` must be a string (the AgentLoom-internal
    # message formatter doesn't handle None even though OpenAI's API does);
    # an empty string serializes as `""` which OpenAI accepts as "no text"
    # alongside tool_calls.
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in calls
        ],
    }


def build_tool_result_messages(
    provider: str, results: list[tuple[ToolCall, str, bool]]
) -> list[dict[str, Any]]:
    """Translate tool outcomes into the role/shape each provider expects."""
    if provider == "anthropic":
        # Single user turn with one tool_result block per call.
        blocks: list[dict[str, Any]] = []
        for call, text, success in results:
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": text,
                    "is_error": not success,
                }
            )
        return [{"role": "user", "content": blocks}]
    if provider == "google":
        return [
            {
                "role": "function",
                "parts": [
                    {
                        "functionResponse": {
                            "name": call.name,
                            "response": {"result": text} if success else {"error": text},
                        }
                    }
                    for call, text, success in results
                ],
            }
        ]
    # OpenAI / Ollama: one tool message per call, keyed by tool_call_id.
    return [{"role": "tool", "tool_call_id": call.id, "content": text} for call, text, _ in results]
