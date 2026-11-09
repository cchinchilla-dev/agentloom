"""Tool-calling helpers — wire-format translation, dispatch, and message synthesis."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from typing import Any

import anyio

from agentloom.core.inter_agent import calling, log_tools
from agentloom.core.models import ToolDefinition
from agentloom.exceptions import ToolNotFoundError
from agentloom.providers.base import ToolCall

# The key a call's arguments hold their raw text under when it is not a JSON object.
INVALID_ARGUMENTS = "__agentloom_invalid_arguments__"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class FailsStep(Exception):  # noqa: N818 — it carries the failure, it is not one
    """Raised by an agent called as a tool: ``error`` fails the calling step.

    Any other exception of a tool goes back to the model as the call's error.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


logger = logging.getLogger("agentloom.steps")


def translate_tools_for_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """OpenAI shape: ``[{"type": "function", "function": {name, description, parameters}}]``."""
    return [{"type": "function", "function": _openai_function(t)} for t in tools]


def _openai_function(tool: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters or {"type": "object", "properties": {}},
    }
    if getattr(tool, "strict", False):
        # Strict output tool: held to its schema as a strict reply is, every property required
        from agentloom.steps._structured import enforce_openai_strict_required

        parameters = copy.deepcopy(function["parameters"])
        enforce_openai_strict_required(parameters)
        function.update(parameters=parameters, strict=True)
    return function


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

    Google groups functions under ``Tool`` objects; AgentLoom has no tool
    grouping (everything is a function), so all go in one declaration list.
    Parameters lose the keys Gemini's schema dialect rejects, as a
    ``responseSchema`` does: an agent's output tool carries
    ``additionalProperties``.
    """
    from agentloom.steps._structured import _strip_unsupported_google_keys

    return [
        {
            "function_declarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": _strip_unsupported_google_keys(copy.deepcopy(t.parameters))
                    or {"type": "object", "properties": {}},
                }
                for t in tools
            ]
        }
    ]


def translate_tools_for_ollama(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Ollama uses the OpenAI shape on supported models."""
    return translate_tools_for_openai(tools)


def translate_tool_choice_for_openai(choice: Any) -> Any:
    """OpenAI ``tool_choice``: ``"auto"`` / ``"required"`` / ``"none"``, or a named function.

    A ``{"name": ...}`` choice becomes ``{"type": "function", "function": {"name": ...}}``.
    """
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


def parse_tool_calls_from_openai(message: dict[str, Any]) -> list[ToolCall]:
    """Parse ``message.tool_calls`` from OpenAI-shaped responses.

    Handles both wire variants seen in the wild:

    * **OpenAI canonical**: ``[{id, type:"function", function:{name, arguments:"<json>"}}]``
      — ``arguments`` is a JSON-encoded string.
    * **Ollama / OpenAI-compatible relays**: ``[{id, function:{name, arguments:{...}}}]``
      — ``type`` may be omitted entirely, and ``arguments`` may already be a
      decoded dict. Treat the absence of ``type`` as ``"function"`` (the
      only call kind dispatched); a non-function explicit ``type`` (e.g.
      ``"code_interpreter"``) is skipped.
    """
    raw_calls = message.get("tool_calls") or []
    calls: list[ToolCall] = []
    for entry in raw_calls:
        entry_type = entry.get("type", "function")
        if entry_type != "function":
            continue
        fn = entry.get("function", {})
        # OpenAI's tool follow-up needs a non-empty id and name, or the next request 400s: skip
        call_id = entry.get("id")
        call_name = fn.get("name")
        if not isinstance(call_id, str) or not call_id.strip():
            logger.warning("Skipping tool call without an id: %r", entry)
            continue
        if not isinstance(call_name, str) or not call_name.strip():
            logger.warning("Skipping tool call without a function name: %r", entry)
            continue
        raw_args = fn.get("arguments")
        if isinstance(raw_args, dict):
            args: dict[str, Any] = raw_args
        elif isinstance(raw_args, str):
            try:
                # vLLM sends "" for a tool without parameters: a call without arguments
                decoded = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                decoded = None
            # Cut short (a reply at its token limit) or not an object: the call gets an error,
            # not ``{}``; kept in the arguments, a recording replays it the same way
            args = decoded if isinstance(decoded, dict) else {INVALID_ARGUMENTS: raw_args}
        else:
            args = {}
        calls.append(ToolCall(id=call_id, name=call_name, arguments=args))
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
    """Parse Gemini ``functionCall`` parts into calls.

    Gemini assigns no ids, so they are synthesized; the result message
    echoes the call's name, so no id round-trip is needed.
    """
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
                signature=part.get("thoughtSignature"),
            )
        )
    return calls


class DeclaredTools:
    """*registry* restricted to the tools a step declares: any other name is not registered.

    The model chooses the tool names it calls, so a name the step does not offer must not run,
    even when the workflow registers that tool for other steps.
    """

    def __init__(self, registry: Any, tools: list[ToolDefinition]) -> None:
        self._registry = registry
        self._names = [tool.name for tool in tools]
        self.tape = getattr(registry, "tape", None)

    def get(self, name: str) -> Any:
        if name not in self._names:
            raise ToolNotFoundError(name, self._names)
        return self._registry.get(name)


def fill_tool_parameters(tools: list[ToolDefinition], registry: Any) -> list[ToolDefinition]:
    """*tools* with the registered tool's schema where a step declares no ``parameters``."""
    filled: list[ToolDefinition] = []
    for tool in tools:
        schema = None if tool.parameters else _registered_schema(tool.name, registry)
        filled.append(tool.model_copy(update={"parameters": schema}) if schema else tool)
    return filled


def _registered_schema(name: str, registry: Any) -> dict[str, Any] | None:
    if registry is None:
        return None
    try:
        schema = getattr(registry.get(name), "parameters_schema", None)
    except KeyError:
        return None  # dispatch reports the unknown tool to the model
    return schema if isinstance(schema, dict) else None


def report_tool_call(
    observer: Any | None,
    *,
    step_id: str,
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    duration_ms: float,
    success: bool,
) -> None:
    """Fire ``on_tool_call`` with hashes of the arguments and the result, never them."""
    hook = getattr(observer, "on_tool_call", None)
    if not callable(hook):
        return
    # Best-effort: an observer failure must not abort the dispatch and fail the step
    try:
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        hook(
            step_id=step_id,
            call_id=call_id,
            tool_name=tool_name,
            args_hash=_hash_text(json.dumps(arguments, sort_keys=True, default=str)),
            result_hash=_hash_text(text),
            duration_ms=duration_ms,
            success=success,
        )
    except Exception:
        logger.debug("on_tool_call hook failed", exc_info=True)


async def dispatch_tool_calls(
    calls: list[ToolCall],
    tool_registry: Any,
    *,
    observer: Any | None = None,
    step_id: str = "",
) -> list[tuple[ToolCall, str, bool]]:
    """Execute each call via the registry in parallel; preserve order.

    Failed calls return ``(call, str(exc), False)`` so the model can
    recover on the next turn rather than aborting the loop. When *observer*
    is provided, fires ``on_tool_call`` per call with hashes of the args
    and result for trace-level observability without leaking PII. A
    ``tape`` on the registry records the results, or serves recorded ones
    in place of running the tools.
    """
    results: list[tuple[ToolCall, str, bool]] = [None] * len(calls)  # type: ignore[list-item]
    tape = getattr(tool_registry, "tape", None)
    served = tape.serve(step_id, calls) if tape is not None else [None] * len(calls)

    async def _run(idx: int, call: ToolCall) -> None:
        calling(idx)  # an agent called as a tool links its events to this call
        start = time.monotonic()
        recorded = served[idx]
        text, success = recorded if recorded is not None else await _execute(call, tool_registry)
        results[idx] = (call, text, success)
        report_tool_call(
            observer,
            step_id=step_id,
            call_id=call.id,
            tool_name=call.name,
            arguments=call.arguments,
            result=text,
            duration_ms=(time.monotonic() - start) * 1000.0,
            success=success,
        )

    try:
        async with anyio.create_task_group() as tg:
            for idx, call in enumerate(calls):
                tg.start_soon(_run, idx, call)
    except ExceptionGroup as group:
        # Raise the call's own error, not anyio's group; a non-retryable one (a stale replay)
        # first, or a transient failure next to it would retry the step
        errors = [getattr(e, "error", e) for e in group.exceptions]
        raise next((e for e in errors if not getattr(e, "is_retryable", True)), errors[0]) from None

    log_tools([(c.name, c.arguments) for c, _, _ in results], [(r, ok) for _, r, ok in results])
    if tape is not None:
        await tape.capture(step_id, results)
    return results


async def _execute(call: ToolCall, tool_registry: Any) -> tuple[str, bool]:
    """Run one call; a lookup or execution failure becomes its result text."""
    try:
        tool = tool_registry.get(call.name)
    except KeyError as e:
        # Its message, without the quotes ``KeyError.__str__`` adds.
        return f"tool '{call.name}' not registered: {e.args[0] if e.args else e}", False
    if INVALID_ARGUMENTS in call.arguments:
        return f"tool arguments are not valid JSON: {call.arguments[INVALID_ARGUMENTS]!r}", False
    try:
        outcome = await tool.execute(**call.arguments)
    except FailsStep:
        raise
    except Exception as e:  # noqa: BLE001 — reported back to the model
        logger.warning("Tool '%s' failed: %s", call.name, e)
        return f"tool execution failed: {e}", False
    return (outcome if isinstance(outcome, str) else json.dumps(outcome, default=str)), True


def build_assistant_message_with_tool_calls(
    provider: str, content: str, calls: list[ToolCall]
) -> dict[str, Any]:
    """Replay the assistant's tool-call decision so the model sees its prior turn."""
    if provider == "anthropic":
        blocks: list[dict[str, Any]] = []
        if content:
            blocks.append({"type": "text", "text": content})
        for c in calls:
            blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
        return {"role": "assistant", "content": blocks}
    if provider == "google":
        # Gemini can return text beside functionCall parts; keep it so the next turn sees it all
        parts: list[dict[str, Any]] = []
        if content:
            parts.append({"text": content})
        for c in calls:
            part: dict[str, Any] = {"functionCall": {"name": c.name, "args": c.arguments}}
            if c.signature:
                # Gemini 3 refuses the next request without it on the same part
                part["thoughtSignature"] = c.signature
            parts.append(part)
        return {"role": "model", "parts": parts}
    if provider == "ollama":
        # Ollama wants arguments as a dict and rejects id/type keys: the OpenAI shape gets a 400
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls],
        }
    # OpenAI canonical: content must be a string (None is rejected); "" is fine beside tool_calls
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


def wire_seeded_tool_turns(provider: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put a stored conversation's tool turns in *provider*'s shape; the tool loop's already are."""
    wired: list[dict[str, Any]] = []
    seeded: dict[str, ToolCall] = {}
    results: list[tuple[ToolCall, str, bool]] = []

    def flush_results() -> None:
        if results:  # together: a provider may want them in a single turn
            wired.extend(build_tool_result_messages(provider, results))
            results.clear()

    for msg in messages:
        call = seeded.get(msg.get("tool_call_id", "")) if msg.get("role") == "tool" else None
        if call is not None:
            results.append((call, msg.get("content") or "", True))
            continue
        flush_results()
        calls = msg.get("tool_calls")
        if calls and "name" in calls[0]:  # dumped from a ``Message``: on the wire it's nested
            parsed = [ToolCall.model_validate(c) for c in calls]
            seeded.update((c.id, c) for c in parsed)
            wired.append(
                build_assistant_message_with_tool_calls(provider, msg.get("content") or "", parsed)
            )
        else:
            wired.append(msg)
    flush_results()
    return wired


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
    if provider == "ollama":
        # Ollama keys the result by name and rejects OpenAI's tool_call_id field
        return [{"role": "tool", "name": call.name, "content": text} for call, text, _ in results]
    # OpenAI canonical: one tool message per call, keyed by tool_call_id.
    return [{"role": "tool", "tool_call_id": call.id, "content": text} for call, text, _ in results]
