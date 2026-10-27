"""Structured-output helpers — wire translation, JSON extraction, validation."""

from __future__ import annotations

import copy
import importlib
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from agentloom.core.models import (
    OUTPUT_TOOL,
    ResponseSchema,
    ResponseSchemaConfigError,
    ToolDefinition,
)

_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _scan_first_json_object(text: str) -> str | None:
    """First balanced ``{...}`` substring, tracking string state."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def load_pydantic_model(dotted_path: str) -> type[BaseModel]:
    """Resolve a dotted path to a Pydantic ``BaseModel`` subclass.

    The path is passed to ``importlib.import_module``, which runs the
    target's top-level code — treat ``response_schema.model`` as the
    same trust surface as the workflow YAML.
    """
    if ":" in dotted_path:
        module_name, _, class_name = dotted_path.rpartition(":")
    else:
        module_name, _, class_name = dotted_path.rpartition(".")
    if not module_name or not class_name:
        raise ResponseSchemaConfigError(
            f"response_schema.model={dotted_path!r}: expected a dotted path like "
            f"'package.module.ClassName'."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ResponseSchemaConfigError(
            f"response_schema.model={dotted_path!r}: cannot import module {module_name!r}: {e}"
        ) from e
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise ResponseSchemaConfigError(
            f"response_schema.model={dotted_path!r}: module {module_name!r} has no "
            f"attribute {class_name!r}."
        ) from e
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise ResponseSchemaConfigError(
            f"response_schema.model={dotted_path!r}: {class_name!r} is not a "
            f"Pydantic BaseModel subclass."
        )
    return cls


def schema_dict_for(response_schema: ResponseSchema, step_id: str) -> dict[str, Any] | None:
    """JSON-Schema dict for *response_schema* (``None`` for ``json_object``).

    Always returns a deep copy so Pydantic's cached class schema and
    inline author dicts stay untouched.
    """
    if response_schema.type == "json_object":
        return None
    if response_schema.type == "pydantic":
        if not response_schema.model:
            raise ResponseSchemaConfigError(
                f"Step {step_id!r}: response_schema.type='pydantic' requires a 'model' dotted path."
            )
        cls = load_pydantic_model(response_schema.model)
        return _normalize_schema_for_strict(copy.deepcopy(cls.model_json_schema()))
    if response_schema.schema_ is None:
        raise ResponseSchemaConfigError(
            f"Step {step_id!r}: response_schema.type='json_schema' requires an inline "
            f"'schema' object."
        )
    return _normalize_schema_for_strict(copy.deepcopy(response_schema.schema_))


def _normalize_schema_for_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Transitively set ``additionalProperties: false`` (OpenAI strict mode)."""

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema


def schema_name_for(response_schema: ResponseSchema, step_id: str) -> str:
    """Schema name for OpenAI strict mode; defaults to the step id."""
    if response_schema.name:
        return response_schema.name
    if response_schema.type == "pydantic" and response_schema.model:
        return response_schema.model.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
    return step_id


def translate_for_openai(response_schema: ResponseSchema, step_id: str) -> dict[str, Any]:
    """OpenAI shape: ``{"type": "json_object"}`` or ``{"type": "json_schema", ...}``."""
    if response_schema.type == "json_object":
        return {"type": "json_object"}
    schema = schema_dict_for(response_schema, step_id)
    if schema is not None and response_schema.strict:
        # Strict mode also requires every property in ``required``.
        _enforce_openai_strict_required(schema)
    payload: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name_for(response_schema, step_id),
            "schema": schema,
            "strict": response_schema.strict,
        },
    }
    if response_schema.description:
        payload["json_schema"]["description"] = response_schema.description
    return payload


def _enforce_openai_strict_required(schema: dict[str, Any]) -> None:
    """In-place: every ``properties`` key gets listed in ``required``."""

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            existing = node.get("required") or []
            seen = set(existing)
            for key in node["properties"]:
                if key not in seen:
                    existing.append(key)
                    seen.add(key)
            node["required"] = existing
        for value in node.values():
            _walk(value)

    _walk(schema)


def translate_for_google(
    response_schema: ResponseSchema, step_id: str
) -> tuple[str, dict[str, Any] | None]:
    """Gemini shape: ``(responseMimeType, responseSchema)`` in snake_case for the adapter remap."""
    schema = schema_dict_for(response_schema, step_id)
    if schema is None:
        return "application/json", None
    return "application/json", _strip_unsupported_google_keys(schema)


# Gemini's ``responseSchema`` parser 400s on these keys (Pydantic-emitted
# noise plus ``additionalProperties``, which our strict normalizer adds for
# OpenAI but Gemini's schema dialect doesn't understand).
_GOOGLE_DROP_KEYS = frozenset({"title", "examples", "default", "additionalProperties"})


def _strip_unsupported_google_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove keys Gemini's ``responseSchema`` rejects."""

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for k in list(node):
                if k in _GOOGLE_DROP_KEYS:
                    node.pop(k)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema


def translate_for_ollama(response_schema: ResponseSchema, step_id: str) -> str | dict[str, Any]:
    """Ollama ``format``: ``"json"`` (free-form) or schema dict (Ollama 0.5+ strict)."""
    if response_schema.type == "json_object":
        return "json"
    schema = schema_dict_for(response_schema, step_id)
    return schema or "json"


def anthropic_system_prefix(response_schema: ResponseSchema, step_id: str) -> str:
    """System-prompt suffix for Anthropic's prefill fallback (no native API)."""
    schema = schema_dict_for(response_schema, step_id)
    if schema is None:
        return (
            "\n\nIMPORTANT: respond with a single JSON object only. No prose, no "
            "markdown fences, no commentary — just the JSON."
        )
    rendered = json.dumps(schema, indent=2)
    return (
        "\n\nIMPORTANT: respond with a single JSON object conforming to this JSON "
        f"Schema:\n```json\n{rendered}\n```\nNo prose, no markdown fences, no "
        "commentary — just the JSON object."
    )


def extract_parsed(content: str) -> Any:
    """Strict ``json.loads`` → code-fence strip → first balanced ``{...}`` span."""
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response content", "", 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = _JSON_CODE_FENCE_RE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Anthropic prefill: response continues after the prefilled ``"{"``.
    if not text.startswith("{"):
        candidate = "{" + text
        span = _scan_first_json_object(candidate)
        if span is not None:
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                pass
    span = _scan_first_json_object(text)
    if span is not None:
        return json.loads(span)
    raise json.JSONDecodeError("no JSON object found", text, 0)


def validate_parsed(parsed: Any, response_schema: ResponseSchema, step_id: str) -> Any:
    """Validate *parsed*; return the coerced value (Pydantic instance or dict)."""
    if response_schema.type == "json_object":
        return parsed
    if response_schema.type == "pydantic":
        cls = load_pydantic_model(response_schema.model or "")
        return cls.model_validate(parsed)
    schema = schema_dict_for(response_schema, step_id)
    if schema is None:
        return parsed
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — jsonschema is a hard dep
        if not isinstance(parsed, dict):
            raise TypeError(
                f"response_schema.type='json_schema' expects an object; got {type(parsed).__name__}"
            ) from None
        return parsed
    jsonschema.validate(parsed, schema)
    return parsed


def _render_validation_error(error: Exception) -> str:
    """One line per Pydantic error, else the error's text. Capped at 800 chars."""
    rendered = str(error)
    if isinstance(error, ValidationError):
        rendered = "\n".join(
            f"- {'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}"
            for err in error.errors()
        )
    if len(rendered) > 800:
        rendered = rendered[:800] + "… (truncated)"
    return rendered


def format_validation_feedback(error: Exception, step_id: str) -> str:
    """Retry message after a validation failure. Capped at 800 chars."""
    return (
        f"Your previous response failed structured-output validation for step "
        f"{step_id!r}. Errors:\n{_render_validation_error(error)}\n\nRespond again "
        f"with a single JSON object that conforms to the declared schema. No prose, "
        f"no markdown fences — just the JSON."
    )


# Sent back when a model that must reply through the output tool answers in text instead.
OUTPUT_TOOL_REMINDER = f"Reply by calling the {OUTPUT_TOOL} tool, with your reply as its arguments."


def output_tool(response_schema: ResponseSchema, step_id: str) -> ToolDefinition:
    """The tool a step with both ``tools`` and a schema delivers its reply through (#120).

    Tool calling works on every provider, a schema enforced next to tools does not:
    the reply comes as the arguments of this tool, validated as a structured reply.
    """
    parameters = schema_dict_for(response_schema, step_id) or {"type": "object"}
    if parameters.get("type") != "object":
        raise ResponseSchemaConfigError(
            f"Step {step_id!r}: a schema delivered through the {OUTPUT_TOOL!r} tool "
            f"must describe an object."
        )
    return ToolDefinition(
        name=OUTPUT_TOOL,
        description=(
            "Deliver your final reply. Call it once, with the reply as its arguments, "
            "when you need no other tool."
        ),
        parameters=parameters,
    )


def format_output_feedback(error: Exception) -> str:
    """The result of an output-tool call whose arguments failed validation."""
    return (
        f"The arguments of {OUTPUT_TOOL} failed validation. Errors:\n"
        f"{_render_validation_error(error)}\n\nCall {OUTPUT_TOOL} again with "
        f"arguments that follow its schema."
    )


__all__ = [
    "OUTPUT_TOOL_REMINDER",
    "anthropic_system_prefix",
    "extract_parsed",
    "format_output_feedback",
    "format_validation_feedback",
    "load_pydantic_model",
    "output_tool",
    "schema_dict_for",
    "schema_name_for",
    "translate_for_google",
    "translate_for_ollama",
    "translate_for_openai",
    "validate_parsed",
]
