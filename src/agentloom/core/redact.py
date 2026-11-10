"""State redaction policy and helpers.

A workflow can declare per-key redaction via ``state_schema:`` in the
YAML, or via the ``AGENTLOOM_REDACT_STATE_KEYS`` env var. The same policy
is applied at every persistence boundary so a key flagged as secret
never lands in a checkpoint file, history record, OTel span attribute,
or webhook payload.

Redaction is one-way: the on-disk artefact carries a stable sentinel
(``"<REDACTED:sha256=...>"``) rather than the plaintext. The in-memory
state stays unredacted so steps that legitimately need the value (an
LLM prompt that interpolates ``{state.api_key}`` against
``api.openai.com``) keep working.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import types
from collections.abc import Collection, Mapping
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

# The sentinel's shape, so consumers detect a redacted value without sniffing strings
_SENTINEL_PREFIX = "<REDACTED:sha256="
_SENTINEL_RE = re.compile(r"^<REDACTED:sha256=[0-9a-f]{16}>$")

ENV_VAR = "AGENTLOOM_REDACT_STATE_KEYS"


class RedactionPolicy:
    """Policy deciding whether a state key (or nested dotted path) is sensitive.

    Matching is glob-based — ``api_key``, ``*token*``, ``_*secret*`` —
    and applied to both the literal key and its dotted form so
    ``{state.api_key}`` and ``state["api_key"]`` resolve consistently.
    """

    def __init__(self, patterns: list[str] | None = None) -> None:
        # No patterns: the policy is falsy and ``redact_state`` returns its input unchanged
        self._patterns = tuple(p for p in (patterns or []) if p)

    def __bool__(self) -> bool:
        return bool(self._patterns)

    @property
    def patterns(self) -> tuple[str, ...]:
        return self._patterns

    def matches(self, key: str) -> bool:
        """Return ``True`` when *key* matches one of the configured glob patterns."""
        return any(fnmatch.fnmatchcase(key, pattern) for pattern in self._patterns)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RedactionPolicy:
        source: Mapping[str, str] = env if env is not None else os.environ
        raw = source.get(ENV_VAR, "")
        patterns = [p.strip() for p in raw.split(",") if p.strip()]
        return cls(patterns)

    def merge(self, other: RedactionPolicy) -> RedactionPolicy:
        """Combine two policies — the result matches if either does."""
        return RedactionPolicy(list(self._patterns) + list(other._patterns))


def _stable_sentinel(value: Any) -> str:
    """Build the ``<REDACTED:sha256=...>`` sentinel for a value.

    The hash is keyed on the value's string form so two state writes of
    the same secret produce the same sentinel — useful when diffing
    checkpoints across runs without ever exposing the plaintext.
    """
    raw = "" if value is None else str(value)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{_SENTINEL_PREFIX}{digest}>"


def is_redacted(value: Any) -> bool:
    """Return whether *value* is a redaction sentinel produced by this module."""
    return isinstance(value, str) and bool(_SENTINEL_RE.match(value))


def carries_sentinel(value: object) -> bool:
    """Return whether a redaction sentinel appears anywhere inside *value*."""
    return _SENTINEL_PREFIX in json.dumps(value, default=str)


def redact_state(state: dict[str, Any], policy: RedactionPolicy) -> dict[str, Any]:
    """Return a copy of *state* with sensitive keys replaced by sentinels.

    Top-level keys are matched directly. Nested dicts are walked
    recursively and matched against their dotted path
    (``credentials.access_token``) so a glob like ``*token*`` masks
    nested entries too. Lists are walked element-wise with their parent's
    path so list-of-secrets cases (``api_keys: [...]``) collapse to a
    list of sentinels.

    Non-string values are still redacted — the sentinel is built from the
    string form of the value so ints / dicts get masked just like
    strings. Non-string dict keys are coerced to ``str`` before pattern
    matching so a state dict deserialized from JSON with int keys does
    not crash ``fnmatch``.

    A self-referential state dict does not loop forever: ``_walk`` tracks
    visited container ids and substitutes a literal ``"<cycle>"`` sentinel
    the second time around. No output records the plaintext, the cycle
    marker included.
    """
    if not policy:
        return state
    walked = _walk(state, policy, prefix="", seen=set())
    # ``_walk`` returns ``Any``; a dict in gives a dict out
    assert isinstance(walked, dict)
    return walked


def redact_json_text(text: str, policy: RedactionPolicy) -> str:
    """Return *text* with its sensitive keys masked, when it holds a JSON object or array.

    Any other text comes back as it was: the policy matches keys, and plain text has none.
    """
    if not policy:
        return text
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if not isinstance(parsed, (dict, list)):
        return text
    walked = _walk(parsed, policy, prefix="", seen=set())
    return json.dumps(walked) if walked != parsed else text


_CYCLE_SENTINEL = "<cycle>"


def _walk(
    value: Any,
    policy: RedactionPolicy,
    *,
    prefix: str,
    seen: set[int],
) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            return _CYCLE_SENTINEL
        seen = seen | {id(value)}
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for k, v in value.items():
            key_str = k if isinstance(k, str) else str(k)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if policy.matches(key_str) or policy.matches(path):
                # Keep an existing sentinel: rehashing it would change it and break diffs
                if is_redacted(v):
                    result[k] = v
                elif isinstance(v, list):
                    result[k] = [iv if is_redacted(iv) else _stable_sentinel(iv) for iv in v]
                elif isinstance(v, dict):
                    result[k] = {
                        ik: (iv if is_redacted(iv) else _stable_sentinel(iv))
                        for ik, iv in v.items()
                    }
                else:
                    result[k] = _stable_sentinel(v)
            else:
                result[k] = _walk(v, policy, prefix=path, seen=seen)
        return result
    if isinstance(value, list):
        return [_walk(item, policy, prefix=prefix, seen=seen) for item in value]
    return value


def redact_model(
    data: dict[str, Any],
    model: type[BaseModel],
    policy: RedactionPolicy,
    *,
    parse_as: Mapping[str, type[BaseModel]] | None = None,
    opaque: Collection[str] = (),
) -> dict[str, Any]:
    """Return a copy of *data*, a dumped *model*, with the workflow's own data redacted.

    The model's fields are never matched: a pattern written for state keys, such as
    ``*token*``, would turn ``max_tokens`` or ``token_usage`` into a string the model
    refuses, and a checkpoint that could not be resumed. Their values are walked
    instead, down to the free-form data a workflow carries (``state``, ``tool_args``,
    webhook headers, a step's ``output``), which is redacted as ``redact_state`` does,
    with paths from where that data starts.

    *parse_as* names free-form fields that hold a model of their own, such as an
    inline subworkflow; the fields named in *opaque*, such as JSON schemas, stay as
    they are.
    """
    if not policy:
        return data
    return _ModelRedactor(policy, parse_as or {}, frozenset(opaque)).fields(data, model)


class _ModelRedactor:
    """Walks a dump along its model's types, so a sentinel only lands where text is accepted."""

    def __init__(
        self,
        policy: RedactionPolicy,
        parse_as: Mapping[str, type[BaseModel]],
        opaque: frozenset[str],
    ) -> None:
        self._policy = policy
        self._parse_as = parse_as
        self._opaque = opaque

    def fields(self, data: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in data.items():
            field = model.model_fields.get(key) if isinstance(key, str) else None
            if field is None:
                # A key the model doesn't declare is the workflow's data
                result[key] = self._free({key: value})[key]
            elif key in self._opaque:
                result[key] = value
            elif key in self._parse_as and isinstance(value, dict):
                result[key] = self.fields(value, self._parse_as[key])
            else:
                result[key] = self.typed(value, field.annotation)
        return result

    def typed(self, value: Any, annotation: Any) -> Any:
        """*value* with what its *annotation* leaves free redacted, the rest walked into."""
        if value is None:
            return None
        origin = get_origin(annotation)
        if origin is Annotated:
            return self.typed(value, get_args(annotation)[0])
        if origin is Union or origin is types.UnionType:
            return self._union(value, [a for a in get_args(annotation) if a is not type(None)])
        if annotation is Any:
            return self._free(value)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self.fields(value, annotation) if isinstance(value, dict) else value
        args = get_args(annotation)
        if origin in (list, tuple, set, frozenset) and isinstance(value, list):
            item = args[0] if args else Any
            return [self.typed(element, item) for element in value]
        if origin is dict and isinstance(value, dict):
            item = args[1] if len(args) == 2 else Any
            if _accepts_text(item):
                return self._free(value)
            return {k: self.typed(v, item) for k, v in value.items()}
        # One of the model's own scalars: never a sentinel
        return value

    def _union(self, value: Any, options: list[Any]) -> Any:
        if any(option is Any for option in options):
            return self._free(value)
        for option in options:
            model = _model_in(option)
            if model is not None and isinstance(value, dict):
                if set(value) <= set(model.model_fields):
                    return self.fields(value, model)
        for option in options:
            if isinstance(value, list) and get_origin(option) in (list, tuple, set, frozenset):
                return self.typed(value, option)
            if isinstance(value, dict) and get_origin(option) is dict:
                return self.typed(value, option)
        return value

    def _free(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return _walk(value, self._policy, prefix="", seen=set())
        return value


def _model_in(annotation: Any) -> type[BaseModel] | None:
    """The model *annotation* names, through ``Annotated``; ``None`` for anything else."""
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _accepts_text(annotation: Any) -> bool:
    """Whether a value of *annotation* may be a string, as a sentinel is."""
    if annotation is Any or annotation is str:
        return True
    if get_origin(annotation) in (Union, types.UnionType):
        return any(_accepts_text(option) for option in get_args(annotation))
    return False


__all__ = [
    "ENV_VAR",
    "RedactionPolicy",
    "carries_sentinel",
    "is_redacted",
    "redact_json_text",
    "redact_model",
    "redact_state",
]
