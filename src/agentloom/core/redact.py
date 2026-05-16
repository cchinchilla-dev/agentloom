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
import os
import re
from typing import Any

# Pattern that matches the sentinel surface so consumers can detect a
# redacted value without sniffing the raw string.
_SENTINEL_PREFIX = "<REDACTED:sha256="
_SENTINEL_RE = re.compile(r"^<REDACTED:sha256=[0-9a-f]{16}>$")

ENV_VAR = "AGENTLOOM_REDACT_STATE_KEYS"


class RedactionPolicy:
    """Decides whether a state key (or nested dotted path) is sensitive.

    Matching is glob-based — ``api_key``, ``*token*``, ``_*secret*`` —
    and applied to both the literal key and its dotted form so
    ``{state.api_key}`` and ``state["api_key"]`` resolve consistently.
    """

    def __init__(self, patterns: list[str] | None = None) -> None:
        # Empty pattern lists silently no-op, which means ``redact_state``
        # is the identity when no policy is configured. Callers that want
        # to opt in entirely build a ``RedactionPolicy`` with explicit
        # patterns.
        self._patterns = tuple(p for p in (patterns or []) if p)

    def __bool__(self) -> bool:
        return bool(self._patterns)

    @property
    def patterns(self) -> tuple[str, ...]:
        return self._patterns

    def matches(self, key: str) -> bool:
        """Return ``True`` when *key* (or any of its dotted prefixes)
        matches one of the configured glob patterns.
        """
        for pattern in self._patterns:
            if fnmatch.fnmatchcase(key, pattern):
                return True
        return False

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> RedactionPolicy:
        env = env if env is not None else os.environ
        raw = env.get(ENV_VAR, "")
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
    """True when *value* is a redaction sentinel produced by this module."""
    return isinstance(value, str) and bool(_SENTINEL_RE.match(value))


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

    A self-referential state dict no longer infinite-loops: ``_walk``
    tracks visited container ids and substitutes a literal
    ``"<cycle>"`` sentinel the second time around. The plaintext is
    never recorded in any output, including the cycle marker.
    """
    if not policy:
        return state
    return _walk(state, policy, prefix="", seen=set())


_CYCLE_SENTINEL = "<cycle>"


def _walk(
    value: Any,
    policy: RedactionPolicy,
    *,
    prefix: str,
    seen: set[int],
) -> Any:
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
                if isinstance(v, list):
                    result[k] = [_stable_sentinel(item) for item in v]
                elif isinstance(v, dict):
                    result[k] = {ik: _stable_sentinel(iv) for ik, iv in v.items()}
                else:
                    result[k] = _stable_sentinel(v)
            else:
                result[k] = _walk(v, policy, prefix=path, seen=seen)
        return result
    if isinstance(value, list):
        return [_walk(item, policy, prefix=prefix, seen=seen) for item in value]
    return value


__all__ = [
    "ENV_VAR",
    "RedactionPolicy",
    "is_redacted",
    "redact_state",
]
