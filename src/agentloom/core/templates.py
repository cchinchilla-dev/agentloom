"""Shared template rendering utilities.

Provides ``SafeFormatDict``, ``DotAccessDict``, and ``DotAccessList`` for
``str.format_map()``-based template rendering used across step executors
and webhook payloads.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agentloom.templates")


class TemplateError(KeyError):
    """Raised in strict mode when a template references a missing variable.

    Subclasses ``KeyError`` so ``str.format_map`` behavior is preserved —
    callers using ``format_map`` see it as the expected exception type.
    """


class DotAccessDict:
    """Wrapper that allows attribute access on a dict for template rendering.

    ``strict=True`` raises :class:`TemplateError` on missing keys; the
    default ``strict=False`` logs a warning and renders an empty string,
    preserving pre-existing workflow compatibility.
    """

    def __init__(self, data: dict[str, object], *, strict: bool = False) -> None:
        self._data = data
        self._strict = strict

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name not in self._data:
            if self._strict:
                raise TemplateError(f"state.{name}")
            logger.warning("Template variable 'state.%s' not found, rendering as empty", name)
            return ""
        value = self._data[name]
        if isinstance(value, dict):
            return DotAccessDict(value, strict=self._strict)
        if isinstance(value, list):
            return DotAccessList(value, strict=self._strict)
        return value

    def __getitem__(self, key: str | int) -> object:
        if isinstance(key, int):
            if self._strict:
                raise TemplateError(f"int index {key} on DotAccessDict")
            return ""
        return self.__getattr__(key)

    def __str__(self) -> str:
        return str(self._data)

    def __format__(self, format_spec: str) -> str:
        # Respect the caller's format_spec — previously ignored, which made
        # `{state.total:.2f}` silently render the raw dict repr.
        if format_spec:
            return format(self._data, format_spec)
        return str(self._data)


class DotAccessList:
    """Wrapper that allows index access on a list for template rendering."""

    def __init__(self, data: list[object], *, strict: bool = False) -> None:
        self._data = data
        self._strict = strict

    def __getitem__(self, index: int | str) -> object:
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError:
                if self._strict:
                    raise TemplateError(f"non-integer index {index!r}") from None
                return ""
        if -len(self._data) <= index < len(self._data):
            value = self._data[index]
            if isinstance(value, dict):
                return DotAccessDict(value, strict=self._strict)
            if isinstance(value, list):
                return DotAccessList(value, strict=self._strict)
            return value
        if self._strict:
            raise TemplateError(f"index {index} out of range")
        return ""

    def __str__(self) -> str:
        return str(self._data)

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return format(self._data, format_spec)
        return str(self._data)


class SafeFormatDict(dict[str, object]):
    """Dict used as ``format_map`` namespace.

    ``strict=False`` (default) returns ``'{key}'`` for missing keys so the
    placeholder shows up in the rendered string. ``strict=True`` raises
    :class:`TemplateError` so callers can turn a typo into a clear failure.
    """

    def __init__(self, *args: object, strict: bool = False, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._strict = strict

    def __missing__(self, key: str) -> str:
        if self._strict:
            raise TemplateError(key)
        return "{" + key + "}"


def build_template_vars(state: dict[str, object], *, strict: bool = False) -> dict[str, object]:
    """Build a flat namespace for ``str.format_map()``.

    Supports both ``{user_input}`` and ``{state.user_input}`` syntax.
    ``strict=True`` propagates to the nested ``DotAccessDict`` / ``Dot``-
    ``AccessList`` wrappers so any missing reference raises.
    """
    flat: dict[str, object] = {}
    flat.update(state)
    flat["state"] = DotAccessDict(state, strict=strict)
    return flat
