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

    Internal storage is name-mangled (``__data`` / ``__strict`` →
    ``_DotAccessDict__data`` / ``_DotAccessDict__strict``) so the wrapper's
    own attributes are not reachable via ``{state._data}`` or
    ``{state._strict}`` in a ``str.format_map`` template.
    """

    def __init__(self, data: dict[str, object], *, strict: bool = False) -> None:
        self.__data = data
        self.__strict = strict

    def __getattr__(self, name: str) -> object:
        # ``__getattr__`` runs only when the normal ``__dict__`` / type
        # lookup misses. Internal access (``self.__data`` / ``self.__strict``)
        # is name-mangled and resolves via the instance dict directly so it
        # never reaches here.
        if name not in self.__data:
            if self.__strict:
                raise TemplateError(f"state.{name}")
            logger.warning("Template variable 'state.%s' not found, rendering as empty", name)
            return ""
        value = self.__data[name]
        if isinstance(value, dict):
            return DotAccessDict(value, strict=self.__strict)
        if isinstance(value, list):
            return DotAccessList(value, strict=self.__strict)
        return value

    def __getitem__(self, key: str | int) -> object:
        if isinstance(key, int):
            if self.__strict:
                raise TemplateError(f"int index {key} on DotAccessDict")
            return ""
        return self.__getattr__(key)

    def __str__(self) -> str:
        return str(self.__data)

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return format(self.__data, format_spec)
        return str(self.__data)


class DotAccessList:
    """Wrapper that allows index access on a list for template rendering."""

    def __init__(self, data: list[object], *, strict: bool = False) -> None:
        self.__data = data
        self.__strict = strict

    def __getitem__(self, index: int | str) -> object:
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError:
                if self.__strict:
                    raise TemplateError(f"non-integer index {index!r}") from None
                return ""
        if -len(self.__data) <= index < len(self.__data):
            value = self.__data[index]
            if isinstance(value, dict):
                return DotAccessDict(value, strict=self.__strict)
            if isinstance(value, list):
                return DotAccessList(value, strict=self.__strict)
            return value
        if self.__strict:
            raise TemplateError(f"index {index} out of range")
        return ""

    def __str__(self) -> str:
        return str(self.__data)

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return format(self.__data, format_spec)
        return str(self.__data)


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
