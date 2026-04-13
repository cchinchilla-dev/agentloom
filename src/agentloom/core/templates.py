"""Shared template rendering utilities.

Provides ``SafeFormatDict``, ``DotAccessDict``, and ``DotAccessList`` for
``str.format_map()``-based template rendering used across step executors
and webhook payloads.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agentloom.templates")


class DotAccessDict:
    """Wrapper that allows attribute access on a dict for template rendering."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name not in self._data:
            logger.warning("Template variable 'state.%s' not found, rendering as empty", name)
            return ""
        value = self._data[name]
        if isinstance(value, dict):
            return DotAccessDict(value)
        if isinstance(value, list):
            return DotAccessList(value)
        return value

    def __getitem__(self, key: str | int) -> object:
        if isinstance(key, int):
            return ""
        return self.__getattr__(key)

    def __str__(self) -> str:
        return str(self._data)

    def __format__(self, format_spec: str) -> str:
        return str(self._data)


class DotAccessList:
    """Wrapper that allows index access on a list for template rendering."""

    def __init__(self, data: list[object]) -> None:
        self._data = data

    def __getitem__(self, index: int | str) -> object:
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError:
                return ""
        if -len(self._data) <= index < len(self._data):
            value = self._data[index]
            if isinstance(value, dict):
                return DotAccessDict(value)
            if isinstance(value, list):
                return DotAccessList(value)
            return value
        return ""

    def __str__(self) -> str:
        return str(self._data)

    def __format__(self, format_spec: str) -> str:
        return str(self._data)


class SafeFormatDict(dict[str, object]):
    """Dict that returns '{key}' for missing keys instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def build_template_vars(state: dict[str, object]) -> dict[str, object]:
    """Build a flat namespace for ``str.format_map()``.

    Supports both ``{user_input}`` and ``{state.user_input}`` syntax.
    """
    flat: dict[str, object] = {}
    flat.update(state)
    flat["state"] = DotAccessDict(state)
    return flat
