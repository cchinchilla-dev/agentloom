"""Shared template rendering utilities.

Provides ``SafeFormatDict``, ``DotAccessDict``, and ``DotAccessList`` for
``str.format_map()``-based template rendering used across step executors
and webhook payloads.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agentloom.templates")


class TemplateError(KeyError):
    """Raised in strict mode when a template references a missing variable.

    Subclasses ``KeyError`` so ``str.format_map`` behaviour is preserved —
    callers using ``format_map`` see it as the expected exception type.
    Non-retryable: a template typo never resolves itself, so the resilience
    layer skips retries.
    """

    is_retryable = False


class _MissingDotAccess:
    """Sentinel that non-strict ``DotAccessDict`` / ``DotAccessList`` return for a missing key.

    A chained reference like ``{state.x.y.z}`` whose first segment is missing
    renders as ``""`` instead of raising ``AttributeError``: every further
    attribute or index access returns the same sentinel, and ``__str__``,
    ``__repr__`` and ``__format__`` render empty whatever the format spec
    (``:.20``) or conversion flag (``!r``).
    """

    __slots__ = ()

    def __getattribute__(self, _name: str) -> _MissingDotAccess:
        # Block every attribute, dunders too, so ``{state.missing.__class__}`` leaks nothing;
        # special methods still work, as Python looks them up on the type
        return self

    def __getitem__(self, _key: Any) -> _MissingDotAccess:
        return self

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        # ``!r`` converts before ``__format__`` runs: ``{state.missing!r}`` must render empty too
        return ""

    def __format__(self, _spec: str) -> str:
        return ""

    def __bool__(self) -> bool:
        return False


_MISSING = _MissingDotAccess()


# Names a template could reach to dump the wrapper's storage (``__dict__``, mangled fields)
_DOTACCESS_BLOCKED_ATTRS = frozenset(
    {
        "__dict__",
        "__init_subclass__",
        "__subclasshook__",
        "__weakref__",
        "_DotAccessDict__data",
        "_DotAccessDict__strict",
        "_DotAccessList__data",
        "_DotAccessList__strict",
    }
)


class DotAccessDict:
    """Wrapper that allows attribute access on a dict for template rendering.

    Two layers of defence keep the wrapper's storage out of template
    output: the storage attribute names are **name-mangled**
    (``__data`` → ``_DotAccessDict__data``) and ``__getattribute__``
    refuses :data:`_DOTACCESS_BLOCKED_ATTRS` so a template that writes
    the mangled name literally — or reaches ``__dict__`` — still gets
    nothing. Internal class methods use :func:`object.__getattribute__`
    to bypass the gate.
    """

    def __init__(self, data: dict[str, object], *, strict: bool = False) -> None:
        self.__data = data
        self.__strict = strict

    def __getattribute__(self, name: str) -> object:
        if name in _DOTACCESS_BLOCKED_ATTRS:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> object:
        data = object.__getattribute__(self, "_DotAccessDict__data")
        strict = object.__getattribute__(self, "_DotAccessDict__strict")
        if name not in data:
            if strict:
                raise TemplateError(f"state.{name}")
            logger.warning("Template variable 'state.%s' not found, rendering as empty", name)
            # The sentinel, not ``""``, so ``{state.x.y.z}`` renders empty rather than raising
            return _MISSING
        value = data[name]
        if isinstance(value, dict):
            return DotAccessDict(value, strict=strict)
        if isinstance(value, list):
            return DotAccessList(value, strict=strict)
        return value

    def __getitem__(self, key: str | int) -> object:
        strict = object.__getattribute__(self, "_DotAccessDict__strict")
        if isinstance(key, int):
            if strict:
                raise TemplateError(f"int index {key} on DotAccessDict")
            # The sentinel, so ``{state[0].foo}`` renders empty too
            return _MISSING
        return self.__getattr__(key)

    def __str__(self) -> str:
        return str(object.__getattribute__(self, "_DotAccessDict__data"))

    def __format__(self, format_spec: str) -> str:
        data = object.__getattribute__(self, "_DotAccessDict__data")
        if not format_spec:
            return str(data)
        # ``dict.__format__`` rejects a non-empty spec: fall back to ``str`` rather than crash
        try:
            return format(data, format_spec)
        except TypeError:
            logger.warning(
                "Template format spec %r ignored on non-scalar dict; rendering as str(...)",
                format_spec,
            )
            return str(data)


class DotAccessList:
    """Wrapper that allows index access on a list for template rendering."""

    def __init__(self, data: list[object], *, strict: bool = False) -> None:
        self.__data = data
        self.__strict = strict

    def __getattribute__(self, name: str) -> object:
        if name in _DOTACCESS_BLOCKED_ATTRS:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getitem__(self, index: int | str) -> object:
        data = object.__getattribute__(self, "_DotAccessList__data")
        strict = object.__getattribute__(self, "_DotAccessList__strict")
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError:
                if strict:
                    raise TemplateError(f"non-integer index {index!r}") from None
                # The sentinel, so ``{state.lst[abc].foo}`` renders empty too
                return _MISSING
        if -len(data) <= index < len(data):
            value = data[index]
            if isinstance(value, dict):
                return DotAccessDict(value, strict=strict)
            if isinstance(value, list):
                return DotAccessList(value, strict=strict)
            return value
        if strict:
            raise TemplateError(f"index {index} out of range")
        # Out of range: the sentinel, as for a missing key
        return _MISSING

    def __str__(self) -> str:
        return str(object.__getattribute__(self, "_DotAccessList__data"))

    def __format__(self, format_spec: str) -> str:
        data = object.__getattribute__(self, "_DotAccessList__data")
        if not format_spec:
            return str(data)
        try:
            return format(data, format_spec)
        except TypeError:
            logger.warning(
                "Template format spec %r ignored on non-scalar list; rendering as str(...)",
                format_spec,
            )
            return str(data)


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
    ``strict=True`` propagates to the nested ``DotAccessDict`` /
    ``DotAccessList`` wrappers so any missing reference raises.
    """
    flat: dict[str, object] = {}
    flat.update(state)
    flat["state"] = DotAccessDict(state, strict=strict)
    return flat
