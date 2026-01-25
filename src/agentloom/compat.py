"""Conditional import mechanism for optional dependencies.

Provides `try_import()` which returns the real module if available,
or a MissingDependencyProxy that raises ImportError with install
instructions on any attribute access.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingDependencyProxy:
    """Proxy object that raises ImportError with install instructions on attribute access."""

    def __init__(self, module_name: str, extra: str = "all") -> None:
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_extra", extra)

    def _raise(self) -> None:
        module_name = object.__getattribute__(self, "_module_name")
        extra = object.__getattribute__(self, "_extra")
        raise ImportError(
            f"'{module_name}' is required for this feature. "
            f"Install with: pip install agentloom[{extra}]"
        )

    def __getattr__(self, name: str) -> None:
        self._raise()

    def __call__(self, *args: object, **kwargs: object) -> None:
        self._raise()

    def __bool__(self) -> bool:
        return False


def try_import(module_name: str, extra: str = "all") -> ModuleType | MissingDependencyProxy:
    """Try to import a module, returning a proxy if it's not installed.

    Args:
        module_name: Fully qualified module name (e.g., "opentelemetry.sdk").
        extra: The pip extra that provides this module (e.g., "observability").

    Returns:
        The real module if available, or a MissingDependencyProxy.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return MissingDependencyProxy(module_name, extra)


def is_available(module_or_proxy: ModuleType | MissingDependencyProxy) -> bool:
    """Check if a try_import result is a real module (True) or a proxy (False)."""
    return not isinstance(module_or_proxy, MissingDependencyProxy)
