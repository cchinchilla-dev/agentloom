"""Tests for conditional import mechanism."""

from __future__ import annotations

import pytest

from agentloom.compat import MissingDependencyProxy, is_available, try_import


class TestTryImport:
    def test_existing_module(self) -> None:
        result = try_import("json")
        assert is_available(result)
        assert hasattr(result, "dumps")

    def test_missing_module(self) -> None:
        result = try_import("nonexistent_module_xyz_12345", extra="observability")
        assert not is_available(result)
        assert isinstance(result, MissingDependencyProxy)

    def test_stdlib_module(self) -> None:
        result = try_import("os.path")
        assert is_available(result)


class TestMissingDependencyProxy:
    def test_getattr_raises(self) -> None:
        proxy = MissingDependencyProxy("fake_module", extra="all")
        with pytest.raises(ImportError, match="fake_module"):
            _ = proxy.some_attribute

    def test_call_raises(self) -> None:
        proxy = MissingDependencyProxy("fake_module")
        with pytest.raises(ImportError, match="pip install"):
            proxy()

    def test_bool_is_false(self) -> None:
        proxy = MissingDependencyProxy("fake_module")
        assert not proxy

    def test_error_message_includes_extra(self) -> None:
        proxy = MissingDependencyProxy("otel", extra="observability")
        with pytest.raises(ImportError, match="agentloom\\[observability\\]"):
            _ = proxy.tracer


class TestIsAvailable:
    def test_real_module(self) -> None:
        import json

        assert is_available(json)

    def test_proxy(self) -> None:
        proxy = MissingDependencyProxy("x")
        assert not is_available(proxy)
