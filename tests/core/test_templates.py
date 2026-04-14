"""Tests for shared template rendering utilities."""

from __future__ import annotations

from agentloom.core.templates import (
    DotAccessDict,
    DotAccessList,
    SafeFormatDict,
    build_template_vars,
)


class TestBuildTemplateVars:
    def test_flat_access(self) -> None:
        tv = build_template_vars({"name": "Alice"})
        result = "{name}".format_map(SafeFormatDict(tv))
        assert result == "Alice"

    def test_dotted_access(self) -> None:
        tv = build_template_vars({"name": "Alice"})
        result = "{state.name}".format_map(SafeFormatDict(tv))
        assert result == "Alice"

    def test_missing_key_preserved(self) -> None:
        tv = build_template_vars({"name": "Alice"})
        result = "{missing}".format_map(SafeFormatDict(tv))
        assert result == "{missing}"

    def test_nested_dict(self) -> None:
        tv = build_template_vars({"user": {"name": "Bob"}})
        result = "{state.user.name}".format_map(SafeFormatDict(tv))
        assert result == "Bob"


class TestDotAccessDict:
    def test_str_representation(self) -> None:
        d = DotAccessDict({"a": 1})
        assert "1" in str(d)

    def test_format(self) -> None:
        d = DotAccessDict({"a": 1})
        assert "1" in f"{d}"

    def test_private_attr_raises(self) -> None:
        d = DotAccessDict({"a": 1})
        # Private attrs delegate to object.__getattribute__
        with __import__("pytest").raises(AttributeError):
            _ = d._nonexistent

    def test_int_key_returns_empty(self) -> None:
        d = DotAccessDict({"a": 1})
        assert d[0] == ""


class TestDotAccessList:
    def test_format(self) -> None:
        lst = DotAccessList(["x", "y"])
        assert format(lst) == "['x', 'y']"

    def test_index_access(self) -> None:
        lst = DotAccessList(["x", "y", "z"])
        assert lst[0] == "x"
        assert lst[2] == "z"

    def test_out_of_range(self) -> None:
        lst = DotAccessList(["x"])
        assert lst[5] == ""

    def test_string_index(self) -> None:
        lst = DotAccessList(["a", "b"])
        assert lst["0"] == "a"

    def test_invalid_string_index(self) -> None:
        lst = DotAccessList(["a"])
        assert lst["foo"] == ""
