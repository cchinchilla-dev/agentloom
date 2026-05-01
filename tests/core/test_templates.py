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

    def test_string_key_delegates(self) -> None:
        d = DotAccessDict({"a": 1})
        assert d["a"] == 1


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


class TestStrictMode:
    """Strict mode turns silent lenient rendering into loud TemplateError."""

    def test_strict_mode_raises_on_missing_key(self) -> None:
        import pytest

        from agentloom.core.templates import TemplateError

        tv = build_template_vars({"name": "Alice"}, strict=True)
        with pytest.raises(TemplateError):
            "{missing}".format_map(SafeFormatDict(tv, strict=True))

    def test_strict_mode_raises_on_missing_state_attr(self) -> None:
        import pytest

        from agentloom.core.templates import TemplateError

        tv = build_template_vars({"name": "Alice"}, strict=True)
        with pytest.raises(TemplateError):
            "{state.missing_attr}".format_map(SafeFormatDict(tv, strict=True))

    def test_lenient_mode_preserves_placeholder(self) -> None:
        tv = build_template_vars({"name": "Alice"})
        # Missing key stays as literal placeholder.
        assert "{missing}".format_map(SafeFormatDict(tv)) == "{missing}"


class TestFormatSpec:
    def test_format_spec_applied_to_nested_dict(self) -> None:
        d = DotAccessDict({"total": 1234.5678})
        # The dict itself is not a number — but `format` on the raw dict
        # would raise. Verify the spec is at least respected for list/dict
        # wrappers by delegating to ``format(self._data, spec)``.
        # Using a numeric leaf exercises the real code path.
        assert f"{d.total:.2f}" == "1234.57"

    def test_format_spec_applied_to_list_element(self) -> None:
        lst = DotAccessList([10, 20, 30])
        assert f"{lst[1]:03d}" == "020"
