"""Tests for shared template rendering utilities."""

from __future__ import annotations

import pytest

from agentloom.core.templates import (
    DotAccessDict,
    DotAccessList,
    SafeFormatDict,
    TemplateError,
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

    def test_private_attr_returns_empty_non_strict(self) -> None:
        # Previously private attrs delegated to ``object.__getattribute__``,
        # which let ``d['_data']`` reach the wrapper's underlying dict via
        # ``__getitem__`` → ``__getattr__``. Every dynamic lookup now goes
        # through the same data-only path: a missing key renders empty
        # under non-strict and raises ``TemplateError`` under strict.
        d = DotAccessDict({"a": 1})
        assert d._nonexistent == ""

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


class TestStrictRaiseBranches:
    """Strict-mode TemplateError branches that the default warn-mode skips."""

    def test_strict_dict_int_index_raises(self) -> None:
        d = DotAccessDict({"a": 1}, strict=True)
        with pytest.raises(TemplateError, match="int index"):
            _ = d[0]  # type: ignore[index]

    def test_strict_list_non_integer_index_raises(self) -> None:
        lst = DotAccessList([1, 2, 3], strict=True)
        with pytest.raises(TemplateError, match="non-integer index"):
            _ = lst["bad"]  # type: ignore[index]

    def test_strict_list_out_of_range_raises(self) -> None:
        lst = DotAccessList([1, 2, 3], strict=True)
        with pytest.raises(TemplateError, match="out of range"):
            _ = lst[99]


class TestFormatSpecOnContainers:
    """``__format__`` honours ``format_spec`` on dict/list wrappers when the
    underlying value supports the spec."""

    def test_dict_format_spec_with_data_supporting_spec(self) -> None:
        d = DotAccessDict({"x": 1234.5678})
        assert format(d, "") == repr({"x": 1234.5678})

    def test_list_format_spec_with_data_supporting_spec(self) -> None:
        lst = DotAccessList([1, 2, 3])
        assert format(lst, "") == repr([1, 2, 3])

    def test_dict_format_spec_non_empty_forwards_to_data(self) -> None:
        # A non-empty spec hits the explicit ``return format(self._data, ...)``
        # path. Use a string-formattable underlying scalar.
        d = DotAccessDict({"x": "hi"})
        # Spec ">5" right-aligns in 5 chars — repr-style strings don't accept
        # this spec, so we wrap a list/dict body with a spec the underlying
        # type accepts: empty string. Instead, exercise the branch via
        # ``__format__`` directly with an empty-string sentinel that still
        # passes the truthy guard? No — empty falls through to else. We need
        # a non-empty spec that ``dict.__format__`` accepts: there is none.
        # Cover the branch by stubbing ``self._data`` with a value that
        # supports the spec.
        d._data = "hi"  # type: ignore[assignment]
        assert format(d, ">5") == "   hi"

    def test_list_format_spec_non_empty_forwards_to_data(self) -> None:
        lst = DotAccessList([1])
        lst._data = "abc"  # type: ignore[assignment]
        assert format(lst, ">5") == "  abc"
