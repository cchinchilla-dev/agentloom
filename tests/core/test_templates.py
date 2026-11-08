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

    def test_private_attr_renders_empty_non_strict(self) -> None:
        # Private names take the same data-only path as any key: missing, they render ``""``.
        d = DotAccessDict({"a": 1})
        assert str(d._nonexistent) == ""
        assert format(d._nonexistent) == ""

    def test_int_key_renders_empty(self) -> None:
        # The sentinel, so a chained ``{state[0].foo}`` also renders empty.
        d = DotAccessDict({"a": 1})
        assert str(d[0]) == ""

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

    def test_out_of_range_renders_empty(self) -> None:
        # The sentinel, so a chained ``{state.lst[99].foo}`` also renders empty.
        lst = DotAccessList(["x"])
        assert str(lst[5]) == ""

    def test_string_index(self) -> None:
        lst = DotAccessList(["a", "b"])
        assert lst["0"] == "a"

    def test_invalid_string_index_renders_empty(self) -> None:
        lst = DotAccessList(["a"])
        assert str(lst["foo"]) == ""


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
        assert "{missing}".format_map(SafeFormatDict(tv)) == "{missing}"


class TestFormatSpec:
    def test_format_spec_applied_to_nested_dict(self) -> None:
        d = DotAccessDict({"total": 1234.5678})
        # A scalar leaf comes back unwrapped, so the spec applies to the float itself.
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
    """The dict and list wrappers pass a format spec on to their value when it takes one."""

    def test_dict_format_spec_with_data_supporting_spec(self) -> None:
        d = DotAccessDict({"x": 1234.5678})
        assert format(d, "") == repr({"x": 1234.5678})

    def test_list_format_spec_with_data_supporting_spec(self) -> None:
        lst = DotAccessList([1, 2, 3])
        assert format(lst, "") == repr([1, 2, 3])

    def test_dict_format_spec_non_empty_forwards_to_data(self) -> None:
        # ``dict`` takes no format spec: a string as storage reaches the forwarding branch.
        d = DotAccessDict({"x": "hi"})
        object.__setattr__(d, "_DotAccessDict__data", "hi")
        assert format(d, ">5") == "   hi"

    def test_list_format_spec_non_empty_forwards_to_data(self) -> None:
        lst = DotAccessList([1])
        object.__setattr__(lst, "_DotAccessList__data", "abc")
        assert format(lst, ">5") == "  abc"


class TestInternalAttributesNotReachableViaTemplate:
    """The wrapper's private storage does not surface through ``str.format_map``.

    Name-mangling (``__data`` → ``_DotAccessDict__data``) puts it out of reach of the short
    attribute names that format syntax supports.
    """

    def test_state_dot_underscore_data_renders_empty(self) -> None:
        tvars = build_template_vars({"user": {"name": "alice", "_password": "secret"}})
        rendered = "{state._data}".format_map(SafeFormatDict(tvars))
        assert rendered == ""
        assert "secret" not in rendered

    def test_state_dot_underscore_strict_renders_empty(self) -> None:
        tvars = build_template_vars({"x": 1})
        rendered = "{state._strict}".format_map(SafeFormatDict(tvars))
        assert rendered == ""

    def test_list_wrapper_short_attribute_not_reachable(self) -> None:
        # The DotAccessList storage is name-mangled too: there is no short ``_data`` attribute.
        from agentloom.core.templates import DotAccessList

        lst = DotAccessList([1, 2, 3])
        assert not hasattr(lst, "_data")
        assert not hasattr(lst, "_strict")

    def test_list_wrapper_blocks_mangled_storage_name(self) -> None:
        from agentloom.core.templates import DotAccessList

        lst = DotAccessList([1, 2, 3])
        with pytest.raises(AttributeError):
            _ = lst._DotAccessList__data
        with pytest.raises(AttributeError):
            _ = lst.__dict__

    @pytest.mark.parametrize(
        "expr",
        [
            "{state.__dict__}",
            "{state._DotAccessDict__data}",
            "{state._DotAccessDict__strict}",
            "{state.user.__dict__}",
            "{state.user._DotAccessDict__data}",
        ],
    )
    def test_python_internals_path_refused(self, expr: str) -> None:
        tvars = build_template_vars({"user": {"name": "alice", "_password": "secret"}})
        rendered = expr.format_map(SafeFormatDict(tvars))
        assert rendered == ""
        assert "alice" not in rendered
        assert "secret" not in rendered
        assert "_DotAccessDict__data" not in rendered

    def test_isinstance_and_str_still_work(self) -> None:
        from agentloom.core.templates import DotAccessDict, DotAccessList

        d = DotAccessDict({"x": 1})
        assert isinstance(d, DotAccessDict)
        assert str(d) == "{'x': 1}"
        lst = DotAccessList([1, 2])
        assert isinstance(lst, DotAccessList)
        assert str(lst) == "[1, 2]"


class TestMissingDeepKeyChain:
    """Non-strict ``{state.x.y.z}`` with a missing intermediate renders as the empty string.

    A missing key returns a chainable sentinel, so the format machinery raises no
    ``AttributeError``, which the engine would retry as a transient error.
    """

    def test_missing_first_segment_renders_empty(self) -> None:
        tv = build_template_vars({"name": "alice"})
        assert "{state.x.y.z}".format_map(SafeFormatDict(tv)) == ""

    def test_missing_first_segment_with_format_spec_renders_empty(self) -> None:
        # The sentinel renders ``""`` whatever the format spec.
        tv = build_template_vars({"x": 1})
        assert "{state.missing.deep:.20}".format_map(SafeFormatDict(tv)) == ""

    def test_missing_first_segment_with_index_renders_empty(self) -> None:
        # Subscript chained off a missing key bottoms out the same way.
        tv = build_template_vars({"x": 1})
        assert "{state.missing[0]}".format_map(SafeFormatDict(tv)) == ""

    def test_missing_segment_strict_raises_at_first_miss(self) -> None:
        tv = build_template_vars({"name": "alice"}, strict=True)
        with pytest.raises(TemplateError, match="state.x"):
            "{state.x.y.z}".format_map(SafeFormatDict(tv, strict=True))

    def test_missing_intermediate_strict_raises_at_first_missing_segment(self) -> None:
        # ``state.x`` and ``state.x.y`` exist: the error names the first missing segment, ``z``.
        tv = build_template_vars({"x": {"y": {}}}, strict=True)
        with pytest.raises(TemplateError, match="state.z"):
            "{state.x.y.z}".format_map(SafeFormatDict(tv, strict=True))

    def test_missing_list_index_chain_renders_empty(self) -> None:
        # As with a missing key, the out-of-range index returns the sentinel and ``.foo`` chains on.
        tv = build_template_vars({"lst": [1, 2, 3]})
        assert "{state.lst[99].foo}".format_map(SafeFormatDict(tv)) == ""

    def test_dict_int_key_chain_renders_empty(self) -> None:
        # An int subscript on a dict returns the sentinel from ``DotAccessDict.__getitem__``.
        tv = build_template_vars({"x": 1})
        assert "{state[0].foo}".format_map(SafeFormatDict(tv)) == ""

    def test_sentinel_is_falsy(self) -> None:
        # Falsy, so a truthy check on a missing key behaves as on an empty value.
        from agentloom.core.templates import _MISSING

        assert bool(_MISSING) is False

    def test_sentinel_blocks_dunder_attribute_access(self) -> None:
        # ``__getattribute__`` returns the sentinel for every name, dunders included.
        tv = build_template_vars({"name": "alice"})
        assert "{state.missing.__class__}".format_map(SafeFormatDict(tv)) == ""
        assert "{state.missing.__dict__}".format_map(SafeFormatDict(tv)) == ""

    def test_sentinel_bottoms_out_on_repr_conversion(self) -> None:
        # ``!r`` runs before ``__format__``, so the sentinel's ``__repr__`` must render empty too.
        tv = build_template_vars({"name": "alice"})
        assert "{state.missing!r}".format_map(SafeFormatDict(tv)) == ""
        assert "{state.missing.deep!r}".format_map(SafeFormatDict(tv)) == ""
        assert "{state.missing!s}".format_map(SafeFormatDict(tv)) == ""
        assert "{state.missing!a}".format_map(SafeFormatDict(tv)) == ""


class TestFormatSpecOnNonScalar:
    """A value that rejects the format spec with ``TypeError`` renders as ``str(value)``."""

    def test_format_spec_on_dict_value_falls_back_to_str(self) -> None:
        # A dict rejects ``.20``: the wrapper falls back to ``str(dict)`` instead of raising.
        tv = build_template_vars({"user": {"name": "alice"}})
        rendered = "{state.user:.20}".format_map(SafeFormatDict(tv))
        assert rendered == str({"name": "alice"})

    def test_format_spec_on_list_value_falls_back_to_str(self) -> None:
        tv = build_template_vars({"items": [1, 2, 3]})
        rendered = "{state.items:.20}".format_map(SafeFormatDict(tv))
        assert rendered == str([1, 2, 3])

    def test_format_spec_on_scalar_leaf_still_works(self) -> None:
        # The fallback runs only on TypeError: a numeric leaf keeps its precise formatting.
        tv = build_template_vars({"n": 1234.5678})
        assert "{state.n:.2f}".format_map(SafeFormatDict(tv)) == "1234.57"


class TestUnicodeAndEscapedBraces:
    """Encoding and escaping corner cases."""

    def test_unicode_nfc_state_key_renders(self) -> None:
        import unicodedata

        key = unicodedata.normalize("NFC", "café")
        tv = build_template_vars({key: "bonjour"})
        rendered = ("{state." + key + "}").format_map(SafeFormatDict(tv))
        assert rendered == "bonjour"

    def test_unicode_nfc_vs_nfd_key_mismatch_renders_empty(self) -> None:
        # Lookup is byte-exact: an NFD key read in NFC renders empty; the docs advise normalizing.
        import unicodedata

        nfd = unicodedata.normalize("NFD", "café")
        nfc = unicodedata.normalize("NFC", "café")
        assert nfd != nfc

        tv = build_template_vars({nfd: "value-stored-nfd"})
        rendered = ("{state." + nfc + "}").format_map(SafeFormatDict(tv))
        assert rendered == ""

    def test_escaped_braces_render_as_literal(self) -> None:
        # ``{{`` / ``}}`` escape a literal brace: no path through the template engine expands them.
        tv = build_template_vars({"x": 1})
        assert "{{state.x}}".format_map(SafeFormatDict(tv)) == "{state.x}"
