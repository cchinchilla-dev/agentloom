"""Regression tests for router expression sandbox.

Covers the CVE-equivalent bypass described in GHSA-c37m-mv4j-972v — arbitrary
code execution through attribute-call chains over `__class__` /
`__subclasses__()` / `__call__` and through the `type` builtin.
"""

from __future__ import annotations

import pytest

from agentloom.exceptions import SecurityError
from agentloom.steps.router import evaluate_expression


class TestRejectsDunderAttributeAccess:
    """Any attribute name starting with `_` must be rejected."""

    @pytest.mark.parametrize(
        "attr",
        [
            "__class__",
            "__base__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__call__",
            "__globals__",
            "__builtins__",
            "__import__",
            "__getattribute__",
            "__reduce__",
            "__dict__",
            "__init_subclass__",
            "__new__",
        ],
    )
    def test_rejects_dunder_attribute(self, attr: str) -> None:
        expr = f"x.{attr}"
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {"x": object()})

    def test_rejects_dunder_in_chained_attribute(self) -> None:
        with pytest.raises(SecurityError):
            evaluate_expression("x.__class__.__mro__", {"x": 0})

    def test_rejects_single_underscore_prefix(self) -> None:
        # Private-by-convention attributes are also reachable into internals;
        # block them wholesale rather than trying to enumerate risky ones.
        with pytest.raises(SecurityError):
            evaluate_expression("x._private", {"x": object()})


class TestRejectsTypeBuiltin:
    """`type` must be removed from the safe-builtins set."""

    def test_rejects_bare_type_call(self) -> None:
        with pytest.raises(SecurityError):
            evaluate_expression("type(1) == int", {})

    def test_rejects_type_inside_expression(self) -> None:
        with pytest.raises(SecurityError):
            evaluate_expression("len(type(x).__mro__) > 0", {"x": 1})


class TestRejectsNonDunderBlocklist:
    """`mro`, `format_map` and similar non-dunder escape routes are blocked."""

    @pytest.mark.parametrize("attr", ["mro", "format_map"])
    def test_rejects_non_dunder_blocklist(self, attr: str) -> None:
        with pytest.raises(SecurityError):
            evaluate_expression(f"x.{attr}", {"x": ""})


class TestRejectsClassInstantiationViaAttributeCall:
    """Attribute calls may not be used to reach the object graph."""

    def test_rejects_call_via_subscript_of_class_chain(self) -> None:
        # Pattern: ().__class__.__bases__[0].__subclasses__()[N].__call__(...)
        # blocked by the dunder-prefix rejection on `__class__`.
        with pytest.raises(SecurityError):
            evaluate_expression(
                "().__class__.__bases__[0].__subclasses__()",
                {},
            )

    def test_rejects_call_on_literal_receiver(self) -> None:
        # Call receivers must be a Name/Attribute/Subscript chain — never a
        # bare literal — so a `(1).bit_length()` style call is out.
        with pytest.raises(SecurityError):
            evaluate_expression("(1).bit_length()", {})

    def test_rejects_keyword_arguments(self) -> None:
        with pytest.raises(SecurityError):
            evaluate_expression("isinstance(x, cls=int)", {"x": 1})


class TestAcceptsDocumentedGrammar:
    """Positive cases that must keep working after the sandbox tightening."""

    def test_state_attribute_equality(self) -> None:
        class State:
            x = "y"

        assert evaluate_expression("state.x == 'y'", {"state": State()}) is True

    def test_len_on_state_items(self) -> None:
        class State:
            items = [1, 2, 3]

        assert evaluate_expression("len(state.items) > 0", {"state": State()}) is True

    def test_isinstance_on_state_foo(self) -> None:
        class State:
            foo = "hello"

        assert evaluate_expression("isinstance(state.foo, str)", {"state": State()}) is True

    def test_arithmetic_and_comparison(self) -> None:
        class State:
            counter = 3

        assert evaluate_expression("state.counter + 1 < 10", {"state": State()}) is True

    def test_boolean_combination(self) -> None:
        class State:
            a = 1
            b = 2

        assert (
            evaluate_expression(
                "state.a > 0 and state.b > 0",
                {"state": State()},
            )
            is True
        )

    def test_membership_against_state_list(self) -> None:
        class State:
            categories = ["billing", "technical"]

        assert evaluate_expression("'billing' in state.categories", {"state": State()}) is True

    def test_subscript_access(self) -> None:
        assert evaluate_expression("x[0] == 1", {"x": [1, 2, 3]}) is True
