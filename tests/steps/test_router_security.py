"""Regression tests for router expression sandbox.

Covers the CVE-equivalent bypass described in GHSA-c37m-mv4j-972v — arbitrary
code execution through attribute-call chains over `__class__` /
`__subclasses__()` / `__call__` and through the `type` builtin.
"""

from __future__ import annotations

import pytest

from agentloom.core.models import Condition, StepDefinition, StepType
from agentloom.core.state import StateManager
from agentloom.exceptions import SecurityError
from agentloom.steps.base import StepContext
from agentloom.steps.router import RouterStep, evaluate_expression


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

    def test_rejects_starred_arguments(self) -> None:
        # `*args` unpacking widens the grammar with no legitimate router use.
        with pytest.raises(SecurityError):
            evaluate_expression("max(*x)", {"x": [1, 2]})


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


class TestGHSAcm37mMv4j972vPayloads:
    """Explicit regression for GHSA-c37m-mv4j-972v.

    Verbatim payloads from the advisory. They MUST raise SecurityError on
    parse — never reach the eval stage.
    """

    def test_payload_1_type_call_via_type_builtin(self) -> None:
        # `type` removed from safe-builtins + `__call__` / `__base__` dunder
        # rejection close this chain at parse time.
        expr = 'type.__call__(type(()).__base__.__subclasses__()[0], ["sh", "-c", "whoami"])'
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {})

    def test_payload_2_direct_class_call(self) -> None:
        # `__class__` dunder rejection blocks this at the first attribute access.
        expr = '().__class__.__base__.__subclasses__()[0].__call__(["sh", "-c", "echo pwned"])'
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {})

    def test_payload_3_catch_warnings_module_builtins_chain(self) -> None:
        # `__class__` / `__mro__` / `__getitem__` / `__call__` / `__builtins__`
        # — every step on the chain is rejected by the dunder filter.
        expr = (
            "''.__class__.__mro__.__getitem__(1).__subclasses__().__getitem__(0)"
            ".__call__()._module.__builtins__.__getitem__('__import__')"
            ".__call__('os').system('id')"
        )
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {})


class TestRouterStepPropagatesSecurityError:
    """`RouterStep.execute()` must surface SecurityError unchanged.

    Wrapping it in a generic `StepError` would hide that the input itself
    was an attack — the caller (engine, observer, audit log) needs to see
    the exact exception type.
    """

    async def test_propagates_security_error_from_condition(self) -> None:
        step = RouterStep()
        context = StepContext(
            step_definition=StepDefinition(
                id="route",
                type=StepType.ROUTER,
                conditions=[
                    Condition(expression="x.__class__", target="ignored"),
                ],
                default="fallback",
            ),
            state_manager=StateManager(initial_state={"x": "hello"}),
        )
        with pytest.raises(SecurityError):
            await step.execute(context)
