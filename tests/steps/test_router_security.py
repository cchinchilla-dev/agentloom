"""Tests for the router expression sandbox, including the GHSA-c37m-mv4j-972v payloads."""

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
        # Private-by-convention names reach internals too: block them all, not a risky few.
        with pytest.raises(SecurityError):
            evaluate_expression("x._private", {"x": object()})


class TestRejectsTypeBuiltin:
    """`type` is not among the safe builtins."""

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
    """Attribute calls cannot reach the object graph."""

    def test_rejects_call_via_subscript_of_class_chain(self) -> None:
        # The dunder rejection on `__class__` stops the chain at its first link.
        with pytest.raises(SecurityError):
            evaluate_expression(
                "().__class__.__bases__[0].__subclasses__()",
                {},
            )

    def test_rejects_call_on_literal_receiver(self) -> None:
        # A call receiver must be a Name, Attribute or Subscript chain, never a literal.
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
    """Positive cases the sandbox accepts."""

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
    """Verbatim payloads from GHSA-c37m-mv4j-972v, each refused at parse time, before eval."""

    def test_payload_1_type_call_via_type_builtin(self) -> None:
        # The missing `type` builtin and the dunder rejection each close this chain at parse time.
        expr = 'type.__call__(type(()).__base__.__subclasses__()[0], ["sh", "-c", "whoami"])'
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {})

    def test_payload_2_direct_class_call(self) -> None:
        # `__class__` dunder rejection blocks this at the first attribute access.
        expr = '().__class__.__base__.__subclasses__()[0].__call__(["sh", "-c", "echo pwned"])'
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {})

    def test_payload_3_catch_warnings_module_builtins_chain(self) -> None:
        # The dunder filter rejects every link of this chain.
        expr = (
            "''.__class__.__mro__.__getitem__(1).__subclasses__().__getitem__(0)"
            ".__call__()._module.__builtins__.__getitem__('__import__')"
            ".__call__('os').system('id')"
        )
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {})


class TestRouterStepPropagatesSecurityError:
    """`RouterStep.execute()` surfaces SecurityError unchanged.

    A generic `StepError` would hide from the engine, observer and audit log that the input
    itself was an attack.
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


class TestRejectsSubscriptDunderBypass:
    """``state['__class__']`` and ``state['_secret']`` are blocked like their attribute forms.

    A string-constant subscript goes through ``_reject_attribute`` as an attribute name does.
    """

    @pytest.mark.parametrize(
        "expr",
        [
            "state['__class__']",
            "state['_data']",
            "state['_secret']",
            "state['_data']['_secret']",
            "state['__init__']",
            "state['__dict__']",
            "state['mro']",
            "state['format_map']",
        ],
    )
    def test_subscript_with_blocked_string_raises(self, expr: str) -> None:
        from agentloom.core.templates import DotAccessDict

        ns = {"state": DotAccessDict({"_secret": "x", "user": "alice"})}
        with pytest.raises(SecurityError):
            evaluate_expression(expr, ns)

    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("state['user']", "alice"),
            ("state['items'][0]", 1),
            ("state['nested']['key']", "value"),
        ],
    )
    def test_non_blocked_subscript_keeps_working(self, expr: str, expected: object) -> None:
        from agentloom.core.templates import DotAccessDict

        ns = {
            "state": DotAccessDict(
                {"user": "alice", "items": [1, 2, 3], "nested": {"key": "value"}}
            )
        }
        assert evaluate_expression(expr, ns) == expected


class TestDotAccessDictDoesNotLeakInternals:
    """``state['_data']`` never returns the wrapper's own underlying dict."""

    def test_subscript_underscored_missing_renders_empty(self) -> None:
        # The ``_MissingDotAccess`` sentinel renders empty and keeps chained access from raising.
        from agentloom.core.templates import DotAccessDict

        d = DotAccessDict({"user": "alice"})
        assert str(d["_data"]) == ""
        assert str(d["__class__"]) == ""
        assert str(d["__init__"]) == ""

    def test_subscript_underscored_present_returns_user_value(self) -> None:
        from agentloom.core.templates import DotAccessDict

        d = DotAccessDict({"_internal": "private-value"})
        assert d["_internal"] == "private-value"

    def test_strict_mode_raises_on_missing_underscored(self) -> None:
        from agentloom.core.templates import DotAccessDict, TemplateError

        d = DotAccessDict({"user": "alice"}, strict=True)
        with pytest.raises(TemplateError):
            _ = d["_data"]
        with pytest.raises(TemplateError):
            _ = d["__class__"]

    def test_user_supplied_underscored_key_still_readable(self) -> None:
        # By design: the router's AST validator, not ``DotAccessDict``, keeps ``_`` keys private.
        from agentloom.core.templates import DotAccessDict

        d = DotAccessDict({"_intentionally_private": "user-data"})
        assert d["_intentionally_private"] == "user-data"
        assert d._intentionally_private == "user-data"


class TestRejectsSubscriptIndirection:
    """Subscript slices must be literal ints or strings, never variables or expressions.

    Otherwise a state value used as the key (``creds[lookup]``) would reach a private
    attribute the validator blocks in its constant form (``creds['_secret']``).
    """

    @pytest.mark.parametrize(
        "expr",
        [
            "x[lookup]",
            "x['_' + 'secret']",
            "x['_sec' + 'ret']",
            "x['_secret' if True else 'x']",
            "x['_' + str(1)]",
            "x[len('abc')]",
        ],
    )
    def test_non_constant_slice_refused(self, expr: str) -> None:
        ns = {"x": {"_secret": "GHOST"}, "lookup": "_secret"}
        with pytest.raises(SecurityError):
            evaluate_expression(expr, ns)

    @pytest.mark.parametrize(
        "expr,want",
        [
            ("state['user']", "alice"),
            ("state['items'][0]", 1),
            ("state['nested']['key']", "value"),
        ],
    )
    def test_constant_subscripts_keep_working(self, expr: str, want: object) -> None:
        from agentloom.core.templates import DotAccessDict

        ns = {
            "state": DotAccessDict(
                {"user": "alice", "items": [1, 2, 3], "nested": {"key": "value"}}
            )
        }
        assert evaluate_expression(expr, ns) == want

    def test_constant_of_invalid_type_refused(self) -> None:
        # Other constant types (None, bool, float, bytes) make no sensible key: a sign of abuse.
        for expr in (
            "x[None]",
            "x[True]",
            "x[1.5]",
        ):
            with pytest.raises(SecurityError):
                evaluate_expression(expr, {"x": {0: "v"}})

    def test_slice_bounds_must_be_int_constants(self) -> None:
        from agentloom.core.templates import DotAccessDict

        ns = {
            "state": DotAccessDict({"items": [1, 2, 3]}),
            "lookup": "_secret",
        }
        try:
            evaluate_expression("state['items'][0:2]", ns)
        except SecurityError:
            pytest.fail("constant-bound slice was wrongly refused")
        except Exception:
            pass  # DotAccessList takes no slice; only the validator's verdict counts here
        with pytest.raises(SecurityError):
            evaluate_expression("state['items'][lookup:2]", ns)


class TestRouterNamespaceDoesNotFlattenStateKeys:
    """Top-level state keys are not bare names in router predicates.

    A state key named ``len`` would shadow the builtin, and any key (``creds``, ``lookup``)
    would be reachable without the ``state.`` prefix.
    """

    async def test_state_key_does_not_shadow_safe_builtin(self) -> None:
        step = RouterStep()
        context = StepContext(
            step_definition=StepDefinition(
                id="route",
                type=StepType.ROUTER,
                conditions=[
                    Condition(expression="len(state.items) == 3", target="ok"),
                ],
                default="fallback",
            ),
            state_manager=StateManager(initial_state={"len": "shadowed", "items": [1, 2, 3]}),
        )
        result = await step.execute(context)
        assert result.output == "ok"

    async def test_bare_state_name_not_resolvable(self) -> None:
        step = RouterStep()
        context = StepContext(
            step_definition=StepDefinition(
                id="route",
                type=StepType.ROUTER,
                conditions=[
                    Condition(expression="severity == 'high'", target="hot"),
                ],
                default="cold",
            ),
            state_manager=StateManager(initial_state={"severity": "high"}),
        )
        from agentloom.exceptions import StepError

        with pytest.raises(StepError):
            await step.execute(context)


class TestRouterStateProxyContract:
    """Every grammar the validator accepts must also resolve at runtime."""

    @pytest.mark.parametrize(
        "state,expr,want",
        [
            ({"user": {"name": "alice"}}, "state.user.name == 'alice'", "match"),
            ({"user": {"name": "alice"}}, "state['user']['name'] == 'alice'", "match"),
            (
                {"items": [{"label": "a"}, {"label": "b"}]},
                "state.items[0].label == 'a'",
                "match",
            ),
            ({"items": [10, 20, 30, 40]}, "len(state.items) == 4", "match"),
        ],
    )
    async def test_validator_accepted_grammars_resolve_at_runtime(
        self, state: dict, expr: str, want: str
    ) -> None:
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[Condition(expression=expr, target=want)],
                default="other",
            ),
            state_manager=StateManager(initial_state=state),
        )
        result = await step.execute(ctx)
        assert result.output == want


class TestSliceUnaryNegativeStep:
    """``state['items'][::-1]`` and friends — slice bounds accept unary ±."""

    @pytest.mark.parametrize(
        "expr",
        [
            "state['items'][::-1]",
            "state['items'][-2:]",
            "state['items'][:-1]",
            "state['items'][1:-1]",
        ],
    )
    def test_negative_step_or_bound_accepted(self, expr: str) -> None:
        from agentloom.steps.router import _validate_expression

        _validate_expression(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "state['items'][:len('x')]",
            "state['items'][lookup:]",
        ],
    )
    def test_non_constant_slice_bound_still_refused(self, expr: str) -> None:
        from agentloom.steps.router import _validate_expression

        with pytest.raises(SecurityError):
            _validate_expression(expr)


class TestRouterProxyRuntimeEdges:
    """Proxy paths the validator-driven tests do not reach: numeric keys, len, iter, ``steps``."""

    async def test_list_proxy_len_via_router_predicate(self) -> None:
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[Condition(expression="len(state.items) == 3", target="ok")],
                default="other",
            ),
            state_manager=StateManager(initial_state={"items": [10, 20, 30]}),
        )
        result = await step.execute(ctx)
        assert result.output == "ok"

    async def test_list_proxy_iter_via_router_predicate(self) -> None:
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[Condition(expression="'a' in state.items", target="ok")],
                default="other",
            ),
            state_manager=StateManager(initial_state={"items": ["a", "b", "c"]}),
        )
        result = await step.execute(ctx)
        assert result.output == "ok"

    async def test_dict_proxy_len_via_router_predicate(self) -> None:
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[Condition(expression="len(state.user) == 3", target="ok")],
                default="other",
            ),
            state_manager=StateManager(
                initial_state={"user": {"name": "alice", "email": "a@x.com", "role": "admin"}}
            ),
        )
        result = await step.execute(ctx)
        assert result.output == "ok"

    async def test_dict_proxy_in_operator_via_router_predicate(self) -> None:
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[Condition(expression="'name' in state.user", target="ok")],
                default="other",
            ),
            state_manager=StateManager(initial_state={"user": {"name": "alice", "role": "admin"}}),
        )
        result = await step.execute(ctx)
        assert result.output == "ok"

    def test_dict_proxy_iter_unit_level(self) -> None:
        # Router expressions cannot loop, so only a Python caller reaches ``__iter__``.
        from agentloom.steps.router import _DictProxy

        proxy = _DictProxy({"name": "alice", "role": "admin"})
        assert sorted(iter(proxy)) == ["name", "role"]

    def test_dict_proxy_numeric_subscript_falls_through(self) -> None:
        # A non-string key, such as an int constant the validator allows, indexes the dict directly.
        from agentloom.steps.router import _DictProxy

        proxy = _DictProxy({1: "numeric-key-value"})
        assert proxy[1] == "numeric-key-value"

    def test_state_proxy_numeric_subscript_falls_through(self) -> None:
        # ``state`` itself indexes the snapshot directly for a non-string key.
        from agentloom.steps.router import _build_state_proxy

        proxy = _build_state_proxy({1: "numeric"})
        assert proxy[1] == "numeric"

    async def test_list_proxy_returns_nested_listproxy(self) -> None:
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[
                    Condition(expression="state.items[0][1] == 'b'", target="ok"),
                ],
                default="other",
            ),
            state_manager=StateManager(initial_state={"items": [["a", "b"], ["c", "d"]]}),
        )
        result = await step.execute(ctx)
        assert result.output == "ok"

    async def test_steps_proxy_returns_prior_step_field(self) -> None:
        # ``steps.<id>.<field>`` lets a router condition read a previous step's output.
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[
                    Condition(expression="steps.prior.output == 'ready'", target="ok"),
                ],
                default="other",
            ),
            state_manager=StateManager(initial_state={"steps": {"prior": {"output": "ready"}}}),
        )
        result = await step.execute(ctx)
        assert result.output == "ok"

    async def test_steps_proxy_returns_scalar_when_step_data_not_dict(self) -> None:
        # ``_StepsProxy`` returns a ``steps`` entry that is not a dict as is, unwrapped.
        step = RouterStep()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="r",
                type=StepType.ROUTER,
                conditions=[Condition(expression="steps.prior == 'flat'", target="ok")],
                default="other",
            ),
            state_manager=StateManager(initial_state={"steps": {"prior": "flat"}}),
        )
        result = await step.execute(ctx)
        assert result.output == "ok"


class TestRouterProxyUnitLevel:
    """Unit coverage of the module-level proxy classes."""

    def test_dict_proxy_attribute_access(self) -> None:
        from agentloom.steps.router import _DictProxy

        p = _DictProxy({"name": "alice"})
        assert p.name == "alice"

    def test_dict_proxy_missing_key_returns_none_wrapped(self) -> None:
        from agentloom.steps.router import _DictProxy

        assert _DictProxy({"a": 1}).missing is None

    def test_list_proxy_len_and_iter(self) -> None:
        from agentloom.steps.router import _ListProxy

        p = _ListProxy([1, 2, 3])
        assert len(p) == 3
        assert list(iter(p)) == [1, 2, 3]

    def test_state_proxy_attribute_access(self) -> None:
        from agentloom.steps.router import _build_state_proxy

        p = _build_state_proxy({"user": {"name": "alice"}})
        assert p.user.name == "alice"
        assert p["user"]["name"] == "alice"


class TestRouterChainedSafeCalls:
    """Chained method calls on safe receivers are accepted.

    ``_safe_receiver`` accepts a chain when every link is an attribute on a Name, Attribute or
    Subscript base and clears the dunder and blocklist filter.
    """

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("state.x.strip().lower()", "hello, world"),
            ("state.x.split(',')[0].strip()", "Hello"),
            ("state.x.lstrip().rstrip().lower()", "hello, world"),
            ("state.x[1:5].lower()", " hel"),
            ("'critical' in state.severity.strip().lower()", True),
        ],
    )
    def test_accepts_chained_safe_calls(self, expr: str, expected: object) -> None:
        # Concrete values check the evaluation too, not just that validation passes.
        class State:
            x = "  Hello, World  "
            severity = "Critical"

        assert evaluate_expression(expr, {"state": State()}) == expected

    def test_chained_lower_strip_eq(self) -> None:
        class State:
            decision = "  APPROVE  "

        assert (
            evaluate_expression("state.decision.strip().lower() == 'approve'", {"state": State()})
            is True
        )

    def test_membership_after_chained_normalisation(self) -> None:
        class State:
            severity = "  Critical  "

        assert (
            evaluate_expression("'critical' in state.severity.strip().lower()", {"state": State()})
            is True
        )

    @pytest.mark.parametrize(
        "expr",
        [
            "''.join(['a', 'b'])",
            "'abc'.upper()",
            "(1).bit_length()",
        ],
    )
    def test_still_rejects_calls_on_literals(self, expr: str) -> None:
        # The base of every chain must be a name, attribute or subscript, never a constant.
        with pytest.raises(SecurityError):
            evaluate_expression(expr, {})

    def test_still_rejects_dunder_after_safe_chain(self) -> None:
        # The dunder filter checks every attribute, so a chain cannot slip ``__class__`` in.
        class State:
            x = "hello"

        with pytest.raises(SecurityError):
            evaluate_expression("state.x.lower().__class__", {"state": State()})

    def test_still_rejects_disallowed_builtins(self) -> None:
        with pytest.raises(SecurityError):
            evaluate_expression("__import__('os').system('x')", {})
