"""Tests for the router step expression evaluation."""

from __future__ import annotations

import pytest

from agentloom.core.models import StepDefinition, StepType
from agentloom.core.state import StateManager
from agentloom.exceptions import SecurityError
from agentloom.steps.base import StepContext
from agentloom.steps.router import RouterStep, _build_state_proxy, evaluate_expression


class TestExpressionEvaluation:
    def test_simple_equality(self) -> None:
        result = evaluate_expression("x == 1", {"x": 1})
        assert result is True

    def test_simple_inequality(self) -> None:
        result = evaluate_expression("x != 1", {"x": 2})
        assert result is True

    def test_string_equality(self) -> None:
        result = evaluate_expression("category == 'billing'", {"category": "billing"})
        assert result is True

    def test_string_equality_false(self) -> None:
        result = evaluate_expression("category == 'billing'", {"category": "technical"})
        assert result is False

    def test_comparison_operators(self) -> None:
        ns = {"x": 10}
        assert evaluate_expression("x > 5", ns) is True
        assert evaluate_expression("x < 5", ns) is False
        assert evaluate_expression("x >= 10", ns) is True
        assert evaluate_expression("x <= 10", ns) is True

    def test_boolean_and(self) -> None:
        result = evaluate_expression("x > 0 and y > 0", {"x": 1, "y": 2})
        assert result is True

    def test_boolean_or(self) -> None:
        result = evaluate_expression("x > 0 or y > 0", {"x": -1, "y": 2})
        assert result is True

    def test_boolean_not(self) -> None:
        result = evaluate_expression("not x", {"x": False})
        assert result is True

    def test_in_operator(self) -> None:
        result = evaluate_expression("'hello' in greeting", {"greeting": "hello world"})
        assert result is True

    def test_not_in_operator(self) -> None:
        result = evaluate_expression("'bye' not in greeting", {"greeting": "hello world"})
        assert result is True


class TestAttributeAccess:
    def test_attribute_access(self) -> None:
        class StateProxy:
            def __init__(self) -> None:
                self.classification = "billing"

        result = evaluate_expression(
            "state.classification == 'billing'",
            {"state": StateProxy()},
        )
        assert result is True

    def test_nested_attribute_access(self) -> None:
        class Inner:
            value = 42

        class Outer:
            inner = Inner()

        result = evaluate_expression("obj.inner.value == 42", {"obj": Outer()})
        assert result is True


class TestSafeBuiltins:
    def test_len(self) -> None:
        result = evaluate_expression("len(items) > 0", {"items": [1, 2, 3]})
        assert result is True

    def test_str(self) -> None:
        result = evaluate_expression("str(x) == '42'", {"x": 42})
        assert result is True

    def test_int(self) -> None:
        result = evaluate_expression("int(x) == 42", {"x": "42"})
        assert result is True

    def test_float(self) -> None:
        result = evaluate_expression("float(x) > 3.0", {"x": "3.14"})
        assert result is True

    def test_bool(self) -> None:
        result = evaluate_expression("bool(x)", {"x": 1})
        assert result is True

    def test_abs(self) -> None:
        result = evaluate_expression("abs(x) == 5", {"x": -5})
        assert result is True

    def test_min_max(self) -> None:
        assert evaluate_expression("min(a, b) == 1", {"a": 1, "b": 2}) is True
        assert evaluate_expression("max(a, b) == 2", {"a": 1, "b": 2}) is True

    def test_isinstance(self) -> None:
        result = evaluate_expression("isinstance(x, str)", {"x": "hello"})
        assert result is True


class TestSafeEvalRestrictions:
    def test_import_not_allowed(self) -> None:
        with pytest.raises(SecurityError, match="not allowed"):
            evaluate_expression("__import__('os')", {})

    def test_exec_not_allowed(self) -> None:
        with pytest.raises(SecurityError, match="not allowed"):
            evaluate_expression("exec('print(1)')", {})

    def test_eval_not_allowed(self) -> None:
        with pytest.raises(SecurityError, match="not allowed"):
            evaluate_expression("eval('1+1')", {})

    def test_open_not_allowed(self) -> None:
        with pytest.raises(SecurityError, match="not allowed"):
            evaluate_expression("open('/etc/passwd')", {})

    def test_lambda_not_allowed(self) -> None:
        with pytest.raises(SecurityError):
            evaluate_expression("(lambda: 1)()", {})

    def test_comprehension_not_allowed(self) -> None:
        with pytest.raises(SecurityError, match="Disallowed"):
            evaluate_expression("[x for x in range(10)]", {})

    def test_syntax_error_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            evaluate_expression("if True: pass", {})

    def test_dunder_builtins_reference_blocked(self) -> None:
        # Dunder names are rejected at validation, before evaluation reaches the empty __builtins__.
        with pytest.raises(SecurityError):
            evaluate_expression("__builtins__", {})


class TestStateContainers:
    """A router reads the state's lists and dicts as it would the native ones."""

    STATE = {
        "tags": ["a", "b", "a"],
        "meta": {"k": [1], "name": "x"},
        "again": {"k": [1]},
        "order": {"items": [7], "get": "field", "values": 3},
        "matrix": [[1], [2]],
    }

    def _eval(self, expression: str) -> object:
        return evaluate_expression(expression, {"state": _build_state_proxy(self.STATE)})

    @pytest.mark.parametrize(
        "expression",
        [
            "state.tags == ['a', 'b', 'a']",
            "['a', 'b', 'a'] == state.tags",
            "state.tags != ['a']",
            "state.meta == {'k': [1], 'name': 'x'}",
            "state.meta.k == [1]",
            "state.meta.k == state.again.k",
            "state.matrix[0] == [1]",
            "state.tags.count('a') == 2",
            "state.tags.index('b') == 1",
            "state.matrix.count([2]) == 1",
            "state.matrix.index(state.again.k) == 0",
            "state.meta.get('name') == 'x'",
            "state.meta.get('k') == [1]",
            "state.meta.get('absent') is None",
            "state.meta.get('absent', 'fallback') == 'fallback'",
            "'k' in state.meta.keys()",
            "[1] in state.meta.values()",
            "('k', [1]) in state.meta.items()",
            "state.get('meta').get('k')[0] == 1",
            "'tags' in state.keys()",
            "state.get('absent', 5) == 5",
        ],
    )
    def test_reads_like_the_native_containers(self, expression: str) -> None:
        assert self._eval(expression) is True

    @pytest.mark.parametrize(
        "expression",
        ["state.tags == ['a']", "state.meta == {'k': [1]}", "state.tags == 'a'"],
    )
    def test_different_containers_are_not_equal(self, expression: str) -> None:
        assert self._eval(expression) is False

    @pytest.mark.parametrize(
        "expression",
        ["state.order.items == [7]", "state.order.get == 'field'", "state.order.values == 3"],
    )
    def test_a_key_wins_over_the_method_of_the_same_name(self, expression: str) -> None:
        assert self._eval(expression) is True

    def test_a_read_never_hands_out_the_state_itself(self) -> None:
        state = {"meta": {"inner": ["x"]}}
        proxy = _build_state_proxy(state)
        values = evaluate_expression("state.meta.values()", {"state": proxy})
        assert values == [["x"]]
        assert not isinstance(values[0], list)


class TestMissingKeyNamedLikeAMethod:
    """``state.items`` with no such key is ``None``; only a call reaches the method."""

    STATE = {"order": {"id": 1}}

    def _eval(self, expression: str) -> object:
        return evaluate_expression(expression, {"state": _build_state_proxy(self.STATE)})

    @pytest.mark.parametrize("name", ["items", "values", "keys", "get"])
    @pytest.mark.parametrize("receiver", ["state", "state.order"])
    @pytest.mark.parametrize(
        "template",
        [
            "{path} is None",
            "{path} == None",
            "not {path}",
            "bool({path}) == False",
            "str({path}) == 'None'",
            "({path} or 'absent') == 'absent'",
            "('set' if {path} else 'absent') == 'absent'",
            "isinstance({path}, int) == False",
        ],
    )
    def test_it_is_none_in_every_expression(self, receiver: str, name: str, template: str) -> None:
        assert self._eval(template.format(path=f"{receiver}.{name}")) is True

    @pytest.mark.parametrize(
        "expression",
        [
            "state.keys() == ['order']",
            "state.order.keys() == ['id']",
            "state.order.values() == [1]",
            "state.order.items() == [('id', 1)]",
            "state.order.get('id') == 1",
            "state.get('order').get('id') == 1",
        ],
    )
    def test_the_call_still_reaches_the_method(self, expression: str) -> None:
        assert self._eval(expression) is True

    def test_a_native_dict_keeps_its_own_method(self) -> None:
        assert evaluate_expression("x.get('a') == 1", {"x": {"a": 1}}) is True

    async def test_a_router_on_the_missing_key_takes_the_default(self) -> None:
        step = StepDefinition(
            id="r",
            type=StepType.ROUTER,
            conditions=[{"expression": "state.items", "target": "has_items"}],
            default="empty",
        )
        context = StepContext(step_definition=step, state_manager=StateManager())
        assert (await RouterStep().execute(context)).output == "empty"
