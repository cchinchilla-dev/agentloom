"""Tests for the router step expression evaluation."""

from __future__ import annotations

import pytest

from agentloom.exceptions import SecurityError
from agentloom.steps.router import evaluate_expression


class TestExpressionEvaluation:
    """Test basic expression evaluation against a namespace."""

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
    """Test attribute-style access for state proxy objects."""

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
    """Test that allowed builtin functions work in expressions."""

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
    """Test that unsafe operations are rejected."""

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
        # Any reference to a dunder identifier is rejected at validation
        # time, even before evaluation would hit the empty __builtins__.
        with pytest.raises(SecurityError):
            evaluate_expression("__builtins__", {})
