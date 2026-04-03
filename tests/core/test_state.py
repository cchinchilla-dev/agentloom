"""Tests for the StateManager module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentloom.core.results import StepResult, StepStatus, TokenUsage
from agentloom.core.state import StateManager


class TestGetSet:
    """Test basic get/set operations."""

    async def test_set_and_get(self) -> None:
        sm = StateManager()
        await sm.set("key", "value")
        result = await sm.get("key")
        assert result == "value"

    async def test_get_default(self) -> None:
        sm = StateManager()
        result = await sm.get("missing", "default_val")
        assert result == "default_val"

    async def test_get_none_default(self) -> None:
        sm = StateManager()
        result = await sm.get("missing")
        assert result is None

    async def test_initial_state(self) -> None:
        sm = StateManager(initial_state={"x": 10, "y": 20})
        assert await sm.get("x") == 10
        assert await sm.get("y") == 20

    async def test_overwrite_value(self) -> None:
        sm = StateManager(initial_state={"key": "old"})
        await sm.set("key", "new")
        assert await sm.get("key") == "new"

    async def test_get_state_snapshot(self) -> None:
        sm = StateManager(initial_state={"a": 1})
        await sm.set("b", 2)
        snapshot = await sm.get_state_snapshot()
        assert snapshot == {"a": 1, "b": 2}

    async def test_snapshot_is_copy(self) -> None:
        sm = StateManager(initial_state={"a": 1})
        snapshot = await sm.get_state_snapshot()
        snapshot["a"] = 999
        assert await sm.get("a") == 1

    def test_sync_get_set(self) -> None:
        sm = StateManager(initial_state={"key": "val"})
        assert sm.get_sync("key") == "val"
        sm.set_sync("key", "new_val")
        assert sm.get_sync("key") == "new_val"

    def test_state_property(self) -> None:
        sm = StateManager(initial_state={"x": 42})
        assert sm.state["x"] == 42


class TestDottedKeys:
    """Test dotted key path resolution for nested state."""

    async def test_set_dotted_key(self) -> None:
        sm = StateManager()
        await sm.set("user.name", "Alice")
        result = await sm.get("user.name")
        assert result == "Alice"

    async def test_set_deep_dotted_key(self) -> None:
        sm = StateManager()
        await sm.set("a.b.c", "deep_value")
        result = await sm.get("a.b.c")
        assert result == "deep_value"

    async def test_get_dotted_key_from_initial_state(self) -> None:
        sm = StateManager(initial_state={"user": {"name": "Bob", "age": 30}})
        assert await sm.get("user.name") == "Bob"
        assert await sm.get("user.age") == 30

    async def test_get_dotted_key_missing_returns_default(self) -> None:
        sm = StateManager(initial_state={"user": {"name": "Bob"}})
        result = await sm.get("user.email", "none@example.com")
        assert result == "none@example.com"

    def test_sync_dotted_keys(self) -> None:
        sm = StateManager()
        sm.set_sync("config.debug", True)
        assert sm.get_sync("config.debug") is True


class TestStepResults:
    """Test step result storage and retrieval."""

    async def test_set_and_get_step_result(self) -> None:
        sm = StateManager()
        result = StepResult(
            step_id="step1",
            status=StepStatus.SUCCESS,
            output="answer text",
            duration_ms=100.0,
            token_usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            cost_usd=0.001,
        )
        await sm.set_step_result("step1", result)
        retrieved = await sm.get_step_result("step1")
        assert retrieved is not None
        assert retrieved.step_id == "step1"
        assert retrieved.status == StepStatus.SUCCESS
        assert retrieved.output == "answer text"

    async def test_get_step_status(self) -> None:
        sm = StateManager()
        result = StepResult(step_id="step1", status=StepStatus.FAILED, error="oops")
        await sm.set_step_result("step1", result)
        status = await sm.get_step_status("step1")
        assert status == StepStatus.FAILED

    async def test_get_missing_step_result(self) -> None:
        sm = StateManager()
        assert await sm.get_step_result("nonexistent") is None

    async def test_get_missing_step_status(self) -> None:
        sm = StateManager()
        assert await sm.get_step_status("nonexistent") is None

    async def test_step_result_stored_in_state(self) -> None:
        sm = StateManager()
        result = StepResult(
            step_id="s1",
            status=StepStatus.SUCCESS,
            output="the output",
        )
        await sm.set_step_result("s1", result)
        # Step output should be accessible via the state
        step_data = await sm.get("steps.s1.output")
        assert step_data == "the output"

    async def test_all_step_results(self) -> None:
        sm = StateManager()
        r1 = StepResult(step_id="a", status=StepStatus.SUCCESS, output="out_a")
        r2 = StepResult(step_id="b", status=StepStatus.SUCCESS, output="out_b")
        await sm.set_step_result("a", r1)
        await sm.set_step_result("b", r2)
        all_results = await sm.all_step_results()
        assert len(all_results) == 2
        assert "a" in all_results
        assert "b" in all_results


class TestArrayIndexPaths:
    """Test array index support in dotted key paths."""

    async def test_get_simple_index(self) -> None:
        sm = StateManager(initial_state={"items": ["a", "b", "c"]})
        assert await sm.get("items[0]") == "a"
        assert await sm.get("items[2]") == "c"

    async def test_get_nested_after_index(self) -> None:
        sm = StateManager(initial_state={"items": [{"name": "Alice"}, {"name": "Bob"}]})
        assert await sm.get("items[0].name") == "Alice"
        assert await sm.get("items[1].name") == "Bob"

    async def test_get_negative_index(self) -> None:
        sm = StateManager(initial_state={"items": [1, 2, 3]})
        assert await sm.get("items[-1]") == 3
        assert await sm.get("items[-2]") == 2

    async def test_get_multi_dimensional(self) -> None:
        sm = StateManager(initial_state={"matrix": [[1, 2], [3, 4]]})
        assert await sm.get("matrix[0][1]") == 2
        assert await sm.get("matrix[1][0]") == 3

    async def test_get_out_of_bounds_returns_default(self) -> None:
        sm = StateManager(initial_state={"items": [1]})
        assert await sm.get("items[5]") is None
        assert await sm.get("items[5]", "fallback") == "fallback"

    async def test_get_negative_out_of_bounds(self) -> None:
        sm = StateManager(initial_state={"items": [1]})
        assert await sm.get("items[-10]") is None

    async def test_get_index_on_non_list(self) -> None:
        sm = StateManager(initial_state={"name": "Alice"})
        assert await sm.get("name[0]") is None

    async def test_get_deep_mixed_path(self) -> None:
        sm = StateManager(initial_state={"a": {"b": [{}, {"c": "val"}]}})
        assert await sm.get("a.b[1].c") == "val"

    async def test_set_at_index(self) -> None:
        sm = StateManager(initial_state={"items": ["old", "keep"]})
        await sm.set("items[0]", "new")
        assert await sm.get("items[0]") == "new"
        assert await sm.get("items[1]") == "keep"

    async def test_set_nested_after_index(self) -> None:
        sm = StateManager(initial_state={"items": [{"name": "old"}]})
        await sm.set("items[0].name", "new")
        assert await sm.get("items[0].name") == "new"

    async def test_set_out_of_bounds_raises(self) -> None:
        sm = StateManager(initial_state={"items": [1]})
        with pytest.raises(IndexError):
            await sm.set("items[5]", "x")

    def test_sync_get_with_index(self) -> None:
        sm = StateManager(initial_state={"items": ["a", "b"]})
        assert sm.get_sync("items[0]") == "a"

    def test_sync_set_with_index(self) -> None:
        sm = StateManager(initial_state={"items": ["old"]})
        sm.set_sync("items[0]", "new")
        assert sm.get_sync("items[0]") == "new"

    async def test_empty_path_raises(self) -> None:
        sm = StateManager()
        with pytest.raises(ValueError, match="must not be empty"):
            await sm.get("")

    async def test_empty_segment_raises(self) -> None:
        sm = StateManager()
        with pytest.raises(ValueError, match="Empty segment"):
            await sm.get("a..b")

    async def test_set_through_scalar_raises_type_error(self) -> None:
        sm = StateManager(initial_state={"items": ["hello"]})
        with pytest.raises(TypeError, match="Expected dict or list"):
            await sm.set("items[0].name", "x")

    async def test_set_string_key_on_list_raises_type_error(self) -> None:
        sm = StateManager(initial_state={"data": [1, 2]})
        with pytest.raises(TypeError, match="Cannot set key"):
            await sm.set("data.foo", "x")


class TestCheckpoint:
    """Test checkpoint save and load functionality."""

    async def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        sm = StateManager(initial_state={"question": "hello"})
        result = StepResult(
            step_id="step1",
            status=StepStatus.SUCCESS,
            output="world",
            duration_ms=50.0,
        )
        await sm.set_step_result("step1", result)

        checkpoint_path = tmp_path / "checkpoint.json"
        await sm.save_checkpoint(checkpoint_path)

        # Verify file exists and is valid JSON
        assert checkpoint_path.exists()
        data = json.loads(checkpoint_path.read_text())
        assert "state" in data
        assert "step_results" in data

    async def test_load_checkpoint_restores_state(self, tmp_path: Path) -> None:
        sm = StateManager(initial_state={"key": "value"})
        result = StepResult(
            step_id="s1",
            status=StepStatus.SUCCESS,
            output="result_data",
        )
        await sm.set_step_result("s1", result)

        checkpoint_path = tmp_path / "checkpoint.json"
        await sm.save_checkpoint(checkpoint_path)

        restored = await StateManager.from_checkpoint(checkpoint_path)
        assert restored.get_sync("key") == "value"

    async def test_load_checkpoint_restores_step_results(self, tmp_path: Path) -> None:
        sm = StateManager()
        result = StepResult(
            step_id="s1",
            status=StepStatus.SUCCESS,
            output="restored_output",
            duration_ms=25.0,
        )
        await sm.set_step_result("s1", result)

        checkpoint_path = tmp_path / "checkpoint.json"
        await sm.save_checkpoint(checkpoint_path)

        restored = await StateManager.from_checkpoint(checkpoint_path)
        step_result = restored._step_results.get("s1")
        assert step_result is not None
        assert step_result.status == StepStatus.SUCCESS
        assert step_result.output == "restored_output"

    async def test_checkpoint_creates_parent_dirs(self, tmp_path: Path) -> None:
        sm = StateManager(initial_state={"a": 1})
        nested_path = tmp_path / "sub" / "dir" / "checkpoint.json"
        await sm.save_checkpoint(nested_path)
        assert nested_path.exists()
