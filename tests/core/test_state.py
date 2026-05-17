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
        # Internal-only sync accessors. See StateManager.{_get,_set}_sync_unsafe
        # docstring — these bypass the async lock and must not be used from
        # concurrent code paths.
        sm = StateManager(initial_state={"key": "val"})
        assert sm._get_sync_unsafe("key") == "val"
        sm._set_sync_unsafe("key", "new_val")
        assert sm._get_sync_unsafe("key") == "new_val"

    def test_state_property(self) -> None:
        sm = StateManager(initial_state={"x": 42})
        assert sm.state["x"] == 42

    def test_public_sync_aliases_removed(self) -> None:
        # The unsafe sync accessors are intentionally underscored. Public
        # `get_sync` / `set_sync` aliases were dropped in 0.5.0 to avoid
        # advertising a lock-bypass API surface.
        sm = StateManager()
        assert not hasattr(sm, "get_sync")
        assert not hasattr(sm, "set_sync")


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
        sm._set_sync_unsafe("config.debug", True)
        assert sm._get_sync_unsafe("config.debug") is True


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

    async def test_set_nested_list_expansion_error_is_clear(self) -> None:
        """Out-of-range list writes must point the user at the fact that lists
        are not auto-expanded — the previous message only said "out of range"."""
        sm = StateManager(initial_state={"items": []})
        with pytest.raises(IndexError, match="not auto-expanded"):
            await sm.set("items[0].name", "x")

    def test_sync_get_with_index(self) -> None:
        sm = StateManager(initial_state={"items": ["a", "b"]})
        assert sm._get_sync_unsafe("items[0]") == "a"

    def test_sync_set_with_index(self) -> None:
        sm = StateManager(initial_state={"items": ["old"]})
        sm._set_sync_unsafe("items[0]", "new")
        assert sm._get_sync_unsafe("items[0]") == "new"

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

    async def test_set_string_key_on_list_raises_state_write_error(self) -> None:
        # 0.5.0 reclassified this error: writing a dotted string segment onto
        # a list intermediate is the same shape of "wrong-type intermediate"
        # as the scalar-overwrite refusal, so it raises ``StateWriteError``
        # uniformly. Pre-0.5.0 callers that caught only ``TypeError`` would
        # have missed this case.
        from agentloom.exceptions import StateWriteError

        sm = StateManager(initial_state={"data": [1, 2]})
        with pytest.raises(StateWriteError, match="not a dict"):
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
        assert restored._get_sync_unsafe("key") == "value"

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


class TestAtomicUpdate:
    """#056 regression — ``StateManager.update`` is atomic across read/modify/write.

    The legacy ``get`` then ``set`` pattern drops the lock between the two
    awaits and collapses parallel writers to 1. The new ``update(key, fn)``
    primitive holds the lock for the full callback invocation; under 50
    parallel writers it must produce final 50, not final 1.
    """

    async def test_update_is_atomic_under_50_parallel_writers(self) -> None:
        import anyio

        sm = StateManager(initial_state={"counter": 0})

        async def bump() -> None:
            await sm.update("counter", lambda c: (c or 0) + 1)

        async with anyio.create_task_group() as tg:
            for _ in range(50):
                tg.start_soon(bump)
        assert await sm.get("counter") == 50

    async def test_update_returns_new_value(self) -> None:
        sm = StateManager(initial_state={"counter": 10})
        new = await sm.update("counter", lambda c: c * 2)
        assert new == 20
        assert await sm.get("counter") == 20

    async def test_update_initialises_missing_key_with_none_seed(self) -> None:
        sm = StateManager()
        result = await sm.update("new_key", lambda c: 7 if c is None else c + 1)
        assert result == 7
        assert await sm.get("new_key") == 7

    async def test_get_then_set_remains_racy(self) -> None:
        """Regression net: we must NOT accidentally over-lock ``get`` + ``set``.

        The contract is documented as racy — anyone needing atomicity must
        switch to ``update``. If this assertion ever starts failing (i.e.
        ``get`` + ``set`` becomes atomic), revisit the lock granularity:
        an unintentional widening makes every read serialise with the
        write queue and tanks throughput in busy workflows.
        """
        import anyio

        sm = StateManager(initial_state={"counter": 0})

        async def bump_via_get_set() -> None:
            cur = await sm.get("counter")
            await anyio.sleep(0.001)
            await sm.set("counter", cur + 1)

        async with anyio.create_task_group() as tg:
            for _ in range(50):
                tg.start_soon(bump_via_get_set)
        assert await sm.get("counter") < 50


class TestDottedScalarOverwriteRefused:
    """#056/F52 regression — dotted write must not silently replace a scalar.

    Pre-0.5.0, writing ``user.name`` when ``state.user`` was a string
    replaced the string with a fresh dict and lost the original value
    with no warning. Now ``_set_nested`` raises ``StateWriteError`` so
    the workflow author either picks a different output key or
    explicitly overwrites ``user`` at the parent path first.
    """

    async def test_refuses_overwrite_of_scalar_parent(self) -> None:
        from agentloom.exceptions import StateWriteError

        sm = StateManager(initial_state={"user": "alice"})
        with pytest.raises(StateWriteError) as exc_info:
            await sm.set("user.name", "bob")
        msg = str(exc_info.value)
        assert "user" in msg and "str" in msg
        # The original scalar must survive the refusal.
        assert await sm.get("user") == "alice"

    async def test_auto_creates_intermediate_when_missing(self) -> None:
        sm = StateManager()
        await sm.set("user.name", "bob")
        assert await sm.get("user.name") == "bob"

    async def test_traverses_existing_dict_intermediate(self) -> None:
        sm = StateManager(initial_state={"user": {"id": 1}})
        await sm.set("user.name", "bob")
        assert await sm.get("user.name") == "bob"
        assert await sm.get("user.id") == 1

    async def test_refusal_mentions_traversed_prefix(self) -> None:
        from agentloom.exceptions import StateWriteError

        sm = StateManager(initial_state={"user": {"profile": "guest"}})
        with pytest.raises(StateWriteError) as exc_info:
            await sm.set("user.profile.name", "bob")
        # The message should point at ``user.profile`` (the scalar that
        # would be overwritten), not just ``user``.
        assert "user.profile" in str(exc_info.value)

    async def test_refuses_traversal_through_list_intermediate(self) -> None:
        """Symmetric to the scalar refusal — a list intermediate with a
        string next-segment is the same "wrong-type intermediate" footgun.

        Pre-fix this leaked a generic ``TypeError("Cannot set key 'name' on
        list ...")``; now ``StateWriteError`` is raised so callers that
        catch the dedicated exception class don't miss the case.
        """
        from agentloom.exceptions import StateWriteError

        sm = StateManager(initial_state={"users": [{"id": 1}]})
        with pytest.raises(StateWriteError) as exc_info:
            await sm.set("users.name", "bob")
        msg = str(exc_info.value)
        assert "users" in msg and "list" in msg
        # Original list survives the refusal.
        assert await sm.get("users") == [{"id": 1}]

    async def test_refuses_string_segment_through_mid_loop_list(self) -> None:
        """Three-segment path through a list intermediate trips the loop-body
        refusal (line 246 — the ``not isinstance(current, dict)`` branch
        inside the ``for part in parts[:-1]`` loop). The two-segment case
        ``users.name`` exits the loop with ``current`` still the root dict
        and trips the final-segment refusal instead; this test pins the
        in-loop branch so both code paths are exercised."""
        from agentloom.exceptions import StateWriteError

        sm = StateManager(initial_state={"users": [{"id": 1}]})
        with pytest.raises(StateWriteError) as exc_info:
            await sm.set("users.profile.name", "bob")
        msg = str(exc_info.value)
        assert "users" in msg and "list" in msg
        assert await sm.get("users") == [{"id": 1}]
