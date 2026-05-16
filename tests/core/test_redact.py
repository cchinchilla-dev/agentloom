"""Tests for the state redaction surface."""

from __future__ import annotations

import pytest

from agentloom.core.redact import (
    ENV_VAR,
    RedactionPolicy,
    is_redacted,
    redact_state,
)


class TestRedactionPolicy:
    def test_empty_policy_is_falsy(self) -> None:
        assert not RedactionPolicy()
        assert not RedactionPolicy([])

    def test_non_empty_policy_is_truthy(self) -> None:
        assert RedactionPolicy(["api_key"])

    def test_match_exact_key(self) -> None:
        policy = RedactionPolicy(["api_key"])
        assert policy.matches("api_key")
        assert not policy.matches("other_key")

    def test_match_glob_anywhere(self) -> None:
        policy = RedactionPolicy(["*token*"])
        assert policy.matches("access_token")
        assert policy.matches("refresh_token_v2")
        assert policy.matches("token")
        assert not policy.matches("user_id")

    def test_match_dotted_path(self) -> None:
        policy = RedactionPolicy(["credentials.access_token"])
        assert policy.matches("credentials.access_token")

    def test_merge_returns_union(self) -> None:
        a = RedactionPolicy(["api_key"])
        b = RedactionPolicy(["*token*"])
        merged = a.merge(b)
        assert merged.matches("api_key")
        assert merged.matches("session_token")

    def test_from_env_parses_csv(self) -> None:
        policy = RedactionPolicy.from_env({ENV_VAR: "api_key , password ,*token*"})
        assert policy.matches("api_key")
        assert policy.matches("password")
        assert policy.matches("refresh_token")

    def test_from_env_empty_when_unset(self) -> None:
        policy = RedactionPolicy.from_env({})
        assert not policy


class TestRedactState:
    def test_identity_when_policy_empty(self) -> None:
        state = {"a": 1, "b": "x"}
        assert redact_state(state, RedactionPolicy()) is state

    def test_redacts_top_level_match(self) -> None:
        state = {"api_key": "sk-leak", "user": "alice"}
        result = redact_state(state, RedactionPolicy(["api_key"]))
        assert is_redacted(result["api_key"])
        assert result["user"] == "alice"

    def test_same_value_produces_same_sentinel(self) -> None:
        s1 = redact_state({"k": "secret"}, RedactionPolicy(["k"]))
        s2 = redact_state({"k": "secret"}, RedactionPolicy(["k"]))
        assert s1["k"] == s2["k"]
        assert is_redacted(s1["k"])

    def test_different_value_produces_different_sentinel(self) -> None:
        s1 = redact_state({"k": "one"}, RedactionPolicy(["k"]))
        s2 = redact_state({"k": "two"}, RedactionPolicy(["k"]))
        assert s1["k"] != s2["k"]

    def test_recurses_into_nested_dict(self) -> None:
        state = {"credentials": {"access_token": "t1", "kind": "bearer"}}
        result = redact_state(state, RedactionPolicy(["*access_token*"]))
        assert is_redacted(result["credentials"]["access_token"])
        assert result["credentials"]["kind"] == "bearer"

    def test_redacts_list_of_secrets(self) -> None:
        state = {"api_keys": ["k1", "k2", "k3"]}
        result = redact_state(state, RedactionPolicy(["api_keys"]))
        assert all(is_redacted(v) for v in result["api_keys"])

    def test_non_string_values_get_sentinel(self) -> None:
        state = {"port": 6379, "data": {"x": 1}}
        result = redact_state(state, RedactionPolicy(["port", "data"]))
        assert is_redacted(result["port"])
        assert isinstance(result["data"], dict)
        assert is_redacted(result["data"]["x"])

    def test_input_state_is_not_mutated(self) -> None:
        state = {"api_key": "sk", "user": "alice"}
        original = dict(state)
        _ = redact_state(state, RedactionPolicy(["api_key"]))
        assert state == original


class TestIsRedacted:
    def test_recognises_sentinel(self) -> None:
        sentinel = redact_state({"k": "x"}, RedactionPolicy(["k"]))["k"]
        assert is_redacted(sentinel)

    def test_rejects_plain_strings(self) -> None:
        assert not is_redacted("plain")
        assert not is_redacted("<REDACTED:other>")
        assert not is_redacted(42)
        assert not is_redacted(None)


class TestCycleSafe:
    """``_walk`` must not infinite-loop on self-referential state."""

    def test_self_reference_does_not_infinite_loop(self) -> None:
        d: dict[str, object] = {"a": 1, "secret": "sk-x"}
        d["self"] = d
        out = redact_state(d, RedactionPolicy(["secret"]))
        assert out["a"] == 1
        assert is_redacted(out["secret"])
        assert out["self"] == "<cycle>"

    def test_list_cycle_does_not_infinite_loop(self) -> None:
        lst: list[object] = [1, 2]
        lst.append(lst)
        d = {"items": lst, "secret": "x"}
        out = redact_state(d, RedactionPolicy(["secret"]))
        assert out["items"][:2] == [1, 2]
        assert out["items"][2] == "<cycle>"


class TestNonStringKeys:
    """``fnmatch.fnmatchcase`` requires str args.

    State deserialized from JSON / pickle / tool output can carry int or
    tuple keys; coerce to ``str`` before matching so a non-string key
    does not crash the entire checkpoint write.
    """

    def test_int_key_matched_via_str_coercion(self) -> None:
        out = redact_state({1: "numeric-keyed"}, RedactionPolicy(["1"]))
        assert is_redacted(out[1])

    def test_tuple_key_does_not_crash(self) -> None:
        out = redact_state({(1, 2): "tuple"}, RedactionPolicy(["api_key"]))
        assert out[(1, 2)] == "tuple"


class TestEngineResumeWarnsOnSentinels:
    """``WorkflowEngine.from_checkpoint`` surfaces redacted keys on resume."""

    async def test_warns_when_state_carries_redaction_sentinel(
        self, caplog: object
    ) -> None:
        import logging
        from unittest.mock import MagicMock

        from agentloom.checkpointing.base import CheckpointData
        from agentloom.core.engine import WorkflowEngine

        caplog.set_level(logging.WARNING, logger="agentloom.engine")  # type: ignore[attr-defined]
        sentinel = redact_state({"api_key": "sk-real"}, RedactionPolicy(["api_key"]))[
            "api_key"
        ]
        cp = CheckpointData(
            workflow_name="resume-test",
            run_id="r1",
            workflow_definition={
                "name": "resume-test",
                "version": "1.0",
                "config": {"provider": "mock", "model": "x"},
                "state": {"api_key": sentinel},
                "steps": [{"id": "s1", "type": "llm_call", "prompt": "hi"}],
            },
            state={"api_key": sentinel, "user": "alice"},
        )
        checkpointer = MagicMock()
        engine = await WorkflowEngine.from_checkpoint(
            checkpoint_data=cp, checkpointer=checkpointer
        )
        messages = [r.getMessage() for r in caplog.records]  # type: ignore[attr-defined]
        assert any("api_key" in m and "redacted" in m for m in messages), messages
        snapshot = await engine.state.get_state_snapshot()
        assert snapshot["api_key"] == sentinel
        assert snapshot["user"] == "alice"

    async def test_no_warning_when_no_sentinels(self, caplog: object) -> None:
        import logging
        from unittest.mock import MagicMock

        from agentloom.checkpointing.base import CheckpointData
        from agentloom.core.engine import WorkflowEngine

        caplog.set_level(logging.WARNING, logger="agentloom.engine")  # type: ignore[attr-defined]
        cp = CheckpointData(
            workflow_name="resume-clean",
            run_id="r2",
            workflow_definition={
                "name": "resume-clean",
                "version": "1.0",
                "config": {"provider": "mock", "model": "x"},
                "state": {"user": "alice"},
                "steps": [{"id": "s1", "type": "llm_call", "prompt": "hi"}],
            },
            state={"user": "alice"},
        )
        await WorkflowEngine.from_checkpoint(
            checkpoint_data=cp, checkpointer=MagicMock()
        )
        for r in caplog.records:  # type: ignore[attr-defined]
            assert "redacted" not in r.getMessage()
