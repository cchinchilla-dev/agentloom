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
