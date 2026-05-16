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


class TestWorkflowDefinitionRejectsUnknownKeys:
    """``WorkflowDefinition`` uses ``extra="forbid"`` so a typo in
    ``state_schema:`` (or any other top-level field) fails loud at parse
    time instead of silently dropping the field.
    """

    def test_typo_in_state_schema_raises(self) -> None:
        from pydantic import ValidationError

        from agentloom.core.models import WorkflowConfig, WorkflowDefinition

        with pytest.raises(ValidationError):
            WorkflowDefinition(
                name="t",
                config=WorkflowConfig(provider="mock", model="x"),
                state={"api_key": "sk"},
                stat_schema={"api_key": {"redact": True}},  # type: ignore[arg-type]
                steps=[],
            )

    def test_arbitrary_unknown_top_level_key_raises(self) -> None:
        from pydantic import ValidationError

        from agentloom.core.models import WorkflowConfig, WorkflowDefinition

        with pytest.raises(ValidationError):
            WorkflowDefinition(
                name="t",
                config=WorkflowConfig(provider="mock", model="x"),
                random_field_that_does_not_exist=True,  # type: ignore[arg-type]
                steps=[],
            )


class TestDoubleRedactionStable:
    """Idempotency: already-redacted value preserved byte-for-byte."""

    def test_already_redacted_value_unchanged(self) -> None:
        policy = RedactionPolicy(["k"])
        once = redact_state({"k": "secret"}, policy)
        twice = redact_state(once, policy)
        assert is_redacted(twice["k"])
        assert twice["k"] == once["k"]

    def test_user_provided_sentinel_shaped_string_is_preserved(self) -> None:
        # A user who legitimately stores ``<REDACTED:sha256=abcdef0123456789>``
        # as a state value shouldn't see it mangled. The redaction policy
        # applies to keys, not values, so a non-flagged key carrying a
        # sentinel-shaped string is left untouched.
        policy = RedactionPolicy(["api_key"])
        out = redact_state(
            {"api_key": "sk-real", "note": "<REDACTED:sha256=abcdef0123456789>"},
            policy,
        )
        assert is_redacted(out["api_key"])
        assert out["note"] == "<REDACTED:sha256=abcdef0123456789>"


class TestRedactionContainerCoverage:
    """Pinned edge cases for container types ``_walk`` does NOT recurse
    into. Documents the supported subset so a regression that quietly
    starts redacting (or fails to redact) tuples / sets / Pydantic
    models is caught.
    """

    def test_tuple_value_passes_through(self) -> None:
        # Tuples are not recursed; the workflow convention is dict /
        # list / scalar values. A flagged key whose value is a tuple
        # gets a single sentinel, not element-wise sentinels.
        out = redact_state({"creds": ("a", "b")}, RedactionPolicy(["creds"]))
        assert is_redacted(out["creds"])

    def test_set_value_passes_through(self) -> None:
        # Sets are unordered and not part of the JSON-serialisable
        # state contract; ensure they don't crash redaction.
        out = redact_state({"creds": {"a", "b"}}, RedactionPolicy(["creds"]))
        assert is_redacted(out["creds"])

    def test_pydantic_model_value_replaced_with_sentinel(self) -> None:
        # When a Pydantic BaseModel ends up in state (rare but legal in
        # Python-built workflows), the flagged key is replaced wholesale
        # with the sentinel — ``str(model)`` is hashed and the original
        # never lands in the output.
        from pydantic import BaseModel

        class Creds(BaseModel):
            api_key: str

        out = redact_state(
            {"creds": Creds(api_key="sk-leak")}, RedactionPolicy(["creds"])
        )
        assert is_redacted(out["creds"])


class TestRedactionGlobMetacharacters:
    """``fnmatch.fnmatchcase`` interprets pattern brackets as character
    classes. Pin the contract so a future migration to a different
    matcher is a deliberate decision.
    """

    def test_bracket_pattern_treated_as_char_class(self) -> None:
        # Pattern ``foo[1]`` matches the key ``foo1``, NOT the literal
        # key ``foo[1]``.
        policy = RedactionPolicy(["foo[1]"])
        out = redact_state({"foo1": "v", "foo[1]": "w"}, policy)
        assert is_redacted(out["foo1"])
        assert out["foo[1]"] == "w"


class TestRedactionEdgeCases:
    """Behaviours pinned during the audit — not bugs, but documented
    contracts that future changes must not regress.
    """

    def test_glob_match_is_case_sensitive(self) -> None:
        # ``fnmatch.fnmatchcase`` is case-sensitive on purpose: pattern
        # ``API_KEY`` does NOT mask ``state.api_key``.
        policy = RedactionPolicy(["API_KEY"])
        out = redact_state({"api_key": "x"}, policy)
        assert out["api_key"] == "x"

    def test_empty_list_value_passes_through(self) -> None:
        # A flagged key whose value is an empty list renders as ``[]`` —
        # the empty container reveals shape but no plaintext. Pinned so
        # any future tightening (e.g. emit a sentinel even for empties)
        # is a deliberate change.
        out = redact_state({"api_keys": []}, RedactionPolicy(["api_keys"]))
        assert out["api_keys"] == []

    def test_empty_dict_value_redacts_to_empty_dict(self) -> None:
        out = redact_state({"creds": {}}, RedactionPolicy(["creds"]))
        assert out["creds"] == {}

    def test_none_value_sentinel_stable(self) -> None:
        # ``str(None)`` is fed into sha256, so redacting two ``None``
        # values produces the SAME sentinel. ``None`` and ``""`` happen
        # to collide because the code maps both to the empty string;
        # pinned so a future ``repr()`` switch is intentional.
        a = redact_state({"k": None}, RedactionPolicy(["k"]))
        b = redact_state({"k": ""}, RedactionPolicy(["k"]))
        assert a["k"] == b["k"]
        assert is_redacted(a["k"])

    def test_pattern_matches_both_bare_and_dotted_path(self) -> None:
        # Pattern matches against the bare key AND the dotted path so
        # ``"access_token"`` redacts both ``state.access_token`` and
        # ``state.credentials.access_token``.
        policy = RedactionPolicy(["access_token"])
        out = redact_state(
            {
                "access_token": "T1",
                "credentials": {"access_token": "T2", "user": "alice"},
            },
            policy,
        )
        assert is_redacted(out["access_token"])
        assert is_redacted(out["credentials"]["access_token"])
        assert out["credentials"]["user"] == "alice"
