"""Tests for the state redaction surface."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from agentloom.core.models import StepDefinition, WorkflowDefinition
from agentloom.core.redact import (
    ENV_VAR,
    RedactionPolicy,
    is_redacted,
    redact_model,
    redact_state,
)
from agentloom.core.results import StepResult


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
    """Non-string keys are coerced to ``str``, the only type ``fnmatch.fnmatchcase`` takes.

    State from JSON, pickle or tool output can carry int or tuple keys, which must not crash the
    checkpoint write.
    """

    def test_int_key_matched_via_str_coercion(self) -> None:
        out = redact_state({1: "numeric-keyed"}, RedactionPolicy(["1"]))
        assert is_redacted(out[1])

    def test_tuple_key_does_not_crash(self) -> None:
        out = redact_state({(1, 2): "tuple"}, RedactionPolicy(["api_key"]))
        assert out[(1, 2)] == "tuple"


class TestEngineResumeWarnsOnSentinels:
    """``WorkflowEngine.from_checkpoint`` surfaces redacted keys on resume."""

    async def test_warns_when_state_carries_redaction_sentinel(self, caplog: object) -> None:
        import logging
        from unittest.mock import MagicMock

        from agentloom.checkpointing.base import CheckpointData
        from agentloom.core.engine import WorkflowEngine

        caplog.set_level(logging.WARNING, logger="agentloom.engine")  # type: ignore[attr-defined]
        sentinel = redact_state({"api_key": "sk-real"}, RedactionPolicy(["api_key"]))["api_key"]
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
        engine = await WorkflowEngine.from_checkpoint(checkpoint_data=cp, checkpointer=checkpointer)
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
        await WorkflowEngine.from_checkpoint(checkpoint_data=cp, checkpointer=MagicMock())
        for r in caplog.records:  # type: ignore[attr-defined]
            assert "redacted" not in r.getMessage()


class TestWorkflowDefinitionRejectsUnknownKeys:
    """``WorkflowDefinition`` forbids unknown keys: a typo fails at parse time, not silently."""

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
        # Redaction applies to keys: a sentinel-shaped value under an unflagged key stays as is.
        policy = RedactionPolicy(["api_key"])
        out = redact_state(
            {"api_key": "sk-real", "note": "<REDACTED:sha256=abcdef0123456789>"},
            policy,
        )
        assert is_redacted(out["api_key"])
        assert out["note"] == "<REDACTED:sha256=abcdef0123456789>"


class TestRedactionContainerCoverage:
    """Container types ``_walk`` does not recurse into: tuples, sets and Pydantic models."""

    def test_tuple_value_passes_through(self) -> None:
        # A flagged tuple gets a single sentinel, not one per element.
        out = redact_state({"creds": ("a", "b")}, RedactionPolicy(["creds"]))
        assert is_redacted(out["creds"])

    def test_set_value_passes_through(self) -> None:
        # Sets fall outside the JSON state contract, but they must not crash redaction.
        out = redact_state({"creds": {"a", "b"}}, RedactionPolicy(["creds"]))
        assert is_redacted(out["creds"])

    def test_pydantic_model_value_replaced_with_sentinel(self) -> None:
        # A flagged model is replaced wholesale by the sentinel of ``str(model)``.
        from pydantic import BaseModel

        class Creds(BaseModel):
            api_key: str

        out = redact_state({"creds": Creds(api_key="sk-leak")}, RedactionPolicy(["creds"]))
        assert is_redacted(out["creds"])


class TestRedactionGlobMetacharacters:
    """Pattern brackets are character classes, as ``fnmatch.fnmatchcase`` reads them."""

    def test_bracket_pattern_treated_as_char_class(self) -> None:
        policy = RedactionPolicy(["foo[1]"])
        out = redact_state({"foo1": "v", "foo[1]": "w"}, policy)
        assert is_redacted(out["foo1"])
        assert out["foo[1]"] == "w"


class TestRedactionEdgeCases:
    """Edge-case contracts of redaction, pinned so that changing one is a deliberate choice."""

    def test_glob_match_is_case_sensitive(self) -> None:
        # Case-sensitive on purpose, as ``fnmatch.fnmatchcase`` is.
        policy = RedactionPolicy(["API_KEY"])
        out = redact_state({"api_key": "x"}, policy)
        assert out["api_key"] == "x"

    def test_empty_list_value_passes_through(self) -> None:
        # An empty container reveals its shape but no plaintext, so it stays ``[]``.
        out = redact_state({"api_keys": []}, RedactionPolicy(["api_keys"]))
        assert out["api_keys"] == []

    def test_empty_dict_value_redacts_to_empty_dict(self) -> None:
        out = redact_state({"creds": {}}, RedactionPolicy(["creds"]))
        assert out["creds"] == {}

    def test_none_value_sentinel_stable(self) -> None:
        # ``None`` and ``""`` share a sentinel: both hash as the empty string.
        a = redact_state({"k": None}, RedactionPolicy(["k"]))
        b = redact_state({"k": ""}, RedactionPolicy(["k"]))
        assert a["k"] == b["k"]
        assert is_redacted(a["k"])

    def test_pattern_matches_both_bare_and_dotted_path(self) -> None:
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


class _Inner(BaseModel):
    token_count: int = 0
    label: str = ""
    extra: dict[str, Any] = {}


class _Outer(BaseModel):
    inner: _Inner | None = None
    items: list[_Inner] = []
    by_name: dict[str, _Inner] = {}
    free: dict[str, Any] = {}
    texts: dict[str, str] = {}
    counts: dict[str, int] = {}
    either: Any | int = None
    choice: str | _Inner = ""
    paths: list[str] | None = None
    table: dict[str, int] | None = None
    raw: Any = None
    tagged: Annotated[_Inner, "tag"] | None = None
    tagged_items: list[Annotated[_Inner, "tag"]] = []
    secrets: dict[str, str] | None = None
    notes: list[dict[str, Any]] | None = None
    maybe_texts: dict[str, str | None] = {}


class TestRedactModel:
    """``redact_model`` redacts the workflow's data and leaves the model's own fields typed."""

    POLICY = RedactionPolicy(["*token*", "api_key"])

    def test_the_models_own_fields_are_never_matched(self) -> None:
        result = StepResult(step_id="s", status="success", time_to_first_token_ms=12.5)
        result.token_usage.prompt_tokens = 7
        out = redact_model(result.model_dump(), StepResult, self.POLICY)
        assert out["token_usage"]["prompt_tokens"] == 7
        assert out["time_to_first_token_ms"] == 12.5
        StepResult.model_validate(out)

    def test_free_form_data_is_redacted_by_key(self) -> None:
        result = StepResult(step_id="s", status="success", output={"api_key": "dummy-1", "n": 1})
        out = redact_model(result.model_dump(), StepResult, self.POLICY)
        assert is_redacted(out["output"]["api_key"])
        assert out["output"]["n"] == 1

    def test_walks_nested_models_lists_and_maps_of_models(self) -> None:
        data = _Outer(
            inner=_Inner(token_count=1),
            items=[_Inner(token_count=2)],
            by_name={"dummy_token": _Inner(token_count=3)},
        ).model_dump()
        out = redact_model(data, _Outer, self.POLICY)
        assert out["inner"]["token_count"] == 1
        assert out["items"][0]["token_count"] == 2
        assert out["by_name"]["dummy_token"]["token_count"] == 3  # a model, never a sentinel

    def test_a_sentinel_lands_only_where_text_is_accepted(self) -> None:
        data = _Outer(
            free={"api_key": 1}, texts={"api_key": "sk"}, counts={"api_key": 5}
        ).model_dump()
        out = redact_model(data, _Outer, self.POLICY)
        assert is_redacted(out["free"]["api_key"])
        assert is_redacted(out["texts"]["api_key"])
        assert out["counts"]["api_key"] == 5
        _Outer.model_validate(out)

    def test_unions_pick_the_option_that_fits(self) -> None:
        data = _Outer(
            either={"api_key": "sk"},
            choice=_Inner(token_count=4),
            paths=["a"],
            table={"api_key": 2},
        ).model_dump()
        out = redact_model(data, _Outer, self.POLICY)
        assert is_redacted(out["either"]["api_key"])  # Any among the options: free-form
        assert out["choice"]["token_count"] == 4
        assert out["paths"] == ["a"]
        assert out["table"] == {"api_key": 2}

        more = _Outer(
            tagged=_Inner(token_count=6, extra={"api_key": "sk"}),
            tagged_items=[_Inner(extra={"api_key": "sk"})],
            secrets={"api_key": "sk"},
            notes=[{"api_key": "sk"}],
            maybe_texts={"api_key": "sk"},
        ).model_dump()
        out = redact_model(more, _Outer, self.POLICY)
        assert out["tagged"]["token_count"] == 6
        assert is_redacted(out["tagged"]["extra"]["api_key"])
        assert is_redacted(out["tagged_items"][0]["extra"]["api_key"])
        assert is_redacted(out["secrets"]["api_key"])
        assert is_redacted(out["notes"][0]["api_key"])
        assert is_redacted(out["maybe_texts"]["api_key"])
        assert redact_model({"choice": "api_key"}, _Outer, self.POLICY) == {"choice": "api_key"}

    def test_a_key_the_model_does_not_declare_is_data(self) -> None:
        out = redact_model({"api_key": "sk", "raw": "x"}, _Outer, self.POLICY)
        assert is_redacted(out["api_key"])
        assert out["raw"] == "x"  # an Any scalar under its own field name

    def test_parse_as_and_opaque(self) -> None:
        step = StepDefinition(
            id="sub",
            type="subworkflow",
            workflow_inline={
                "name": "child",
                "state": {"api_key": "dummy-child"},
                "steps": [{"id": "a", "type": "llm_call", "prompt": "p", "max_tokens": 50}],
            },
            tools=[{"name": "t", "parameters": {"properties": {"api_key": {"type": "string"}}}}],
        )
        out = redact_model(
            step.model_dump(),
            StepDefinition,
            self.POLICY,
            parse_as={"workflow_inline": WorkflowDefinition},
            opaque={"parameters"},
        )
        child = out["workflow_inline"]
        assert is_redacted(child["state"]["api_key"])
        assert child["steps"][0]["max_tokens"] == 50
        WorkflowDefinition.model_validate(child)
        assert out["tools"][0]["parameters"]["properties"]["api_key"] == {"type": "string"}

    def test_a_map_of_models_keeps_its_keys_and_values(self) -> None:
        wf = WorkflowDefinition(
            name="w",
            state_schema={"*token*": {"redact": True}},
            steps=[StepDefinition(id="a", type="llm_call", prompt="p")],
        )
        out = redact_model(wf.model_dump(), WorkflowDefinition, self.POLICY)
        assert out["state_schema"] == {"*token*": {"redact": True}}

    def test_no_policy_returns_the_dump(self) -> None:
        data = {"api_key": "sk"}
        assert redact_model(data, _Outer, RedactionPolicy()) is data
