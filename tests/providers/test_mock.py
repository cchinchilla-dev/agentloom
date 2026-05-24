"""Tests for MockProvider."""

from __future__ import annotations

import json
import time

import pytest

from agentloom.providers.mock import MockProvider, prompt_hash


@pytest.fixture
def responses_file(tmp_path):
    path = tmp_path / "responses.json"
    data = {
        "step_one": {
            "content": "hello from step_one",
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            "cost_usd": 0.0005,
            "latency_ms": 20.0,
            "finish_reason": "stop",
        },
        prompt_hash([{"role": "user", "content": "hash me"}], "gpt-4o-mini"): {
            "content": "by hash",
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "finish_reason": "stop",
        },
    }
    path.write_text(json.dumps(data))
    return path


async def test_matches_by_step_id(responses_file):
    provider = MockProvider(responses_file=responses_file)
    r = await provider.complete(
        messages=[{"role": "user", "content": "anything"}],
        model="gpt-4o-mini",
        step_id="step_one",
    )
    assert r.content == "hello from step_one"
    assert r.usage.total_tokens == 12
    assert r.cost_usd == 0.0005
    assert provider.calls[0]["matched"] is True


async def test_matches_by_prompt_hash(responses_file):
    provider = MockProvider(responses_file=responses_file)
    r = await provider.complete(
        messages=[{"role": "user", "content": "hash me"}], model="gpt-4o-mini"
    )
    assert r.content == "by hash"


async def test_prompt_hash_differentiates_model():
    # Same messages, different model — must produce different keys so that
    # recordings cannot collide across models.
    msgs = [{"role": "user", "content": "hello"}]
    assert prompt_hash(msgs, "gpt-4o-mini") != prompt_hash(msgs, "gpt-4o")


async def test_prompt_hash_differentiates_temperature():
    msgs = [{"role": "user", "content": "hello"}]
    assert prompt_hash(msgs, "m", temperature=0.1) != prompt_hash(msgs, "m", temperature=0.9)


async def test_prompt_hash_differentiates_max_tokens():
    msgs = [{"role": "user", "content": "hello"}]
    assert prompt_hash(msgs, "m", max_tokens=100) != prompt_hash(msgs, "m", max_tokens=500)


async def test_prompt_hash_stable_across_invocations():
    # Pydantic-aware serialization: equal inputs hash equal regardless of
    # instance identity or dict ordering.
    msgs_1 = [{"role": "user", "content": "hello"}]
    msgs_2 = [{"content": "hello", "role": "user"}]
    assert prompt_hash(msgs_1, "m", 0.5, 100) == prompt_hash(msgs_2, "m", 0.5, 100)


async def test_default_response_when_no_match(tmp_path):
    provider = MockProvider(default_response="fallback!")
    r = await provider.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert r.content == "fallback!"
    assert r.provider == "mock"
    assert provider.calls[0]["matched"] is False


async def test_replay_latency(responses_file):
    provider = MockProvider(responses_file=responses_file, latency_model="replay")
    start = time.perf_counter()
    await provider.complete(
        messages=[{"role": "user", "content": "x"}], model="m", step_id="step_one"
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms >= 15.0  # recorded 20ms, allow slack


async def test_constant_latency(tmp_path):
    provider = MockProvider(latency_model="constant", latency_ms=10.0)
    start = time.perf_counter()
    await provider.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert (time.perf_counter() - start) * 1000.0 >= 8.0


async def test_normal_latency_deterministic_with_seed(tmp_path):
    provider = MockProvider(latency_model="normal", latency_ms=5.0, seed=42)
    await provider.complete(messages=[{"role": "user", "content": "x"}], model="m")
    # just ensures the gaussian branch runs without error


async def test_observer_receives_step_id_match(responses_file):
    calls = []

    class Obs:
        def on_mock_replay(self, workflow_name, step_id, matched_by):
            calls.append((workflow_name, step_id, matched_by))

    provider = MockProvider(responses_file=responses_file, observer=Obs(), workflow_name="wf1")
    await provider.complete(
        messages=[{"role": "user", "content": "x"}], model="m", step_id="step_one"
    )
    assert calls == [("wf1", "step_one", "step_id")]


async def test_observer_receives_prompt_hash_and_default(responses_file):
    calls = []

    class Obs:
        def on_mock_replay(self, workflow_name, step_id, matched_by):
            calls.append((workflow_name, step_id, matched_by))

    provider = MockProvider(responses_file=responses_file, observer=Obs(), workflow_name="wf")
    # Matches by prompt_hash — the fixture keys on (messages, "gpt-4o-mini").
    await provider.complete(messages=[{"role": "user", "content": "hash me"}], model="gpt-4o-mini")
    await provider.complete(messages=[{"role": "user", "content": "nothing-matches"}], model="m")
    assert [c[2] for c in calls] == ["prompt_hash", "default"]


def test_rejects_non_object_responses_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        MockProvider(responses_file=path)


def test_canonical_default_uses_model_dump_for_pydantic_objects() -> None:
    """`_canonical_default` should call ``model_dump()`` on Pydantic instances
    so the prompt hash depends on the public field shape, not the repr."""
    from pydantic import BaseModel

    from agentloom.providers.mock import _canonical_default

    class Cfg(BaseModel):
        x: int
        y: str

    out = _canonical_default(Cfg(x=1, y="hello"))
    assert out == {"x": 1, "y": "hello"}


def test_canonical_default_falls_back_to_str_for_plain_objects() -> None:
    from agentloom.providers.mock import _canonical_default

    class Plain:
        def __repr__(self) -> str:
            return "Plain(123)"

    assert _canonical_default(Plain()) == "Plain(123)"


class TestStrictReplay:
    """Issue #063: strict mode turns a recording miss / prompt drift into
    a hard error so a green replay cannot silently answer the wrong
    prompt. ``agentloom replay`` sets ``strict=True``."""

    def _write(self, tmp_path, payload):
        path = tmp_path / "rec.json"
        path.write_text(json.dumps(payload))
        return path

    async def test_strict_miss_raises_recording_mismatch(self, tmp_path) -> None:
        from agentloom.exceptions import RecordingMismatchError

        path = self._write(tmp_path, {"_version": 2, "other": {"content": "x"}})
        provider = MockProvider(responses_file=path, strict=True)
        with pytest.raises(RecordingMismatchError, match="No recorded response"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}], model="m", step_id="missing"
            )

    async def test_strict_prompt_drift_raises(self, tmp_path) -> None:
        from agentloom.exceptions import RecordingMismatchError

        # Entry keyed by step id but recorded against a different prompt.
        recorded_hash = prompt_hash([{"role": "user", "content": "ORIGINAL"}], "m", None, None, {})
        path = self._write(
            tmp_path,
            {"_version": 2, "s": {"content": "x", "request_hash": recorded_hash}},
        )
        provider = MockProvider(responses_file=path, strict=True)
        with pytest.raises(RecordingMismatchError, match="does not match the recording"):
            await provider.complete(
                messages=[{"role": "user", "content": "EDITED"}], model="m", step_id="s"
            )

    async def test_strict_match_succeeds(self, tmp_path) -> None:
        messages = [{"role": "user", "content": "hi"}]
        recorded_hash = prompt_hash(messages, "m", None, None, {})
        path = self._write(
            tmp_path,
            {"_version": 2, "s": {"content": "answer", "request_hash": recorded_hash}},
        )
        provider = MockProvider(responses_file=path, strict=True)
        r = await provider.complete(messages=messages, model="m", step_id="s")
        assert r.content == "answer"

    async def test_non_strict_miss_returns_default(self, tmp_path, caplog) -> None:
        path = self._write(tmp_path, {"_version": 2, "other": {"content": "x"}})
        provider = MockProvider(responses_file=path, strict=False, default_response="DEFAULT")
        r = await provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="m", step_id="missing"
        )
        # Non-strict keeps the dev fallback, but warns so a CI assertion
        # passing on the placeholder is at least visible.
        assert r.content == "DEFAULT"
        assert any("no recorded response" in rec.message.lower() for rec in caplog.records)


class TestRecordingSchemaValidation:
    """Issue #063: a malformed / wrong-version recording is rejected at
    load time (F29) — pre-0.5.0 it loaded silently and every lookup fell
    through to the placeholder default."""

    async def test_malformed_recording_rejected(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"not": "valid"}))
        with pytest.raises(ValueError, match="must be a response object"):
            MockProvider(responses_file=path)

    async def test_invalid_json_recording_rejected(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json at all")
        with pytest.raises(ValueError, match="not valid JSON"):
            MockProvider(responses_file=path)

    async def test_recording_v1_version_rejected(self, tmp_path) -> None:
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"_version": 1, "s": {"content": "x"}}))
        with pytest.raises(ValueError, match="needs v2"):
            MockProvider(responses_file=path)

    async def test_recording_v2_accepted(self, tmp_path) -> None:
        path = tmp_path / "ok.json"
        path.write_text(json.dumps({"_version": 2, "s": {"content": "x"}}))
        provider = MockProvider(responses_file=path)
        assert provider.responses_file == path

    async def test_recording_list_turns_accepted(self, tmp_path) -> None:
        # Multi-turn tool loops record a list of response objects.
        path = tmp_path / "turns.json"
        path.write_text(json.dumps({"_version": 2, "s": [{"content": "a"}, {"content": "b"}]}))
        provider = MockProvider(responses_file=path)
        assert isinstance(provider._responses["s"], list)

    async def test_step_id_entry_with_non_dict_value_falls_through(self, tmp_path) -> None:
        # Recording schema validation rejects non-dict / non-list entries
        # on load, but defensive code in ``_lookup`` still guards against
        # a programmatic mutation that bypasses the loader. A non-dict /
        # non-list entry behaves as a miss — never as a crash.
        path = tmp_path / "ok.json"
        path.write_text(json.dumps({"_version": 2, "s": {"content": "x"}}))
        provider = MockProvider(responses_file=path, default_response="DEFAULT")
        # Bypass the loader to simulate the defensive branch.
        provider._responses["s"] = "not a dict"  # type: ignore[assignment]
        r = await provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="m", step_id="s"
        )
        assert r.content == "DEFAULT"
