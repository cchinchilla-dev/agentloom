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
        prompt_hash([{"role": "user", "content": "hash me"}]): {
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
    await provider.complete(messages=[{"role": "user", "content": "hash me"}], model="m")
    await provider.complete(messages=[{"role": "user", "content": "nothing-matches"}], model="m")
    assert [c[2] for c in calls] == ["prompt_hash", "default"]


def test_rejects_non_object_responses_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        MockProvider(responses_file=path)
