"""Tests for RecordingProvider."""

from __future__ import annotations

import json

from agentloom.providers.mock import MockProvider, prompt_hash
from agentloom.providers.recorder import RecordingProvider


async def test_records_then_replays(tmp_path):
    # Source provider: a mock with a known response
    src_file = tmp_path / "src.json"
    src_file.write_text(
        json.dumps(
            {
                "s1": {
                    "content": "captured",
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                    "cost_usd": 0.002,
                    "finish_reason": "stop",
                }
            }
        )
    )
    source = MockProvider(responses_file=src_file)

    recording_path = tmp_path / "recorded.json"
    recorder = RecordingProvider(source, recording_path)

    messages = [{"role": "user", "content": "record me"}]
    r = await recorder.complete(messages=messages, model="gpt-4o-mini", step_id="s1")
    assert r.content == "captured"

    # File is flushed per-call
    data = json.loads(recording_path.read_text())
    assert "s1" in data
    assert data["s1"]["content"] == "captured"
    assert data["s1"]["usage"]["total_tokens"] == 7
    assert "latency_ms" in data["s1"]

    # Replay via a fresh MockProvider from the recorded file
    replay = MockProvider(responses_file=recording_path)
    r2 = await replay.complete(messages=messages, model="gpt-4o-mini", step_id="s1")
    assert r2.content == "captured"


async def test_records_by_prompt_hash_when_no_step_id(tmp_path):
    source = MockProvider(default_response="ok")
    recording_path = tmp_path / "out.json"
    recorder = RecordingProvider(source, recording_path)

    messages = [{"role": "user", "content": "nohash"}]
    await recorder.complete(messages=messages, model="m")

    data = json.loads(recording_path.read_text())
    assert prompt_hash(messages) in data


async def test_close_flushes_and_closes_wrapped(tmp_path):
    source = MockProvider(default_response="x")
    recording_path = tmp_path / "out.json"
    recorder = RecordingProvider(source, recording_path)
    await recorder.close()
    # file exists (empty dict) after close
    assert recording_path.exists()
    assert json.loads(recording_path.read_text()) == {}


async def test_observer_notified_on_capture(tmp_path):
    source = MockProvider(default_response="ok")
    source.name = "det"
    events = []

    class Obs:
        def on_recording_capture(self, step_id, provider, model, latency_s):
            events.append((step_id, provider, model, latency_s))

    recorder = RecordingProvider(source, tmp_path / "out.json", observer=Obs())
    await recorder.complete(messages=[{"role": "user", "content": "hi"}], model="m", step_id="s1")
    assert len(events) == 1
    step_id, provider, model, latency_s = events[0]
    assert step_id == "s1"
    assert provider == "det"
    assert model == "m"
    assert latency_s >= 0.0


async def test_multiple_recorders_same_path_do_not_clobber(tmp_path):
    """Regression: two RecordingProviders on the same path must accumulate,
    not overwrite each other's entries on close()."""
    path = tmp_path / "shared.json"

    rec_a = RecordingProvider(MockProvider(default_response="a"), path)
    rec_b = RecordingProvider(MockProvider(default_response="b"), path)

    await rec_a.complete(messages=[{"role": "user", "content": "m1"}], model="m", step_id="from_a")
    await rec_b.complete(messages=[{"role": "user", "content": "m2"}], model="m", step_id="from_b")

    # close() in reverse order — the one with the emptier history must not wipe
    await rec_b.close()
    await rec_a.close()

    data = json.loads(path.read_text())
    assert "from_a" in data
    assert "from_b" in data


async def test_merges_with_existing_file(tmp_path):
    recording_path = tmp_path / "out.json"
    recording_path.write_text(json.dumps({"prior": {"content": "old"}}))

    source = MockProvider(default_response="new")
    recorder = RecordingProvider(source, recording_path)
    await recorder.complete(
        messages=[{"role": "user", "content": "a"}], model="m", step_id="new_step"
    )

    data = json.loads(recording_path.read_text())
    assert "prior" in data
    assert "new_step" in data


async def test_ignores_corrupted_existing_file_on_init(tmp_path):
    recording_path = tmp_path / "corrupt.json"
    recording_path.write_text("{ not valid json")

    source = MockProvider(default_response="ok")
    recorder = RecordingProvider(source, recording_path)
    await recorder.complete(messages=[{"role": "user", "content": "hi"}], model="m", step_id="s1")

    data = json.loads(recording_path.read_text())
    assert "s1" in data


async def test_flush_recovers_from_corrupted_file(tmp_path):
    recording_path = tmp_path / "corrupt.json"
    source = MockProvider(default_response="ok")
    recorder = RecordingProvider(source, recording_path)
    # Corrupt the file after init but before flush
    recording_path.write_text("{ still bad")

    await recorder.complete(messages=[{"role": "user", "content": "hi"}], model="m", step_id="s1")
    data = json.loads(recording_path.read_text())
    assert "s1" in data


async def test_stream_delegates_to_wrapped(tmp_path):
    called = {}

    class FakeStream:
        def __init__(self):
            self.name = "fake"
            self.api_key = None
            self.base_url = None

        async def stream(self, **kwargs):
            called.update(kwargs)
            return "stream-result"

        async def complete(self, **kwargs):
            raise NotImplementedError

        def supports_model(self, model):
            return model == "fake-model"

        async def close(self):
            pass

    recorder = RecordingProvider(FakeStream(), tmp_path / "r.json")  # type: ignore[arg-type]
    result = await recorder.stream(messages=[{"role": "user", "content": "hi"}], model="fake-model")
    assert result == "stream-result"
    assert called["model"] == "fake-model"


async def test_supports_model_delegates_to_wrapped(tmp_path):
    source = MockProvider(default_response="ok")
    source.name = "src"
    recorder = RecordingProvider(source, tmp_path / "r.json")
    assert recorder.supports_model("any-model") is True
