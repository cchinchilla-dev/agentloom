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
    assert prompt_hash(messages, "m") in data


async def test_close_flushes_and_closes_wrapped(tmp_path):
    source = MockProvider(default_response="x")
    recording_path = tmp_path / "out.json"
    recorder = RecordingProvider(source, recording_path)
    await recorder.close()
    # File exists after close. Only the version envelope is present since no
    # completions were recorded.
    assert recording_path.exists()
    data = json.loads(recording_path.read_text())
    assert data.get("_version") == 2
    assert [k for k in data if not k.startswith("_")] == []


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


async def test_stream_recording_produces_replayable_entry(tmp_path):
    """Streamed completions must be captured under the same key format as
    ``complete()`` so that the resulting file is fully replayable."""

    from collections.abc import AsyncIterator

    from agentloom.core.results import TokenUsage
    from agentloom.providers.base import StreamResponse

    class FakeStreamingProvider:
        name = "fake"
        api_key = ""
        base_url = ""

        async def stream(self, **kwargs) -> StreamResponse:
            sr = StreamResponse(model=kwargs["model"], provider="fake")

            async def _gen() -> AsyncIterator[str]:
                for chunk in ["hello", " ", "world"]:
                    yield chunk
                sr.usage = TokenUsage(prompt_tokens=1, completion_tokens=3, total_tokens=4)
                sr.cost_usd = 0.01
                sr.finish_reason = "stop"

            sr._set_iterator(_gen())
            return sr

        async def complete(self, **kwargs):
            raise NotImplementedError

        def supports_model(self, model):
            return True

        async def close(self):
            pass

    recorder = RecordingProvider(FakeStreamingProvider(), tmp_path / "r.json")  # type: ignore[arg-type]
    sr = await recorder.stream(
        messages=[{"role": "user", "content": "hi"}],
        model="fake-model",
        step_id="s_stream",
    )
    chunks = [c async for c in sr]
    assert "".join(chunks) == "hello world"

    data = json.loads((tmp_path / "r.json").read_text())
    assert "s_stream" in data
    assert data["s_stream"]["content"] == "hello world"
    assert data["s_stream"]["usage"]["total_tokens"] == 4
    assert data["s_stream"]["finish_reason"] == "stop"

    # Replayable via MockProvider.
    replay = MockProvider(responses_file=tmp_path / "r.json")
    r = await replay.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="fake-model",
        step_id="s_stream",
    )
    assert r.content == "hello world"


async def test_supports_model_delegates_to_wrapped(tmp_path):
    source = MockProvider(default_response="ok")
    source.name = "src"
    recorder = RecordingProvider(source, tmp_path / "r.json")
    assert recorder.supports_model("any-model") is True


async def test_concurrent_recording_persists_all_entries(tmp_path):
    """N parallel complete() calls on one RecordingProvider must all appear
    in the output file — no last-writer-wins, no dict-iteration races."""

    import anyio

    source = MockProvider(default_response="r")
    recording_path = tmp_path / "parallel.json"
    recorder = RecordingProvider(source, recording_path)

    async def _one(i: int) -> None:
        await recorder.complete(
            messages=[{"role": "user", "content": f"msg-{i}"}],
            model="m",
            step_id=f"step-{i}",
        )

    async with anyio.create_task_group() as tg:
        for i in range(10):
            tg.start_soon(_one, i)

    data = json.loads(recording_path.read_text())
    for i in range(10):
        assert f"step-{i}" in data, f"missing step-{i} under concurrency"


async def test_concurrent_recording_does_not_raise_dict_iteration_error(tmp_path):
    """Merging recorded entries during flush must not iterate a dict that is
    being mutated by another coroutine."""

    import anyio

    source = MockProvider(default_response="r")
    recording_path = tmp_path / "parallel.json"
    recorder = RecordingProvider(source, recording_path)

    async def _loop(prefix: str) -> None:
        for i in range(20):
            await recorder.complete(
                messages=[{"role": "user", "content": f"{prefix}-{i}"}],
                model="m",
                step_id=f"{prefix}-{i}",
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_loop, "a")
        tg.start_soon(_loop, "b")
        tg.start_soon(_loop, "c")

    data = json.loads(recording_path.read_text())
    for prefix in ("a", "b", "c"):
        for i in range(20):
            assert f"{prefix}-{i}" in data
