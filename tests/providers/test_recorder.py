"""Tests for RecordingProvider."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agentloom.core.attempts import Attempt, enter_attempt
from agentloom.core.models import ResponseSchema
from agentloom.core.redact import RedactionPolicy, is_redacted
from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.base import BaseProvider, ProviderResponse, ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider, prompt_hash
from agentloom.providers.openai import OpenAIProvider
from agentloom.providers.recorder import RecordingProvider
from agentloom.providers.replay_or_live import ReplayCut, ReplayOrLiveProvider


async def test_records_then_replays(tmp_path):
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
    """Two RecordingProviders on one path accumulate, not overwrite each other, on ``close()``."""
    path = tmp_path / "shared.json"

    rec_a = RecordingProvider(MockProvider(default_response="a"), path)
    rec_b = RecordingProvider(MockProvider(default_response="b"), path)

    await rec_a.complete(messages=[{"role": "user", "content": "m1"}], model="m", step_id="from_a")
    await rec_b.complete(messages=[{"role": "user", "content": "m2"}], model="m", step_id="from_b")

    # ``rec_a`` closes last and never saw ``from_b``: its flush must merge, not overwrite.
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
    """A streamed completion is recorded under the same key as ``complete()``, so it replays."""

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
    """Parallel ``complete()`` calls all reach the file: no last writer wins, no iteration race."""

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
    """The flush merge never iterates a dict another coroutine is mutating."""

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


async def test_stream_with_none_iterator_returns_no_chunks(tmp_path):
    """A wrapped stream with a ``None`` iterator yields no chunks instead of crashing."""
    from agentloom.providers.base import BaseProvider, StreamResponse
    from agentloom.providers.recorder import RecordingProvider

    class NullStreamProvider(BaseProvider):
        name = "null-stream"

        async def complete(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def stream(self, messages, model, **kwargs):  # type: ignore[no-untyped-def]
            sr = StreamResponse(model=model, provider=self.name)
            return sr

        def supports_model(self, model: str) -> bool:
            return True

    recorder = RecordingProvider(NullStreamProvider(), tmp_path / "rec.json")
    sr = await recorder.stream(
        messages=[{"role": "user", "content": "x"}],
        model="m",
    )
    chunks = [c async for c in sr]
    assert chunks == []
    await recorder.close()


def test_a_recording_is_merged_with_the_shared_lock_held(tmp_path) -> None:
    # Every writer of a recording (the recorders, the tool tape) merges under one lock.
    from agentloom.providers.recorder import _FILE_LOCK, merge_into_recording

    held: list[bool] = []
    merge_into_recording(tmp_path / "rec.json", lambda merged: held.append(_FILE_LOCK.locked()))
    assert held == [True]


async def _turns(recorder: RecordingProvider, step_id: str, *prompts: str) -> None:
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        await recorder.complete(messages=messages, model="m", step_id=step_id)


async def _attempt(recorder: RecordingProvider, failed: Attempt | None = None) -> Attempt:
    """Start an attempt of step ``s`` the way the engine does; ``failed`` makes it a retry."""
    running = enter_attempt(())
    await recorder.begin_attempt("s", failed)
    return running


async def test_a_retry_drops_the_turns_of_the_failed_attempt(tmp_path) -> None:
    # The engine signals each attempt: a retry forgets what the failed attempt recorded.
    path = tmp_path / "rec.json"
    recorder = RecordingProvider(MockProvider(default_response="ok"), path)
    first = await _attempt(recorder)
    await _turns(recorder, "s", "a", "b")
    await _attempt(recorder, first)
    assert "s" not in json.loads(path.read_text())
    await _turns(recorder, "s", "c")
    assert json.loads(path.read_text())["s"]["request_hash"] == prompt_hash(
        [{"role": "user", "content": "c"}], "m"
    )


async def test_a_retry_keeps_the_turns_of_earlier_runs_of_the_step(tmp_path) -> None:
    path = tmp_path / "rec.json"
    recorder = RecordingProvider(MockProvider(default_response="ok"), path)
    for runs in (["a"], ["b", "c"]):  # the step runs twice in a loop, once per turn list
        run = await _attempt(recorder)
        await _turns(recorder, "s", *runs)
    await _attempt(recorder, run)  # the second run failed: back to one turn
    assert isinstance(json.loads(path.read_text())["s"], dict)
    await _turns(recorder, "s", "d", "e")
    third = await _attempt(recorder)
    await _turns(recorder, "s", "f")
    await _attempt(recorder, third)  # a third run fails: its one turn goes
    hashes = [t["request_hash"] for t in json.loads(path.read_text())["s"]]
    assert hashes == [prompt_hash([{"role": "user", "content": p}], "m") for p in "ade"]


async def test_a_retry_of_an_attempt_that_recorded_nothing_keeps_the_recording(tmp_path) -> None:
    path = tmp_path / "rec.json"
    recorder = RecordingProvider(MockProvider(default_response="ok"), path)
    await _attempt(recorder)
    await _turns(recorder, "s", "a")
    second = await _attempt(recorder)  # the second run fails before its first turn
    await _attempt(recorder, second)
    assert json.loads(path.read_text())["s"]["request_hash"] == prompt_hash(
        [{"role": "user", "content": "a"}], "m"
    )


GO = [{"role": "user", "content": "go"}]


class TestAliasedModel:
    """A recording hashes the model a step asked for, not the provider's name for it."""

    @staticmethod
    def _recording_gateway(tmp_path, live):
        gateway = ProviderGateway()
        gateway.register(live, priority=0, model_aliases={"asked": "provider-id"})
        gateway.wrap_providers(lambda p: RecordingProvider(p, tmp_path / "rec.json"))
        return gateway

    @staticmethod
    def _replay_gateway(tmp_path):
        gateway = ProviderGateway()
        gateway.register(MockProvider(responses_file=tmp_path / "rec.json", strict=True))
        return gateway

    async def test_complete_replays_strictly(self, tmp_path):
        live = MockProvider(default_response="answer")
        await self._recording_gateway(tmp_path, live).complete(GO, "asked", step_id="s")
        assert live.calls[0]["model"] == "provider-id"

        replayed = await self._replay_gateway(tmp_path).complete(GO, "asked", step_id="s")
        assert replayed.content == "answer"

    async def test_stream_replays_strictly(self, tmp_path):
        live = MockProvider(default_response="answer")
        stream = await self._recording_gateway(tmp_path, live).stream(GO, "asked", step_id="s")
        assert [chunk async for chunk in stream] == ["answer"]

        replayed = await self._replay_gateway(tmp_path).complete(GO, "asked", step_id="s")
        assert replayed.content == "answer"

    async def test_embed_replays_strictly(self, tmp_path):
        live = MockProvider()
        await self._recording_gateway(tmp_path, live).embed(["a"], "asked", step_id="s")

        replayed = await self._replay_gateway(tmp_path).embed(["a"], "asked", step_id="s")
        assert len(replayed.embeddings) == 1

    async def test_a_changed_request_is_still_caught(self, tmp_path):
        live = MockProvider(default_response="answer")
        await self._recording_gateway(tmp_path, live).complete(GO, "asked", step_id="s")

        with pytest.raises(RecordingMismatchError):
            await self._replay_gateway(tmp_path).complete(GO, "other", step_id="s")

    async def test_a_cut_replays_an_aliased_step(self, tmp_path):
        live = MockProvider(default_response="answer")
        await self._recording_gateway(tmp_path, live).complete(GO, "asked", step_id="s")

        fresh = MockProvider(default_response="live answer")
        cut = ReplayCut(MockProvider(responses_file=tmp_path / "rec.json", strict=True))
        gateway = ProviderGateway()
        gateway.register(fresh, priority=0, model_aliases={"asked": "provider-id"})
        gateway.wrap_providers(lambda p: ReplayOrLiveProvider(p, cut))

        response = await gateway.complete(GO, "asked", step_id="s")
        assert response.content == "answer"
        assert [(c.source, c.reason) for c in cut.calls] == [("replay", "hit")]
        assert fresh.calls == []


TOKEN = {"id": "c1", "name": "fetch", "arguments": {"url": "u", "api_token": "dummy-secret"}}


class TestRedaction:
    """The run's redaction policy masks the tool arguments an entry holds."""

    POLICY = RedactionPolicy(["*token*"])

    @staticmethod
    def _source(tmp_path):
        src = tmp_path / "src.json"
        src.write_text(json.dumps({"s": {"content": "", "tool_calls": [TOKEN]}}))
        return MockProvider(responses_file=src)

    async def test_complete_masks_tool_arguments_and_hashes_the_real_request(self, tmp_path):
        out = tmp_path / "rec.json"
        recorder = RecordingProvider(self._source(tmp_path), out, redaction=self.POLICY)
        response = await recorder.complete(GO, "m", step_id="s")

        assert response.tool_calls[0].arguments["api_token"] == "dummy-secret"
        assert "dummy-secret" not in out.read_text()
        entry = json.loads(out.read_text())["s"]
        assert is_redacted(entry["tool_calls"][0]["arguments"]["api_token"])
        assert entry["tool_calls"][0]["arguments"]["url"] == "u"
        assert entry["request_hash"] == prompt_hash(GO, "m", None, None, {})

    async def test_stream_masks_tool_arguments(self, tmp_path):
        class _Tooled(MockProvider):
            async def stream(self, *args, **kwargs):
                sr = await super().stream(*args, **kwargs)
                sr.tool_calls = [ToolCall(**TOKEN)]
                return sr

        out = tmp_path / "rec.json"
        recorder = RecordingProvider(_Tooled(), out, redaction=self.POLICY)
        stream = await recorder.stream(GO, "m", step_id="s")
        assert [chunk async for chunk in stream]
        assert "dummy-secret" not in out.read_text()

    async def test_a_replayed_entry_is_masked_without_touching_its_source(self, tmp_path):
        recording = tmp_path / "plain.json"
        plain = RecordingProvider(self._source(tmp_path), recording)
        await plain.complete(GO, "m", step_id="s")
        source = MockProvider(responses_file=recording, strict=True)

        out = tmp_path / "rec.json"
        await RecordingProvider(source, out, redaction=self.POLICY).complete(GO, "m", step_id="s")

        assert "dummy-secret" not in out.read_text()
        assert source._responses["s"]["tool_calls"][0]["arguments"]["api_token"] == "dummy-secret"

    async def test_without_a_policy_the_arguments_stay(self, tmp_path):
        out = tmp_path / "rec.json"
        await RecordingProvider(self._source(tmp_path), out).complete(GO, "m", step_id="s")
        assert "dummy-secret" in out.read_text()

    async def test_strict_replay_goes_on_after_a_masked_tool_call(self, tmp_path):
        out = tmp_path / "rec.json"
        src = tmp_path / "src.json"
        src.write_text(
            json.dumps({"s": [{"content": "", "tool_calls": [TOKEN]}, {"content": "end"}]})
        )
        recorder = RecordingProvider(MockProvider(responses_file=src), out, redaction=self.POLICY)

        def _after(arguments):
            call = {"id": "c1", "type": "function"}
            call["function"] = {"name": "fetch", "arguments": json.dumps(arguments)}
            done = {"role": "tool", "tool_call_id": "c1", "content": "page"}
            return [*GO, {"role": "assistant", "content": "", "tool_calls": [call]}, done]

        first = await recorder.complete(GO, "m", step_id="s")
        await recorder.complete(_after(first.tool_calls[0].arguments), "m", step_id="s")

        replay = MockProvider(responses_file=out, strict=True)
        served = await replay.complete(GO, "m", step_id="s")
        assert is_redacted(served.tool_calls[0].arguments["api_token"])
        answer = await replay.complete(_after(served.tool_calls[0].arguments), "m", step_id="s")
        assert answer.content == "end"


class TestAdapterKwargs:
    """What the adapter needs reaches it, though the hash leaves it out."""

    SCHEMA = ResponseSchema(type="json_schema", schema_={"type": "object", "properties": {}})

    @respx.mock
    async def test_an_inline_schema_is_named_after_the_step_while_recording(self, tmp_path):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )
        out = tmp_path / "rec.json"
        adapter = OpenAIProvider(api_key="dummy-key")
        recorder = RecordingProvider(adapter, out)
        await recorder.complete(
            GO,
            "gpt-4o-mini",
            step_id="classify",
            agentloom_step_id="classify",
            agentloom_response_schema=self.SCHEMA,
        )
        await adapter.close()

        body = json.loads(route.calls.last.request.content)
        assert body["response_format"]["json_schema"]["name"] == "classify"
        assert "step_id" not in body

    async def test_stream_forwards_it_and_the_hash_leaves_it_out(self, tmp_path):
        class _Inner(BaseProvider):
            name = "inner"

            def __init__(self) -> None:
                super().__init__()
                self.kwargs: dict[str, object] = {}

            async def complete(self, messages, model, temperature=None, max_tokens=None, **kwargs):
                self.kwargs = kwargs
                return ProviderResponse(content="ok", model=model, provider=self.name)

        out = tmp_path / "rec.json"
        inner = _Inner()
        stream = await RecordingProvider(inner, out).stream(
            GO, "m", step_id="s", agentloom_step_id="s"
        )
        assert [chunk async for chunk in stream] == ["ok"]

        assert inner.kwargs == {"agentloom_step_id": "s"}
        assert json.loads(out.read_text())["s"]["request_hash"] == prompt_hash(
            GO, "m", None, None, {}
        )
