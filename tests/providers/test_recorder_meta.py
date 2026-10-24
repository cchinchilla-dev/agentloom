"""Recordings keep each response's metadata, outside the request hash (#166)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from agentloom.core.attempts import enter_attempt
from agentloom.providers.base import BaseProvider, ProviderResponse, ResponseMeta
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import RecordingProvider
from tests.providers.test_tool_tape_retries import _AgentFailingOnce, _Clock, _run

if TYPE_CHECKING:
    from pathlib import Path

ASK = [{"role": "user", "content": "q"}]
META = ResponseMeta(
    temperature=0.2,
    seed=7,
    system_fingerprint="fp_1",
    request_id="req_1",
    rate_limit={"x-ratelimit-remaining-requests": "99"},
    received_at=datetime(2026, 10, 24, 12, 0, tzinfo=UTC),
)
# What a recording keeps of META: only the fields set, the time in ISO 8601.
RECORDED = {
    "temperature": 0.2,
    "seed": 7,
    "system_fingerprint": "fp_1",
    "request_id": "req_1",
    "rate_limit": {"x-ratelimit-remaining-requests": "99"},
    "received_at": "2026-10-24T12:00:00Z",
}


class _Reporting(BaseProvider):
    """Answers with ``meta``, as the HTTP adapters do; streams through the base fallback."""

    name = "reporting"

    def __init__(self, meta: ResponseMeta | None) -> None:
        super().__init__()
        self.meta = meta

    async def complete(self, messages: list[dict[str, Any]], model: str, *_: Any, **__: Any) -> Any:
        return ProviderResponse(content="a", model=model, provider=self.name, meta=self.meta)


async def _record(path: Path, meta: ResponseMeta | None, stream: bool = False) -> ProviderResponse:
    recorder = RecordingProvider(_Reporting(meta), path)
    if not stream:
        return await recorder.complete(ASK, "m", step_id="s")
    sr = await recorder.stream(ASK, "m", step_id="s")
    async for _ in sr:
        pass
    return sr.to_provider_response()


@pytest.mark.parametrize("stream", [False, True], ids=["complete", "stream"])
async def test_recorder_persists_meta(tmp_path: Path, stream: bool) -> None:
    path = tmp_path / "rec.json"
    response = await _record(path, META, stream)
    assert response.meta == META
    assert json.loads(path.read_text())["s"]["meta"] == RECORDED
    # A strict replay serves it back: the metadata is not part of the request hash.
    replayed = await MockProvider(responses_file=path, strict=True).complete(ASK, "m", step_id="s")
    assert replayed.meta == META


async def test_meta_absent_is_backwards_compatible(tmp_path: Path) -> None:
    with_meta, without = tmp_path / "with.json", tmp_path / "without.json"
    await _record(with_meta, META)
    await _record(without, None)
    entry = json.loads(without.read_text())["s"]
    assert "meta" not in entry
    assert entry["request_hash"] == json.loads(with_meta.read_text())["s"]["request_hash"]
    replay = MockProvider(responses_file=without, strict=True)
    replayed = await replay.complete(ASK, "m", step_id="s")
    assert replayed.meta is None


async def test_a_stream_through_the_gateway_keeps_its_meta() -> None:
    gateway = ProviderGateway()
    gateway.register(_Reporting(META))
    sr = await gateway.stream(ASK, "m")
    async for _ in sr:
        pass
    await gateway.close()
    assert sr.to_provider_response().meta == META


async def test_a_retried_step_records_its_attempt_number(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    await _run(RecordingProvider(_AgentFailingOnce(), path), _Clock(), None)
    # The failed first attempt left no turns; the retry's turns say they come from attempt 2.
    assert [turn.get("meta") for turn in json.loads(path.read_text())["s"]] == [{"attempt": 2}] * 2


async def test_an_attempt_number_is_that_of_the_recording_run(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    entry = {"content": "a", "model": "m", "provider": "mock", "meta": {"seed": 7, "attempt": 3}}
    source.write_text(json.dumps({"_version": 2, "s": entry, "t": entry}))
    path = tmp_path / "rec.json"
    recorder = RecordingProvider(MockProvider(responses_file=source), path)
    await recorder.complete(ASK, "m", step_id="s")
    assert json.loads(path.read_text())["s"]["meta"] == {"seed": 7}
    enter_attempt((), 2)
    await recorder.complete(ASK, "m", step_id="t")
    assert json.loads(path.read_text())["t"]["meta"] == {"seed": 7, "attempt": 2}
