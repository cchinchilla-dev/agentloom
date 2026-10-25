"""Replay up to a cut point, then run live (#165)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import StepDefinition, StepType, WorkflowConfig, WorkflowDefinition
from agentloom.core.results import WorkflowResult, WorkflowStatus
from agentloom.providers.base import BaseProvider, EmbeddingResponse, ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import RecordingProvider
from agentloom.providers.replay_or_live import ReplayCut, ReplayOrLiveProvider, cut_steps


class _Versioned(BaseProvider):
    """Answers with its version and the prompt; an HTTP adapter, so it never gets ``step_id``."""

    name = "versioned"

    def __init__(self, version: str) -> None:
        super().__init__()
        self.version = version
        self.prompts: list[str] = []

    async def complete(self, messages: list[dict[str, Any]], model: str, **kw: Any) -> Any:
        assert "step_id" not in kw
        self.prompts.append(messages[-1]["content"])
        content = f"{self.version}:{messages[-1]['content']}"
        return ProviderResponse(content=content, model=model, provider=self.name)

    async def embed(self, inputs: list[str], model: str, **kw: Any) -> EmbeddingResponse:
        return EmbeddingResponse(embeddings=[[1.0]], model=model, provider=self.name)


def _step(step_id: str, prompt: str, *after: str) -> StepDefinition:
    return StepDefinition(
        id=step_id, type=StepType.LLM_CALL, prompt=prompt, output=step_id, depends_on=list(after)
    )


def _workflow(b_prompt: str = "B {state.a}") -> WorkflowDefinition:
    steps = [_step("a", "A"), _step("b", b_prompt, "a"), _step("c", "C {state.b}", "b")]
    return WorkflowDefinition(
        name="cut",
        config=WorkflowConfig(provider="versioned", model="m"),
        steps=[*steps, _step("d", "D")],
    )


async def _run(workflow: WorkflowDefinition, provider: BaseProvider) -> WorkflowResult:
    gateway = ProviderGateway()
    gateway.register(provider, models=["m"])
    result = await WorkflowEngine(workflow=workflow, provider_gateway=gateway).run()
    await gateway.close()
    return result


async def _replay(
    path: Path, forced: set[str], workflow: WorkflowDefinition | None = None
) -> tuple[WorkflowResult, ReplayCut, _Versioned]:
    cut = ReplayCut(MockProvider(responses_file=path, strict=True), forced)
    live = _Versioned("v2")
    return await _run(workflow or _workflow(), ReplayOrLiveProvider(live, cut)), cut, live


@pytest.fixture
async def recording(tmp_path: Path) -> Path:
    path = tmp_path / "rec.json"
    assert (await _run(_workflow(), RecordingProvider(_Versioned("v1"), path))).status == "success"
    return path


def _sources(cut: ReplayCut) -> list[tuple[str, int, str, str]]:
    return sorted((c.step_id, c.call, c.source, c.reason) for c in cut.calls)


async def test_cut_point_serves_prefix_from_recording(recording: Path) -> None:
    forced = cut_steps(_workflow(), "b", [])
    result, _, live = await _replay(recording, forced)

    assert forced == {"b", "c"}
    assert result.status == WorkflowStatus.SUCCESS
    state = result.final_state
    assert (state["a"], state["d"]) == ("v1:A", "v1:D")  # the independent branch replays too
    assert state["b"] == "v2:B v1:A"
    assert state["c"] == "v2:C v2:B v1:A"
    assert live.prompts == ["B v1:A", "C v2:B v1:A"]


async def test_first_hash_mismatch_switches_to_live(recording: Path) -> None:
    _, cut, live = await _replay(recording, set(), _workflow("B changed {state.a}"))

    assert _sources(cut) == [
        ("a", 1, "replay", "hit"),
        ("b", 1, "live", "miss"),
        ("c", 1, "live", "miss"),
        ("d", 1, "replay", "hit"),
    ]
    assert live.prompts == ["B changed v1:A", "C v2:B changed v1:A"]


async def test_force_live_step_ignores_matching_recording(recording: Path) -> None:
    result, cut, _ = await _replay(recording, cut_steps(_workflow(), None, ["d"]))

    assert result.final_state["d"] == "v2:D"
    assert ("d", 1, "live", "forced") in _sources(cut)


async def test_call_sources_are_reported(recording: Path) -> None:
    _, cut, _ = await _replay(recording, {"b", "c"})

    assert _sources(cut) == [
        ("a", 1, "replay", "hit"),
        ("b", 1, "live", "forced"),
        ("c", 1, "live", "forced"),
        ("d", 1, "replay", "hit"),
    ]
    assert "  b #1: live (forced)" in cut.summary()


async def test_unused_recorded_turns_are_reported(recording: Path) -> None:
    entries = json.loads(recording.read_text())
    entries["a"] = [entries["a"], entries["a"]]  # two turns recorded, one call made
    entries["d"] = [entries["d"], entries["d"]]  # d runs live, so its turns are not counted
    recording.write_text(json.dumps(entries))
    _, cut, _ = await _replay(recording, {"d"})

    assert cut.unused_turns() == {"a": (1, 2)}
    assert "  a: served 1 of 2 recorded turns" in cut.summary()


async def test_steps_of_a_forced_subworkflow_run_live(tmp_path: Path) -> None:
    inline = {"name": "child", "steps": [{"id": "inner", "type": "llm_call", "prompt": "I"}]}
    sub = StepDefinition(id="sub", type=StepType.SUBWORKFLOW, workflow_inline=inline)
    workflow = _workflow().model_copy(update={"steps": [_step("a", "A"), sub]})
    path = tmp_path / "rec.json"
    await _run(workflow, RecordingProvider(_Versioned("v1"), path))
    _, cut, _ = await _replay(path, {"sub"}, workflow)

    assert _sources(cut) == [("a", 1, "replay", "hit"), ("inner", 1, "live", "forced")]


def test_cut_steps_rejects_unknown_steps() -> None:
    with pytest.raises(ValueError, match="No step 'x', 'y' in workflow 'cut'"):
        cut_steps(_workflow(), "x", ["a", "y"])


async def test_stream_and_embed_go_through_the_cut(recording: Path) -> None:
    cut = ReplayCut(MockProvider(responses_file=recording, strict=True))
    provider = ReplayOrLiveProvider(_Versioned("v2"), cut)
    hi = [{"role": "user", "content": "A"}]

    sr = await provider.stream(hi, "m", step_id="a")
    assert [chunk async for chunk in sr] == ["v1:A"]
    embedded = await provider.embed(["x"], "m", step_id="e")
    assert embedded.embeddings == [[1.0]]
    assert _sources(cut) == [("a", 1, "replay", "hit"), ("e", 1, "live", "miss")]
    assert provider.supports_model("m")
    await provider.close()
