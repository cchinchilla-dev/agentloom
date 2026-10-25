"""When every provider fails, the gateway's error keeps the status they share (#182)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    RetryConfig,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowResult, WorkflowStatus
from agentloom.exceptions import ProviderError, RateLimitError
from agentloom.providers.base import BaseProvider, EmbeddingResponse, ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider

HI = [{"role": "user", "content": "hi"}]
METHODS = ["complete", "stream", "embed"]
NO_WAIT = RetryConfig(max_retries=1, backoff_max=0, jitter=False)


class _Failing(BaseProvider):
    """Fails every call with ``status`` (``None``: an error without one), as an adapter would."""

    def __init__(self, name: str, status: int | None, *, embeds: bool = True) -> None:
        super().__init__()
        self.name = name
        self._status = status
        self._embeds = embeds
        self.calls = 0

    def _error(self) -> ProviderError:
        if self._status == 429:
            return RateLimitError(self.name)
        return ProviderError(self.name, "refused", status_code=self._status)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.calls += 1
        raise self._error()

    async def embed(
        self, inputs: list[str], model: str, dimensions: int | None = None, **kwargs: Any
    ) -> EmbeddingResponse:
        self.calls += 1
        if not self._embeds:
            raise NotImplementedError("no embeddings")
        raise self._error()


def _gateway(*providers: BaseProvider) -> ProviderGateway:
    gateway = ProviderGateway()
    for priority, provider in enumerate(providers):
        gateway.register(provider, priority=priority, circuit_fail_threshold=1)
    return gateway


async def _fail(gateway: ProviderGateway, method: str) -> ProviderError:
    with pytest.raises(ProviderError, match="All providers failed") as info:
        if method == "embed":
            await gateway.embed(["hi"], "m")
        else:
            await getattr(gateway, method)(HI, "m")
    return info.value


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("status", [400, 429])
async def test_all_failed_with_one_status_carries_it(method: str, status: int) -> None:
    error = await _fail(_gateway(_Failing("first", status), _Failing("second", status)), method)

    assert error.status_code == status
    assert isinstance(error.__cause__, ProviderError)
    assert error.__cause__.provider == "second"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("other", [503, None])
async def test_all_failed_with_mixed_statuses_has_no_status(method: str, other: int | None) -> None:
    error = await _fail(_gateway(_Failing("first", other), _Failing("second", 400)), method)

    assert error.status_code is None
    assert error.__cause__ is None


@pytest.mark.parametrize("method", METHODS)
async def test_an_open_circuit_leaves_no_status(method: str) -> None:
    gateway = _gateway(_Failing("first", 400), _Failing("second", 400))
    gateway._providers[0].circuit_breaker.record_failure()  # threshold 1: open

    error = await _fail(gateway, method)

    assert "circuit is open" in str(error)
    assert error.status_code is None


async def test_a_provider_without_embeddings_leaves_no_status() -> None:
    gateway = _gateway(_Failing("first", 400, embeds=False), _Failing("second", 400))

    error = await _fail(gateway, "embed")

    assert "has no embeddings" in str(error)
    assert error.status_code is None


async def _run(step: StepDefinition, provider: BaseProvider) -> WorkflowResult:
    gateway = ProviderGateway()
    gateway.register(provider)  # default breaker: the retry reaches the provider again
    workflow = WorkflowDefinition(
        name="w",
        config=WorkflowConfig(provider="mock", model="m"),
        state={"docs": ["alpha"]},
        steps=[step],
    )
    return await WorkflowEngine(workflow=workflow, provider_gateway=gateway).run()


def _mock(tmp_path: Path, first: dict[str, Any]) -> MockProvider:
    recording = tmp_path / "rec.json"
    recording.write_text(json.dumps({"s": [first, {"content": "served by the retry"}]}))
    return MockProvider(responses_file=recording)


def _llm_step(**fields: Any) -> StepDefinition:
    return StepDefinition(
        id="s", type=StepType.LLM_CALL, prompt="hi", output="out", retry=NO_WAIT, **fields
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_a_400_through_the_gateway_is_not_retried(tmp_path: Path, stream: bool) -> None:
    mock = _mock(tmp_path, {"fault": {"type": "http_error", "status_code": 400}})

    result = await _run(_llm_step(stream=stream), mock)

    assert result.status == WorkflowStatus.FAILED
    assert result.step_results["s"].error_classification == "permanent"
    assert len(mock.calls) == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_a_503_through_the_gateway_is_still_retried(tmp_path: Path, stream: bool) -> None:
    mock = _mock(tmp_path, {"fault": {"type": "http_error", "status_code": 503}})

    result = await _run(_llm_step(stream=stream), mock)

    assert result.status == WorkflowStatus.SUCCESS
    assert result.final_state["out"] == "served by the retry"
    assert len(mock.calls) == 2


async def test_a_stream_cut_midway_is_retried(tmp_path: Path) -> None:
    cut = {"content": "one two", "fault": {"type": "stream_truncate", "after_chunks": 1}}
    mock = _mock(tmp_path, cut)

    result = await _run(_llm_step(stream=True), mock)

    assert result.status == WorkflowStatus.SUCCESS
    assert len(mock.calls) == 2


async def test_an_embed_step_does_not_retry_a_400() -> None:
    provider = _Failing("embedder", 400)
    step = StepDefinition(
        id="s", type=StepType.EMBED, inputs="state.docs", output="out", retry=NO_WAIT
    )

    result = await _run(step, provider)

    assert result.status == WorkflowStatus.FAILED
    assert result.step_results["s"].error_classification == "permanent"
    assert provider.calls == 1
