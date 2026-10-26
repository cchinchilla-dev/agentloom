"""An llm_call step marks the failures it returns transient or permanent (#182)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    ResponseSchema,
    RetryConfig,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowResult, WorkflowStatus
from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider


class _BrokenStream(BaseProvider):
    """Starts a stream, then fails it with a 400 after the first chunk."""

    name = "broken"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        raise NotImplementedError

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        self.calls += 1

        async def chunks() -> AsyncIterator[str]:
            yield "partial "
            raise ProviderError(self.name, "invalid request", status_code=400)

        sr = StreamResponse(model=model, provider=self.name)
        sr._set_iterator(chunks())
        return sr


async def _run_with(provider: BaseProvider, **fields: Any) -> WorkflowResult:
    gateway = ProviderGateway()
    gateway.register(provider)
    retry = RetryConfig(max_retries=1, backoff_max=0, jitter=False)
    step = StepDefinition(
        id="s", type=StepType.LLM_CALL, prompt="hi", output="out", retry=retry, **fields
    )
    workflow = WorkflowDefinition(
        name="w", config=WorkflowConfig(provider="mock", model="m"), steps=[step]
    )
    return await WorkflowEngine(workflow=workflow, provider_gateway=gateway).run()


async def _run(tmp_path: Path, content: str, **fields: Any) -> tuple[WorkflowResult, MockProvider]:
    recording = tmp_path / "rec.json"
    recording.write_text(json.dumps({"s": {"content": content}}))
    mock = MockProvider(responses_file=recording)
    return await _run_with(mock, **fields), mock


async def test_structured_output_out_of_attempts_is_not_retried(tmp_path: Path) -> None:
    schema = ResponseSchema(type="json_object")

    result, mock = await _run(tmp_path, "not json", response_schema=schema)

    assert result.status == WorkflowStatus.FAILED
    assert result.step_results["s"].error_classification == "permanent"
    assert len(mock.calls) == 2  # the step's own validation attempts, and no engine retry


async def test_a_schema_the_step_cannot_use_is_permanent(tmp_path: Path) -> None:
    # Resolves, but is no Pydantic model: the schema is wrong, whatever the answer.
    schema = ResponseSchema(type="pydantic", model="collections:OrderedDict")

    for stream in (False, True):
        result, mock = await _run(tmp_path, '{"a": 1}', response_schema=schema, stream=stream)

        assert result.status == WorkflowStatus.FAILED
        assert result.step_results["s"].error_classification == "permanent"
        assert len(mock.calls) == 1


async def test_a_stream_that_fails_midway_with_a_400_is_not_retried() -> None:
    provider = _BrokenStream()

    result = await _run_with(provider, stream=True)

    assert result.status == WorkflowStatus.FAILED
    assert result.step_results["s"].error_classification == "permanent"
    assert provider.calls == 1


async def test_a_streamed_step_retries_invalid_structured_output(tmp_path: Path) -> None:
    # A stream is validated once, at its end, with no attempts of its own: the engine's
    # retry is its second attempt, as the non-streamed step's validation loop is.
    recording = tmp_path / "rec.json"
    recording.write_text(json.dumps({"s": [{"content": "not json"}, {"content": '{"a": 1}'}]}))
    mock = MockProvider(responses_file=recording)

    result = await _run_with(mock, response_schema=ResponseSchema(type="json_object"), stream=True)

    assert result.status == WorkflowStatus.SUCCESS
    assert result.final_state["out"] == {"a": 1}
    assert len(mock.calls) == 2
