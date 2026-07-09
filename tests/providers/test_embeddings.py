"""Tests for embeddings (#118) across providers, gateway, mock, and step."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowStatus
from agentloom.core.state import StateManager
from agentloom.providers.anthropic import AnthropicProvider
from agentloom.providers.base import BaseProvider, EmbeddingResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.google import GoogleProvider
from agentloom.providers.mock import MockProvider, _pseudo_vector
from agentloom.providers.ollama import OllamaProvider
from agentloom.providers.openai import OpenAIProvider
from agentloom.tools.registry import ToolRegistry


def _embed_workflow(
    *,
    responses_file: str | None = None,
    dimensions: int | None = None,
    output_key: str = "vectors",
) -> WorkflowDefinition:
    """Single-step embed workflow used across the integration tests."""
    return WorkflowDefinition(
        name="embed-test",
        config=WorkflowConfig(
            provider="mock",
            model="text-embedding-3-small",
            responses_file=responses_file,
        ),
        state={"docs": ["alpha", "beta", "gamma"]},
        steps=[
            StepDefinition(
                id="vectorize",
                type=StepType.EMBED,
                inputs="state.docs",
                dimensions=dimensions,
                output=output_key,
            )
        ],
    )


async def _run(
    workflow: WorkflowDefinition, gateway: ProviderGateway
) -> tuple[WorkflowStatus, dict[str, Any]]:
    sm = StateManager(initial_state=dict(workflow.state))
    engine = WorkflowEngine(
        workflow=workflow,
        state_manager=sm,
        provider_gateway=gateway,
        tool_registry=ToolRegistry(),
    )
    result = await engine.run()
    return result.status, result.final_state


class TestBaseProviderEmbed:
    """The default ``embed()`` raises so silent no-ops don't slip through."""

    async def test_base_provider_embed_raises_not_implemented(self) -> None:
        class _NoEmbed(BaseProvider):
            name = "noembed"

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        with pytest.raises(NotImplementedError, match="noembed"):
            await _NoEmbed().embed(inputs=["hi"], model="foo")


class TestOpenAIEmbed:
    """OpenAI hits ``/v1/embeddings`` and returns vectors ordered by index."""

    @respx.mock
    async def test_openai_embed_returns_vectors(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                        {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                    ],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                },
            )

        respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=_handler)
        provider = OpenAIProvider(api_key="sk-test")
        r = await provider.embed(inputs=["a", "b"], model="text-embedding-3-small")
        assert isinstance(r, EmbeddingResponse)
        assert r.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert r.provider == "openai"
        assert r.usage.prompt_tokens == 4
        await provider.close()

    @respx.mock
    async def test_openai_embed_forwards_dimensions(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "text-embedding-3-small",
                    "data": [{"index": 0, "embedding": [0.0] * 64}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )

        respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=_handler)
        provider = OpenAIProvider(api_key="sk-test")
        await provider.embed(inputs=["x"], model="text-embedding-3-small", dimensions=64)
        assert captured["payload"]["dimensions"] == 64
        await provider.close()

    @respx.mock
    async def test_openai_embed_batches_large_input(self, monkeypatch: Any) -> None:
        # Force a tiny batch size so a 5-input call splits into 3 batches;
        # verifies the batching loop preserves order across chunks.
        monkeypatch.setattr(OpenAIProvider, "_EMBED_BATCH_SIZE", 2)
        request_count = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            body = json.loads(request.content)
            data = [
                {"index": i, "embedding": [float(request_count), float(i)]}
                for i, _ in enumerate(body["input"])
            ]
            n = len(body["input"])
            return httpx.Response(
                200,
                json={
                    "model": "text-embedding-3-small",
                    "data": data,
                    "usage": {"prompt_tokens": n, "total_tokens": n},
                },
            )

        respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=_handler)
        provider = OpenAIProvider(api_key="sk-test")
        r = await provider.embed(inputs=["a", "b", "c", "d", "e"], model="text-embedding-3-small")
        assert request_count == 3  # ceil(5 / 2)
        assert len(r.embeddings) == 5
        assert r.usage.prompt_tokens == 5
        await provider.close()

    def test_openai_supports_embedding_model_prefix(self) -> None:
        # ``supports_model`` must claim embedding models so the gateway
        # routes ``text-embedding-3-small`` to OpenAI, not the LLM path.
        provider = OpenAIProvider(api_key="sk-test")
        assert provider.supports_model("text-embedding-3-small")
        assert provider.supports_model("text-embedding-ada-002")


class TestGoogleEmbed:
    """Google uses ``batchEmbedContents`` and returns per-input values."""

    @respx.mock
    async def test_google_embed_returns_vectors(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        {"values": [0.1, 0.2]},
                        {"values": [0.3, 0.4]},
                    ]
                },
            )

        respx.post(url__regex=r".+batchEmbedContents.+").mock(side_effect=_handler)
        provider = GoogleProvider(api_key="g-test")
        r = await provider.embed(inputs=["a", "b"], model="text-embedding-004")
        assert r.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert r.provider == "google"
        await provider.close()

    @respx.mock
    async def test_google_embed_forwards_dimensions_and_task_type(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [{"values": [0.0, 0.0]}]})

        respx.post(url__regex=r".+batchEmbedContents.+").mock(side_effect=_handler)
        provider = GoogleProvider(api_key="g-test")
        await provider.embed(
            inputs=["x"],
            model="text-embedding-004",
            dimensions=256,
            task_type="RETRIEVAL_QUERY",
        )
        req = captured["payload"]["requests"][0]
        assert req["outputDimensionality"] == 256
        assert req["taskType"] == "RETRIEVAL_QUERY"
        await provider.close()


class TestOllamaEmbed:
    """Ollama uses ``/api/embed`` with a list input."""

    @respx.mock
    async def test_ollama_embed_returns_vectors(self) -> None:
        respx.post("http://localhost:11434/api/embed").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "nomic-embed-text",
                    "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                    "prompt_eval_count": 6,
                },
            )
        )
        provider = OllamaProvider(base_url="http://localhost:11434")
        r = await provider.embed(inputs=["a", "b"], model="nomic-embed-text")
        assert r.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert r.cost_usd == 0.0
        assert r.usage.prompt_tokens == 6
        await provider.close()


class TestAnthropicEmbed:
    """Anthropic has no embeddings endpoint — the refusal is explicit."""

    async def test_anthropic_embed_raises_with_hint(self) -> None:
        provider = AnthropicProvider(api_key="k")
        with pytest.raises(NotImplementedError, match="Anthropic"):
            await provider.embed(inputs=["x"], model="claude-haiku-4-5")


class TestGatewayEmbed:
    """Gateway routing, fallback across providers, and NotImplementedError skip."""

    async def test_gateway_embed_delegates_to_registered_provider(self) -> None:
        class _Fake(BaseProvider):
            name = "fake"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[float(len(t))] for t in inputs],
                    model=model,
                    provider=self.name,
                )

        gw = ProviderGateway()
        gw.register(_Fake(), priority=0)
        r = await gw.embed(inputs=["hi", "world"], model="x")
        assert r.embeddings == [[2.0], [5.0]]

    async def test_gateway_falls_back_on_not_implemented(self) -> None:
        # Anthropic-style provider first in the chain must be skipped
        # cleanly (no CB failure) and the second provider serves the request.
        class _NoEmbed(BaseProvider):
            name = "anth-like"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        class _Yes(BaseProvider):
            name = "yes"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[1.0]] * len(inputs),
                    model=model,
                    provider=self.name,
                )

        gw = ProviderGateway()
        anth = _NoEmbed()
        yes = _Yes()
        gw.register(anth, priority=0)
        gw.register(yes, priority=1)
        r = await gw.embed(inputs=["a", "b"], model="x")
        assert r.provider == "yes"
        # Anthropic-style provider's circuit breaker was NOT tripped by
        # the NotImplementedError — its failure count stays at zero.
        anth_entry = next(e for e in gw._providers if e.provider is anth)
        assert anth_entry.circuit_breaker._failure_count == 0

    async def test_gateway_all_providers_missing_embed_raises(self) -> None:
        from agentloom.exceptions import ProviderError

        class _NoEmbed(BaseProvider):
            name = "n"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        gw = ProviderGateway()
        gw.register(_NoEmbed(), priority=0)
        with pytest.raises(ProviderError, match="All providers failed for embed"):
            await gw.embed(inputs=["x"], model="anything")


class TestMockEmbed:
    """MockProvider serves recordings or synthesises pseudo-vectors."""

    def test_pseudo_vector_deterministic(self) -> None:
        v1 = _pseudo_vector("hello", 8)
        v2 = _pseudo_vector("hello", 8)
        assert v1 == v2
        assert len(v1) == 8
        assert all(-1.0 <= x <= 1.0 for x in v1)

    async def test_mock_embed_serves_recording(self, tmp_path: Any) -> None:
        rec = tmp_path / "rec.json"
        rec.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "s": {
                        "embeddings": [[0.5, 0.5], [0.6, 0.6]],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 4, "total_tokens": 4},
                        "cost_usd": 0.0001,
                    },
                }
            )
        )
        p = MockProvider(responses_file=rec)
        r = await p.embed(inputs=["hello", "world"], model="text-embedding-3-small", step_id="s")
        assert r.embeddings == [[0.5, 0.5], [0.6, 0.6]]
        assert r.cost_usd == 0.0001
        assert r.usage.prompt_tokens == 4

    async def test_mock_embed_pseudo_fallback_when_no_recording(self, tmp_path: Any) -> None:
        rec = tmp_path / "rec.json"
        rec.write_text(json.dumps({"_version": 2}))
        p = MockProvider(responses_file=rec)
        r = await p.embed(inputs=["alpha", "beta"], model="x", dimensions=8, step_id="s")
        assert len(r.embeddings) == 2
        assert len(r.embeddings[0]) == 8
        # Deterministic: same input → same vector.
        r2 = await p.embed(inputs=["alpha", "beta"], model="x", dimensions=8, step_id="s")
        assert r.embeddings == r2.embeddings

    async def test_mock_strict_raises_on_missing_recording(self, tmp_path: Any) -> None:
        from agentloom.exceptions import RecordingMismatchError

        rec = tmp_path / "rec.json"
        rec.write_text(json.dumps({"_version": 2}))
        p = MockProvider(responses_file=rec, strict=True)
        with pytest.raises(RecordingMismatchError, match="No embed recording"):
            await p.embed(inputs=["x"], model="text-embedding-3-small", step_id="miss")


class TestEmbedStepIntegration:
    """The step reads inputs from state and writes vectors back."""

    async def test_embed_step_writes_vectors_to_state(self, tmp_path: Any) -> None:
        rec = tmp_path / "rec.json"
        rec.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "vectorize": {
                        "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 5, "total_tokens": 5},
                        "cost_usd": 0.0,
                    },
                }
            )
        )
        wf = _embed_workflow(responses_file=str(rec), dimensions=2)
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=rec), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        assert final["vectors"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    async def test_embed_step_rejects_missing_inputs_field(self) -> None:
        # An embed step without ``inputs:`` fails at execute time with a
        # clear error rather than a mysterious ``NoneType`` traceback.
        wf = WorkflowDefinition(
            name="bad",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"docs": ["a"]},
            steps=[StepDefinition(id="v", type=StepType.EMBED)],
        )
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        status, _ = await _run(wf, gw)
        assert status == WorkflowStatus.FAILED

    async def test_embed_step_rejects_wrong_shape(self) -> None:
        # ``inputs`` resolving to something that isn't str / list[str]
        # is a state-schema bug — the step surfaces it as FAILED.
        wf = WorkflowDefinition(
            name="bad-shape",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"docs": {"not": "a-list"}},
            steps=[
                StepDefinition(id="v", type=StepType.EMBED, inputs="state.docs", output="vectors")
            ],
        )
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        status, _ = await _run(wf, gw)
        assert status == WorkflowStatus.FAILED

    async def test_embed_step_accepts_single_string_input(self, tmp_path: Any) -> None:
        # ``state.foo`` may resolve to a single str — coerce to a
        # 1-element batch instead of failing the shape check.
        wf = WorkflowDefinition(
            name="one",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"doc": "hello"},
            steps=[
                StepDefinition(
                    id="v",
                    type=StepType.EMBED,
                    inputs="state.doc",
                    dimensions=4,
                    output="vec",
                )
            ],
        )
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        assert len(final["vec"]) == 1

    async def test_embed_step_rejects_non_state_reference(self) -> None:
        # The dotted-path resolver rejects references that don't start
        # with ``state.`` — no eval, no template rendering.
        wf = WorkflowDefinition(
            name="ref",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"docs": ["a"]},
            steps=[StepDefinition(id="v", type=StepType.EMBED, inputs="docs", output="vec")],
        )
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        status, _ = await _run(wf, gw)
        assert status == WorkflowStatus.FAILED

    async def test_embed_step_rejects_missing_state_path(self) -> None:
        wf = WorkflowDefinition(
            name="ref",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"docs": ["a"]},
            steps=[
                StepDefinition(
                    id="v",
                    type=StepType.EMBED,
                    inputs="state.missing.field",
                    output="vec",
                )
            ],
        )
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        status, _ = await _run(wf, gw)
        assert status == WorkflowStatus.FAILED


class TestEmbedStepObserver:
    """The step calls ``observer.on_embedding_call`` after a successful run."""

    async def test_observer_hook_fires_with_shape_and_tokens(self, tmp_path: Any) -> None:
        from agentloom.observability.noop import NoopObserver

        recorded: list[dict[str, Any]] = []

        class _Observer(NoopObserver):
            def on_embedding_call(self, **kwargs: Any) -> None:
                recorded.append(kwargs)

        rec = tmp_path / "rec.json"
        rec.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "vectorize": {
                        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 4, "total_tokens": 4},
                        "cost_usd": 0.0,
                    },
                }
            )
        )
        wf = _embed_workflow(responses_file=str(rec))
        wf.state = {"docs": ["a", "b"]}
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=rec), priority=0)
        sm = StateManager(initial_state=dict(wf.state))
        engine = WorkflowEngine(
            workflow=wf,
            state_manager=sm,
            provider_gateway=gw,
            tool_registry=ToolRegistry(),
            observer=_Observer(),
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS
        assert len(recorded) == 1
        call = recorded[0]
        assert call["provider"] == "mock"
        assert call["dimensions"] == 2
        assert call["input_count"] == 2
        assert call["prompt_tokens"] == 4

    async def test_observer_hook_failure_does_not_break_step(self, tmp_path: Any) -> None:
        # A raising observer must not turn a successful embed into a failure —
        # the ``contextlib.suppress(Exception)`` guard covers hook errors.
        from agentloom.observability.noop import NoopObserver

        class _Boom(NoopObserver):
            def on_embedding_call(self, **kwargs: Any) -> None:
                raise RuntimeError("exporter down")

        rec = tmp_path / "rec.json"
        rec.write_text(json.dumps({"_version": 2}))
        wf = _embed_workflow(responses_file=str(rec), dimensions=4)
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=rec), priority=0)
        sm = StateManager(initial_state=dict(wf.state))
        engine = WorkflowEngine(
            workflow=wf,
            state_manager=sm,
            provider_gateway=gw,
            tool_registry=ToolRegistry(),
            observer=_Boom(),
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.SUCCESS


class TestEmbedStepInputBracesStripped:
    """``inputs: {state.foo}`` — the resolver tolerates the templated form."""

    async def test_state_ref_wrapped_in_braces_still_resolves(self, tmp_path: Any) -> None:
        wf = WorkflowDefinition(
            name="braces",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"docs": ["one", "two"]},
            steps=[
                StepDefinition(
                    id="v",
                    type=StepType.EMBED,
                    inputs="{state.docs}",
                    dimensions=4,
                    output="vec",
                )
            ],
        )
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        assert len(final["vec"]) == 2


class TestRecordingProviderEmbed:
    """``RecordingProvider`` captures embed calls so ``--record`` works."""

    async def test_recorder_wraps_embed_and_writes_recording(self, tmp_path: Any) -> None:
        from agentloom.providers.recorder import RecordingProvider

        class _Wrapped(BaseProvider):
            name = "wrapped"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[0.1, 0.2], [0.3, 0.4]],
                    model=model,
                    provider=self.name,
                )

        out = tmp_path / "rec.json"
        rec = RecordingProvider(_Wrapped(), out)
        r = await rec.embed(inputs=["a", "b"], model="text-embedding-3-small", step_id="v")
        assert r.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        await rec.close()
        loaded = json.loads(out.read_text())
        assert loaded["_version"] == 2
        assert loaded["v"]["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert loaded["v"]["model"] == "text-embedding-3-small"

    async def test_recorded_file_replays_through_mock(self, tmp_path: Any) -> None:
        # End-to-end: record an embed via wrapper, then load the file with
        # MockProvider strict=True and verify the same vectors come back.
        from agentloom.providers.recorder import RecordingProvider

        class _Wrapped(BaseProvider):
            name = "wrapped"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[0.5, 0.5]],
                    model=model,
                    provider=self.name,
                )

        out = tmp_path / "rec.json"
        rec = RecordingProvider(_Wrapped(), out)
        await rec.embed(inputs=["hello"], model="x", step_id="v")
        await rec.close()

        mock = MockProvider(responses_file=out, strict=True)
        r = await mock.embed(inputs=["hello"], model="x", step_id="v")
        assert r.embeddings == [[0.5, 0.5]]


class TestGatewayEmbedResilience:
    """Rate-limit + circuit-breaker interactions on the embed path."""

    async def test_rate_limit_error_does_not_trip_breaker(self) -> None:
        from agentloom.exceptions import ProviderError, RateLimitError

        class _Throttled(BaseProvider):
            name = "throttled"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                raise RateLimitError(provider="throttled", retry_after_s=1.0)

        class _Ok(BaseProvider):
            name = "ok"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[1.0]] * len(inputs),
                    model=model,
                    provider=self.name,
                )

        gw = ProviderGateway()
        # Use rate limiter to make the throttled path realistic.
        gw.register(_Throttled(), priority=0, max_rpm=6000, max_tpm=6_000_000)
        gw.register(_Ok(), priority=1)
        r = await gw.embed(inputs=["a"], model="x")
        assert r.provider == "ok"
        # Throttled provider's breaker must NOT have recorded a failure —
        # rate limits are load signals, not faults.
        throttled_entry = next(e for e in gw._providers if e.provider.name == "throttled")
        assert throttled_entry.circuit_breaker._failure_count == 0

        # Now put ``ok`` first and verify a ProviderError bubbles up when
        # only throttled candidates remain.
        gw2 = ProviderGateway()
        gw2.register(_Throttled(), priority=0)
        with pytest.raises(ProviderError, match="rate-limited"):
            await gw2.embed(inputs=["a"], model="x")

    async def test_circuit_breaker_trips_after_repeated_failures(self) -> None:
        from agentloom.exceptions import ProviderError

        call_count = [0]

        class _Flaky(BaseProvider):
            name = "flaky"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                call_count[0] += 1
                raise RuntimeError("upstream 500")

        gw = ProviderGateway()
        gw.register(_Flaky(), priority=0, circuit_fail_threshold=3, circuit_reset_timeout=60.0)
        # Two failed calls open the breaker on the third (threshold=3
        # means the fourth attempt is refused).
        for _ in range(3):
            with pytest.raises(ProviderError):
                await gw.embed(inputs=["x"], model="anything")
        entry = gw._providers[0]
        assert entry.circuit_breaker._failure_count >= 3
        # Next call: breaker is now OPEN, provider isn't invoked at all.
        calls_before = call_count[0]
        with pytest.raises(ProviderError, match="circuit is open"):
            await gw.embed(inputs=["x"], model="anything")
        assert call_count[0] == calls_before

    async def test_falls_back_when_first_candidate_circuit_is_open(self) -> None:
        from agentloom.resilience.circuit_breaker import CircuitState

        class _Open(BaseProvider):
            name = "open-cb"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                raise AssertionError("Must not be called — circuit is open")

        class _Ok(BaseProvider):
            name = "ok"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[2.0]] * len(inputs),
                    model=model,
                    provider=self.name,
                )

        gw = ProviderGateway()
        gw.register(_Open(), priority=0)
        gw.register(_Ok(), priority=1)
        # Force the first provider's circuit into OPEN state.
        gw._providers[0].circuit_breaker._state = CircuitState.OPEN
        gw._providers[0].circuit_breaker._last_state_change = 999999999.0
        r = await gw.embed(inputs=["a"], model="x")
        assert r.provider == "ok"

    async def test_provider_call_hooks_fire_with_step_id(self) -> None:
        from agentloom.observability.noop import NoopObserver

        starts: list[dict[str, Any]] = []
        ends: list[dict[str, Any]] = []

        class _Obs(NoopObserver):
            def on_provider_call_start(self, **kwargs: Any) -> None:
                starts.append(kwargs)

            def on_provider_call_end(self, **kwargs: Any) -> None:
                ends.append(kwargs)

        class _Ok(BaseProvider):
            name = "hooked"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[0.1]],
                    model=model,
                    provider=self.name,
                    usage=__import__("agentloom.core.results", fromlist=["TokenUsage"]).TokenUsage(
                        prompt_tokens=7, total_tokens=7
                    ),
                )

        gw = ProviderGateway()
        gw.register(_Ok(), priority=0)
        gw.set_observer(_Obs())
        await gw.embed(inputs=["hi"], model="x", step_id="s1")
        assert len(starts) == 1 and starts[0]["step_id"] == "s1"
        assert len(ends) == 1 and ends[0]["prompt_tokens"] == 7


class TestEmbeddingCostTracking:
    """Cost from embed calls flows into ``WorkflowResult.total_cost_usd``."""

    async def test_embedding_cost_tracked_in_total_cost(self, tmp_path: Any) -> None:
        rec = tmp_path / "rec.json"
        rec.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "vectorize": {
                        "embeddings": [[0.1], [0.2], [0.3]],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                        "cost_usd": 0.00042,
                    },
                }
            )
        )
        wf = _embed_workflow(responses_file=str(rec))
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=rec), priority=0)
        sm = StateManager(initial_state=dict(wf.state))
        engine = WorkflowEngine(
            workflow=wf,
            state_manager=sm,
            provider_gateway=gw,
            tool_registry=ToolRegistry(),
        )
        result = await engine.run()
        assert result.total_cost_usd == pytest.approx(0.00042)
        assert result.total_tokens == 3


class TestGatewayEmbedContract:
    """Providers must return one vector per input; the gateway rejects mismatches."""

    async def test_len_mismatch_raises_provider_error(self) -> None:
        # A provider returning fewer vectors than inputs would silently
        # write misaligned data to state; the gateway wraps the CB call so
        # the mismatch surfaces as a ``ProviderError`` and the fallback
        # chain moves on rather than mis-mapping documents to vectors.
        from agentloom.exceptions import ProviderError

        class _Bad(BaseProvider):
            name = "bad"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[0.1]],  # 1 vector for N inputs
                    model=model,
                    provider=self.name,
                )

        gw = ProviderGateway()
        gw.register(_Bad(), priority=0)
        with pytest.raises(ProviderError, match="provider contract violation"):
            await gw.embed(inputs=["a", "b", "c"], model="x")


class TestGatewayEmbedObserverHooksOnErrorPaths:
    """``on_provider_call_end`` fires with ``error=<Class>`` on every error path."""

    async def test_rate_limit_error_fires_end_hook_with_error_label(self) -> None:
        from agentloom.exceptions import RateLimitError
        from agentloom.observability.noop import NoopObserver

        ends: list[dict[str, Any]] = []

        class _Obs(NoopObserver):
            def on_provider_call_start(self, **k: Any) -> None:
                pass

            def on_provider_call_end(self, **k: Any) -> None:
                ends.append(k)

        class _Throttled(BaseProvider):
            name = "throttled"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                raise RateLimitError("throttled", "rate limit")

        class _Yes(BaseProvider):
            name = "yes"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(embeddings=[[0.9]], model=model, provider=self.name)

        gw = ProviderGateway()
        gw.set_observer(_Obs())
        gw.register(_Throttled(), priority=0)
        gw.register(_Yes(), priority=1)
        r = await gw.embed(inputs=["x"], model="anything", step_id="s")
        assert r.provider == "yes"
        # First candidate's end_hook fires with error="RateLimitError";
        # the throttled provider is not marked as a fault (no
        # on_provider_error), so the second candidate serves the call.
        assert any(e.get("error") == "RateLimitError" for e in ends)

    async def test_generic_exception_fires_end_hook_and_error_hook(self) -> None:
        from agentloom.observability.noop import NoopObserver

        ends: list[dict[str, Any]] = []
        errors: list[tuple[str, str]] = []

        class _Obs(NoopObserver):
            def on_provider_call_start(self, **k: Any) -> None:
                pass

            def on_provider_call_end(self, **k: Any) -> None:
                ends.append(k)

            def on_provider_error(self, provider: str, error: str, **k: Any) -> None:
                errors.append((provider, error))

        class _Broken(BaseProvider):
            name = "broken"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                raise ValueError("upstream schema drift")

        class _Yes(BaseProvider):
            name = "yes"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(embeddings=[[0.1]], model=model, provider=self.name)

        gw = ProviderGateway()
        gw.set_observer(_Obs())
        gw.register(_Broken(), priority=0)
        gw.register(_Yes(), priority=1)
        r = await gw.embed(inputs=["hi"], model="x", step_id="s")
        assert r.provider == "yes"
        assert any(e.get("error") == "ValueError" for e in ends)
        assert ("broken", "ValueError") in errors

    async def test_not_implemented_fires_end_hook_with_error_label(self) -> None:
        from agentloom.observability.noop import NoopObserver

        ends: list[dict[str, Any]] = []

        class _Obs(NoopObserver):
            def on_provider_call_start(self, **k: Any) -> None:
                pass

            def on_provider_call_end(self, **k: Any) -> None:
                ends.append(k)

        class _NoEmbed(BaseProvider):
            name = "noembed"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        class _Yes(BaseProvider):
            name = "yes"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(embeddings=[[0.1]], model=model, provider=self.name)

        gw = ProviderGateway()
        gw.set_observer(_Obs())
        gw.register(_NoEmbed(), priority=0)
        gw.register(_Yes(), priority=1)
        r = await gw.embed(inputs=["hi"], model="x", step_id="s")
        assert r.provider == "yes"
        assert any(e.get("error") == "NotImplementedError" for e in ends)


class TestEmbedStepGuardrails:
    """Edge paths in ``EmbedStep``: missing gateway, provider raise, empty state segment."""

    async def test_step_fails_when_gateway_is_none(self) -> None:
        from agentloom.exceptions import StepError
        from agentloom.steps.embed import EmbedStep

        step = StepDefinition(
            id="v",
            type=StepType.EMBED,
            inputs="state.docs",
            output="vec",
        )
        base_step = EmbedStep()

        class _Ctx:
            def __init__(self) -> None:
                self.step_definition = step
                self.provider_gateway = None
                self.state_manager = StateManager(initial_state={"docs": ["a"]})
                self.workflow_model = "m"
                self.observer = None

        with pytest.raises(StepError, match="No provider gateway"):
            await base_step.execute(_Ctx())  # type: ignore[arg-type]

    async def test_step_returns_failed_when_gateway_raises(self, tmp_path: Any) -> None:
        # Gateway.embed raising a non-``StepError`` bubbles as a FAILED
        # step result (with ``error`` populated) rather than propagating —
        # the engine's retry loop then re-attempts within budget.
        from agentloom.core.results import StepStatus
        from agentloom.exceptions import ProviderError
        from agentloom.steps.embed import EmbedStep

        step = StepDefinition(
            id="v",
            type=StepType.EMBED,
            inputs="state.docs",
            output="vec",
        )

        class _BrokenGateway:
            async def embed(self, **kwargs: Any) -> EmbeddingResponse:
                raise ProviderError("gateway", "all providers failed")

        class _Ctx:
            def __init__(self) -> None:
                self.step_definition = step
                self.provider_gateway = _BrokenGateway()
                self.state_manager = StateManager(initial_state={"docs": ["a"]})
                self.workflow_model = "m"
                self.observer = None

        result = await EmbedStep().execute(_Ctx())  # type: ignore[arg-type]
        assert result.status == StepStatus.FAILED
        assert "all providers failed" in (result.error or "")

    async def test_step_rejects_empty_state_segment(self) -> None:
        # ``inputs: state..docs`` — the double-dot creates an empty path
        # segment which the resolver catches before the gateway is called.
        from agentloom.core.results import StepStatus

        wf = WorkflowDefinition(
            name="empty",
            config=WorkflowConfig(provider="mock", model="x"),
            state={"docs": ["a"]},
            steps=[
                StepDefinition(
                    id="v",
                    type=StepType.EMBED,
                    inputs="state..docs",
                    output="vec",
                )
            ],
        )
        gw = ProviderGateway()
        gw.register(MockProvider(), priority=0)
        sm = StateManager(initial_state=dict(wf.state))
        engine = WorkflowEngine(
            workflow=wf,
            state_manager=sm,
            provider_gateway=gw,
            tool_registry=ToolRegistry(),
        )
        result = await engine.run()
        assert result.status == WorkflowStatus.FAILED
        failed = next(s for s in result.step_results.values() if s.status == StepStatus.FAILED)
        assert "empty path segment" in (failed.error or "")


class TestObserverEmbeddingHook:
    """``NoopObserver`` and ``WorkflowObserver`` expose ``on_embedding_call``."""

    def test_noop_observer_on_embedding_call_is_a_pass(self) -> None:
        from agentloom.observability.noop import NoopObserver

        # Just needs to not raise — noop by contract.
        NoopObserver().on_embedding_call(
            provider="p", model="m", dimensions=8, input_count=1, prompt_tokens=3
        )

    def test_workflow_observer_records_metric_when_metrics_installed(self) -> None:
        from agentloom.observability.observer import WorkflowObserver

        recorded: list[tuple[str, str, int]] = []

        class _Metrics:
            def record_embedding_call(self, provider: str, model: str, dimensions: int) -> None:
                recorded.append((provider, model, dimensions))

        obs = WorkflowObserver()
        obs._metrics = _Metrics()  # type: ignore[assignment]
        obs.on_embedding_call(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            input_count=1,
            prompt_tokens=3,
        )
        assert recorded == [("openai", "text-embedding-3-small", 1536)]


class TestMetricsRecordEmbeddingCall:
    """``MetricsManager.record_embedding_call`` — disabled/OTel/no-dim paths."""

    def test_disabled_manager_is_a_noop(self) -> None:
        from agentloom.observability.metrics import MetricsManager

        m = MetricsManager(enabled=False)
        # No exception, no counter — the guard fires before any backend
        # lookup so ``_embedding_counter`` staying ``None`` is fine.
        m.record_embedding_call("openai", "text-embedding-3-small", 1536)

    def test_otel_backend_increments_counter_and_records_histogram(self) -> None:
        from agentloom.observability.metrics import MetricsManager

        counter_hits: list[tuple[int, dict[str, str]]] = []
        histogram_hits: list[tuple[int, dict[str, str]]] = []

        class _Counter:
            def add(self, value: int, attrs: dict[str, str]) -> None:
                counter_hits.append((value, attrs))

        class _Histogram:
            def record(self, value: int, attrs: dict[str, str]) -> None:
                histogram_hits.append((value, attrs))

        m = MetricsManager(enabled=True)
        m._enabled = True
        m._backend = "otel"
        m._embedding_counter = _Counter()  # type: ignore[assignment]
        m._embedding_dimensions_histogram = _Histogram()  # type: ignore[assignment]

        m.record_embedding_call("openai", "text-embedding-3-small", 1536)
        assert counter_hits == [(1, {"provider": "openai", "model": "text-embedding-3-small"})]
        assert histogram_hits == [(1536, {"provider": "openai", "model": "text-embedding-3-small"})]

    def test_otel_backend_skips_histogram_when_dimensions_zero(self) -> None:
        # ``dimensions=0`` means the caller did not request truncation —
        # recording ``0`` would skew the p50 downward, so skip the
        # histogram write while still incrementing the counter.
        from agentloom.observability.metrics import MetricsManager

        counter_hits: list[tuple[int, dict[str, str]]] = []
        histogram_hits: list[tuple[int, dict[str, str]]] = []

        class _Counter:
            def add(self, value: int, attrs: dict[str, str]) -> None:
                counter_hits.append((value, attrs))

        class _Histogram:
            def record(self, value: int, attrs: dict[str, str]) -> None:
                histogram_hits.append((value, attrs))

        m = MetricsManager(enabled=True)
        m._enabled = True
        m._backend = "otel"
        m._embedding_counter = _Counter()  # type: ignore[assignment]
        m._embedding_dimensions_histogram = _Histogram()  # type: ignore[assignment]

        m.record_embedding_call("ollama", "nomic-embed-text", 0)
        assert len(counter_hits) == 1
        assert histogram_hits == []


class TestProviderEmbedHTTPErrorPaths:
    """``httpx.HTTPError`` inside ``embed`` becomes a ``ProviderError`` with the class name."""

    @respx.mock
    async def test_openai_embed_wraps_http_error(self) -> None:
        from agentloom.exceptions import ProviderError

        respx.post("https://api.openai.com/v1/embeddings").mock(
            side_effect=httpx.ConnectError("boom")
        )
        p = OpenAIProvider(api_key="k")
        with pytest.raises(ProviderError, match="ConnectError"):
            await p.embed(inputs=["a"], model="text-embedding-3-small")
        await p.close()

    @respx.mock
    async def test_google_embed_wraps_http_error(self) -> None:
        from agentloom.exceptions import ProviderError

        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-embedding-001:batchEmbedContents"
        ).mock(side_effect=httpx.ConnectError("boom"))
        p = GoogleProvider(api_key="k")
        with pytest.raises(ProviderError, match="ConnectError"):
            await p.embed(inputs=["a"], model="gemini-embedding-001")
        await p.close()

    @respx.mock
    async def test_ollama_embed_wraps_http_error(self) -> None:
        from agentloom.exceptions import ProviderError

        respx.post("http://localhost:11434/api/embed").mock(side_effect=httpx.ConnectError("boom"))
        p = OllamaProvider()
        with pytest.raises(ProviderError, match="ConnectError"):
            await p.embed(inputs=["a"], model="nomic-embed-text")
        await p.close()


class TestGatewayEmbedNoCandidates:
    """A model with no matching provider raises early — clearer than a fallback-chain summary."""

    async def test_no_candidates_raises_provider_error(self) -> None:
        from agentloom.exceptions import ProviderError

        gw = ProviderGateway()  # nothing registered
        with pytest.raises(ProviderError, match="No provider registered"):
            await gw.embed(inputs=["hi"], model="unknown-model")


class TestGoogleEmbedTitleKwarg:
    """``title`` kwarg lands on each per-input request body."""

    @respx.mock
    async def test_google_embed_forwards_title(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]})

        respx.post(url__regex=r".+batchEmbedContents.+").mock(side_effect=_handler)
        provider = GoogleProvider(api_key="g-test")
        await provider.embed(
            inputs=["hello"],
            model="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            title="Doc A",
        )
        req = captured["payload"]["requests"][0]
        assert req["title"] == "Doc A"
        assert req["taskType"] == "RETRIEVAL_DOCUMENT"
        await provider.close()


class TestSupportsModelNarrowness:
    """``supports_model`` must not claim other providers' embedding SKUs."""

    def test_openai_does_not_claim_google_embedding_004(self) -> None:
        # ``text-embedding-004`` is Google's SKU; OpenAI's matcher must
        # skip it so gateway fallback doesn't route Google model strings
        # to the OpenAI adapter.
        p = OpenAIProvider(api_key="k")
        assert not p.supports_model("text-embedding-004")
        assert p.supports_model("text-embedding-3-small")
        assert p.supports_model("text-embedding-ada-002")

    def test_google_does_not_claim_openai_text_embedding_3(self) -> None:
        # ``text-embedding-3-small`` is OpenAI territory; Gemini's matcher
        # must skip it. ``embedding-001`` (bare) and any ``gemini-*`` are
        # legitimate Google matches.
        p = GoogleProvider(api_key="k")
        assert not p.supports_model("text-embedding-3-small")
        assert not p.supports_model("text-embedding-004")
        assert p.supports_model("gemini-embedding-001")
        assert p.supports_model("embedding-001")


class TestStepDefinitionDimensionsValidation:
    """``dimensions`` must be a positive integer or omitted."""

    def test_zero_dimensions_rejected_at_parse(self) -> None:
        # A workflow author who typos ``dimensions: 0`` gets a clean
        # parse-time error rather than a provider-specific HTTP 400 at
        # runtime.
        with pytest.raises(Exception, match="greater than or equal to 1"):
            StepDefinition(
                id="v",
                type=StepType.EMBED,
                inputs="state.docs",
                output="vec",
                dimensions=0,
            )

    def test_negative_dimensions_rejected_at_parse(self) -> None:
        with pytest.raises(Exception, match="greater than or equal to 1"):
            StepDefinition(
                id="v",
                type=StepType.EMBED,
                inputs="state.docs",
                output="vec",
                dimensions=-8,
            )

    def test_positive_dimensions_accepted(self) -> None:
        s = StepDefinition(
            id="v",
            type=StepType.EMBED,
            inputs="state.docs",
            output="vec",
            dimensions=256,
        )
        assert s.dimensions == 256
