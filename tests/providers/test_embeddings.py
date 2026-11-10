"""Tests for embeddings across providers, gateway, mock, and step."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

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
        # The gateway picks embed candidates by ``supports_model``.
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

        respx.post(url__regex=r".+batchEmbedContents$").mock(side_effect=_handler)
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

        respx.post(url__regex=r".+batchEmbedContents$").mock(side_effect=_handler)
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
        # A provider without ``embed()`` is skipped without a circuit-breaker failure.
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
    """MockProvider serves recordings or synthesizes pseudo-vectors."""

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

    def test_embed_step_rejects_missing_inputs_field(self) -> None:
        with pytest.raises(ValueError, match="requires 'inputs'"):
            StepDefinition(id="v", type=StepType.EMBED)

    async def test_embed_step_rejects_wrong_shape(self) -> None:
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
        # A lone string is embedded as a one-element batch.
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
        # Only ``state.`` paths resolve: no eval, no template rendering.
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


class TestEmbedRequestHash:
    """A recorded embedding carries the hash of its request, and strict replay checks it."""

    @staticmethod
    def _wrapped(vectors: int) -> BaseProvider:
        class _Wrapped(BaseProvider):
            name = "wrapped"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[0.5, 0.5]] * vectors, model=model, provider=self.name
                )

        return _Wrapped()

    async def _record(self, tmp_path: Any) -> Any:
        from agentloom.providers.recorder import RecordingProvider

        out = tmp_path / "rec.json"
        recorder = RecordingProvider(self._wrapped(2), out)
        await recorder.embed(inputs=["a", "b"], model="x", dimensions=2, step_id="v")
        await recorder.close()
        return out

    async def test_the_recording_holds_the_hash(self, tmp_path: Any) -> None:
        out = await self._record(tmp_path)
        assert len(json.loads(out.read_text())["v"]["request_hash"]) == 64

    async def test_the_same_request_replays(self, tmp_path: Any) -> None:
        mock = MockProvider(responses_file=await self._record(tmp_path), strict=True)
        r = await mock.embed(inputs=["a", "b"], model="x", dimensions=2, step_id="v")
        assert r.embeddings == [[0.5, 0.5], [0.5, 0.5]]

    @pytest.mark.parametrize(
        "changed",
        [
            {"inputs": ["a", "c"], "model": "x", "dimensions": 2},
            {"inputs": ["a", "b"], "model": "y", "dimensions": 2},
            {"inputs": ["a", "b"], "model": "x", "dimensions": 4},
        ],
        ids=["input", "model", "dimensions"],
    )
    async def test_a_changed_request_fails_strict_replay(
        self, tmp_path: Any, changed: dict[str, Any]
    ) -> None:
        from agentloom.exceptions import RecordingMismatchError

        mock = MockProvider(responses_file=await self._record(tmp_path), strict=True)
        with pytest.raises(RecordingMismatchError, match="does not match the recording"):
            await mock.embed(step_id="v", **changed)

    async def test_a_changed_request_warns_in_lenient_replay(
        self, tmp_path: Any, caplog: Any
    ) -> None:
        mock = MockProvider(responses_file=await self._record(tmp_path))
        r = await mock.embed(inputs=["a", "c"], model="x", dimensions=2, step_id="v")
        assert r.embeddings == [[0.5, 0.5], [0.5, 0.5]]
        assert any("changed since the recording" in rec.getMessage() for rec in caplog.records)

    async def test_a_recording_without_a_hash_still_replays(self, tmp_path: Any) -> None:
        rec = tmp_path / "rec.json"
        rec.write_text(json.dumps({"v": {"embeddings": [[0.5, 0.5]]}}))
        mock = MockProvider(responses_file=rec, strict=True)
        assert (await mock.embed(inputs=["a"], model="x", step_id="v")).embeddings == [[0.5, 0.5]]

    async def test_another_number_of_inputs_is_a_mismatch_at_once(self, tmp_path: Any) -> None:
        from agentloom.exceptions import RecordingMismatchError

        rec = tmp_path / "rec.json"
        rec.write_text(json.dumps({"v": {"embeddings": [[0.5], [0.6], [0.7]]}}))
        gateway = ProviderGateway()
        gateway.register(MockProvider(responses_file=rec, strict=True), priority=0)
        with pytest.raises(RecordingMismatchError, match="3 vectors for 2 inputs"):
            await gateway.embed(inputs=["a", "b"], model="x", step_id="v")


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
        # Rate limits are load signals, not faults: the breaker records nothing.
        throttled_entry = next(e for e in gw._providers if e.provider.name == "throttled")
        assert throttled_entry.circuit_breaker._failure_count == 0

        # With no other candidate, the rate limit surfaces as a ProviderError.
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
        # Threshold 3: the third failure opens the breaker, so the fourth call is refused.
        for _ in range(3):
            with pytest.raises(ProviderError):
                await gw.embed(inputs=["x"], model="anything")
        entry = gw._providers[0]
        assert entry.circuit_breaker._failure_count >= 3
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
        # Fewer vectors than inputs would silently mis-map documents to vectors in state.
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
        # The double dot leaves an empty path segment, caught before the gateway is called.
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
        # The guard returns before touching ``_embedding_counter``, which stays ``None``.
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
        # 0 means no truncation was requested; recording it would skew the histogram down.
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

        gw = ProviderGateway()
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

        respx.post(url__regex=r".+batchEmbedContents$").mock(side_effect=_handler)
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
        # ``text-embedding-004`` is a Google SKU despite the ``text-embedding-`` prefix.
        p = OpenAIProvider(api_key="k")
        assert not p.supports_model("text-embedding-004")
        assert p.supports_model("text-embedding-3-small")
        assert p.supports_model("text-embedding-ada-002")

    def test_google_does_not_claim_openai_text_embedding_3(self) -> None:
        p = GoogleProvider(api_key="k")
        assert not p.supports_model("text-embedding-3-small")
        assert not p.supports_model("text-embedding-004")
        assert p.supports_model("gemini-embedding-001")
        assert p.supports_model("embedding-001")


class TestStepDefinitionDimensionsValidation:
    """``dimensions`` must be a positive integer or omitted."""

    def test_zero_dimensions_rejected_at_parse(self) -> None:
        # A parse-time error beats a provider-specific HTTP 400 at run time.
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


class TestProviderCompleteHTTPErrorPathsUseFormatter:
    """``httpx.HTTPError`` inside ``complete()`` becomes a ``ProviderError`` with the class name."""

    @respx.mock
    async def test_openai_complete_wraps_http_error(self) -> None:
        from agentloom.exceptions import ProviderError

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("boom")
        )
        p = OpenAIProvider(api_key="k")
        with pytest.raises(ProviderError, match="ConnectError"):
            await p.complete(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")
        await p.close()

    @respx.mock
    async def test_anthropic_complete_wraps_http_error(self) -> None:
        from agentloom.exceptions import ProviderError

        respx.post("https://api.anthropic.com/v1/messages").mock(
            side_effect=httpx.ConnectError("boom")
        )
        p = AnthropicProvider(api_key="k")
        with pytest.raises(ProviderError, match="ConnectError"):
            await p.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-haiku-4-5-20251001",
            )
        await p.close()

    @respx.mock
    async def test_google_complete_wraps_http_error(self) -> None:
        from agentloom.exceptions import ProviderError

        respx.post(url__regex=r".+generateContent$").mock(side_effect=httpx.ConnectError("boom"))
        p = GoogleProvider(api_key="k")
        with pytest.raises(ProviderError, match="ConnectError"):
            await p.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="gemini-2.5-flash",
            )
        await p.close()


class TestMockEmbedHashLookup:
    """Recordings without ``step_id`` keys fall back to hash-based lookup."""

    async def test_embed_recording_keyed_by_hash_still_serves(self, tmp_path: Any) -> None:
        # The recorder keys an embed call made without a step id by this hash.
        from agentloom.providers.mock import prompt_hash

        h = prompt_hash(
            [{"embed_inputs": ["hello"], "model": "text-embedding-3-small", "dimensions": None}],
            "text-embedding-3-small",
            None,
            None,
            {},
        )
        rec = tmp_path / "rec.json"
        rec.write_text(
            json.dumps(
                {
                    "_version": 2,
                    h: {
                        "embeddings": [[0.42]],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                        "cost_usd": 0.0,
                    },
                }
            )
        )
        provider = MockProvider(responses_file=rec)
        r = await provider.embed(
            inputs=["hello"],
            model="text-embedding-3-small",
            step_id="unregistered-step",
        )
        assert r.embeddings == [[0.42]]


class TestGatewayEmbedCircuitOpenMidCall:
    """``CircuitOpenError`` raised inside ``_call`` (race) falls back cleanly."""

    async def test_circuit_open_error_during_call_moves_to_next_candidate(self) -> None:
        # A concurrent request can open the breaker between the pre-check and the call.
        from agentloom.exceptions import CircuitOpenError

        class _FakeCB:
            state = "OPEN"

            def _maybe_transition_to_half_open(self) -> Any:
                # Closed at pre-check, so the gateway reaches ``call()``, which raises.
                from agentloom.providers.gateway import CircuitState

                return CircuitState.CLOSED

            async def call(self, fn: Any, exclude: Any = None) -> Any:
                raise CircuitOpenError("throttled")

        class _First(BaseProvider):
            name = "first"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[0.0]] * len(inputs), model=model, provider=self.name
                )

        class _Second(BaseProvider):
            name = "second"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(
                    embeddings=[[9.9]] * len(inputs), model=model, provider=self.name
                )

        gw = ProviderGateway()
        gw.register(_First(), priority=0)
        gw.register(_Second(), priority=1)
        gw._providers[0].circuit_breaker = _FakeCB()  # type: ignore[assignment]
        r = await gw.embed(inputs=["hi"], model="x")
        assert r.provider == "second"


class TestEmbeddingsAreNotChat:
    """An embedding call reports the ``embeddings`` operation to the observer."""

    async def test_gateway_hooks_carry_the_operation(self) -> None:
        class _Ok(BaseProvider):
            name = "hooked"

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, messages, model, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            async def embed(self, inputs, model, dimensions=None, **kwargs):  # type: ignore[override]
                return EmbeddingResponse(embeddings=[[0.1]], model=model, provider=self.name)

        gw = ProviderGateway()
        gw.register(_Ok(), priority=0)
        observer = MagicMock()
        gw.set_observer(observer)

        await gw.embed(inputs=["hi"], model="x", step_id="s1")

        assert observer.on_provider_call_start.call_args.kwargs["operation"] == "embeddings"
        assert observer.on_provider_call_end.call_args.kwargs["operation"] == "embeddings"
        observer.on_circuit_state.assert_called_once_with("hooked", "closed")

    async def test_tokens_of_an_embed_step_carry_it(self, tmp_path: Any) -> None:
        rec = tmp_path / "rec.json"
        rec.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "vectorize": {
                        "embeddings": [[0.1], [0.2], [0.3]],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                        "cost_usd": 0.0,
                    },
                }
            )
        )
        wf = _embed_workflow(responses_file=str(rec))
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=rec), priority=0)
        observer = MagicMock()
        engine = WorkflowEngine(
            workflow=wf, provider_gateway=gw, tool_registry=ToolRegistry(), observer=observer
        )

        result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        observer.on_tokens.assert_called_once()
        assert observer.on_tokens.call_args.kwargs["operation"] == "embeddings"
