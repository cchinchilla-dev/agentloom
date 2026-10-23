"""Generic OpenAI-compatible endpoint provider (#48): vLLM, TGI, LM Studio, a gateway."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import ValidationError

from agentloom.core.models import WorkflowConfig
from agentloom.exceptions import ProviderError
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.generic_openai import GenericOpenAIProvider
from agentloom.providers.openai import OpenAIProvider

BASE = "http://vllm.local:8000/v1"
MODEL = "meta-llama/Llama-3-70B"
HI = [{"role": "user", "content": "hi"}]
REPLY = {
    "choices": [{"message": {"content": "hola"}, "finish_reason": "stop"}],
    "model": MODEL,
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}


@respx.mock
async def test_completes_any_model_at_its_base_url_without_the_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=REPLY)
    )
    provider = GenericOpenAIProvider(base_url=BASE)
    assert provider.supports_model(MODEL)
    result = await provider.complete(messages=HI, model=MODEL)
    await provider.close()

    assert (result.content, result.provider, result.cost_usd) == ("hola", "generic_openai", 0.0)
    request = route.calls[0].request
    assert json.loads(request.content)["model"] == MODEL
    assert "authorization" not in request.headers


@respx.mock
async def test_sends_the_key_it_is_given() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=REPLY)
    )
    provider = GenericOpenAIProvider(base_url=BASE, api_key="YOUR_VLLM_KEY_HERE")
    await provider.complete(messages=HI, model=MODEL)
    await provider.close()
    assert route.calls[0].request.headers["authorization"] == "Bearer YOUR_VLLM_KEY_HERE"


@respx.mock
async def test_streams_from_its_base_url() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"ho"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"la"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, content=sse.encode())
    )
    provider = GenericOpenAIProvider(base_url=BASE)
    sr = await provider.stream(messages=HI, model=MODEL)
    assert [chunk async for chunk in sr] == ["ho", "la"]
    assert (sr.content, sr.provider) == ("hola", "generic_openai")
    await provider.close()


@pytest.mark.parametrize(
    ("given", "used"),
    [
        ("http://vllm.local:8000", "http://vllm.local:8000/v1"),
        ("http://127.0.0.1:8090/s/ep1/agent/v1", "http://127.0.0.1:8090/s/ep1/agent/v1"),
    ],
)
def test_base_url_gets_v1_only_on_a_bare_host(given: str, used: str) -> None:
    assert GenericOpenAIProvider(base_url=given).base_url == used


def test_a_base_url_is_required() -> None:
    with pytest.raises(ValueError, match="needs a base_url"):
        GenericOpenAIProvider(base_url="")


@respx.mock
async def test_openai_sends_no_authorization_header_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=REPLY)
    )
    provider = OpenAIProvider()
    await provider.complete(messages=HI, model="gpt-4o-mini")
    await provider.close()
    assert "authorization" not in route.calls[0].request.headers


def test_config_needs_a_base_url_for_generic_openai() -> None:
    with pytest.raises(ValidationError, match="needs config.base_url"):
        WorkflowConfig(provider="generic_openai")


@pytest.mark.parametrize("field", ["base_url", "api_key_env"])
def test_config_refuses_endpoint_fields_with_another_provider(field: str) -> None:
    with pytest.raises(ValidationError, match="apply only to provider 'generic_openai'"):
        WorkflowConfig(provider="openai", **{field: "x"})


def test_cli_registers_the_endpoint_with_the_key_its_variable_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentloom.cli.run import _setup_generic_openai

    monkeypatch.setenv("VLLM_API_KEY", "YOUR_VLLM_KEY_HERE")
    gateway = ProviderGateway()
    config = WorkflowConfig(
        provider="generic_openai", base_url=BASE, model=MODEL, api_key_env="VLLM_API_KEY"
    )
    _setup_generic_openai(gateway, config)
    (entry,) = gateway._providers
    assert isinstance(entry.provider, GenericOpenAIProvider)
    assert (entry.priority, entry.provider.base_url) == (0, BASE)
    assert entry.provider.api_key == "YOUR_VLLM_KEY_HERE"


def test_cli_refuses_a_key_variable_that_is_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentloom.cli.run import _setup_generic_openai

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    config = WorkflowConfig(provider="generic_openai", base_url=BASE, api_key_env="VLLM_API_KEY")
    with pytest.raises(ProviderError, match="VLLM_API_KEY"):
        _setup_generic_openai(ProviderGateway(), config)


def test_cli_refuses_a_provider_override_without_a_base_url() -> None:
    from agentloom.cli.run import _setup_generic_openai

    config = WorkflowConfig(provider="openai")
    config.provider = "generic_openai"  # what ``--provider generic_openai`` does after parsing
    with pytest.raises(ProviderError, match="needs config.base_url"):
        _setup_generic_openai(ProviderGateway(), config)


def test_cli_leaves_other_providers_alone() -> None:
    from agentloom.cli.run import _setup_generic_openai

    gateway = ProviderGateway()
    _setup_generic_openai(gateway, WorkflowConfig(provider="openai"))
    assert gateway._providers == []
