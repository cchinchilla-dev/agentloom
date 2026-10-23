"""Any OpenAI-compatible endpoint (vLLM, TGI, LM Studio, a gateway) through the OpenAI adapter."""

from __future__ import annotations

from typing import Any

from agentloom.providers.openai import OpenAIProvider


class GenericOpenAIProvider(OpenAIProvider):
    """OpenAI chat-completions adapter for a self-hosted or third-party endpoint.

    ``base_url`` is required and every model name is accepted: the server
    decides what it serves. The key comes only from ``api_key`` (a workflow
    names its variable in ``config.api_key_env``), never from
    ``OPENAI_API_KEY``, so an OpenAI key is not sent to another server.
    Without a key, requests carry no ``Authorization`` header.
    """

    name = "generic_openai"
    _API_KEY_ENV = None

    def __init__(self, base_url: str, api_key: str = "", **kwargs: Any) -> None:
        if not base_url:
            raise ValueError("GenericOpenAIProvider needs a base_url")
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    def supports_model(self, model: str) -> bool:
        return True
