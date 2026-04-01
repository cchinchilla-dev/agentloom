"""Fixtures for e2e tests against live LLM providers."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from agentloom.providers.ollama import OllamaProvider

E2E_OLLAMA_MODEL = "qwen2.5:0.5b"


@pytest.fixture
async def ollama_provider() -> AsyncIterator[OllamaProvider]:
    """OllamaProvider pointed at the local (or CI) Ollama instance."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    provider = OllamaProvider(api_key="", base_url=base_url)
    yield provider
    await provider.close()


@pytest.fixture
def ollama_model() -> str:
    """Model name used across e2e tests."""
    return E2E_OLLAMA_MODEL
