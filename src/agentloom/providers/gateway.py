"""Provider Gateway — unified interface with fallback."""

from __future__ import annotations

import logging
from typing import Any

from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("agentloom.gateway")


class ProviderEntry:
    def __init__(self, provider: BaseProvider, priority: int = 0, is_fallback: bool = False, models: list[str] | None = None) -> None:
        self.provider = provider
        self.priority = priority
        self.is_fallback = is_fallback
        self.models = models or []


class ProviderGateway:
    """Central provider routing with fallback."""

    def __init__(self) -> None:
        self._providers: list[ProviderEntry] = []

    def register(self, provider: BaseProvider, priority: int = 0, is_fallback: bool = False, models: list[str] | None = None, **kwargs: Any) -> None:
        entry = ProviderEntry(provider=provider, priority=priority, is_fallback=is_fallback, models=models)
        self._providers.append(entry)
        self._providers.sort(key=lambda e: e.priority)

    async def complete(self, messages: list[dict[str, str]], model: str, temperature: float | None = None, max_tokens: int | None = None, **kwargs: Any) -> ProviderResponse:
        candidates = [e for e in self._providers if e.provider.supports_model(model)]
        fallbacks = [e for e in self._providers if e.is_fallback and e not in candidates]
        all_c = candidates + fallbacks
        if not all_c:
            raise ProviderError("gateway", f"No provider for model \'{model}\'")
        errors: list[str] = []
        for entry in all_c:
            try:
                return await entry.provider.complete(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
            except Exception as e:
                errors.append(f"Provider \'{entry.provider.name}\': {e}")
        raise ProviderError("gateway", "All providers failed: " + "; ".join(errors))

    async def close(self) -> None:
        for entry in self._providers:
            await entry.provider.close()
