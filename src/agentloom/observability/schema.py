"""Single source of truth for OTel span, attribute, and metric names.

Downstream trace consumers (AgentAnvil, Grafana dashboards, Jaeger plugins)
parse these strings. Every span name, span attribute, and metric name used
anywhere in the engine lives here — no raw string literals for telemetry
keys allowed outside this module.

Attribute naming follows the OpenTelemetry **GenAI semantic conventions**
where applicable (``gen_ai.*``); AgentLoom-specific fields use the
``workflow.*``, ``step.*``, ``tool.*``, or ``agentloom.*`` prefixes.

The names are defined in ``agentloom.contracts._semconv`` and published as
``agentloom.contracts.experimental`` (which warns: the GenAI conventions are
still in development). This module re-exports them for the runtime without a
warning and adds the provider-name mapping.
"""

from __future__ import annotations

from agentloom.contracts._semconv import GenAIOperationName as GenAIOperationName
from agentloom.contracts._semconv import GenAIProviderName as GenAIProviderName
from agentloom.contracts._semconv import MetricName as MetricName
from agentloom.contracts._semconv import SpanAttr as SpanAttr
from agentloom.contracts._semconv import SpanName as SpanName

# Map AgentLoom provider class names (``Provider.name``) to the canonical
# OTel ``gen_ai.provider.name`` value. Provider classes keep their short
# names internally; the observer / gateway translate at telemetry time.
_PROVIDER_NAME_TO_GENAI: dict[str, str] = {
    "openai": GenAIProviderName.OPENAI.value,
    "anthropic": GenAIProviderName.ANTHROPIC.value,
    "google": GenAIProviderName.GCP_GEMINI.value,
    "ollama": GenAIProviderName.OLLAMA.value,
    "mock": GenAIProviderName.MOCK.value,
}


def to_genai_provider_name(provider_name: str) -> str:
    """Translate an AgentLoom provider name to the OTel canonical form."""
    return _PROVIDER_NAME_TO_GENAI.get(provider_name, provider_name)


__all__ = [
    "GenAIOperationName",
    "GenAIProviderName",
    "MetricName",
    "SpanAttr",
    "SpanName",
    "to_genai_provider_name",
]
