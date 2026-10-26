"""Experimental contracts: telemetry names and the inter-agent event log.

The OpenTelemetry span, attribute and metric names follow the GenAI semantic
conventions, which are still in development, so a minor release may rename
them when the conventions change. The inter-agent event models (#135) and
their JSON Schema (``SCHEMA_PATH``, ``SCHEMA_URI``) may change while they
settle. Importing this module emits a ``FutureWarning`` once per process; the
runtime uses the same names through ``agentloom.observability.schema`` and
``agentloom.core.inter_agent`` without one.
"""

from __future__ import annotations

import warnings

from agentloom.contracts._inter_agent import (
    SCHEMA_PATH,
    SCHEMA_URI,
    InterAgentEvent,
    ResponseSentEvent,
    StateTransitionEvent,
    TaskReceivedEvent,
    TaskSentEvent,
    ToolCallEvent,
    ToolReturnEvent,
)
from agentloom.contracts._semconv import (
    GenAIOperationName,
    GenAIProviderName,
    MetricName,
    SpanAttr,
    SpanName,
)

warnings.warn(
    "agentloom.contracts.experimental tracks the OpenTelemetry GenAI semantic conventions, "
    "which are still in development, and the inter-agent event log: these names may change "
    "in a minor release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_URI",
    "GenAIOperationName",
    "GenAIProviderName",
    "InterAgentEvent",
    "MetricName",
    "ResponseSentEvent",
    "SpanAttr",
    "SpanName",
    "StateTransitionEvent",
    "TaskReceivedEvent",
    "TaskSentEvent",
    "ToolCallEvent",
    "ToolReturnEvent",
]
