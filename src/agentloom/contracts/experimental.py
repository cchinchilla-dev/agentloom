"""Experimental contracts: OpenTelemetry span, attribute and metric names.

They follow the OpenTelemetry GenAI semantic conventions, which are still in
development, so a minor release may rename them when the conventions change.
Importing this module emits a ``FutureWarning`` once per process; the runtime
uses the same names through ``agentloom.observability.schema`` without one.
"""

from __future__ import annotations

import warnings

from agentloom.contracts._semconv import (
    GenAIOperationName,
    GenAIProviderName,
    MetricName,
    SpanAttr,
    SpanName,
)

warnings.warn(
    "agentloom.contracts.experimental tracks the OpenTelemetry GenAI semantic conventions, "
    "which are still in development: these names may change in a minor release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["GenAIOperationName", "GenAIProviderName", "MetricName", "SpanAttr", "SpanName"]
