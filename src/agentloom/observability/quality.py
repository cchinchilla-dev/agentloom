"""Emit :class:`QualityAnnotation` data as OTel spans.

The workflow span closes inside the engine before the caller sees
:class:`WorkflowResult`; retroactively adding attributes to a closed span
is not viable. Instead, each annotation becomes a standalone span named
``quality:<target>`` carrying ``workflow.run_id`` and ``workflow.name`` so
trace consumers can filter by run_id to group quality spans with the
original workflow trace.

The OTel call path is optional — the AgentLoom observer hook is a no-op
when no tracer is configured, so ``result.annotate(...)`` remains safe to
call in offline/test scenarios.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentloom.observability.schema import SpanAttr, SpanName

if TYPE_CHECKING:
    from agentloom.core.results import QualityAnnotation, WorkflowResult


# Span-attribute types OTel takes as-is; anything else is JSON-serialized.
_OTEL_PRIMITIVE_TYPES = (str, bool, int, float)


def _coerce_attribute_value(value: Any) -> Any:
    """Make *value* safe to pass to ``span.set_attribute``.

    OTel accepts primitives (``str`` / ``bool`` / ``int`` / ``float``)
    and homogeneous lists of those. Anything else — dicts, sets, custom
    objects, or mixed-type lists — is serialized to a JSON string so the
    attribute survives the export pipeline. ``json.dumps`` falls back to
    ``str(value)`` for objects without a JSON encoding.
    """
    if isinstance(value, _OTEL_PRIMITIVE_TYPES):
        return value
    # OTel arrays must be homogeneous; ``type(v) is`` keeps bools apart from ints.
    if isinstance(value, (list, tuple)) and value:
        first_type = type(value[0])
        if first_type in _OTEL_PRIMITIVE_TYPES and all(type(v) is first_type for v in value):
            return list(value)
    elif isinstance(value, (list, tuple)):
        # Empty list is trivially homogeneous and OTel-safe.
        return []
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def emit_quality_annotation(
    annotation: QualityAnnotation,
    tracing: Any,
    *,
    run_id: str,
    workflow_name: str,
) -> None:
    """Emit a single annotation as a standalone ``quality:<target>`` span.

    *tracing* is any object with a ``start_span(name, attributes=...)`` and
    ``end_span(span)`` surface — :class:`TracingManager` satisfies this,
    as does a MagicMock in tests. Pass ``None`` (or any falsy value) to
    no-op.
    """
    if not tracing:
        return
    attrs: dict[str, Any] = {
        SpanAttr.WORKFLOW_RUN_ID: run_id,
        SpanAttr.WORKFLOW_NAME: workflow_name,
        SpanAttr.QUALITY_SCORE: annotation.quality_score,
        SpanAttr.QUALITY_SOURCE: annotation.source,
        SpanAttr.QUALITY_TARGET: annotation.target,
    }
    # One span attribute per metadata key, queryable without JSON parsing.
    for key, value in annotation.metadata.items():
        attrs[f"{SpanAttr.QUALITY_METADATA_PREFIX}{key}"] = _coerce_attribute_value(value)

    span = tracing.start_span(
        SpanName.QUALITY.format(target=annotation.target),
        attributes=attrs,
    )
    end = getattr(tracing, "end_span", None)
    if end is not None:
        end(span)
    else:
        span.end()


def emit_quality_annotations(
    result: WorkflowResult,
    tracing: Any,
) -> None:
    """Emit every annotation attached to *result*."""
    for annotation in result.annotations:
        emit_quality_annotation(
            annotation,
            tracing,
            run_id=result.run_id,
            workflow_name=result.workflow_name,
        )


__all__ = [
    "emit_quality_annotation",
    "emit_quality_annotations",
]
