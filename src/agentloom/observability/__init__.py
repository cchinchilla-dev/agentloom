"""Observability layer — optional, degrades gracefully without dependencies.

Install with: pip install agentloom[observability]
"""

from agentloom.observability.cost_tracker import CostTracker
from agentloom.observability.noop import NoopMeter, NoopSpan, NoopTracer

__all__ = ["CostTracker", "NoopMeter", "NoopSpan", "NoopTracer"]
