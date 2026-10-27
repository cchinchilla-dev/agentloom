"""Pydantic models for ``examples/36_structured_output.yaml`` and ``examples/41_agent.yaml``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Classification(BaseModel):
    """Intent classification with confidence + rationale."""

    label: Literal["question", "complaint", "feedback", "request"] = Field(
        ..., description="The single intent label this message best fits."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's self-reported confidence (0–1)."
    )
    rationale: str = Field(..., description="One-sentence justification for the chosen label.")


class SupportReply(BaseModel):
    """The reply of the support desk agent in ``examples/agents/support.yaml``."""

    answer: str = Field(..., description="The answer for the customer, in one or two sentences.")
    refund: Literal["eligible", "not_eligible", "not_requested"] = Field(
        ..., description="Whether the customer asked for money back, and if so the decision."
    )
