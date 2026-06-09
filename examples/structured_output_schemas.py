"""Pydantic models for ``examples/36_structured_output.yaml``."""

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
