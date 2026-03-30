"""Model pricing table for cost calculation."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelPricing(BaseModel):
    """Pricing for a specific model (cost per 1K tokens)."""

    input_cost_per_1k: float
    output_cost_per_1k: float


_BUNDLED_PRICING_PATH = Path(__file__).parent / "pricing.yaml"


def _load_pricing_yaml(path: Path) -> dict[str, ModelPricing]:
    """Load pricing from a YAML file."""
    raw: dict[str, dict[str, float]] = yaml.safe_load(path.read_text())
    return {
        model: ModelPricing(input_cost_per_1k=vals["input"], output_cost_per_1k=vals["output"])
        for model, vals in raw.items()
    }


def load_pricing(custom_path: str | None = None) -> dict[str, ModelPricing]:
    """Load pricing table from YAML.

    Resolution order:
      1. ``custom_path`` argument (explicit caller override)
      2. ``AGENTLOOM_PRICING_FILE`` env var
      3. Bundled ``pricing.yaml`` shipped with the package

    Returns:
        Mapping of model name to :class:`ModelPricing`.
    """
    path_str = custom_path or os.environ.get("AGENTLOOM_PRICING_FILE")
    if path_str:
        return _load_pricing_yaml(Path(path_str))
    return _load_pricing_yaml(_BUNDLED_PRICING_PATH)


# Module-level table — populated on first import.
DEFAULT_PRICING: dict[str, ModelPricing] = load_pricing()


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_table: dict[str, ModelPricing] | None = None,
) -> float:
    """Calculate the cost of an LLM call.

    Args:
        model: Model name.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.
        pricing_table: Custom pricing table (defaults to DEFAULT_PRICING).

    Returns:
        Cost in USD. Returns 0.0 for unknown models.
    """
    table = pricing_table or DEFAULT_PRICING

    # Try exact match first, then prefix matching
    pricing = table.get(model)
    if pricing is None:
        for key, p in table.items():
            if model.startswith(key):
                pricing = p
                break

    if pricing is None:
        return 0.0

    input_cost = (prompt_tokens / 1000) * pricing.input_cost_per_1k
    output_cost = (completion_tokens / 1000) * pricing.output_cost_per_1k
    return input_cost + output_cost
