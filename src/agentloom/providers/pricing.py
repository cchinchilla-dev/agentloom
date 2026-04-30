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
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        msg = f"Pricing YAML '{path}' must be a mapping of model entries, got {type(raw).__name__}"
        raise ValueError(msg)
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

    # Try exact match first, then prefix matching. Check longer keys before
    # shorter ones so ``gpt-4o-mini`` is never claimed by a plain ``gpt-4``
    # entry when both are present.
    pricing = table.get(model)
    if pricing is None:
        for key in sorted(table, key=len, reverse=True):
            if model.startswith(key):
                pricing = table[key]
                break

    if pricing is None:
        return 0.0

    input_cost = (prompt_tokens / 1000) * pricing.input_cost_per_1k
    output_cost = (completion_tokens / 1000) * pricing.output_cost_per_1k
    return input_cost + output_cost
