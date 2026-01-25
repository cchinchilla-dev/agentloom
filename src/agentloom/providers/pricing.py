"""Model pricing table for cost calculation."""

from __future__ import annotations

from pydantic import BaseModel


class ModelPricing(BaseModel):
    """Pricing for a specific model (cost per 1K tokens)."""

    input_cost_per_1k: float
    output_cost_per_1k: float


# Default pricing table (USD per 1K tokens) — updated as of 2025
# TODO: load from a yaml config instead of hardcoding
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(input_cost_per_1k=0.0025, output_cost_per_1k=0.01),
    "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.03),
    "gpt-4": ModelPricing(input_cost_per_1k=0.03, output_cost_per_1k=0.06),
    "gpt-3.5-turbo": ModelPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    "o1": ModelPricing(input_cost_per_1k=0.015, output_cost_per_1k=0.06),
    "o1-mini": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.012),
    # Anthropic
    "claude-opus-4-20250514": ModelPricing(input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    "claude-sonnet-4-20250514": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-3-5-sonnet-20241022": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-3-5-haiku-20241022": ModelPricing(input_cost_per_1k=0.0008, output_cost_per_1k=0.004),
    "claude-3-haiku-20240307": ModelPricing(input_cost_per_1k=0.00025, output_cost_per_1k=0.00125),
    # Google
    "gemini-2.0-flash": ModelPricing(input_cost_per_1k=0.0001, output_cost_per_1k=0.0004),
    "gemini-1.5-flash": ModelPricing(input_cost_per_1k=0.000075, output_cost_per_1k=0.0003),
    "gemini-1.5-pro": ModelPricing(input_cost_per_1k=0.00125, output_cost_per_1k=0.005),
    # Ollama (local — free)
    "llama3": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "llama3.1": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "mistral": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "phi3": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "qwen2": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
}


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
            if model.startswith(key) or key.startswith(model):
                pricing = p
                break

    if pricing is None:
        return 0.0

    input_cost = (prompt_tokens / 1000) * pricing.input_cost_per_1k
    output_cost = (completion_tokens / 1000) * pricing.output_cost_per_1k
    return input_cost + output_cost
