"""Model pricing table for cost calculation."""

from __future__ import annotations

from pydantic import BaseModel


class ModelPricing(BaseModel):
    """Pricing for a specific model (cost per 1K tokens)."""

    input_cost_per_1k: float
    output_cost_per_1k: float


# Default pricing table (USD per 1K tokens) — updated March 2026
# TODO: load from a yaml config instead of hardcoding
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # OpenAI — GPT-5.4 (March 2026)
    "gpt-5.4": ModelPricing(input_cost_per_1k=0.00250, output_cost_per_1k=0.01500),
    "gpt-5.4-mini": ModelPricing(input_cost_per_1k=0.00075, output_cost_per_1k=0.00450),
    "gpt-5.4-nano": ModelPricing(input_cost_per_1k=0.00020, output_cost_per_1k=0.00125),
    # OpenAI — GPT-4.1 (April 2025)
    "gpt-4.1": ModelPricing(input_cost_per_1k=0.00200, output_cost_per_1k=0.00800),
    # OpenAI — GPT-4o family
    "gpt-4o": ModelPricing(input_cost_per_1k=0.00250, output_cost_per_1k=0.01000),
    "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.00060),
    # OpenAI — reasoning models
    "o3": ModelPricing(input_cost_per_1k=0.00200, output_cost_per_1k=0.00800),
    "o4-mini": ModelPricing(input_cost_per_1k=0.00110, output_cost_per_1k=0.00440),
    # Anthropic — Claude 4.6 (Feb 2026)
    "claude-opus-4-6": ModelPricing(input_cost_per_1k=0.00500, output_cost_per_1k=0.02500),
    "claude-sonnet-4-6": ModelPricing(input_cost_per_1k=0.00300, output_cost_per_1k=0.01500),
    # Anthropic — Claude 4.5 (Sept-Nov 2025)
    "claude-opus-4-5-20251101": ModelPricing(input_cost_per_1k=0.00500, output_cost_per_1k=0.02500),
    "claude-sonnet-4-5-20250929": ModelPricing(
        input_cost_per_1k=0.00300, output_cost_per_1k=0.01500
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_cost_per_1k=0.00100, output_cost_per_1k=0.00500
    ),
    # Anthropic — Claude 4.1 / 4 (legacy)
    "claude-opus-4-1-20250805": ModelPricing(input_cost_per_1k=0.01500, output_cost_per_1k=0.07500),
    "claude-opus-4-20250514": ModelPricing(input_cost_per_1k=0.01500, output_cost_per_1k=0.07500),
    "claude-sonnet-4-20250514": ModelPricing(input_cost_per_1k=0.00300, output_cost_per_1k=0.01500),
    # Google — Gemini 3 (2026)
    "gemini-3.1-pro": ModelPricing(input_cost_per_1k=0.00200, output_cost_per_1k=0.01200),
    "gemini-3-flash": ModelPricing(input_cost_per_1k=0.00050, output_cost_per_1k=0.00300),
    # Google — Gemini 2.5 (mid 2025)
    "gemini-2.5-pro": ModelPricing(input_cost_per_1k=0.00125, output_cost_per_1k=0.01000),
    "gemini-2.5-flash": ModelPricing(input_cost_per_1k=0.00030, output_cost_per_1k=0.00250),
    # Google — Gemini 2.0 (deprecated, sunset June 2026)
    "gemini-2.0-flash": ModelPricing(input_cost_per_1k=0.00010, output_cost_per_1k=0.00040),
    # Ollama (local — free)
    "qwen3": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "llama3.3": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "llama3.1": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "phi4": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "deepseek-r1": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "mistral": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
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
