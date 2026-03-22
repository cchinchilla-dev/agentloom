"""Tests for pricing calculation."""

from __future__ import annotations

from agentloom.providers.pricing import DEFAULT_PRICING, ModelPricing, calculate_cost


class TestCalculateCost:
    def test_exact_model_match(self) -> None:
        cost = calculate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0

    def test_prefix_match(self) -> None:
        cost = calculate_cost("gpt-4o-mini-2024-07-18", prompt_tokens=1000, completion_tokens=500)
        # Should match "gpt-4o-mini" via prefix
        assert cost > 0

    def test_unknown_model_returns_zero(self) -> None:
        cost = calculate_cost("totally-unknown-model", prompt_tokens=1000, completion_tokens=500)
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        cost = calculate_cost("gpt-4o-mini", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_ollama_is_free(self) -> None:
        cost = calculate_cost("phi4", prompt_tokens=10000, completion_tokens=5000)
        assert cost == 0.0

    def test_custom_pricing_table(self) -> None:
        custom = {"my-model": ModelPricing(input_cost_per_1k=1.0, output_cost_per_1k=2.0)}
        cost = calculate_cost(
            "my-model",
            prompt_tokens=1000,
            completion_tokens=1000,
            pricing_table=custom,
        )
        assert cost == 3.0  # 1.0 + 2.0

    def test_no_bidirectional_match(self) -> None:
        """Short model name should NOT match longer keys."""
        cost = calculate_cost("gpt", prompt_tokens=1000, completion_tokens=500)
        # "gpt" does NOT start with any key (keys are "gpt-4o-mini", "gpt-4o", etc.)
        # And we removed key.startswith(model), so this should return 0
        assert cost == 0.0


class TestDefaultPricingTable:
    def test_has_openai_models(self) -> None:
        assert "gpt-4o-mini" in DEFAULT_PRICING
        assert "gpt-4o" in DEFAULT_PRICING

    def test_has_anthropic_models(self) -> None:
        assert "claude-haiku-4-5-20251001" in DEFAULT_PRICING
        assert "claude-sonnet-4-6" in DEFAULT_PRICING

    def test_has_google_models(self) -> None:
        assert "gemini-2.5-flash" in DEFAULT_PRICING

    def test_has_ollama_models(self) -> None:
        assert "phi4" in DEFAULT_PRICING
        assert DEFAULT_PRICING["phi4"].input_cost_per_1k == 0.0
