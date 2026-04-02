"""Tests for pricing calculation."""

from __future__ import annotations

import tempfile

import pytest

from agentloom.providers.pricing import DEFAULT_PRICING, ModelPricing, calculate_cost, load_pricing


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


class TestLoadPricing:
    def test_bundled_yaml_loads(self) -> None:
        table = load_pricing()
        assert len(table) > 0
        assert "gpt-4o-mini" in table

    def test_custom_yaml_path(self) -> None:
        yaml_content = """\
my-model:
  input: 1.0
  output: 2.0
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            table = load_pricing(custom_path=f.name)
        assert "my-model" in table
        assert table["my-model"].input_cost_per_1k == 1.0
        assert table["my-model"].output_cost_per_1k == 2.0

    def test_env_var_pricing_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_content = """\
env-model:
  input: 0.5
  output: 0.75
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            monkeypatch.setenv("AGENTLOOM_PRICING_FILE", f.name)
            table = load_pricing()
        assert "env-model" in table

    def test_custom_path_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit custom_path takes precedence over env var."""
        env_yaml = "env-model:\n  input: 0.1\n  output: 0.2\n"
        custom_yaml = "custom-model:\n  input: 0.5\n  output: 0.6\n"

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as env_f:
            env_f.write(env_yaml)
            env_f.flush()
            monkeypatch.setenv("AGENTLOOM_PRICING_FILE", env_f.name)

            with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as custom_f:
                custom_f.write(custom_yaml)
                custom_f.flush()
                table = load_pricing(custom_path=custom_f.name)

        assert "custom-model" in table
        assert "env-model" not in table

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pricing(custom_path="/nonexistent/pricing.yaml")
