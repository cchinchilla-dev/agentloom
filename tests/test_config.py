"""Tests for config module."""

from __future__ import annotations

import tempfile

from agentloom.config import AgentLoomConfig, ProviderConfig, load_config


class TestAgentLoomConfig:
    def test_defaults(self) -> None:
        cfg = AgentLoomConfig()
        assert cfg.default_provider == "openai"
        assert cfg.log_level == "INFO"
        assert cfg.budget_limit_usd is None
        assert cfg.max_concurrent_steps == 10

    def test_custom_values(self) -> None:
        cfg = AgentLoomConfig(budget_limit_usd=5.0, log_format="text")
        assert cfg.budget_limit_usd == 5.0
        assert cfg.log_format == "text"


class TestProviderConfig:
    def test_defaults(self) -> None:
        pc = ProviderConfig(name="test")
        assert pc.api_key == ""
        assert pc.priority == 0
        assert pc.timeout == 30.0

    def test_custom_config(self) -> None:
        pc = ProviderConfig(name="openai", api_key="sk-xxx", priority=1)
        assert pc.name == "openai"
        assert pc.api_key == "sk-xxx"


class TestLoadConfig:
    def test_no_path_returns_defaults(self) -> None:
        cfg = load_config(None)
        assert isinstance(cfg, AgentLoomConfig)
        assert cfg.default_provider == "openai"

    def test_load_from_yaml(self) -> None:
        yaml_content = """\
log_level: DEBUG
default_provider: ollama
budget_limit_usd: 1.5
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        assert cfg.log_level == "DEBUG"
        assert cfg.default_provider == "ollama"
        assert cfg.budget_limit_usd == 1.5

    def test_load_empty_yaml_returns_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            cfg = load_config(f.name)
        assert isinstance(cfg, AgentLoomConfig)
