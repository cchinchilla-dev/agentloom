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


class TestEnvVarOverrides:
    def test_log_level_override(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("AGENTLOOM_LOG_LEVEL", "DEBUG")
        try:
            cfg = load_config()
            assert cfg.log_level == "DEBUG"
        finally:
            mp.undo()

    def test_default_provider_override(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("AGENTLOOM_DEFAULT_PROVIDER", "anthropic")
        try:
            cfg = load_config()
            assert cfg.default_provider == "anthropic"
        finally:
            mp.undo()

    def test_budget_limit_override(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("AGENTLOOM_BUDGET_LIMIT", "2.5")
        try:
            cfg = load_config()
            assert cfg.budget_limit_usd == 2.5
        finally:
            mp.undo()

    def test_bool_coercion_true(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("AGENTLOOM_CHECKPOINT", "true")
        try:
            cfg = load_config()
            assert cfg.checkpoint_enabled is True
        finally:
            mp.undo()

    def test_bool_coercion_false(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("AGENTLOOM_CHECKPOINT", "no")
        try:
            cfg = load_config()
            assert cfg.checkpoint_enabled is False
        finally:
            mp.undo()

    def test_env_overrides_yaml(self, monkeypatch: object) -> None:
        """Env vars take precedence over YAML file values."""
        import pytest

        yaml_content = "log_level: WARNING\n"
        mp = pytest.MonkeyPatch()
        mp.setenv("AGENTLOOM_LOG_LEVEL", "ERROR")
        try:
            with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
                f.write(yaml_content)
                f.flush()
                cfg = load_config(f.name)
            assert cfg.log_level == "ERROR"
        finally:
            mp.undo()

    def test_max_concurrent_steps_override(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("AGENTLOOM_MAX_CONCURRENT_STEPS", "20")
        try:
            cfg = load_config()
            assert cfg.max_concurrent_steps == 20
        finally:
            mp.undo()


class TestProviderDiscovery:
    def test_discovers_openai(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("OPENAI_API_KEY", "sk-test")
        try:
            cfg = load_config()
            names = [p.name for p in cfg.providers]
            assert "openai" in names
            assert "ollama" in names  # always present
        finally:
            mp.undo()

    def test_discovers_anthropic(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        try:
            cfg = load_config()
            names = [p.name for p in cfg.providers]
            assert "anthropic" in names
        finally:
            mp.undo()

    def test_ollama_always_present(self) -> None:
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" in names

    def test_ollama_is_fallback(self) -> None:
        cfg = load_config()
        ollama = next(p for p in cfg.providers if p.name == "ollama")
        assert ollama.is_fallback is True

    def test_default_provider_gets_priority_zero(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("OPENAI_API_KEY", "sk-test")
        mp.setenv("AGENTLOOM_DEFAULT_PROVIDER", "openai")
        try:
            cfg = load_config()
            openai_cfg = next(p for p in cfg.providers if p.name == "openai")
            assert openai_cfg.priority == 0
        finally:
            mp.undo()

    def test_non_default_gets_higher_priority(self, monkeypatch: object) -> None:
        import pytest

        mp = pytest.MonkeyPatch()
        mp.setenv("OPENAI_API_KEY", "sk-test")
        mp.setenv("AGENTLOOM_DEFAULT_PROVIDER", "anthropic")
        try:
            cfg = load_config()
            openai_cfg = next(p for p in cfg.providers if p.name == "openai")
            assert openai_cfg.priority > 0
        finally:
            mp.undo()

    def test_yaml_providers_skip_discovery(self) -> None:
        """When providers are in the config file, auto-discovery is skipped."""
        yaml_content = """\
providers:
  - name: custom
    api_key: sk-custom
    models: ["my-model"]
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        assert len(cfg.providers) == 1
        assert cfg.providers[0].name == "custom"
