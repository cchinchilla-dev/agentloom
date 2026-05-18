"""Tests for config module."""

from __future__ import annotations

import tempfile

import pytest

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

    def test_default_provider_override_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = load_config(default_provider_override="anthropic")
        assert cfg.default_provider == "anthropic"


class TestEnvVarOverrides:
    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_LOG_LEVEL", "DEBUG")
        cfg = load_config()
        assert cfg.log_level == "DEBUG"

    def test_default_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_DEFAULT_PROVIDER", "anthropic")
        cfg = load_config()
        assert cfg.default_provider == "anthropic"

    def test_budget_limit_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_BUDGET_LIMIT", "2.5")
        cfg = load_config()
        assert cfg.budget_limit_usd == 2.5

    def test_bool_coercion_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_CHECKPOINT", "true")
        cfg = load_config()
        assert cfg.checkpoint_enabled is True

    def test_bool_coercion_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_CHECKPOINT", "no")
        cfg = load_config()
        assert cfg.checkpoint_enabled is False

    def test_bool_coercion_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_CHECKPOINT", "treu")
        with pytest.raises(ValueError, match="Invalid boolean value"):
            load_config()

    def test_env_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars take precedence over YAML file values."""
        yaml_content = "log_level: WARNING\n"
        monkeypatch.setenv("AGENTLOOM_LOG_LEVEL", "ERROR")
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        assert cfg.log_level == "ERROR"

    def test_max_concurrent_steps_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_MAX_CONCURRENT_STEPS", "20")
        cfg = load_config()
        assert cfg.max_concurrent_steps == 20


class TestProviderDiscovery:
    def test_discovers_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Ensure the opt-in is OFF so the assertion below is meaningful —
        # parent shells with ``AGENTLOOM_OLLAMA_FALLBACK`` set would
        # otherwise mask the regression net.
        monkeypatch.delenv("AGENTLOOM_OLLAMA_FALLBACK", raising=False)
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "openai" in names
        # Ollama is opt-in as of 0.5.0 — without the env var the
        # secondary fallback path must NOT be registered.
        assert "ollama" not in names

    def test_discovers_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "anthropic" in names

    def test_ollama_not_registered_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pre-0.5.0 Ollama was auto-registered as a global fallback, so
        every primary-provider failure produced an error chain that
        mentioned Ollama even for users who didn't run it. Registration
        is now gated behind ``AGENTLOOM_OLLAMA_FALLBACK``."""
        monkeypatch.delenv("AGENTLOOM_OLLAMA_FALLBACK", raising=False)
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" not in names

    def test_ollama_registered_when_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Opt-in path: ``AGENTLOOM_OLLAMA_FALLBACK=1`` re-enables the
        pre-0.5.0 behaviour for users who DO run Ollama and want it as a
        catch-all fallback."""
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", "1")
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" in names
        ollama = next(p for p in cfg.providers if p.name == "ollama")
        assert ollama.is_fallback is True

    @pytest.mark.parametrize("falsey", ["0", "false", "no", "off", ""])
    def test_ollama_not_registered_for_falsey_flag(
        self, monkeypatch: pytest.MonkeyPatch, falsey: str
    ) -> None:
        """The flag honours boolean coercion: ``0`` / ``false`` / ``no`` /
        ``off`` / empty do NOT opt in. A bare non-empty-string check
        would treat ``AGENTLOOM_OLLAMA_FALLBACK=0`` as enabled and
        surprise a user who explicitly disabled the fallback."""
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", falsey)
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" not in names

    def test_ollama_not_registered_for_unrecognised_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognised flag value fails closed — discovery does not
        abort, and Ollama stays unregistered."""
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", "maybe")
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" not in names

    def test_ollama_registered_when_explicit_in_yaml(self) -> None:
        """Explicit YAML config bypasses auto-discovery — users who list
        ``ollama`` under ``providers:`` keep the path even without the
        env var, since that's the unambiguous opt-in."""
        yaml_content = """\
providers:
  - name: ollama
    base_url: http://localhost:11434
    is_fallback: true
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        names = [p.name for p in cfg.providers]
        assert "ollama" in names

    def test_ollama_registered_when_default_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A workflow that names ``ollama`` as the primary provider
        registers it even without the opt-in flag — otherwise the user
        would see "no provider for ollama" with no clear path to fix it.
        Pre-0.5.0 path preserved for users who run Ollama primary."""
        monkeypatch.delenv("AGENTLOOM_OLLAMA_FALLBACK", raising=False)
        cfg = load_config(default_provider_override="ollama")
        names = [p.name for p in cfg.providers]
        assert "ollama" in names

    def test_default_provider_gets_priority_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AGENTLOOM_DEFAULT_PROVIDER", "openai")
        cfg = load_config()
        openai_cfg = next(p for p in cfg.providers if p.name == "openai")
        assert openai_cfg.priority == 0

    def test_non_default_gets_higher_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AGENTLOOM_DEFAULT_PROVIDER", "anthropic")
        cfg = load_config()
        openai_cfg = next(p for p in cfg.providers if p.name == "openai")
        assert openai_cfg.priority > 0

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
