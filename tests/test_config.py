"""Tests for config module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agentloom.config import (
    AgentLoomConfig,
    ProviderConfig,
    load_config,
    resolve_config_path,
)
from agentloom.exceptions import ConfigError


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

    @pytest.mark.usefixtures("latin1_locale")
    def test_a_config_file_is_read_as_utf8_whatever_the_locale(self, tmp_path: Path) -> None:
        path = tmp_path / "agentloom.yaml"
        path.write_text("# configuración — local\nlog_level: DEBUG\n", encoding="utf-8")

        assert load_config(str(path)).log_level == "DEBUG"

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
        # A parent shell with ``AGENTLOOM_OLLAMA_FALLBACK`` set would mask the last assertion.
        monkeypatch.delenv("AGENTLOOM_OLLAMA_FALLBACK", raising=False)
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "openai" in names
        # The Ollama fallback is opt-in: absent without the env var.
        assert "ollama" not in names

    def test_discovers_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "anthropic" in names

    def test_ollama_not_registered_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ollama is no default fallback: users without it would see it in every error chain."""
        monkeypatch.delenv("AGENTLOOM_OLLAMA_FALLBACK", raising=False)
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" not in names

    def test_ollama_registered_when_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AGENTLOOM_OLLAMA_FALLBACK=1`` registers Ollama as a catch-all fallback."""
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
        """The flag is coerced as a boolean, so ``AGENTLOOM_OLLAMA_FALLBACK=0`` does not opt in."""
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", falsey)
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" not in names

    def test_ollama_not_registered_for_unrecognised_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognized flag value fails closed: discovery goes on without Ollama."""
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", "maybe")
        cfg = load_config()
        names = [p.name for p in cfg.providers]
        assert "ollama" not in names

    def test_ollama_registered_when_explicit_in_yaml(self) -> None:
        """Listing ``ollama`` under ``providers:`` is an explicit opt-in: no env var needed."""
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
        """Ollama as the primary provider registers without the flag; else no provider serves it."""
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

    def test_yaml_providers_skip_discovery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When providers are in the config file, auto-discovery is skipped."""
        monkeypatch.setenv("OPENAI_API_KEY", "from-env-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env-anthropic")
        path = tmp_path / "agentloom.yaml"
        path.write_text("providers:\n  - name: anthropic\n    models: [my-model]\n")
        cfg = load_config(path)
        assert [p.name for p in cfg.providers] == ["anthropic"]
        assert cfg.providers[0].models == ["my-model"]
        # The key comes from the provider's usual variable, never from the file
        assert cfg.providers[0].api_key == "from-env-anthropic"


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agentloom.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestConfigFile:
    """The ``providers:`` block of ``agentloom.yaml``: what it takes and what it refuses."""

    def test_base_url_from_the_file_then_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        path = _config(
            tmp_path,
            "providers:\n"
            "  - {name: openai, base_url: 'https://proxy.example/v1'}\n"
            "  - {name: ollama}\n"
            "  - {name: google}\n",
        )
        urls = {p.name: p.base_url for p in load_config(path).providers}
        assert urls == {
            "openai": "https://proxy.example/v1",
            "ollama": "http://ollama.internal:11434",
            "google": "",
        }

    def test_listed_ollama_needs_no_opt_in_and_unlisted_stays_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", "1")
        listed = load_config(_config(tmp_path, "providers: [{name: ollama, priority: 5}]\n"))
        assert [(p.name, p.priority, p.api_key) for p in listed.providers] == [("ollama", 5, "")]
        unlisted = load_config(_config(tmp_path, "providers: [{name: openai}]\n"))
        assert [p.name for p in unlisted.providers] == ["openai"]

    @pytest.mark.parametrize(
        ("name", "hint"),
        [("anthropic", "set ANTHROPIC_API_KEY instead"), ("ollama", "Ollama takes none")],
    )
    def test_api_key_in_the_file_is_refused(self, tmp_path: Path, name: str, hint: str) -> None:
        path = _config(tmp_path, f"providers:\n  - {{name: {name}, api_key: YOUR_KEY_HERE}}\n")
        with pytest.raises(ConfigError, match=hint):
            load_config(path)

    @pytest.mark.parametrize("entry", ["{name: opneai}", "{models: [x]}", "just-a-string"])
    def test_unknown_provider_is_refused(self, tmp_path: Path, entry: str) -> None:
        path = _config(tmp_path, f"providers:\n  - {entry}\n")
        with pytest.raises(ConfigError, match="not one of anthropic, google, ollama, openai"):
            load_config(path)

    @pytest.mark.parametrize(
        ("text", "field"),
        [
            ("providers: [{name: openai, modles: [x]}]\n", "providers.0.modles"),
            ("provders: []\n", "provders"),
        ],
    )
    def test_unknown_key_is_refused(self, tmp_path: Path, text: str, field: str) -> None:
        with pytest.raises(ConfigError, match=field):
            load_config(_config(tmp_path, text))

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("providers: [{name: openai\n", "while parsing"),
            ("- a list\n", "must hold a mapping"),
            ("providers: {name: openai}\n", "providers must be a list"),
        ],
    )
    def test_unreadable_file_is_refused_in_one_line(
        self, tmp_path: Path, text: str, message: str
    ) -> None:
        with pytest.raises(ConfigError, match=message) as exc_info:
            load_config(_config(tmp_path, text))
        assert "\n" not in str(exc_info.value)


class TestResolveConfigPath:
    """``--config`` wins, then ``AGENTLOOM_CONFIG``, then ``./agentloom.yaml``."""

    @pytest.fixture(autouse=True)
    def _in_tmp(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENTLOOM_CONFIG", raising=False)

    def test_nothing_to_read(self) -> None:
        assert resolve_config_path() is None

    def test_agentloom_yaml_in_the_working_directory(self, tmp_path: Path) -> None:
        (tmp_path / "agentloom.yaml").write_text("providers: []\n")
        assert resolve_config_path() == (tmp_path / "agentloom.yaml").resolve()

    def test_env_var_wins_over_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "agentloom.yaml").write_text("providers: []\n")
        other = tmp_path / "other.yaml"
        other.write_text("providers: []\n")
        monkeypatch.setenv("AGENTLOOM_CONFIG", str(other))
        assert resolve_config_path() == other.resolve()

    def test_explicit_path_wins_over_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        flag = tmp_path / "flag.yaml"
        flag.write_text("providers: []\n")
        monkeypatch.setenv("AGENTLOOM_CONFIG", str(tmp_path / "env.yaml"))
        assert resolve_config_path(flag) == flag.resolve()

    def test_empty_env_var_counts_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTLOOM_CONFIG", "")
        assert resolve_config_path() is None

    @pytest.mark.parametrize("via_env", [True, False])
    def test_missing_file_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, via_env: bool
    ) -> None:
        missing = tmp_path / "gone.yaml"
        if via_env:
            monkeypatch.setenv("AGENTLOOM_CONFIG", str(missing))
        with pytest.raises(ConfigError, match="not found"):
            resolve_config_path(None if via_env else missing)
