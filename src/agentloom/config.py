"""Global configuration for AgentLoom."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str
    api_key: str = ""
    base_url: str = ""
    models: list[str] = Field(default_factory=list)
    priority: int = 0
    is_fallback: bool = False
    max_retries: int = 3
    timeout: float = 30.0


_PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "o4-mini"],
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-haiku-4-5-20251001"],
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "models": ["gemini-2.5-flash"],
    },
    "ollama": {
        "env_key": "",
        "base_url": "http://localhost:11434",
        "is_fallback": True,
    },
}


class AgentLoomConfig(BaseModel):
    """Global configuration, loaded from env vars or config file."""

    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    observability_enabled: bool = True
    default_provider: str = "openai"
    max_concurrent_steps: int = 10
    budget_limit_usd: float | None = None
    checkpoint_enabled: bool = False
    checkpoint_dir: str = ".agentloom/checkpoints"
    providers: list[ProviderConfig] = Field(default_factory=list)


_ENV_MAP: dict[str, tuple[str, type]] = {
    "LOG_LEVEL": ("log_level", str),
    "LOG_FORMAT": ("log_format", str),
    "OBSERVABILITY": ("observability_enabled", bool),
    "DEFAULT_PROVIDER": ("default_provider", str),
    "MAX_CONCURRENT_STEPS": ("max_concurrent_steps", int),
    "BUDGET_LIMIT": ("budget_limit_usd", float),
    "CHECKPOINT": ("checkpoint_enabled", bool),
    "CHECKPOINT_DIR": ("checkpoint_dir", str),
}

_ENV_PREFIX = "AGENTLOOM_"


def _coerce(value: str, target_type: type) -> object:
    """Convert an env var string to the target type."""
    if target_type is bool:
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        msg = f"Invalid boolean value for configuration: {value!r}"
        raise ValueError(msg)
    return target_type(value)


def _apply_env_overrides(data: dict[str, object]) -> dict[str, object]:
    """Apply AGENTLOOM_* env var overrides to raw config data."""
    for suffix, (field, target_type) in _ENV_MAP.items():
        env_val = os.environ.get(f"{_ENV_PREFIX}{suffix}")
        if env_val is not None:
            data[field] = _coerce(env_val, target_type)
    return data


def discover_providers(default_provider: str) -> list[ProviderConfig]:
    """Auto-discover providers from API-key env vars."""
    providers: list[ProviderConfig] = []

    for name, defaults in _PROVIDER_DEFAULTS.items():
        env_key = str(defaults.get("env_key", ""))
        is_fallback = bool(defaults.get("is_fallback", False))

        # Providers that require an API key are skipped when the key is absent.
        if env_key and not os.environ.get(env_key):
            continue

        api_key = os.environ.get(env_key, "") if env_key else ""
        default_base = str(defaults.get("base_url", ""))
        base_url = os.environ.get(f"{name.upper()}_BASE_URL", default_base)
        raw_models = defaults.get("models", [])
        models = list(raw_models) if isinstance(raw_models, list) else []

        providers.append(
            ProviderConfig(
                name=name,
                api_key=api_key,
                base_url=base_url,
                models=models,
                priority=0 if default_provider == name else (100 if is_fallback else 10),
                is_fallback=is_fallback,
            )
        )

    return providers


def load_config(
    config_path: str | None = None,
    default_provider_override: str | None = None,
) -> AgentLoomConfig:
    """Load configuration with env var overrides and provider auto-discovery.

    Resolution order (later wins):
      1. Built-in defaults
      2. YAML config file (if provided)
      3. ``AGENTLOOM_*`` environment variables
      4. ``default_provider_override`` (if given)

    Providers are auto-discovered from API-key env vars when no
    providers are defined in the config file.

    Args:
        config_path: Path to agentloom.yaml config file.
        default_provider_override: Override the default provider (e.g. from CLI flag).

    Returns:
        Validated AgentLoomConfig instance.
    """
    data: dict[str, object] = {}

    if config_path is not None:
        from pathlib import Path

        import yaml

        raw = yaml.safe_load(Path(config_path).read_text())
        if raw:
            data = raw

    data = _apply_env_overrides(data)
    config = AgentLoomConfig.model_validate(data)

    if default_provider_override is not None:
        config.default_provider = default_provider_override

    # Auto-discover providers only when none are explicitly configured.
    if not config.providers:
        config.providers = discover_providers(config.default_provider)

    return config
