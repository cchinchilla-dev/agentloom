"""Global configuration for AgentLoom."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from agentloom.exceptions import ConfigError

CONFIG_ENV = "AGENTLOOM_CONFIG"
DEFAULT_CONFIG_FILE = "agentloom.yaml"


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    model_config = ConfigDict(extra="forbid")

    name: str
    api_key: str = ""
    base_url: str = ""
    models: list[str] = Field(default_factory=list)
    priority: int = 0
    is_fallback: bool = False
    # A requested model id -> the id this provider knows it by; the gateway translates per candidate
    model_aliases: dict[str, str] = Field(default_factory=dict)


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
    # Opt-in only: an implicit fallback adds a localhost:11434 call to every primary failure.
    "ollama": {
        "env_key": "AGENTLOOM_OLLAMA_FALLBACK",
        "base_url": "http://localhost:11434",
        "is_fallback": True,
    },
}


def default_model(provider: str) -> str | None:
    """The model *provider* runs when a workflow names none; ``None`` if it has no default."""
    models = _PROVIDER_DEFAULTS.get(provider, {}).get("models")
    return str(models[0]) if isinstance(models, list) and models else None


class AgentLoomConfig(BaseModel):
    """Global configuration, loaded from env vars or config file."""

    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderConfig] = Field(default_factory=list)


_TRUTHY_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag_enabled(name: str) -> bool:
    """True when an ``AGENTLOOM_*`` boolean opt-in flag is set truthy.

    ``1/true/yes/on`` enable; an unset var, an empty string, or any falsey /
    unrecognized value does not. This never raises: a typo in an opt-in flag
    fails closed rather than aborting provider discovery.
    """
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in _TRUTHY_FLAG_VALUES


def discover_providers(default_provider: str) -> list[ProviderConfig]:
    """Auto-discover providers from API-key env vars.

    Skip rule for ``env_key``:
    * Providers whose ``env_key`` names an API key (``OPENAI_API_KEY``,
      ``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``) are skipped when the
      key is missing — they cannot make calls without credentials. Any
      non-empty value counts as the credential.
    * Ollama's ``env_key`` is the ``AGENTLOOM_OLLAMA_FALLBACK`` opt-in
      flag, not a credential. It is skipped by default, but registered
      when either (a) the flag holds a truthy value (``1/true/yes/on``;
      ``0/false/no/off`` and unrecognized values do not opt in), or
      (b) the workflow has explicitly named ``ollama`` as its primary
      provider. Without (b) a workflow that says ``provider: ollama``
      would silently end up with no registered providers and an opaque
      "no provider for ollama" error.
    """
    providers: list[ProviderConfig] = []

    for name, defaults in _PROVIDER_DEFAULTS.items():
        env_key = str(defaults.get("env_key", ""))
        is_fallback = bool(defaults.get("is_fallback", False))

        if env_key:
            if name == "ollama":
                # A flag, not a key: ``AGENTLOOM_OLLAMA_FALLBACK=0`` must not register Ollama.
                enabled = _env_flag_enabled(env_key)
            else:
                enabled = bool(os.environ.get(env_key))
            # Ollama as the workflow's primary registers without the flag, else no provider exists.
            if not enabled and not (name == "ollama" and default_provider == "ollama"):
                continue

        # Ollama's ``env_key`` is a flag, not a credential: never copy its value into ``api_key``.
        api_key = "" if name == "ollama" else (os.environ.get(env_key, "") if env_key else "")
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


def resolve_config_path(explicit: str | Path | None = None) -> Path | None:
    """Return the config file to load, as an absolute path, or ``None``.

    *explicit* (``--config``) wins, then the ``AGENTLOOM_CONFIG`` env var, then
    ``./agentloom.yaml`` when it exists.

    Raises:
        ConfigError: If *explicit* or the env var names a file that does not exist.
    """
    chosen = explicit if explicit is not None else os.environ.get(CONFIG_ENV) or None
    if chosen is not None:
        path = Path(chosen).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file {str(path)!r} not found")
        return path.resolve()
    default = Path(DEFAULT_CONFIG_FILE)
    return default.resolve() if default.is_file() else None


def _read_config_file(path: str | Path) -> dict[str, object]:
    """The file's mapping, with each provider entry checked before validation."""
    import yaml

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"config file {str(path)!r}: {' '.join(str(e).split())}") from e
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {str(path)!r} must hold a mapping")
    providers = raw.get("providers") or []
    if not isinstance(providers, list):
        raise ConfigError(f"config file {str(path)!r}: providers must be a list")
    for i, entry in enumerate(providers):
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or name not in _PROVIDER_DEFAULTS:
            known = ", ".join(sorted(_PROVIDER_DEFAULTS))
            raise ConfigError(
                f"config file {str(path)!r}: providers[{i}] names {name!r}, not one of {known}"
            )
        if isinstance(entry, dict) and "api_key" in entry:
            env_key = _PROVIDER_DEFAULTS[name].get("env_key")
            where = "Ollama takes none" if name == "ollama" else f"set {env_key} instead"
            raise ConfigError(
                f"config file {str(path)!r}: providers[{i}] sets api_key, which is read "
                f"from the environment only: {where}"
            )
    return raw


def _with_environment(provider: ProviderConfig) -> ProviderConfig:
    """A provider from the file, with its key and default base URL from the environment."""
    defaults = _PROVIDER_DEFAULTS[provider.name]
    env_key = str(defaults.get("env_key", ""))
    api_key = "" if provider.name == "ollama" else os.environ.get(env_key, "")
    base_url = provider.base_url or os.environ.get(
        f"{provider.name.upper()}_BASE_URL", str(defaults.get("base_url", ""))
    )
    return provider.model_copy(update={"api_key": api_key, "base_url": base_url})


def load_config(
    config_path: str | Path | None = None,
    default_provider_override: str | None = None,
) -> AgentLoomConfig:
    """Load the providers from the config file, or discover them from the environment.

    A ``providers:`` block in the file replaces auto-discovery; each provider
    takes its API key from its usual environment variable, never from the
    file. Without the block, providers are discovered from those variables.

    Raises:
        ConfigError: If the file cannot be read, or holds an unknown key or
            provider, or an ``api_key``.
    """
    data: dict[str, object] = {}

    if config_path is not None:
        data = _read_config_file(config_path)

    try:
        config = AgentLoomConfig.model_validate(data)
    except PydanticValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        source = f"config file {str(config_path)!r}" if config_path else "configuration"
        raise ConfigError(f"{source}: {problems}") from e

    if config.providers:
        config.providers = [_with_environment(p) for p in config.providers]
    else:
        config.providers = discover_providers(default_provider_override or "openai")

    return config
