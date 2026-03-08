"""Global configuration for AgentLoom."""

from __future__ import annotations

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


# TODO: support AGENTFORGE_ env var prefix (pydantic-settings would be nice here)
def load_config(config_path: str | None = None) -> AgentLoomConfig:
    """Load configuration from a YAML file or return defaults.

    Args:
        config_path: Path to agentloom.yaml config file.

    Returns:
        Validated AgentLoomConfig instance.
    """
    if config_path is None:
        return AgentLoomConfig()

    from pathlib import Path

    import yaml  # noqa: TCH002

    raw = yaml.safe_load(Path(config_path).read_text())
    if raw is None:
        return AgentLoomConfig()
    return AgentLoomConfig.model_validate(raw)
