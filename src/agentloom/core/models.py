"""Pydantic models for workflow and step definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StepType(StrEnum):
    """Supported step types."""

    LLM_CALL = "llm_call"
    TOOL = "tool"
    ROUTER = "router"
    SUBWORKFLOW = "subworkflow"


class Attachment(BaseModel):
    """Multi-modal attachment for an LLM call step.

    ``source`` may be a URL, a local file path, or a raw base64 string.
    Template variables (e.g. ``{state.image_url}``) are resolved at runtime.

    Supported types:

    * ``image`` — JPEG, PNG, GIF, WebP (all providers)
    * ``pdf`` — PDF documents (Anthropic, Google)
    * ``audio`` — WAV, MP3, OGG, FLAC (OpenAI, Google)
    """

    type: Literal["image", "pdf", "audio"] = "image"
    source: str
    media_type: str | None = None
    fetch: Literal["local", "provider"] = "local"


class RetryConfig(BaseModel):
    """Retry configuration for a step."""

    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0


class Condition(BaseModel):
    """A routing condition: expression + target step."""

    expression: str
    target: str


class StepDefinition(BaseModel):
    """Definition of a single workflow step."""

    id: str
    type: StepType
    depends_on: list[str] = Field(default_factory=list)

    # LLM call fields
    model: str | None = None
    system_prompt: str | None = None
    prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    # Tool fields
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)

    # Router fields
    conditions: list[Condition] = Field(default_factory=list)
    default: str | None = None

    # Subworkflow fields
    workflow_path: str | None = None
    workflow_inline: dict[str, Any] | None = None

    # Multimodal attachments (images, etc.)
    attachments: list[Attachment] = Field(default_factory=list)

    # Streaming (None = inherit from workflow config)
    stream: bool | None = None

    # Output mapping
    output: str | None = None

    # Per-step config
    timeout: float | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)


class SandboxConfig(BaseModel):
    """Sandbox configuration for built-in tools.

    When enabled, shell commands are validated against an allowlist
    and file operations are restricted to allowed paths.
    """

    enabled: bool = False
    allowed_commands: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)
    readable_paths: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)
    allow_network: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    max_write_bytes: int | None = None


class WorkflowConfig(BaseModel):
    """Workflow-level configuration."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_retries: int = 3
    budget_usd: float | None = None
    timeout: float | None = None
    max_concurrent_steps: int = 10
    stream: bool = False
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


class WorkflowDefinition(BaseModel):
    """Complete workflow definition — the top-level schema for YAML files."""

    name: str
    version: str = "1.0"
    description: str = ""
    config: WorkflowConfig = Field(default_factory=WorkflowConfig)
    state: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepDefinition]

    def get_step(self, step_id: str) -> StepDefinition | None:
        """Get a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def step_ids(self) -> list[str]:
        """Return all step IDs in definition order."""
        return [s.id for s in self.steps]
