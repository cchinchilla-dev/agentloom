"""Pydantic models for workflow and step definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentloom.resilience.retry import DEFAULT_RETRYABLE_STATUS_CODES


class StepType(StrEnum):
    """Supported step types."""

    LLM_CALL = "llm_call"
    TOOL = "tool"
    ROUTER = "router"
    SUBWORKFLOW = "subworkflow"
    APPROVAL_GATE = "approval_gate"


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
    """Retry configuration for a step.

    ``retryable_status_codes`` controls whether a provider exception
    (anything exposing a ``status_code`` attribute, including
    ``ProviderError`` / ``RateLimitError`` / ``httpx.HTTPStatusError``)
    triggers a retry. Exceptions without a status code are retried by
    default — they're typically transient network errors. A 4xx client
    error not in this list (400/401/403/404) is **not** retried, which
    avoids burning the retry budget on permanent failures.
    """

    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    jitter: bool = True
    retryable_status_codes: list[int] = Field(
        default_factory=lambda: list(DEFAULT_RETRYABLE_STATUS_CODES)
    )


class Condition(BaseModel):
    """A routing condition: expression + target step."""

    expression: str
    target: str


class WebhookConfig(BaseModel):
    """Webhook notification config for approval gates."""

    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: str | None = None
    timeout: float = 30.0


class ThinkingConfig(BaseModel):
    """Reasoning / thinking configuration for a step.

    Activates provider-side reasoning from YAML so a workflow author doesn't
    have to drop into Python kwargs. The same config is translated per
    provider:

    - **OpenAI o-series** — reasoning is implicit in the model name; this
      config is currently ignored (kept here so the YAML stays uniform).
    - **Anthropic** — sends ``thinking: {type: "enabled", budget_tokens}``.
    - **Google Gemini 2.5+** — sends ``generationConfig.thinkingConfig``
      with ``thinkingBudget`` (from ``budget_tokens``), ``thinkingLevel``
      (from ``level``), ``includeThoughts`` (from ``capture_reasoning``).
    - **Ollama 0.9+** — sends top-level ``think: <level>`` if ``level`` is
      set, else ``think: true``.

    ``capture_reasoning`` controls whether the chain-of-thought trace is
    exposed via ``ProviderResponse.reasoning_content``. Honoured by all
    providers that surface a trace: Anthropic drops ``type="thinking"``
    blocks when set to ``False``; Gemini omits ``includeThoughts`` from
    the request so the server never sends thought summaries; Ollama
    drops ``message.thinking`` and inline ``<think>...</think>`` tags
    (the visible answer is still cleaned up). OpenAI keeps the trace
    server-side regardless, so the field has no effect there.
    """

    enabled: bool = False
    budget_tokens: int | None = None
    level: Literal["low", "medium", "high"] | None = None
    capture_reasoning: bool = True


class ToolDefinition(BaseModel):
    """LLM-callable tool declaration on an ``llm_call`` step (#116).

    ``parameters`` is a JSON Schema object describing the function's
    arguments. The provider adapters translate this declaration into the
    wire format their API expects (OpenAI/Ollama use it as-is, Anthropic
    nests it under ``input_schema``, Google nests it under
    ``function_declarations``). Tool dispatch resolves ``name`` against
    the workflow's ``tool_registry`` so a built-in or user-registered
    tool runs the call.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


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

    # Approval gate fields (timeout enforced by callback server — #42)
    timeout_seconds: int | None = None
    on_timeout: Literal["approve", "reject"] | None = None
    notify: WebhookConfig | None = None

    # Multimodal attachments (images, etc.)
    attachments: list[Attachment] = Field(default_factory=list)

    # Streaming (None = inherit from workflow config)
    stream: bool | None = None

    # Output mapping
    output: str | None = None

    # Per-step config
    timeout: float | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Reasoning / extended thinking
    thinking: ThinkingConfig | None = None

    # Tool calling (#116) — LLM picks tools at runtime
    tools: list[ToolDefinition] = Field(default_factory=list)
    # ``tool_choice``: "auto" | "required" | "none" | {"name": "..."}.
    # Auto lets the model decide; required forces a tool call; none
    # disables tools for this turn (useful for tool-augmented chats that
    # want a final summary without further calls).
    tool_choice: Any = "auto"
    max_tool_iterations: int = 5


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
    allowed_schemes: list[str] = Field(default_factory=lambda: ["http", "https"])
    max_write_bytes: int | None = None
    danger_opt_in: list[str] = Field(default_factory=list)


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

    # MockProvider configuration (used only when provider == "mock")
    responses_file: str | None = None
    latency_model: Literal["constant", "normal", "replay"] = "constant"
    latency_ms: float = 0.0

    # Observability
    capture_prompts: bool = (
        False  # When true, llm_call spans emit a span event with the rendered prompt
    )


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
