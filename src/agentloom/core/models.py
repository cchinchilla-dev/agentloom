"""Pydantic models for workflow and step definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """LLM-callable tool declared on an ``llm_call`` step.

    ``parameters`` is a JSON Schema object; provider adapters translate it
    to each API's native shape. ``name`` resolves against the workflow's
    ``tool_registry`` for dispatch.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolChoiceByName(BaseModel):
    """Pin tool selection to a specific function: ``{"name": "..."}``."""

    name: str


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
    # Opt-in to fresh-state isolation. With ``isolated_state: false`` (the
    # default for backwards compatibility), parent state propagates DOWN
    # into the child and the child's entire final state propagates UP as
    # the parent's ``output:`` value — handy for trivial helper
    # subworkflows but leaky for anything resembling encapsulation.
    # ``isolated_state: true`` seeds the child only from the child's own
    # ``state:`` block plus the explicit ``input:`` mapping below, and
    # only the keys listed in ``return_keys`` (default: all top-level
    # keys the child wrote) surface back through ``output:``.
    isolated_state: bool = False
    input: dict[str, Any] = Field(default_factory=dict)
    return_keys: list[str] | None = None

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

    # Tool calling — LLM picks tools at runtime. Constrained union so YAML
    # typos fail fast at parse time instead of silently coercing to AUTO.
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: Literal["auto", "required", "none"] | ToolChoiceByName = "auto"
    max_tool_iterations: int = Field(default=5, ge=1)


class SandboxConfig(BaseModel):
    """Sandbox configuration for built-in tools.

    When enabled, shell commands are validated against an allowlist
    and file operations are restricted to allowed paths.

    Webhook delivery (``approval_gate.notify.url``) always passes through
    the sandbox: when ``enabled`` is true, the URL must satisfy
    ``allow_network`` / ``allowed_schemes`` / ``allowed_domains``; when
    ``enabled`` is false, a built-in deny-list still blocks loopback,
    link-local, and RFC 1918 destinations. Workflows that genuinely need to
    notify an in-cluster service can opt out via
    ``allow_internal_webhook_targets``.
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
    allow_internal_webhook_targets: bool = False


class WorkflowConfig(BaseModel):
    """Workflow-level configuration."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_retries: int = 3
    budget_usd: float | None = None
    timeout: float | None = None
    # ``ge=1`` catches both ``0`` (which makes ``anyio.CapacityLimiter(0)``
    # block every ``start_soon`` and deadlocks the workflow with no timeout)
    # and negatives (which used to surface as a cryptic
    # ``total_tokens must be >= 0`` from the fallback result construction).
    # ``le=1024`` is a sanity ceiling: workflows needing more concurrency
    # per layer can either split layers or open an issue — past this point
    # the layer-loop scheduling overhead dominates and individual provider
    # rate limits will be the real bottleneck.
    max_concurrent_steps: int = Field(default=10, ge=1, le=1024)
    stream: bool = False
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    # When a step (or a router) ends in FAILED, ``skip_downstream`` marks every
    # transitive dependent as SKIPPED so they don't fire side-effect tools,
    # webhooks, or LLM calls against partial state. ``continue`` preserves the
    # pre-0.5.0 behaviour (dependents still run) for best-effort fan-outs that
    # explicitly want to swallow failures.
    on_step_failure: Literal["skip_downstream", "continue"] = "skip_downstream"

    # When ``True``, two or more parallel-eligible steps writing the same
    # ``output:`` key abort at parse time. When ``False`` (default), the parser
    # only emits a warning — silent last-writer-wins is the pre-0.5.0
    # behaviour that several workflows already depend on, so opt-in only.
    strict_outputs: bool = False

    # MockProvider configuration (used only when provider == "mock")
    responses_file: str | None = None
    latency_model: Literal["constant", "normal", "replay"] = "constant"
    latency_ms: float = 0.0

    # Observability
    capture_prompts: bool = (
        False  # When true, llm_call spans emit a span event with the rendered prompt
    )


class StateKeyConfig(BaseModel):
    """Per-key state metadata, currently used only for redaction.

    YAML usage::

        state:
          api_key: "..."
        state_schema:
          api_key: { redact: true }

    Glob keys are supported (``"*token*"``); ``redact: true`` causes the
    value to be replaced with a stable ``<REDACTED:sha256=...>`` sentinel
    in every persisted artefact (checkpoint, run history, OTel span event,
    webhook body). The in-memory state stays plaintext so steps that
    legitimately need the secret keep working.
    """

    redact: bool = False


class WorkflowDefinition(BaseModel):
    """Complete workflow definition — the top-level schema for YAML files.

    Unknown top-level keys are refused at parse time (``extra="forbid"``).
    A common silent failure mode pre-fix was a typo like ``stat_schema:``
    instead of ``state_schema:`` — Pydantic's default would have dropped
    the unknown key, leaving the redaction policy empty and shipping
    every flagged secret to disk in plaintext.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0"
    description: str = ""
    config: WorkflowConfig = Field(default_factory=WorkflowConfig)
    state: dict[str, Any] = Field(default_factory=dict)
    state_schema: dict[str, StateKeyConfig] = Field(default_factory=dict)
    steps: list[StepDefinition]

    @model_validator(mode="after")
    def _validate_step_ids_unique(self) -> WorkflowDefinition:
        """Refuse duplicate ``id:`` values across the top-level step list.

        Until 0.4.0 the parser accepted two steps with the same ``id``
        silently; only one would surface in ``final_state.steps`` after
        execution and the other was lost without warning. A workflow author
        who renamed a step and forgot to rename a reference could ship a
        workflow where steps shadow each other — typically caught only when
        the missing step's output went unused downstream and the run came
        back with garbled state.
        """
        seen: dict[str, int] = {}
        dups: list[tuple[str, list[int]]] = []
        for i, s in enumerate(self.steps):
            if s.id in seen:
                dups.append((s.id, [seen[s.id], i]))
            else:
                seen[s.id] = i
        if dups:
            msg = "; ".join(f"id={d[0]!r} at indices {d[1]}" for d in dups)
            raise ValueError(f"Duplicate step ids: {msg}")
        return self

    def get_step(self, step_id: str) -> StepDefinition | None:
        """Get a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def step_ids(self) -> list[str]:
        """Return all step IDs in definition order."""
        return [s.id for s in self.steps]

    def redaction_patterns(self) -> list[str]:
        """Glob patterns flagged ``redact: true`` in the state schema."""
        return [key for key, cfg in self.state_schema.items() if cfg.redact]
