"""Pydantic models for workflow and step definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from agentloom.contracts.stable import Attachment as Attachment
from agentloom.contracts.stable import Message as Message
from agentloom.contracts.stable import ToolDefinition as ToolDefinition
from agentloom.resilience.retry import DEFAULT_RETRYABLE_STATUS_CODES


class StepType(StrEnum):
    """Supported step types."""

    LLM_CALL = "llm_call"
    TOOL = "tool"
    ROUTER = "router"
    SUBWORKFLOW = "subworkflow"
    APPROVAL_GATE = "approval_gate"
    EMBED = "embed"
    AGENT = "agent"


class RetryConfig(BaseModel):
    """Retry configuration for a step.

    ``retryable_status_codes`` controls whether a provider exception
    (anything exposing a ``status_code`` attribute, including
    ``ProviderError`` / ``RateLimitError`` / ``httpx.HTTPStatusError``)
    triggers a retry. Exceptions without a status code are retried by
    default — they're typically transient network errors. A 4xx client
    error not in this list (400/401/403/404) is **not** retried, which
    avoids burning the retry budget on permanent failures.

    Without ``max_retries``, a step takes its workflow's ``config.max_retries``.
    """

    max_retries: int = Field(default=3, ge=0)
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
      config is ignored (accepted so the YAML stays uniform).
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


class Conversation(BaseModel):
    """A typed multi-turn conversation stored in workflow state.

    Workflows reference a conversation by dotted state path
    (``conversation: state.chat``) on an ``llm_call`` step. The step
    loads the messages, calls the provider with the full history, and
    appends the assistant reply (plus any tool calls) back into the same
    state key.

    ``token_budget`` bounds the message list; ``trim_policy`` decides how
    to shrink it when the budget is exceeded. When ``None``, the
    conversation grows unbounded (checkpoint size is the effective cap).

    ``summary`` is populated when the ``summarize_oldest`` policy runs,
    so a resumed workflow can rebuild the "here's what happened earlier"
    context without re-summarizing.

    See :class:`agentloom.steps.llm_call.LLMCallStep` for how the
    conversation is consumed at step-execute time.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_budget: int | None = Field(default=None, ge=1)
    trim_policy: Literal["drop_oldest", "drop_pairs", "summarize_oldest"] = "drop_oldest"
    summary: str | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_message_dicts(cls, value: Any) -> Any:
        """Accept plain-dict messages from YAML/checkpoint payloads.

        Keys set to ``None`` (``content`` aside) are dropped, so a payload with
        ``"tool_calls": null`` or ``"metadata": null`` takes the field defaults.
        """
        if not isinstance(value, list):
            return value
        coerced: list[Any] = []
        for entry in value:
            if isinstance(entry, Message):
                coerced.append(entry)
                continue
            if isinstance(entry, dict):
                coerced.append({k: v for k, v in entry.items() if v is not None or k == "content"})
                continue
            coerced.append(entry)
        return coerced

    def token_count(self, model: str = "") -> int:
        """Sum of per-message token estimates.

        See :meth:`Message.token_count` for the heuristic's caveats.
        """
        return sum(m.token_count(model) for m in self.messages)


class ToolChoiceByName(BaseModel):
    """Pin tool selection to a specific function: ``{"name": "..."}``."""

    name: str


class ResponseSchema(BaseModel):
    """Structured-output contract for an ``llm_call`` step.

    Three modes: ``json_object`` (any JSON), ``json_schema`` (inline
    schema), ``pydantic`` (dotted-path model; parsed value is the
    instance). Providers with a native API (OpenAI, Google, Ollama
    0.5+) enforce server-side; Anthropic uses prefill + client-side
    validation.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["pydantic", "json_schema", "json_object"] = "json_object"
    model: str | None = None
    # ``schema_`` avoids shadowing Pydantic's ``BaseModel.schema()``; YAML uses the alias
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    name: str | None = None
    description: str | None = None
    strict: bool = True

    @model_validator(mode="after")
    def _validate_mode_specific_fields(self) -> ResponseSchema:
        """Refuse companion fields that don't match the chosen mode."""
        if self.type == "pydantic":
            if not self.model:
                raise ValueError(
                    "response_schema.type='pydantic' requires a 'model' dotted path "
                    "(e.g. 'examples.schemas.Classification')."
                )
            if self.schema_ is not None:
                raise ValueError(
                    "response_schema.type='pydantic' must not also set 'schema'; "
                    "the schema is derived from the Pydantic model."
                )
        elif self.type == "json_schema":
            if self.model is not None:
                raise ValueError(
                    "response_schema.type='json_schema' must not set 'model'; "
                    "the model field is only meaningful in pydantic mode."
                )
            if not self.schema_:
                raise ValueError(
                    "response_schema.type='json_schema' requires an inline 'schema' object."
                )
            if self.schema_.get("additionalProperties") is True:
                raise ValueError(
                    "response_schema: 'additionalProperties: true' is incompatible with "
                    "OpenAI's strict json_schema mode. Set it to false (or remove the "
                    "key — strict mode treats it as false by default)."
                )
        elif self.model is not None or self.schema_ is not None:
            raise ValueError(
                "response_schema.type='json_object' must not set 'model' or "
                "'schema'; neither is meaningful when no schema is enforced."
            )
        return self


class ResponseSchemaConfigError(ValueError):
    """Workflow-author error in a :class:`ResponseSchema` — never retried."""


# The tool through which a step with both ``tools`` and a schema delivers its reply.
OUTPUT_TOOL = "final_result"


class AgentMemory(BaseModel):
    """The conversation budget and trim policy an agent applies on each of its turns.

    Each one left out keeps the conversation's own.
    """

    model_config = ConfigDict(extra="forbid")

    token_budget: int | None = Field(default=None, ge=1)
    trim_policy: Literal["drop_oldest", "drop_pairs", "summarize_oldest"] | None = None


class LoopPolicy(BaseModel):
    """How many model calls an agent's turn may make, and what reaching the cap does."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=5, ge=1)
    on_max_iterations: Literal["error", "warn_and_return"] = "error"


class AgentConstraints(BaseModel):
    """Limits that fail an agent's turn when it breaks them."""

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: float | None = Field(default=None, gt=0)
    max_latency_ms: int | None = Field(default=None, ge=1)
    # Plain text, matched case-insensitively against the reply.
    forbidden_patterns: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)


# A function name every provider accepts (OpenAI, Anthropic and Gemini's rules together).
_FUNCTION_NAME = r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$"


class AgentTool(BaseModel):
    """Another agent offered as a tool that takes a single ``message``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_FUNCTION_NAME)
    agent: AgentDefinition
    description: str | None = None

    @field_validator("agent", mode="before")
    @classmethod
    def _load_agent(cls, value: Any) -> Any:
        from agentloom.core.agent import resolve_agent

        return resolve_agent(value)


def _tool_kind(value: Any) -> str:
    return "name" if isinstance(value, str) else "agent"


# A tool name, or another agent; tagged so a malformed entry reports one error, not one per kind.
AgentToolEntry = Annotated[
    Annotated[str, Tag("name")] | Annotated[AgentTool, Tag("agent")],
    Discriminator(_tool_kind),
]


class AgentDefinition(BaseModel):
    """A reusable agent: system prompt, tools, memory, loop policy and constraints.

    An ``agent`` step runs one turn of it as an ``llm_call`` with the agent's
    settings; :class:`agentloom.core.agent.Agent` runs one from Python.
    """

    model_config = ConfigDict(extra="forbid")

    # Also the message name of its turns, which OpenAI refuses outside these characters.
    name: str = Field(pattern=_FUNCTION_NAME)
    description: str = ""
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # Names from the workflow's tool registry, and other agents.
    tools: list[AgentToolEntry] = Field(default_factory=list)
    memory: AgentMemory | None = None
    loop_policy: LoopPolicy = Field(default_factory=LoopPolicy)
    output_schema: ResponseSchema | None = None
    constraints: AgentConstraints = Field(default_factory=AgentConstraints)

    @model_validator(mode="after")
    def _validate_tool_names(self) -> AgentDefinition:
        names = self.tool_names()
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(f"Agent {self.name!r}: duplicate tool names {duplicated}")
        if self.output_schema is not None and OUTPUT_TOOL in names:
            raise ValueError(
                f"Agent {self.name!r}: the tool name {OUTPUT_TOOL!r} is reserved for "
                f"the reply of an agent with an output_schema"
            )
        return self

    def tool_names(self) -> list[str]:
        """The names the agent's tools go by, in declaration order."""
        return [t if isinstance(t, str) else t.name for t in self.tools]


class StepDefinition(BaseModel):
    """Definition of a single workflow step.

    Unknown keys are refused at parse time (``extra="forbid"``) so a typo
    like ``workflow:`` for ``workflow_inline:`` fails at ``agentloom
    validate`` with the offending key named, instead of being silently
    dropped and surfacing a cryptic run-time error.
    """

    model_config = ConfigDict(extra="forbid")

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
    # false: the child inherits parent state and returns all of its own as ``output``;
    # true: it gets only its ``state:`` plus ``input``, and returns ``return_keys`` (default all).
    isolated_state: bool = False
    input: dict[str, Any] = Field(default_factory=dict)
    return_keys: list[str] | None = None

    # Approval gate fields; nothing enforces timeout_seconds / on_timeout yet (#210).
    timeout_seconds: int | None = None
    on_timeout: Literal["approve", "reject"] | None = None
    notify: WebhookConfig | None = None

    # Multimodal attachments (images, etc.)
    attachments: list[Attachment] = Field(default_factory=list)

    # Streaming (None = inherit from workflow config)
    stream: bool | None = None

    # Output mapping
    output: str | None = None

    # Per-step config; 0 means no limit
    timeout: float | None = Field(default=None, ge=0)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Reasoning / extended thinking
    thinking: ThinkingConfig | None = None

    # Tool calling: a constrained ``tool_choice`` fails a YAML typo at parse time, not as auto
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: Literal["auto", "required", "none"] | ToolChoiceByName = "auto"
    max_tool_iterations: int = Field(default=5, ge=1)

    # Constrains the reply to a JSON shape; parsed value lands on state.
    response_schema: ResponseSchema | None = None

    # Embed: ``inputs`` is a state path to a ``list[str]``; ``dimensions`` truncates where supported
    inputs: str | None = None
    dimensions: int | None = Field(default=None, ge=1)

    # State path to a :class:`Conversation` the reply is appended to; ``output`` gets the raw text
    conversation: str | None = None
    # With a conversation: tag replies with ``Message.name``, re-assert ``system_prompt`` each turn
    speaker: str | None = None

    # Inline, or a path resolved like ``workflow_path``; its settings replace the llm_call fields
    agent: AgentDefinition | None = None

    @field_validator("agent", mode="before")
    @classmethod
    def _load_agent(cls, value: Any) -> Any:
        from agentloom.core.agent import resolve_agent

        return resolve_agent(value)

    @model_validator(mode="after")
    def _validate_agent_fields(self) -> StepDefinition:
        """An ``agent`` step names its agent; the agent's settings come from its definition."""
        if self.type != StepType.AGENT:
            if self.agent is not None:
                raise ValueError(f"Step {self.id!r}: 'agent' applies only to agent steps")
            return self
        if self.agent is None:
            raise ValueError(f"Agent step {self.id!r} requires an 'agent'")
        # A dumped step (a checkpoint) lists every field: only a value other than the default counts
        fields = type(self).model_fields
        overridden = sorted(
            name
            for name in self.model_fields_set & _AGENT_SETTINGS
            if getattr(self, name) != fields[name].get_default(call_default_factory=True)
        )
        if overridden:
            raise ValueError(
                f"Agent step {self.id!r}: {overridden} come from the agent's definition "
                f"and cannot be set on the step"
            )
        return self

    @model_validator(mode="after")
    def _validate_tools_and_response_schema_are_exclusive(self) -> StepDefinition:
        """Refuse ``tools`` + ``response_schema`` — they compete for the next turn."""
        if self.type == StepType.LLM_CALL and self.tools and self.response_schema is not None:
            raise ValueError(
                f"Step {self.id!r}: 'tools' and 'response_schema' cannot be set "
                f"on the same llm_call step. Pick one — either let the model "
                f"call tools (free-form reply) or constrain its reply to a "
                f"JSON schema (no tool dispatch)."
            )
        return self


# StepDefinition fields an agent step takes from its agent instead.
_AGENT_SETTINGS = frozenset(
    {
        "model",
        "system_prompt",
        "temperature",
        "max_tokens",
        "thinking",
        "tools",
        "tool_choice",
        "max_tool_iterations",
        "response_schema",
        "speaker",
    }
)


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
    """Workflow-level configuration.

    Unknown keys are refused at parse time (``extra="forbid"``). Mock responses
    come from ``responses_file`` (a path to a recording), never an inline
    ``responses:`` list.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    # Endpoint of ``provider: generic_openai``; its key comes from env var ``api_key_env``.
    base_url: str | None = None
    api_key_env: str | None = None
    # For each step whose retry block does not set its own
    max_retries: int = Field(default=3, ge=0)
    budget_usd: float | None = None
    timeout: float | None = Field(default=None, ge=0)  # 0 means no limit, as a step's
    # ge=1: ``CapacityLimiter(0)`` blocks every start, a deadlock; le=1024 is a sanity ceiling
    max_concurrent_steps: int = Field(default=10, ge=1, le=1024)
    stream: bool = False
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    # ``skip_downstream`` keeps dependents of a failed step from acting on partial state
    on_step_failure: Literal["skip_downstream", "continue"] = "skip_downstream"

    # Opt-in: parallel steps sharing an ``output:`` key fail parsing; some rely on last-writer-wins
    strict_outputs: bool = False

    # MockProvider configuration (used only when provider == "mock")
    responses_file: str | None = None
    latency_model: Literal["constant", "normal", "replay"] = "constant"
    latency_ms: float = 0.0

    # Observability
    capture_prompts: bool = (
        False  # When true, llm_call spans emit a span event with the rendered prompt
    )

    @model_validator(mode="after")
    def _endpoint_only_for_generic_openai(self) -> WorkflowConfig:
        if self.provider == "generic_openai":
            if not self.base_url:
                raise ValueError("provider 'generic_openai' needs config.base_url")
        elif self.base_url is not None or self.api_key_env is not None:
            raise ValueError(
                "config.base_url and config.api_key_env apply only to provider 'generic_openai'"
            )
        return self


class StateKeyConfig(BaseModel):
    """Per-key state metadata; ``redact`` is its only setting.

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

    Unknown top-level keys are refused at parse time (``extra="forbid"``): a typo
    such as ``stat_schema:`` would otherwise leave the redaction policy empty and
    write every flagged secret to disk in plaintext.
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

        Two steps with one ``id`` shadow each other: only one would surface in
        ``final_state.steps``, the other lost without warning.
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

    @model_validator(mode="after")
    def _apply_workflow_max_retries(self) -> WorkflowDefinition:
        """Give a step without its own ``retry.max_retries`` the ``config.max_retries``.

        Copies the step instead of changing it: a step object passed in may be
        shared with a workflow whose config differs.
        """
        self.steps = [
            step
            if "max_retries" in step.retry.model_fields_set
            else step.model_copy(
                update={
                    "retry": step.retry.model_copy(update={"max_retries": self.config.max_retries})
                }
            )
            for step in self.steps
        ]
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
