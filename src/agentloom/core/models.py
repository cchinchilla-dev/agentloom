"""Pydantic models for workflow and step definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentloom.resilience.retry import DEFAULT_RETRYABLE_STATUS_CODES


class StepType(StrEnum):
    """Supported step types."""

    LLM_CALL = "llm_call"
    TOOL = "tool"
    ROUTER = "router"
    SUBWORKFLOW = "subworkflow"
    APPROVAL_GATE = "approval_gate"
    EMBED = "embed"


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


class ToolCallSpec(BaseModel):
    """A tool decision persisted inside a :class:`Message`.

    Structurally identical to :class:`agentloom.providers.base.ToolCall`
    but re-declared here to avoid a circular import from ``core`` into
    ``providers``. Conversion helpers on :class:`Message` translate to /
    from the provider-side ``ToolCall`` when the LLM step needs to emit
    the wire-format assistant turn.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single turn inside a :class:`Conversation`.

    ``role`` follows the OpenAI/Anthropic chat conventions:

    - ``system``: instruction turn (usually kept as the first message and
      never trimmed away by the built-in policies).
    - ``user``: end-user (or simulated user) input.
    - ``assistant``: model reply — may carry ``tool_calls`` when the model
      picked one or more tools.
    - ``tool``: result of a previous tool call, paired with
      ``tool_call_id``.

    ``name`` is optional and identifies the speaker for multi-agent
    threads (e.g. ``alice`` vs ``bob``). Providers that support ``name``
    (OpenAI, some Anthropic tool-loop shapes) forward it; providers that
    don't drop it silently with a debug log.

    ``metadata`` carries free-form turn-level annotations (timestamp,
    agent_id, tool call index) that ride along the checkpoint but never
    reach the provider.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def token_count(self, model: str = "") -> int:
        """Rough token-count estimate for budget bookkeeping.

        Not authoritative — the value is only ever used to decide whether
        the conversation exceeds ``token_budget`` before a call, so a
        conservative under-count is fine. The heuristic follows the same
        1 token ≈ 4 characters approximation the OpenAI docs recommend
        for English + light per-message overhead (role marker + name
        marker) so single-word turns don't hash to 0 tokens. ``model`` is
        accepted for forward compatibility with a tokenizer-backed
        implementation but currently ignored.
        """
        del model
        char_count = len(self.content)
        for call in self.tool_calls:
            char_count += len(call.name)
            # Approximate JSON length of the tool arguments without paying
            # the ``json.dumps`` cost on every trim check.
            for key, value in call.arguments.items():
                char_count += len(str(key)) + len(str(value)) + 4
        overhead = 4  # role marker
        if self.name:
            overhead += 2 + len(self.name)
        if self.tool_call_id:
            overhead += 4 + len(self.tool_call_id)
        return max(1, (char_count + 3) // 4 + overhead)


class Conversation(BaseModel):
    """A typed multi-turn conversation stored in workflow state.

    Workflows reference a conversation by dotted state path
    (``conversation: state.chat``) on an ``llm_call`` step. The engine
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

        Pydantic normally handles this via ``model_validate``, but keeping
        the coercion explicit lets us survive minor schema drift — a
        recording made before ``metadata`` existed still round-trips
        without a manual migration.
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

        Delegates to :meth:`Message.token_count`; see there for the
        heuristic's caveats.
        """
        return sum(m.token_count(model) for m in self.messages)


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
    # ``schema_`` avoids shadowing Pydantic's deprecated ``BaseModel.schema()``;
    # YAML authors write ``schema:`` via the alias.
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

    # Constrains the reply to a JSON shape; parsed value lands on state.
    response_schema: ResponseSchema | None = None

    # Embed step fields — ``inputs`` is a dotted state path (e.g.
    # ``state.documents``) that resolves to a ``list[str]``; ``dimensions``
    # requests a truncated vector on providers that honour it (OpenAI
    # ``text-embedding-3-*``, Google ``gemini-embedding-001``). ``ge=1``
    # rejects ``0`` / negative at parse time so failures don't leak to
    # provider-specific 400s at runtime.
    inputs: str | None = None
    dimensions: int | None = Field(default=None, ge=1)

    # Conversation-history primitive. ``conversation`` is a dotted state
    # path (e.g. ``state.chat``) that resolves to a :class:`Conversation`.
    # The step loads the messages, appends the rendered ``prompt`` as a
    # user turn (skipped when ``prompt`` is empty and the conversation
    # already has a trailing user message), sends the full list to the
    # provider, and appends the assistant reply (plus any tool calls)
    # back to the same path before persisting. When set, ``output`` is
    # still honoured but stores the raw assistant text — the conversation
    # itself carries the structured turn.
    conversation: str | None = None
    # Multi-agent speaker name. When set (only meaningful with
    # ``conversation``), the appended user turn and the assistant reply
    # are tagged with ``Message.name`` so a shared conversation can carry
    # several distinct agents (OpenAI forwards ``name`` natively; the
    # other adapters prepend ``"[<speaker>] …"`` inline). Also lets the
    # step's ``system_prompt`` re-assert each turn — without a speaker the
    # system turn is inserted once, but a multi-agent workflow needs each
    # agent's own instruction on its turn.
    speaker: str | None = None

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

    Unknown keys are refused at parse time (``extra="forbid"``). This
    also rejects the half-supported ``config.responses:`` field — mock
    responses are configured via ``responses_file`` (a path to a
    recording), never an inline ``responses:`` list.
    """

    model_config = ConfigDict(extra="forbid")

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
