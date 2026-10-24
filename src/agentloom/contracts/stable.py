"""Stable contracts: the models and protocol other packages build on.

Covered by semantic versioning: a breaking change here needs a major release.
This module imports only pydantic, so consumers that parse AgentLoom traces,
recordings or checkpoints, or wrap AgentLoom workflows, can depend on it
without loading the runtime. The runtime modules that used to define these
models (``core.models``, ``core.results``, ``providers.base``) re-export them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


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


class ToolCall(BaseModel):
    """A function-call decision returned by the model.

    ``id`` round-trips on the follow-up tool-result message (OpenAI:
    ``tool_call_id``, Anthropic: ``tool_use_id``). ``arguments`` is
    already JSON-decoded — adapters parse before constructing this.
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
    tool_calls: list[ToolCall] = Field(default_factory=list)
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


class ToolDefinition(BaseModel):
    """LLM-callable tool declared on an ``llm_call`` step.

    ``parameters`` is a JSON Schema object; provider adapters translate it
    to each API's native shape. ``name`` resolves against the workflow's
    ``tool_registry`` for dispatch.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token usage for an LLM call.

    ``reasoning_tokens`` is populated only when the provider reports a
    separate reasoning / thinking token count — OpenAI o-series via
    ``completion_tokens_details.reasoning_tokens`` and Gemini 2.5+ via
    ``usageMetadata.thoughtsTokenCount``. Anthropic and Ollama do not
    expose a separate count today (Anthropic rolls thinking into
    ``output_tokens``; Ollama emits a single ``eval_count``), so the
    field stays ``0`` for those providers — the chain-of-thought trace
    is still surfaced via ``ProviderResponse.reasoning_content``. When
    populated, providers bill these tokens at the output rate, so
    cost calculation must include them.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def billable_completion_tokens(self) -> int:
        """Tokens billed at the provider's output rate.

        Always equal to ``completion_tokens + reasoning_tokens`` — exposed
        as a property so downstream cost code does not have to remember
        which field to add.
        """
        return self.completion_tokens + self.reasoning_tokens


class ResponseMeta(BaseModel):
    """What a provider call sent and got back besides the answer.

    The sampling parameters are the ones in the request body, after the
    adapter dropped or renamed what its API does not take. The identifiers
    are those the provider returned: ``system_fingerprint`` from
    OpenAI-compatible servers, ``model_version`` from Gemini, and the
    request id from a response header or, on Gemini, the body.
    ``rate_limit`` keeps only the rate-limit and ``retry-after`` headers.
    ``attempt`` is set in recordings only, when the engine retried the step.
    """

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    system_fingerprint: str | None = None
    model_version: str | None = None
    request_id: str | None = None
    rate_limit: dict[str, str] = Field(default_factory=dict)
    received_at: datetime | None = None
    attempt: int | None = None


class ProviderResponse(BaseModel):
    """Unified response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None
    # Reasoning / chain-of-thought text. Populated when the provider
    # exposes the trace and the caller did not opt out via
    # ``ThinkingConfig.capture_reasoning``. OpenAI o-series keeps the
    # trace server-side, so this stays ``None``; Anthropic concatenates
    # ``type="thinking"`` blocks; Gemini surfaces ``thought=true`` parts;
    # Ollama returns ``message.thinking`` (or strips inline
    # ``<think>...</think>`` tags as a fallback).
    reasoning_content: str | None = None
    # Empty when the model didn't pick a tool. When non-empty, ``content``
    # may be empty — the LLM step dispatches each call and re-prompts.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Parsed structured output (dict or Pydantic instance). ``None`` for
    # free-form steps. Populated by ``LLMCallStep``, not the adapters.
    # Typed ``object | None`` — callers treat it as opaque; consumers
    # that need a specific shape do their own ``isinstance`` narrowing.
    parsed: object | None = None
    # ``None`` when the adapter reports nothing (embeddings, the mock
    # replaying an entry recorded without it).
    meta: ResponseMeta | None = None


@runtime_checkable
class WorkflowHandle(Protocol):
    """A runnable workflow, whatever runtime implements it.

    Structural: any object with these members satisfies it, without
    depending on AgentLoom (a LangGraph or CrewAI wrapper, for instance).
    ``agentloom.core.handle.WorkflowAdapter`` wraps an AgentLoom workflow.
    """

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema of the inputs ``run`` accepts."""

    @property
    def output_schema(self) -> dict[str, Any]:
        """JSON Schema of the outputs ``run`` returns."""

    async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run once with ``inputs`` and return the outputs."""


__all__ = [
    "Attachment",
    "Message",
    "ProviderResponse",
    "ResponseMeta",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "WorkflowHandle",
]
