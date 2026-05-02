"""Single source of truth for OTel span, attribute, and metric names.

Downstream trace consumers (AgentTest, Grafana dashboards, Jaeger plugins)
parse these strings. Every span name, span attribute, and metric name used
anywhere in the engine lives here — no raw string literals for telemetry
keys allowed outside this module.

Attribute naming follows the OpenTelemetry **GenAI semantic conventions**
where applicable (``gen_ai.*``); AgentLoom-specific fields use the
``workflow.*``, ``step.*``, ``tool.*``, or ``agentloom.*`` prefixes.

When the public ``agentloom-contracts`` package is extracted in a future
release, this file is the unit that moves — keeping everything in one
place makes that migration a rename rather than a grep.
"""

from __future__ import annotations

from enum import StrEnum


class SpanName:
    """Span name templates.

    Callers use ``.format(**fields)`` to fill in the dynamic segment,
    e.g. ``SpanName.STEP.format(step_id="classify")``.
    """

    # AgentLoom orchestration spans — not part of GenAI conventions (these
    # cover workflow-level dispatch and step-level orchestration, not the
    # inference call itself).
    WORKFLOW = "workflow:{workflow_name}"
    STEP = "step:{step_id}"
    AGENT = "agent:{agent_name}"

    # GenAI inference / tool spans follow the canonical OTel template
    # ``{gen_ai.operation.name} {model}`` so Jaeger / Grafana GenAI dashboards
    # auto-correlate them by operation + model.
    GEN_AI_INFERENCE = "{operation_name} {model}"  # e.g. "chat gpt-4o-mini"
    GEN_AI_TOOL_CALL = "execute_tool {tool_name}"


class SpanAttr:
    """Span attribute keys.

    Grouped by namespace:

    * ``workflow.*`` / ``step.*`` — AgentLoom orchestration metadata.
    * ``gen_ai.*`` — OpenTelemetry GenAI semantic conventions
      (registry-canonical names; the registry is currently flagged
      "experimental / development" — names are not yet "stable" per
      OTel maturity tags but are widely adopted).
    * ``tool.*`` — Tool-call details surfaced by tool_step and agent loops.
    * ``agentloom.*`` — AgentLoom-specific attributes that don't map onto
      an existing OTel namespace (prompt metadata, approval decisions,
      webhook status, record/replay origin, quality annotations).
    """

    # Workflow-level
    WORKFLOW_NAME = "workflow.name"
    WORKFLOW_RUN_ID = "workflow.run_id"
    WORKFLOW_STATUS = "workflow.status"
    WORKFLOW_DURATION_MS = "workflow.duration_ms"
    WORKFLOW_TOTAL_TOKENS = "workflow.total_tokens"
    WORKFLOW_TOTAL_COST_USD = "workflow.total_cost_usd"

    # Step-level
    STEP_ID = "step.id"
    STEP_TYPE = "step.type"
    STEP_STATUS = "step.status"
    STEP_DURATION_MS = "step.duration_ms"
    STEP_COST_USD = "step.cost_usd"
    STEP_ERROR = "step.error"

    # OTel general semantic conventions (not GenAI-specific). ``error.type`` is
    # conditionally required on any span representing a failed operation —
    # set on inference (provider) spans alongside ``step.error`` so OTel-aware
    # consumers can filter on the standard attribute.
    ERROR_TYPE = "error.type"
    STEP_STREAM = "step.stream"
    STEP_ATTACHMENTS = "step.attachments"

    # GenAI semantic conventions — aligned with the OTel registry as of
    # the May 2026 spec.
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"  # array of strings
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"

    # Provider call attempts (gateway-level; one span per fallback attempt)
    PROVIDER_ATTEMPT = "agentloom.provider.attempt"
    PROVIDER_ATTEMPT_OUTCOME = "agentloom.provider.attempt_outcome"

    # Tool calls
    TOOL_NAME = "tool.name"
    TOOL_ARGS_HASH = "tool.args_hash"
    TOOL_RESULT_HASH = "tool.result_hash"
    TOOL_SUCCESS = "tool.success"

    # Prompt metadata (AgentLoom-specific, no full-prompt capture by default)
    PROMPT_HASH = "agentloom.prompt.hash"
    PROMPT_LENGTH_CHARS = "agentloom.prompt.length_chars"
    PROMPT_TEMPLATE_ID = "agentloom.prompt.template_id"
    PROMPT_TEMPLATE_VARS = "agentloom.prompt.template_vars"
    # Span event name emitted when ``WorkflowConfig.capture_prompts`` is on.
    # Carries the rendered prompt as event payload (not an attribute, to
    # stay clear of attribute-size limits and easy to filter in OTLP).
    PROMPT_CAPTURED_EVENT = "agentloom.prompt.captured"

    # Conversation (reserved for the conversation primitive in later phases)
    CONVERSATION_TURN_COUNT = "agentloom.conversation.turn_count"
    CONVERSATION_TOKEN_COUNT = "agentloom.conversation.token_count"

    # Approval gate / webhook / record-replay
    APPROVAL_DECISION = "agentloom.approval_gate.decision"
    WEBHOOK_STATUS = "agentloom.webhook.status"
    WEBHOOK_LATENCY_S = "agentloom.webhook.latency_s"
    MOCK_MATCHED_BY = "agentloom.mock.matched_by"
    MOCK_STEP_ID = "agentloom.mock.step_id"
    RECORDING_PROVIDER = "agentloom.recording.provider"
    RECORDING_MODEL = "agentloom.recording.model"
    RECORDING_LATENCY_S = "agentloom.recording.latency_s"

    # Quality annotations (see issue #59)
    QUALITY_SCORE = "agentloom.quality.score"
    QUALITY_SOURCE = "agentloom.quality.source"


class MetricName:
    """Prometheus / OTel metric names.

    Names mirror exactly what ``observability/metrics.py`` emits to OTel
    and what ``deploy/grafana/dashboards/agentloom.json`` queries. A
    ``test_metric_names_match_emissions`` regression test pins this so
    drift becomes a CI failure, not a silent dashboard breakage.
    """

    # Workflow
    WORKFLOW_RUNS_TOTAL = "agentloom_workflow_runs_total"
    WORKFLOW_DURATION_SECONDS = "agentloom_workflow_duration_seconds"
    COST_USD_TOTAL = "agentloom_cost_usd_total"

    # Step
    STEP_EXECUTIONS_TOTAL = "agentloom_step_executions_total"
    STEP_DURATION_SECONDS = "agentloom_step_duration_seconds"
    ATTACHMENTS_TOTAL = "agentloom_attachments_total"

    # Provider gateway
    PROVIDER_CALLS_TOTAL = "agentloom_provider_calls_total"
    PROVIDER_LATENCY_SECONDS = "agentloom_provider_latency_seconds"
    PROVIDER_ERRORS_TOTAL = "agentloom_provider_errors_total"

    # Tokens / streaming
    TOKENS_TOTAL = "agentloom_tokens_total"
    STREAM_RESPONSES_TOTAL = "agentloom_stream_responses_total"
    TIME_TO_FIRST_TOKEN_SECONDS = "agentloom_time_to_first_token_seconds"

    # Resilience gauges
    CIRCUIT_BREAKER_STATE = "agentloom_circuit_breaker_state"
    BUDGET_REMAINING_USD = "agentloom_budget_remaining_usd"

    # HITL / record-replay
    APPROVAL_GATES_TOTAL = "agentloom_approval_gates_total"
    WEBHOOK_DELIVERIES_TOTAL = "agentloom_webhook_deliveries_total"
    WEBHOOK_LATENCY_SECONDS = "agentloom_webhook_latency_seconds"
    MOCK_REPLAYS_TOTAL = "agentloom_mock_replays_total"
    RECORDING_CAPTURES_TOTAL = "agentloom_recording_captures_total"
    RECORDING_LATENCY_SECONDS = "agentloom_recording_latency_seconds"


class GenAIOperationName(StrEnum):
    """Standard values for the ``gen_ai.operation.name`` attribute.

    ``chat`` covers ordinary conversational completions (the default for
    AgentLoom ``llm_call`` steps). The other values are listed for
    forward-compat with future steps (embeddings, agent loops, tool
    execution) so call sites don't have to invent strings.
    """

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
    EXECUTE_TOOL = "execute_tool"
    INVOKE_AGENT = "invoke_agent"
    INVOKE_WORKFLOW = "invoke_workflow"


class GenAIProviderName(StrEnum):
    """Standard values for the ``gen_ai.provider.name`` attribute.

    Matches the OTel GenAI registry. Reusing the exact spec strings means
    Grafana / Jaeger GenAI dashboards auto-correlate without per-site
    relabel rules. ``ollama`` and ``mock`` are AgentLoom-local extensions
    of the registry — the spec is non-exhaustive and explicitly allows
    custom values.
    """

    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI_INFERENCE = "azure.ai.inference"
    AZURE_AI_OPENAI = "azure.ai.openai"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    GCP_GEMINI = "gcp.gemini"
    GCP_GEN_AI = "gcp.gen_ai"
    GCP_VERTEX_AI = "gcp.vertex_ai"
    GROQ = "groq"
    IBM_WATSONX_AI = "ibm.watsonx.ai"
    MISTRAL_AI = "mistral_ai"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    X_AI = "x_ai"
    # AgentLoom-local custom values (not part of the OTel registry).
    OLLAMA = "ollama"
    MOCK = "mock"


# Map AgentLoom provider class names (``Provider.name``) to the canonical
# OTel ``gen_ai.provider.name`` value. Provider classes keep their short
# names internally; the observer / gateway translate at telemetry time.
_PROVIDER_NAME_TO_GENAI: dict[str, str] = {
    "openai": GenAIProviderName.OPENAI.value,
    "anthropic": GenAIProviderName.ANTHROPIC.value,
    "google": GenAIProviderName.GCP_GEMINI.value,
    "ollama": GenAIProviderName.OLLAMA.value,
    "mock": GenAIProviderName.MOCK.value,
}


def to_genai_provider_name(provider_name: str) -> str:
    """Translate an AgentLoom provider name to the OTel canonical form."""
    return _PROVIDER_NAME_TO_GENAI.get(provider_name, provider_name)


__all__ = [
    "GenAIOperationName",
    "GenAIProviderName",
    "MetricName",
    "SpanAttr",
    "SpanName",
    "to_genai_provider_name",
]
