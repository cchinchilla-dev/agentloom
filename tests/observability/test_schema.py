"""Regression tests for the centralized OTel schema module.

These pin the public string values so downstream consumers (AgentTest,
Grafana dashboards) can depend on them. A rename here should be a
deliberate, reviewed change — not an accidental edit to a comment.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agentloom.observability.observer import WorkflowObserver
from agentloom.observability.schema import (
    GenAIOperationName,
    GenAIProviderName,
    MetricName,
    SpanAttr,
    SpanName,
    to_genai_provider_name,
)


class TestSpanAttrContract:
    """GenAI semantic convention keys must be exactly the canonical
    ``gen_ai.*`` names from the OTel registry — no deprecated aliases."""

    def test_gen_ai_attribute_keys(self) -> None:
        assert SpanAttr.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
        assert SpanAttr.GEN_AI_PROVIDER_NAME == "gen_ai.provider.name"
        assert SpanAttr.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
        assert SpanAttr.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert SpanAttr.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        assert (
            SpanAttr.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS == "gen_ai.usage.reasoning.output_tokens"
        )
        assert SpanAttr.GEN_AI_RESPONSE_FINISH_REASONS == "gen_ai.response.finish_reasons"
        assert SpanAttr.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK == "gen_ai.response.time_to_first_chunk"

    def test_workflow_and_step_keys(self) -> None:
        assert SpanAttr.WORKFLOW_NAME == "workflow.name"
        assert SpanAttr.WORKFLOW_RUN_ID == "workflow.run_id"
        assert SpanAttr.STEP_ID == "step.id"

    def test_agentloom_specific_keys_namespaced(self) -> None:
        assert SpanAttr.PROMPT_HASH.startswith("agentloom.")
        assert SpanAttr.APPROVAL_DECISION.startswith("agentloom.")
        assert SpanAttr.WEBHOOK_STATUS.startswith("agentloom.")
        assert SpanAttr.RECORDING_PROVIDER.startswith("agentloom.")


class TestSpanNameTemplates:
    def test_workflow_span_template(self) -> None:
        assert SpanName.WORKFLOW.format(workflow_name="wf") == "workflow:wf"

    def test_step_span_template(self) -> None:
        assert SpanName.STEP.format(step_id="s1") == "step:s1"

    def test_inference_span_template(self) -> None:
        # Canonical OTel template: "{operation_name} {model}".
        assert (
            SpanName.GEN_AI_INFERENCE.format(operation_name="chat", model="gpt-4o-mini")
            == "chat gpt-4o-mini"
        )


class TestGenAIEnums:
    def test_provider_name_canonical_values(self) -> None:
        # Spec values from the OTel registry — must match exactly.
        assert GenAIProviderName.OPENAI == "openai"
        assert GenAIProviderName.ANTHROPIC == "anthropic"
        assert GenAIProviderName.GCP_GEMINI == "gcp.gemini"

    def test_operation_name_canonical_values(self) -> None:
        assert GenAIOperationName.CHAT == "chat"
        assert GenAIOperationName.EMBEDDINGS == "embeddings"
        assert GenAIOperationName.EXECUTE_TOOL == "execute_tool"

    def test_provider_name_translation(self) -> None:
        # AgentLoom internal names map to OTel canonical names.
        assert to_genai_provider_name("google") == "gcp.gemini"
        assert to_genai_provider_name("openai") == "openai"
        assert to_genai_provider_name("anthropic") == "anthropic"
        # Unknown providers pass through verbatim.
        assert to_genai_provider_name("unknown") == "unknown"


class TestMetricNameCentralization:
    def test_metric_names_prefixed_with_agentloom(self) -> None:
        for attr_name, value in vars(MetricName).items():
            if attr_name.isupper() and isinstance(value, str):
                assert value.startswith("agentloom_"), (
                    f"{attr_name}={value!r} must start with 'agentloom_'"
                )

    def test_metric_names_match_emissions(self) -> None:
        """``MetricName`` constants must mirror exactly what ``metrics.py``
        emits to OTel — drift breaks the Grafana dashboard silently. This
        test scans the source for ``agentloom_*`` literal names and
        asserts every one is declared in the schema."""
        import re
        from pathlib import Path

        metrics_src = (
            Path(__file__).parent.parent.parent
            / "src"
            / "agentloom"
            / "observability"
            / "metrics.py"
        ).read_text()
        emitted = set(re.findall(r'"(agentloom_[a-z_]+)"', metrics_src))
        declared = {
            value
            for attr_name, value in vars(MetricName).items()
            if attr_name.isupper() and isinstance(value, str)
        }
        missing = emitted - declared
        assert not missing, (
            f"metrics.py emits these names that aren't in MetricName: {sorted(missing)}"
        )


class TestObserverEmitsRunIdAndGenAI:
    def test_workflow_span_includes_run_id(self) -> None:
        tracing = MagicMock()
        span = MagicMock()
        tracing.start_span.return_value = span

        observer = WorkflowObserver(tracing=tracing)
        observer.on_workflow_start("wf1", run_id="run-abc")

        _, kwargs = tracing.start_span.call_args
        attrs = kwargs["attributes"]
        assert attrs[SpanAttr.WORKFLOW_NAME] == "wf1"
        assert attrs[SpanAttr.WORKFLOW_RUN_ID] == "run-abc"

    def test_step_span_includes_gen_ai_attributes(self) -> None:
        tracing = MagicMock()
        span = MagicMock()
        tracing.start_span.return_value = span

        observer = WorkflowObserver(tracing=tracing)
        observer.on_workflow_start("wf1", run_id="r1")
        observer.on_step_start("s1", "llm_call", stream=False)
        observer.on_step_end(
            "s1",
            "llm_call",
            "success",
            duration_ms=100.0,
            cost_usd=0.01,
            prompt_tokens=20,
            completion_tokens=20,
            model="gpt-4o-mini",
            provider="openai",
            finish_reason="stop",
            prompt_hash="abc123",
            prompt_length_chars=42,
            prompt_template_id="wf1:s1",
            prompt_template_vars="state.query",
        )

        keys = {c.args[0] for c in span.set_attribute.call_args_list}
        assert SpanAttr.GEN_AI_PROVIDER_NAME in keys
        assert SpanAttr.GEN_AI_OPERATION_NAME in keys
        assert SpanAttr.GEN_AI_REQUEST_MODEL in keys
        assert SpanAttr.GEN_AI_USAGE_INPUT_TOKENS in keys
        assert SpanAttr.GEN_AI_USAGE_OUTPUT_TOKENS in keys
        assert SpanAttr.GEN_AI_RESPONSE_FINISH_REASONS in keys
        assert SpanAttr.PROMPT_HASH in keys
        assert SpanAttr.PROMPT_TEMPLATE_ID in keys
        # finish_reasons must be the spec-mandated array form.
        attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert attrs[SpanAttr.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]

    def test_provider_child_span_start_end_pair(self) -> None:
        tracing = MagicMock()
        span = MagicMock()
        tracing.start_span.return_value = span

        observer = WorkflowObserver(tracing=tracing)
        observer.on_provider_call_start(
            "s1",
            "openai",
            "gpt-4o-mini",
            temperature=0.2,
            max_tokens=500,
        )
        observer.on_provider_call_end(
            "s1",
            "openai",
            "gpt-4o-mini",
            latency_s=0.4,
            prompt_tokens=10,
            completion_tokens=15,
        )
        # Span was started with the canonical OTel inference name.
        name, kwargs = tracing.start_span.call_args
        assert name[0] == "chat gpt-4o-mini"
        attrs = kwargs["attributes"]
        assert attrs[SpanAttr.GEN_AI_OPERATION_NAME] == "chat"
        assert attrs[SpanAttr.GEN_AI_PROVIDER_NAME] == "openai"
        assert attrs[SpanAttr.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
        # End closed it.
        tracing.end_span.assert_called_once_with(span)


class TestPromptAttributesOnLLMStepSpan:
    """``llm_call`` step spans must carry the prompt-metadata attributes
    promised by #125: hash, length_chars, template_id, template_vars."""

    def test_prompt_hash_present_on_llm_step_span(self) -> None:
        tracing = MagicMock()
        span = MagicMock()
        tracing.start_span.return_value = span
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("answer", "llm_call")
        observer.on_step_end(
            "answer",
            "llm_call",
            "success",
            120.0,
            prompt_hash="abc123",
            prompt_length_chars=88,
            prompt_template_id="wf:answer",
            prompt_template_vars="state.q,state.ctx",
        )
        attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert attrs[SpanAttr.PROMPT_HASH] == "abc123"
        assert attrs[SpanAttr.PROMPT_LENGTH_CHARS] == 88

    def test_template_vars_used_listed_on_span(self) -> None:
        tracing = MagicMock()
        span = MagicMock()
        tracing.start_span.return_value = span
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("answer", "llm_call")
        observer.on_step_end(
            "answer",
            "llm_call",
            "success",
            10.0,
            prompt_template_vars="state.q,state.ctx,state.user",
        )
        attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert attrs[SpanAttr.PROMPT_TEMPLATE_VARS] == "state.q,state.ctx,state.user"


class TestCapturePromptsFlag:
    """``observability.capture_prompts`` is off by default; when on, the
    full rendered prompt rides on the span as an OTel event (not an
    attribute, to avoid blowing the attribute size budget)."""

    def test_capture_prompts_flag_off_by_default(self) -> None:
        from agentloom.core.models import WorkflowConfig

        cfg = WorkflowConfig()
        assert cfg.capture_prompts is False

    def test_capture_prompts_flag_on_emits_span_event(self) -> None:
        # When the flag is enabled, ``llm_call`` calls
        # ``observer.attach_step_event`` with the rendered prompt — verified
        # at the observer hook level here; the engine wiring is covered by
        # the integration smoke and the prompt-flag flows through
        # ``StepContext.capture_prompts``.
        tracing = MagicMock()
        span = MagicMock()
        tracing.start_span.return_value = span
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("answer", "llm_call")
        observer.attach_step_event(
            "answer",
            "agentloom.prompt.captured",
            {"prompt": "rendered text", "system_prompt": ""},
        )
        span.add_event.assert_called_once_with(
            "agentloom.prompt.captured",
            {"prompt": "rendered text", "system_prompt": ""},
        )


class TestSchemaModuleIsSingleSourceOfTruth:
    """Linter-style guard: span / metric attribute literals only live in
    ``observability/schema.py`` so external consumers can rely on a stable
    contract instead of grepping for ad-hoc strings."""

    def test_no_string_literal_span_attrs_outside_schema_module(self) -> None:
        # Linter-style guard. Targets the high-signal namespaces only —
        # ``gen_ai.*`` (OTel GenAI conventions) and the AgentLoom
        # attribute prefixes that ``schema.py`` declares — so that a new
        # call site reaching for an ad-hoc attribute string fails CI
        # instead of silently fragmenting the contract. Logger names
        # (``agentloom.engine`` etc.) are pure package paths and don't
        # match the attribute regexes by design.
        import re
        from pathlib import Path

        src_root = Path(__file__).parent.parent.parent / "src" / "agentloom"
        attr_patterns = [
            re.compile(r'"gen_ai\.[a-z_.]+?"'),
            re.compile(
                r'"agentloom\.(?:prompt|approval_gate|webhook|mock|recording|quality|conversation|provider)\.[a-z_.]+?"'
            ),
        ]
        offenders: list[tuple[str, int, str]] = []
        for path in src_root.rglob("*.py"):
            if path.name == "schema.py":
                continue
            text = path.read_text()
            for line_no, line in enumerate(text.splitlines(), start=1):
                for pat in attr_patterns:
                    for match in pat.findall(line):
                        offenders.append((str(path.relative_to(src_root)), line_no, match))
        assert not offenders, (
            "Span / attribute literals found outside schema.py — these must "
            "go through ``SpanAttr.*``:\n" + "\n".join(f"  {p}:{ln}  {m}" for p, ln, m in offenders)
        )


class TestGenAIProviderNameTranslation:
    """Every concrete provider's internal name must translate to a
    ``GenAIProviderName`` value via ``to_genai_provider_name`` so the
    canonical OTel attribute lights up Grafana / Jaeger GenAI dashboards
    without per-site relabel rules."""

    def test_registered_providers_translate_to_canonical_names(self) -> None:
        from agentloom.providers.anthropic import AnthropicProvider
        from agentloom.providers.google import GoogleProvider
        from agentloom.providers.mock import MockProvider
        from agentloom.providers.ollama import OllamaProvider
        from agentloom.providers.openai import OpenAIProvider

        valid = {member.value for member in GenAIProviderName}
        for provider_cls in (
            OpenAIProvider,
            AnthropicProvider,
            GoogleProvider,
            OllamaProvider,
            MockProvider,
        ):
            translated = to_genai_provider_name(provider_cls.name)
            assert translated in valid, (
                f"{provider_cls.__name__}.name={provider_cls.name!r} "
                f"translates to {translated!r}, not in GenAIProviderName"
            )
