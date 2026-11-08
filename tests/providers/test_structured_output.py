"""Structured output across providers: translation, parsing, validation and retries."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
import pytest
import respx
from pydantic import BaseModel, Field, ValidationError

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    ResponseSchema,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowStatus
from agentloom.core.state import StateManager
from agentloom.providers.anthropic import AnthropicProvider
from agentloom.providers.base import ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.google import GoogleProvider
from agentloom.providers.mock import MockProvider
from agentloom.providers.ollama import OllamaProvider
from agentloom.providers.openai import OpenAIProvider
from agentloom.steps._structured import (
    extract_parsed,
    format_validation_feedback,
    load_pydantic_model,
    schema_dict_for,
    translate_for_google,
    translate_for_ollama,
    translate_for_openai,
    validate_parsed,
)
from agentloom.tools.registry import ToolRegistry


class _Classification(BaseModel):
    """Shared Pydantic shape used by the structured-output tests."""

    label: Literal["question", "complaint", "feedback"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


# At module level so the test bodies stay within the 100-char line limit.
_GOOD_JSON = '{"label": "complaint", "confidence": 0.9, "rationale": "ok"}'
_BAD_JSON_LABEL = '{"label": "not-a-real-label", "confidence": 2.0, "rationale": "x"}'
_FIXED_JSON = '{"label": "complaint", "confidence": 0.5, "rationale": "fixed"}'


def _workflow_with_schema(
    *,
    schema: ResponseSchema,
    provider: str = "mock",
    model: str = "gpt-4o-mini",
    responses_file: str | None = None,
    max_retries: int = 0,
) -> WorkflowDefinition:
    """Single-step workflow that uses a ``response_schema``."""
    from agentloom.core.models import RetryConfig

    return WorkflowDefinition(
        name="structured-test",
        config=WorkflowConfig(
            provider=provider,
            model=model,
            responses_file=responses_file,
        ),
        state={"input": "My order from yesterday never arrived."},
        steps=[
            StepDefinition(
                id="classify",
                type=StepType.LLM_CALL,
                prompt="Classify: {state.input}",
                response_schema=schema,
                output="classification",
                retry=RetryConfig(max_retries=max_retries),
            )
        ],
    )


async def _run(
    workflow: WorkflowDefinition, gateway: ProviderGateway
) -> tuple[WorkflowStatus, dict[str, Any]]:
    """Boilerplate engine wiring for the integration tests."""
    sm = StateManager(initial_state=dict(workflow.state))
    engine = WorkflowEngine(
        workflow=workflow,
        state_manager=sm,
        provider_gateway=gateway,
        tool_registry=ToolRegistry(),
    )
    result = await engine.run()
    return result.status, result.final_state


class TestResponseSchemaModel:
    """``ResponseSchema`` Pydantic parse-time invariants."""

    def test_pydantic_mode_requires_model_field(self) -> None:
        with pytest.raises(ValidationError, match="dotted path"):
            ResponseSchema(type="pydantic")

    def test_pydantic_mode_rejects_inline_schema(self) -> None:
        with pytest.raises(ValidationError, match="must not also set 'schema'"):
            ResponseSchema(
                type="pydantic",
                model="x.Y",
                schema_={"type": "object"},
            )

    def test_json_schema_mode_requires_inline_schema(self) -> None:
        with pytest.raises(ValidationError, match="inline 'schema' object"):
            ResponseSchema(type="json_schema")

    def test_json_object_mode_accepts_neither_field(self) -> None:
        rs = ResponseSchema(type="json_object")
        assert rs.type == "json_object"

    def test_yaml_alias_schema_is_honored(self) -> None:
        # YAML authors write ``schema:``, not ``schema_:``.
        inline = {"type": "object", "properties": {"x": {"type": "string"}}}
        rs = ResponseSchema.model_validate({"type": "json_schema", "schema": inline})
        assert rs.schema_ == inline

    def test_additionalproperties_true_rejected_for_openai_json_schema(self) -> None:
        # OpenAI strict mode 400s on it; refusing it at parse time keeps workflows portable.
        with pytest.raises(ValidationError, match="additionalProperties: true"):
            ResponseSchema(
                type="json_schema",
                schema_={"type": "object", "additionalProperties": True},
            )


class TestSchemaTranslators:
    """Provider-agnostic translation helpers."""

    def test_openai_pydantic_translation(self) -> None:
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        out = translate_for_openai(rs, "classify")
        assert out["type"] == "json_schema"
        assert out["json_schema"]["name"] == "_Classification"
        assert out["json_schema"]["strict"] is True
        assert out["json_schema"]["schema"]["additionalProperties"] is False

    def test_openai_json_object_translation(self) -> None:
        rs = ResponseSchema(type="json_object")
        assert translate_for_openai(rs, "step") == {"type": "json_object"}

    def test_google_translation_drops_pydantic_titles(self) -> None:
        # Gemini 400s on ``title``, ``examples``, ``default`` and ``additionalProperties``.
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "title": "Foo",
                "properties": {"x": {"type": "string", "title": "X-field", "default": "abc"}},
                "required": ["x"],
            },
        )
        mime, schema = translate_for_google(rs, "step")
        assert mime == "application/json"
        assert schema is not None
        assert "title" not in schema
        assert "title" not in schema["properties"]["x"]
        assert "default" not in schema["properties"]["x"]
        # The strict normalizer adds ``additionalProperties: false``; Gemini's dialect rejects it.
        assert "additionalProperties" not in schema
        assert "additionalProperties" not in schema["properties"]["x"]

    def test_ollama_json_object_returns_literal_json(self) -> None:
        # ``"json"`` is Ollama's documented value for free-form JSON.
        rs = ResponseSchema(type="json_object")
        assert translate_for_ollama(rs, "step") == "json"

    def test_ollama_strict_returns_schema_dict(self) -> None:
        # Ollama 0.5+ takes a full JSON Schema under the same ``format`` key.
        rs = ResponseSchema(
            type="json_schema",
            schema_={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        )
        out = translate_for_ollama(rs, "step")
        assert isinstance(out, dict)
        assert out["type"] == "object"

    def test_schema_normalizes_additionalproperties_false(self) -> None:
        rs = ResponseSchema(
            type="json_schema",
            schema_={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        schema = schema_dict_for(rs, "step")
        assert schema is not None
        assert schema["additionalProperties"] is False


class TestExtractParsed:
    """JSON-extraction fallbacks."""

    def test_strict_json_parses(self) -> None:
        assert extract_parsed('{"a": 1}') == {"a": 1}

    def test_code_fence_is_stripped(self) -> None:
        assert extract_parsed('```json\n{"a": 1}\n```') == {"a": 1}

    def test_anthropic_prefill_continuation_parses(self) -> None:
        # The adapter reattaches the prefilled ``{``; the parser copes when one does not.
        assert extract_parsed('"label": "complaint", "confidence": 0.9}') == {
            "label": "complaint",
            "confidence": 0.9,
        }

    def test_prose_around_json_parses(self) -> None:
        # The first balanced ``{...}`` span wins.
        assert extract_parsed('Here it is:\n{"a": 1}\nThanks.') == {"a": 1}

    def test_empty_content_raises_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            extract_parsed("")


class TestValidateParsed:
    """Per-mode validation behaviour."""

    def test_json_object_mode_passes_through(self) -> None:
        rs = ResponseSchema(type="json_object")
        assert validate_parsed({"anything": 1}, rs, "step") == {"anything": 1}

    @pytest.mark.parametrize("parsed", [None, [1, 2], "text", 3])
    def test_json_object_mode_refuses_what_is_no_object(self, parsed: Any) -> None:
        """A reply of ``null`` is no reply: it is asked for again, as a schema's mismatch."""
        rs = ResponseSchema(type="json_object")
        with pytest.raises(TypeError, match="expects an object"):
            validate_parsed(parsed, rs, "step")

    def test_pydantic_coerces_to_model_instance(self) -> None:
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        out = validate_parsed(
            {"label": "complaint", "confidence": 0.92, "rationale": "x"}, rs, "step"
        )
        assert isinstance(out, _Classification)
        assert out.confidence == 0.92

    def test_pydantic_validation_failure_raises(self) -> None:
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        with pytest.raises(ValidationError):
            validate_parsed({"label": "complaint", "confidence": 5.0, "rationale": "x"}, rs, "step")

    def test_json_schema_validation_runs(self) -> None:
        import jsonschema

        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        with pytest.raises(jsonschema.ValidationError):
            validate_parsed({"x": "not-an-int"}, rs, "step")

    def test_json_schema_valid_input_passes_through(self) -> None:
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        assert validate_parsed({"x": 42}, rs, "step") == {"x": 42}


class TestLoadPydanticModel:
    """Dotted-path loader behaviour."""

    def test_load_valid_path(self) -> None:
        cls = load_pydantic_model("tests.providers.test_structured_output._Classification")
        assert cls is _Classification

    def test_load_colon_form(self) -> None:
        # ``module:ClassName`` keeps YAML clear when the model lives in a nested module.
        cls = load_pydantic_model("tests.providers.test_structured_output:_Classification")
        assert cls is _Classification

    def test_unknown_module_raises_config_error(self) -> None:
        from agentloom.core.models import ResponseSchemaConfigError

        with pytest.raises(ResponseSchemaConfigError, match="cannot import"):
            load_pydantic_model("definitely.not.a.real.module:X")

    def test_unknown_class_raises_config_error(self) -> None:
        from agentloom.core.models import ResponseSchemaConfigError

        with pytest.raises(ResponseSchemaConfigError, match="has no attribute"):
            load_pydantic_model("tests.providers.test_structured_output:NotAClass")

    def test_non_pydantic_class_raises_config_error(self) -> None:
        from agentloom.core.models import ResponseSchemaConfigError

        with pytest.raises(ResponseSchemaConfigError, match="not a Pydantic"):
            load_pydantic_model("collections:OrderedDict")

    def test_malformed_path_raises_config_error(self) -> None:
        from agentloom.core.models import ResponseSchemaConfigError

        with pytest.raises(ResponseSchemaConfigError, match="dotted path"):
            load_pydantic_model("NoModule")


class TestFormatValidationFeedback:
    """Retry-prompt formatter shape."""

    def test_pydantic_errors_render_as_bullets(self) -> None:
        try:
            _Classification.model_validate({"label": "xxx", "confidence": 5, "rationale": ""})
        except ValidationError as exc:
            msg = format_validation_feedback(exc, "step")
        assert "step" in msg
        assert "label" in msg
        assert "Respond again" in msg

    def test_long_errors_truncate(self) -> None:
        # A non-Pydantic error skips the bullet formatter: the simplest way to hit the cap.
        long_msg = "detail " * 200
        msg = format_validation_feedback(ValueError(long_msg), "step")
        assert "(truncated)" in msg


class TestProviderPropagation:
    """Each adapter must honour the structured-output kwargs on the wire."""

    @respx.mock
    async def test_response_schema_json_schema_native_openai(self) -> None:
        # The strict bit makes the API enforce the shape server-side.
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": _GOOD_JSON,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                },
            )

        respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_handler)
        provider = OpenAIProvider(api_key="sk-test")
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            agentloom_response_schema=rs,
            agentloom_step_id="classify",
        )
        rf = captured["payload"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["name"] == "_Classification"
        # Added by the translator, so portable schemas need not set it.
        assert rf["json_schema"]["schema"]["additionalProperties"] is False
        await provider.close()

    @respx.mock
    async def test_response_schema_json_schema_native_google(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": (
                                            '{"label": "complaint", '
                                            '"confidence": 0.7, "rationale": "x"}'
                                        ),
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 3,
                    },
                },
            )

        respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.+").mock(
            side_effect=_handler
        )
        provider = GoogleProvider(api_key="g-test")
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-1.5-flash",
            agentloom_response_schema=rs,
            agentloom_step_id="classify",
        )
        gen_cfg = captured["payload"]["generationConfig"]
        assert gen_cfg["responseMimeType"] == "application/json"
        assert "responseSchema" in gen_cfg
        # Gemini-incompatible Pydantic-emitted keys must have been stripped.
        assert "title" not in gen_cfg["responseSchema"]
        await provider.close()

    @respx.mock
    async def test_response_schema_anthropic_prefill_fallback(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "claude-3-5-sonnet",
                    "content": [
                        {
                            "type": "text",
                            "text": '"label":"complaint","confidence":0.8,"rationale":"y"}',
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                    "stop_reason": "end_turn",
                },
            )

        respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=_handler)
        provider = AnthropicProvider(api_key="anth-test")
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        response = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-3-5-sonnet-20240620",
            agentloom_response_schema=rs,
            agentloom_step_id="classify",
        )
        msgs = captured["payload"]["messages"]
        # The prefilled assistant turn makes the model continue the object from ``{``.
        assert msgs[-1] == {"role": "assistant", "content": "{"}
        # The system prompt gained the JSON-only instruction.
        assert "JSON" in captured["payload"]["system"]
        # ``ProviderResponse.content`` is reattached to a complete JSON string.
        assert response.content.startswith("{")
        await provider.close()

    @respx.mock
    async def test_response_schema_ollama_format_field(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "message": {
                        "role": "assistant",
                        "content": ('{"label": "complaint", "confidence": 0.8, "rationale": "x"}'),
                    },
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 2,
                },
            )

        respx.post("http://localhost:11434/api/chat").mock(side_effect=_handler)
        provider = OllamaProvider(base_url="http://localhost:11434")
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        )
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            agentloom_response_schema=rs,
            agentloom_step_id="classify",
        )
        fmt = captured["payload"]["format"]
        assert isinstance(fmt, dict)
        assert fmt["type"] == "object"
        await provider.close()


class TestEndToEndIntegration:
    """LLMCallStep wiring against MockProvider."""

    async def test_response_schema_pydantic_coerces_to_model_instance(self, tmp_path: Any) -> None:
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "classify": {
                        "content": '{"label": "complaint", "confidence": 0.92, "rationale": "ok"}',
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        "cost_usd": 0.0,
                        "latency_ms": 0,
                    },
                }
            )
        )
        wf = _workflow_with_schema(
            schema=ResponseSchema(
                type="pydantic",
                model="tests.providers.test_structured_output._Classification",
            ),
            responses_file=str(recording),
        )
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        assert isinstance(final["classification"], _Classification)
        assert final["classification"].label == "complaint"

    async def test_response_schema_validation_failure_triggers_retry(self, tmp_path: Any) -> None:
        # A list of turns gives the mock a bad answer first, then the fixed one.
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "classify": [
                        {
                            "content": _BAD_JSON_LABEL,
                            "model": "gpt-4o-mini",
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 2,
                                "total_tokens": 3,
                            },
                            "cost_usd": 0.0,
                            "latency_ms": 0,
                        },
                        {
                            "content": _FIXED_JSON,
                            "model": "gpt-4o-mini",
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 2,
                                "total_tokens": 3,
                            },
                            "cost_usd": 0.0,
                            "latency_ms": 0,
                        },
                    ],
                }
            )
        )
        wf = _workflow_with_schema(
            schema=ResponseSchema(
                type="pydantic",
                model="tests.providers.test_structured_output._Classification",
            ),
            responses_file=str(recording),
            max_retries=2,
        )
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        assert final["classification"].label == "complaint"

    async def test_response_schema_validation_failure_after_max_retries_fails_step(
        self, tmp_path: Any
    ) -> None:
        # A single entry serves the same bad JSON to every retry.
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "classify": {
                        "content": _BAD_JSON_LABEL,
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        "cost_usd": 0.0,
                        "latency_ms": 0,
                    },
                }
            )
        )
        wf = _workflow_with_schema(
            schema=ResponseSchema(
                type="pydantic",
                model="tests.providers.test_structured_output._Classification",
            ),
            responses_file=str(recording),
            max_retries=1,
        )
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, _ = await _run(wf, gw)
        assert status == WorkflowStatus.FAILED

    async def test_mock_fallback_returns_empty_json_when_schema_requested(
        self, tmp_path: Any
    ) -> None:
        # With a schema and no entry, the mock answers ``{}``, not its ``"Mock response"``.
        recording = tmp_path / "rec.json"
        recording.write_text(json.dumps({"_version": 2}))
        wf = _workflow_with_schema(
            schema=ResponseSchema(type="json_object"),
            responses_file=str(recording),
            max_retries=0,
        )
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        assert final["classification"] == {}

    async def test_state_holds_parsed_value_not_raw_string(self, tmp_path: Any) -> None:
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "classify": {
                        "content": '{"label": "feedback", "confidence": 0.5, "rationale": "ok"}',
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        "cost_usd": 0.0,
                        "latency_ms": 0,
                    },
                }
            )
        )
        wf = _workflow_with_schema(
            schema=ResponseSchema(type="json_object"),
            responses_file=str(recording),
        )
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        # ``json_object`` mode returns a dict (no Pydantic coercion).
        assert isinstance(final["classification"], dict)
        assert final["classification"]["label"] == "feedback"


class TestProviderResponseParsedField:
    """The provider response shape carries the parsed value through the gateway."""

    def test_parsed_defaults_to_none(self) -> None:
        # Free-form steps leave it ``None``, so consumers can tell structured output apart.
        r = ProviderResponse(content="hello", model="x", provider="y")
        assert r.parsed is None

    def test_parsed_accepts_arbitrary_value(self) -> None:
        # Typed ``Any``: a dict, a Pydantic instance or a list, as the schema declares.
        r = ProviderResponse(content="{}", model="x", provider="y", parsed={"a": 1})
        assert r.parsed == {"a": 1}


class TestStreamingStructuredOutput:
    """Streaming-mode structured output parses end-to-end after the stream closes."""

    async def test_stream_parses_accumulated_content(self, tmp_path: Any) -> None:
        # The mock's ``stream`` wraps ``complete``; the step parses once the stream closes.
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "classify": {
                        "content": '{"label": "complaint", "confidence": 0.3, "rationale": "x"}',
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        "cost_usd": 0.0,
                        "latency_ms": 0,
                    },
                }
            )
        )
        wf = _workflow_with_schema(
            schema=ResponseSchema(
                type="pydantic",
                model="tests.providers.test_structured_output._Classification",
            ),
            responses_file=str(recording),
        )
        wf.config.stream = True
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, final = await _run(wf, gw)
        assert status == WorkflowStatus.SUCCESS
        assert isinstance(final["classification"], _Classification)
        assert final["classification"].label == "complaint"

    async def test_stream_invalid_json_fails_step(self, tmp_path: Any) -> None:
        # No mid-stream retry: unparseable content fails the step at once.
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "classify": {
                        "content": "not json at all",
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        "cost_usd": 0.0,
                        "latency_ms": 0,
                    },
                }
            )
        )
        wf = _workflow_with_schema(
            schema=ResponseSchema(type="json_object"),
            responses_file=str(recording),
        )
        wf.config.stream = True
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, _ = await _run(wf, gw)
        assert status == WorkflowStatus.FAILED


class TestStructuredHelperEdgeCases:
    """Coverage for the defensive branches in :mod:`_structured`."""

    def test_schema_dict_for_pydantic_without_model_raises(self) -> None:
        # Direct callers, not only YAML-loaded workflows, get a clean error.
        from agentloom.core.models import ResponseSchemaConfigError

        rs = ResponseSchema.model_construct(type="pydantic", model=None)
        with pytest.raises(ResponseSchemaConfigError, match="dotted path"):
            schema_dict_for(rs, "step")

    def test_schema_dict_for_json_schema_without_schema_raises(self) -> None:
        from agentloom.core.models import ResponseSchemaConfigError

        rs = ResponseSchema.model_construct(type="json_schema", schema_=None)
        with pytest.raises(ResponseSchemaConfigError, match="inline 'schema' object"):
            schema_dict_for(rs, "step")

    def test_normalize_schema_walks_lists_and_nested_objects(self) -> None:
        # Objects nested in arrays get ``additionalProperties: false`` too: strict at every level.
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"x": {"type": "string"}}},
                    }
                },
            },
        )
        schema = schema_dict_for(rs, "step")
        assert schema is not None
        assert schema["additionalProperties"] is False
        assert schema["properties"]["items"]["items"]["additionalProperties"] is False

    def test_translate_for_google_passes_through_simple_schemas(self) -> None:
        rs = ResponseSchema(
            type="json_schema",
            schema_={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        )
        mime, schema = translate_for_google(rs, "step")
        assert mime == "application/json"
        assert schema is not None
        assert schema["type"] == "object"

    def test_translate_for_google_inlines_local_refs(self) -> None:
        """Gemini's dialect has no ``$defs`` or ``$ref``: a nested model is inlined."""
        item = {
            "type": "object",
            "description": "An item.",
            "properties": {"tag": {"$ref": "#/definitions/Tag"}},
        }
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
                    "main": {"$ref": "#/$defs/Item", "description": "The first one."},
                },
                "$defs": {"Item": item},
                "definitions": {"Tag": {"type": "string", "title": "Tag"}},
            },
        )
        _, schema = translate_for_google(rs, "step")
        inlined = {
            "type": "object",
            "description": "An item.",
            "properties": {"tag": {"type": "string"}},
        }
        assert schema == {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": inlined},
                "main": {**inlined, "description": "The first one."},
            },
        }

    def test_translate_for_google_decodes_escaped_names(self) -> None:
        """A JSON Pointer writes ``/`` as ``~1`` and ``~`` as ``~0``, in that order (RFC 6901)."""
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A~1B"}, "b": {"$ref": "#/$defs/x~01y"}},
                "$defs": {"A/B": {"type": "string"}, "x~1y": {"type": "integer"}},
            },
        )
        _, schema = translate_for_google(rs, "step")
        assert schema == {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }

    def test_translate_for_google_adds_the_constraints_beside_a_ref(self) -> None:
        """JSON Schema applies the keywords beside a ``$ref`` with its definition, not instead."""
        order = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {
                    "x": {
                        "$ref": "#/$defs/Order",
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code", "id"],
                    }
                },
                "$defs": {"Order": order},
            },
        )
        _, schema = translate_for_google(rs, "step")
        assert schema is not None
        assert schema["properties"]["x"] == {
            "type": "object",
            "properties": {"id": {"type": "string"}, "code": {"type": "string"}},
            "required": ["id", "code"],
        }

    @pytest.mark.parametrize(
        "beside",
        [{"properties": {"id": {"type": "integer"}}}, {"minProperties": 2}],
        ids=["a property", "a keyword"],
    )
    def test_translate_for_google_leaves_a_ref_its_neighbours_contradict(
        self, beside: dict[str, Any]
    ) -> None:
        """Both cannot hold in Gemini's dialect, which has no ``allOf``: it refuses the ``$ref``."""
        order = {"type": "object", "properties": {"id": {"type": "string"}}, "minProperties": 1}
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Order", **beside}},
                "$defs": {"Order": order},
            },
        )
        _, schema = translate_for_google(rs, "step")
        assert schema is not None
        assert schema["properties"]["x"] == {"$ref": "#/$defs/Order", **beside}

    def test_translate_for_google_leaves_a_recursive_ref(self) -> None:
        """A model that contains itself cannot be inlined: Gemini refuses it with a 400."""
        node = {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/Node"}},
                "$defs": {"Node": node},
            },
        )
        _, schema = translate_for_google(rs, "step")
        assert schema is not None
        assert schema["properties"]["root"]["properties"]["child"] == {"$ref": "#/$defs/Node"}
        assert "$defs" not in schema

    def test_translate_for_google_approximates_the_constraints_gemini_lacks(self) -> None:
        """Gemini 400s on these: what its dialect can say stays, local validation does the rest."""
        pair = [{"type": "string"}, {"type": "integer"}]
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "a"},
                    "one": {"type": "integer", "const": 1},
                    "n": {"type": "integer", "exclusiveMinimum": 0},
                    "m": {"type": "number", "exclusiveMaximum": 5, "maximum": 4},
                    "flag": {"type": "integer", "exclusiveMinimum": True},
                    "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                    "pair": {"type": "array", "prefixItems": pair, "minItems": 2, "maxItems": 2},
                    "const": {"type": "string"},
                },
            },
        )
        _, schema = translate_for_google(rs, "step")
        assert schema == {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["a"]},
                "one": {"type": "integer"},
                "n": {"type": "integer", "minimum": 0},
                "m": {"type": "number", "maximum": 4},
                "flag": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "pair": {"type": "array", "items": {"anyOf": pair}, "minItems": 2, "maxItems": 2},
                "const": {"type": "string"},
            },
        }

    def test_translate_for_google_json_object_returns_no_schema(self) -> None:
        rs = ResponseSchema(type="json_object")
        mime, schema = translate_for_google(rs, "step")
        assert mime == "application/json"
        assert schema is None

    def test_format_validation_feedback_non_pydantic_error(self) -> None:
        # A non-Pydantic error, such as a non-object reply's ``TypeError``, renders as its text.
        msg = format_validation_feedback(ValueError("not a dict"), "step")
        assert "not a dict" in msg
        assert "step" in msg

    def test_load_pydantic_model_with_empty_string_raises(self) -> None:
        # ``validate_parsed`` passes ``model or ""``, so an empty path must fail with a clean error.
        from agentloom.core.models import ResponseSchemaConfigError

        with pytest.raises(ResponseSchemaConfigError):
            load_pydantic_model("")


class _MixedRequired(BaseModel):
    """Helper for the OpenAI strict-required test (module-level for the loader)."""

    mandatory: str
    optional: str | None = None


class TestExtractParsedBalancedScanner:
    """The depth scanner finds the first balanced JSON object, where a greedy ``{.*}`` fails."""

    def test_multiple_json_objects_returns_first(self) -> None:
        # A greedy regex would fuse the two objects into one malformed span.
        assert extract_parsed('Here: {"a": 1} and: {"b": 2}') == {"a": 1}

    def test_prose_with_stray_brace_after_object(self) -> None:
        # A greedy regex would take the ``}`` in the prose as the close of the JSON.
        assert extract_parsed('{"x": 1} (note the }) here') == {"x": 1}

    def test_string_with_braces_does_not_unbalance(self) -> None:
        # The scanner tracks string state, so a ``}`` inside a string leaves the depth alone.
        assert extract_parsed('{"a": "use } carefully", "b": 2}') == {
            "a": "use } carefully",
            "b": 2,
        }

    def test_escaped_quote_inside_string(self) -> None:
        # The string-state tracker has to honour backslash escapes too.
        assert extract_parsed('{"a": "he said \\"hi\\"", "b": 1}') == {
            "a": 'he said "hi"',
            "b": 1,
        }

    def test_anthropic_prefill_with_trailing_prose(self) -> None:
        # What Anthropic returns after the reattached ``{``: the object's tail, then prose.
        text = '"label": "x", "confidence": 0.1, "rationale": "y"} Hope!'
        assert extract_parsed(text) == {
            "label": "x",
            "confidence": 0.1,
            "rationale": "y",
        }


class TestOpenAIStrictRequired:
    """OpenAI strict mode requires every property in ``required``."""

    def test_pydantic_schema_with_optional_field_gets_full_required(self) -> None:
        # Pydantic leaves an optional field out of ``required``; OpenAI strict mode 400s on that.
        rs = ResponseSchema(
            type="pydantic",
            model="tests.providers.test_structured_output._MixedRequired",
        )
        out = translate_for_openai(rs, "step")
        required = out["json_schema"]["schema"]["required"]
        assert set(required) == {"mandatory", "optional"}

    def test_non_strict_mode_preserves_original_required(self) -> None:
        # ``strict: false`` opts out of the strict rules: ``required`` stays as written.
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            },
            strict=False,
        )
        out = translate_for_openai(rs, "step")
        assert out["json_schema"]["schema"]["required"] == ["a"]
        assert out["json_schema"]["strict"] is False

    def test_nested_object_required_propagates(self) -> None:
        # The walker descends into nested objects, as a nested BaseModel field produces.
        rs = ResponseSchema(
            type="json_schema",
            schema_={
                "type": "object",
                "properties": {
                    "inner": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                        "required": ["a"],
                    }
                },
                "required": [],
            },
        )
        out = translate_for_openai(rs, "step")
        outer_required = out["json_schema"]["schema"]["required"]
        inner_required = out["json_schema"]["schema"]["properties"]["inner"]["required"]
        assert set(outer_required) == {"inner"}
        assert set(inner_required) == {"a", "b"}


class TestDeepCopyIsolation:
    """Schema translation does not leak its changes into Pydantic's class cache."""

    def test_pydantic_schema_cache_unaffected_by_translator(self) -> None:
        # ``model_json_schema()`` is cached on the class: a mutation would reach every later call.
        before = _Classification.model_json_schema()
        before_props = before.get("additionalProperties")
        rs = ResponseSchema(
            type="pydantic",
            model="tests.providers.test_structured_output._Classification",
        )
        _ = translate_for_openai(rs, "step")
        after = _Classification.model_json_schema()
        assert after.get("additionalProperties") == before_props

    def test_inline_schema_not_mutated(self) -> None:
        # The author's inline dict must not pick up runtime decorations either.
        author_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        rs = ResponseSchema(type="json_schema", schema_=author_schema)
        translate_for_openai(rs, "step")
        assert "additionalProperties" not in author_schema


class TestToolsAndResponseSchemaAreExclusive:
    """``tools`` and ``response_schema`` on one ``llm_call`` step are refused at parse time."""

    def test_both_tools_and_response_schema_rejected(self) -> None:
        from agentloom.core.models import ToolDefinition

        with pytest.raises(ValidationError, match="'tools' and 'response_schema'"):
            StepDefinition(
                id="conflicting",
                type=StepType.LLM_CALL,
                prompt="hi",
                tools=[ToolDefinition(name="t", description="", parameters={"type": "object"})],
                response_schema=ResponseSchema(type="json_object"),
            )

    def test_tools_alone_is_fine(self) -> None:
        from agentloom.core.models import ToolDefinition

        StepDefinition(
            id="tools-only",
            type=StepType.LLM_CALL,
            prompt="hi",
            tools=[ToolDefinition(name="t", description="", parameters={"type": "object"})],
        )

    def test_response_schema_alone_is_fine(self) -> None:
        StepDefinition(
            id="schema-only",
            type=StepType.LLM_CALL,
            prompt="hi",
            response_schema=ResponseSchema(type="json_object"),
        )


class TestProviderStreamingPropagation:
    """Streaming adapters must translate ``agentloom_response_schema`` on stream()."""

    @respx.mock
    async def test_openai_stream_forwards_response_format(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            body = 'data: {"choices":[{"delta":{"content":"{\\"x\\":1}"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=body.encode())

        respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_handler)
        provider = OpenAIProvider(api_key="sk-test")
        rs = ResponseSchema(type="json_object")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            agentloom_response_schema=rs,
            agentloom_step_id="s",
        )
        async for _ in sr:
            pass
        assert captured["payload"]["response_format"] == {"type": "json_object"}
        await provider.close()

    @respx.mock
    async def test_anthropic_stream_prefills_and_forwards_schema(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            body = (
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","delta":{"text":"\\"x\\":1}"}}\n\n'
                "event: message_delta\n"
                'data: {"type":"message_delta","delta":'
                '{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
            )
            return httpx.Response(200, content=body.encode())

        respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=_handler)
        provider = AnthropicProvider(api_key="anth-test")
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-3-5-sonnet-20240620",
            agentloom_response_schema=rs,
            agentloom_step_id="s",
        )
        chunks: list[str] = []
        async for chunk in sr:
            chunks.append(chunk)
        # First chunk gets the prefilled ``{`` reattached.
        assert chunks[0].startswith("{")
        # Payload includes the prefill assistant turn and the JSON-only system prompt.
        msgs = captured["payload"]["messages"]
        assert msgs[-1] == {"role": "assistant", "content": "{"}
        assert "JSON" in captured["payload"]["system"]
        await provider.close()

    @respx.mock
    async def test_google_stream_forwards_generation_config_schema(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            body = (
                'data: {"candidates":[{"content":{"parts":[{"text":"{\\"x\\":1}"}]},'
                '"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,'
                '"candidatesTokenCount":2,"totalTokenCount":3}}\n\n'
            )
            return httpx.Response(200, content=body.encode())

        respx.post(url__regex=r".+streamGenerateContent.+").mock(side_effect=_handler)
        provider = GoogleProvider(api_key="g-test")
        rs = ResponseSchema(
            type="json_schema",
            schema_={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        )
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-1.5-flash",
            agentloom_response_schema=rs,
            agentloom_step_id="s",
        )
        async for _ in sr:
            pass
        gen_cfg = captured["payload"]["generationConfig"]
        assert gen_cfg["responseMimeType"] == "application/json"
        assert "responseSchema" in gen_cfg
        await provider.close()

    @respx.mock
    async def test_ollama_stream_forwards_format(self) -> None:
        captured: dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            body = (
                '{"message":{"role":"assistant","content":"{}"}, "done":false}\n'
                '{"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":2}\n'
            )
            return httpx.Response(200, content=body.encode())

        respx.post("http://localhost:11434/api/chat").mock(side_effect=_handler)
        provider = OllamaProvider(base_url="http://localhost:11434")
        rs = ResponseSchema(type="json_object")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3.1",
            agentloom_response_schema=rs,
            agentloom_step_id="s",
        )
        async for _ in sr:
            pass
        assert captured["payload"]["format"] == "json"
        await provider.close()


class TestExtractParsedEdges:
    """Balanced-brace scanner branches not covered elsewhere."""

    def test_scan_ignores_escaped_quote_inside_string(self) -> None:
        # The ``}`` after ``\"`` is still in the string; the prose prefix forces the scanner path.
        assert extract_parsed(r'note: {"key": "he said \" then }", "n": 1}') == {
            "key": 'he said " then }',
            "n": 1,
        }

    def test_scan_handles_backslash_before_regular_char(self) -> None:
        # A literal backslash-n: the escape flag is set on ``\`` and cleared on ``n``.
        assert extract_parsed(r'note: {"k": "a\nb"}') == {"k": "a\nb"}

    def test_stray_close_and_prefill_candidate_fallthrough(self) -> None:
        # With the prefilled ``{`` the first span is ``{some prose }``, which does not parse;
        # the plain scan then skips the ``}`` at depth zero and finds the real object.
        assert extract_parsed('some prose } and then {"good": 1}') == {"good": 1}

    def test_code_fence_with_malformed_json_falls_through(self) -> None:
        # A fence with malformed JSON falls through to the scanner instead of raising there.
        text = '```json\n{"bad": nope}\n```\nreal: {"good": 1}'
        # The scanner's first balanced span is the same ``{"bad": nope}``, so it still raises.
        with pytest.raises(json.JSONDecodeError):
            extract_parsed(text)

    def test_code_fence_with_nested_object_matches_outer_span(self) -> None:
        # The lazy fence regex backtracks to the closing fence, so the nested object parses there.
        assert extract_parsed('```json {"a": {"b": 1}} ```') == {"a": {"b": 1}}


class TestStructuredHelpersMisc:
    """Remaining single-line branches in :mod:`_structured`."""

    def test_schema_name_explicit_wins(self) -> None:
        from agentloom.steps._structured import schema_name_for

        rs = ResponseSchema(type="json_object", name="ExplicitName")
        assert schema_name_for(rs, "step") == "ExplicitName"

    def test_openai_translation_carries_description(self) -> None:
        rs = ResponseSchema(
            type="json_schema",
            schema_={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            description="Free-form doc",
        )
        out = translate_for_openai(rs, "step")
        assert out["json_schema"]["description"] == "Free-form doc"

    def test_anthropic_prefix_json_object_mode_has_no_schema(self) -> None:
        from agentloom.steps._structured import anthropic_system_prefix

        rs = ResponseSchema(type="json_object")
        prefix = anthropic_system_prefix(rs, "step")
        assert "JSON object only" in prefix
        assert "```json" not in prefix

    def test_ollama_translation_pydantic_returns_schema(self) -> None:
        rs = ResponseSchema(
            type="pydantic", model="tests.providers.test_structured_output._Classification"
        )
        out = translate_for_ollama(rs, "step")
        assert isinstance(out, dict)
        assert out["type"] == "object"


class TestStreamingValidationErrorPaths:
    """Streaming end-of-stream validation surfaces failures uniformly."""

    async def test_stream_response_schema_config_error_fails_step(self, tmp_path: Any) -> None:
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "classify": {
                        "content": '{"any": "shape"}',
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                        "cost_usd": 0.0,
                        "latency_ms": 0,
                    },
                }
            )
        )
        wf = _workflow_with_schema(
            schema=ResponseSchema(
                type="pydantic",
                # Importable, but not a Pydantic class: validation raises ResponseSchemaConfigError.
                model="collections:OrderedDict",
            ),
            responses_file=str(recording),
        )
        wf.config.stream = True
        gw = ProviderGateway()
        gw.register(MockProvider(responses_file=recording, workflow_name=wf.name), priority=0)
        status, _ = await _run(wf, gw)
        assert status == WorkflowStatus.FAILED


class TestCLISysPathPrepend:
    """CLI prepends CWD so dotted-path Pydantic models resolve from the repo root."""

    def test_cli_run_inserts_cwd_at_index_zero(self, tmp_path: Any, monkeypatch: Any) -> None:
        # A schema module shipped next to the YAML must import: the cwd leads ``sys.path``.
        import sys

        from typer.testing import CliRunner

        from agentloom.cli.main import app

        wf = tmp_path / "wf.yaml"
        wf.write_text(
            "name: cwd-path-test\n"
            "config: {provider: mock, model: gpt-4o-mini}\n"
            "steps:\n"
            "  - {id: s, type: llm_call, prompt: hi}\n"
        )
        monkeypatch.chdir(tmp_path)
        target = str(tmp_path.resolve())
        if target in sys.path:
            sys.path.remove(target)
        runner = CliRunner()
        result = runner.invoke(app, ["run", str(wf), "--lite"])
        assert result.exit_code == 0
        assert sys.path[0] == target
        runner.invoke(app, ["run", str(wf), "--lite"])
        assert sys.path.count(target) == 1


class TestResponseSchemaCompanionFieldRefusal:
    """Companion fields must match the mode."""

    def test_json_schema_mode_rejects_stale_model(self) -> None:
        with pytest.raises(ValidationError, match="must not set 'model'"):
            ResponseSchema(
                type="json_schema",
                model="some.path.Thing",
                schema_={"type": "object"},
            )

    def test_json_object_mode_rejects_model(self) -> None:
        with pytest.raises(ValidationError, match="must not set 'model' or 'schema'"):
            ResponseSchema(type="json_object", model="some.path.Thing")

    def test_json_object_mode_rejects_schema(self) -> None:
        with pytest.raises(ValidationError, match="must not set 'model' or 'schema'"):
            ResponseSchema(type="json_object", schema_={"type": "object"})
