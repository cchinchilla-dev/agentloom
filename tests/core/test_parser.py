"""Tests for the WorkflowParser module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError as PydanticValidationError

from agentloom.core.models import WorkflowConfig, WorkflowDefinition
from agentloom.core.parser import WorkflowParser
from agentloom.exceptions import ValidationError

SIMPLE_YAML = """
name: yaml-test
version: "1.0"
config:
  provider: mock
  model: mock-model
state:
  question: "What is Python?"
steps:
  - id: answer
    type: llm_call
    prompt: "Answer: {state.question}"
    output: answer
"""

INVALID_YAML_CYCLE = """
name: cycle-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
    depends_on: [b]
  - id: b
    type: llm_call
    prompt: "b"
    depends_on: [a]
"""

INVALID_YAML_MISSING_REF = """
name: missing-ref-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
    depends_on: [nonexistent]
"""


class TestYAMLParsing:
    """Test parsing valid YAML into WorkflowDefinition."""

    def test_parse_simple_yaml(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert workflow.name == "yaml-test"

    def test_parse_simple_yaml_steps(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert len(workflow.steps) == 1
        assert workflow.steps[0].id == "answer"

    def test_parse_simple_yaml_config(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert workflow.config.provider == "mock"
        assert workflow.config.model == "mock-model"

    def test_parse_simple_yaml_state(self) -> None:
        workflow = WorkflowParser.from_yaml(SIMPLE_YAML)
        assert workflow.state["question"] == "What is Python?"

    def test_parse_from_dict(self) -> None:
        data = {
            "name": "dict-test",
            "steps": [{"id": "s1", "type": "llm_call", "prompt": "test"}],
        }
        workflow = WorkflowParser.from_dict(data)
        assert workflow.name == "dict-test"


class TestValidationErrors:
    """Test that invalid YAML raises proper errors."""

    def test_invalid_yaml_syntax(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml("invalid: yaml: content: [")

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml("name: test\n")

    def test_non_mapping_yaml(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml("- just\n- a\n- list\n")

    def test_invalid_step_type(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(
                "name: t\nsteps:\n  - id: x\n    type: invalid\n    prompt: y\n"
            )


class TestMissingReferences:
    def test_missing_depends_on_reference(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(INVALID_YAML_MISSING_REF)

    def test_valid_depends_on_reference(self) -> None:
        yaml_str = """
name: valid-ref
steps:
  - id: a
    type: llm_call
    prompt: "a"
  - id: b
    type: llm_call
    prompt: "b"
    depends_on: [a]
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        assert len(workflow.steps) == 2


class TestDAGBuild:
    def test_build_dag_simple(self) -> None:
        yaml_str = """
name: dag-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
  - id: b
    type: llm_call
    prompt: "b"
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        dag = WorkflowParser.build_dag(workflow)
        assert set(dag.nodes) == {"a", "b"}

    def test_build_dag_with_deps(self) -> None:
        yaml_str = """
name: dag-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
  - id: b
    type: llm_call
    prompt: "b"
    depends_on: [a]
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        dag = WorkflowParser.build_dag(workflow)
        assert dag.predecessors("b") == {"a"}

    def test_build_dag_cycle_raises_error(self) -> None:
        """A cycle fails at the reference or the DAG validation stage."""
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(INVALID_YAML_CYCLE)


class TestParseTimeInvariants:
    """Parse-time checks on step ids, ``max_concurrent_steps`` and shared ``output:`` keys.

    A duplicate id would shadow the other step, and parallel steps sharing an ``output:`` key
    would lose writes to last-writer-wins.
    """

    def test_rejects_duplicate_step_ids(self) -> None:
        yaml_str = """
name: dup
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: first}
  - {id: a, type: llm_call, prompt: second}
"""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowParser.from_yaml(yaml_str)
        msg = str(exc_info.value)
        # The id and both indices let the author find the two copies.
        assert "a" in msg
        assert "[0, 1]" in msg

    @pytest.mark.parametrize(
        "yaml_str",
        [
            "name: w\nconfig: {provider: mock, model: x, max_retries: -1}\n"
            "steps:\n  - {id: a, type: llm_call, prompt: x}\n",
            "name: w\nconfig: {provider: mock, model: x}\n"
            "steps:\n  - {id: a, type: llm_call, prompt: x, retry: {max_retries: -1}}\n",
        ],
        ids=["workflow", "step"],
    )
    def test_rejects_negative_max_retries(self, yaml_str: str) -> None:
        # A negative count would run the step zero times
        with pytest.raises(ValidationError, match="max_retries"):
            WorkflowParser.from_yaml(yaml_str)

    def test_shared_step_keeps_each_workflow_max_retries(self) -> None:
        from agentloom.core.models import StepDefinition, WorkflowConfig, WorkflowDefinition

        step = StepDefinition(id="a", type="llm_call", prompt="x")
        none = WorkflowDefinition(name="w0", config=WorkflowConfig(max_retries=0), steps=[step])
        two = WorkflowDefinition(name="w2", config=WorkflowConfig(max_retries=2), steps=[step])
        assert none.steps[0].retry.max_retries == 0
        assert two.steps[0].retry.max_retries == 2
        assert "max_retries" not in step.retry.model_fields_set

    @pytest.mark.parametrize("bad", [0, -1, -100, 9999999])
    def test_rejects_invalid_max_concurrent_steps(self, bad: int) -> None:
        yaml_str = f"""
name: bad-concurrent
config:
  provider: mock
  model: x
  max_concurrent_steps: {bad}
steps:
  - {{id: a, type: llm_call, prompt: x}}
"""
        with pytest.raises(ValidationError):
            WorkflowParser.from_yaml(yaml_str)

    def test_accepts_boundary_max_concurrent_steps_1(self) -> None:
        yaml_str = """
name: boundary
config: {provider: mock, model: x, max_concurrent_steps: 1}
steps:
  - {id: a, type: llm_call, prompt: x}
"""
        wf = WorkflowParser.from_yaml(yaml_str)
        assert wf.config.max_concurrent_steps == 1

    def test_warns_on_parallel_steps_sharing_output_key(self) -> None:
        import warnings

        yaml_str = """
name: dup-output
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared}
"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WorkflowParser.from_yaml(yaml_str)
            messages = [str(w.message) for w in caught]
        assert any("shared" in m and "'a'" in m and "'b'" in m for m in messages)

    def test_no_warning_when_owners_chain_via_depends_on(self) -> None:
        """Sequential overwrite is intentional — no warning."""
        import warnings

        yaml_str = """
name: seq
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared, depends_on: [a]}
"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WorkflowParser.from_yaml(yaml_str)
            messages = [str(w.message) for w in caught if "shared" in str(w.message)]
        assert messages == []

    def test_strict_outputs_promotes_warning_to_error(self) -> None:
        yaml_str = """
name: strict
config: {provider: mock, model: x, strict_outputs: true}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared}
"""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowParser.from_yaml(yaml_str)
        assert "shared" in str(exc_info.value)

    def test_warns_on_partial_chain_owners(self) -> None:
        """Three owners with a partial dependency chain still collide.

        ``b`` depends on ``a`` but can run alongside ``c``, so the warning names ``b`` and ``c``
        even though ``b`` has ``a`` in its ancestry.
        """
        import warnings

        yaml_str = """
name: partial-chain
config: {provider: mock, model: x}
steps:
  - {id: a, type: llm_call, prompt: x, output: shared}
  - {id: b, type: llm_call, prompt: x, output: shared, depends_on: [a]}
  - {id: c, type: llm_call, prompt: x, output: shared}
"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            WorkflowParser.from_yaml(yaml_str)
            messages = [str(w.message) for w in caught if "shared" in str(w.message)]
        assert any("'b'" in m and "'c'" in m for m in messages)


class TestStrictExtraKeys:
    """``StepDefinition`` and ``WorkflowConfig`` reject unknown keys at parse time.

    A typo fails ``agentloom validate`` with the field named instead of being dropped silently.
    """

    def test_step_definition_rejects_unknown_key(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from agentloom.core.models import StepDefinition, StepType

        with pytest.raises(PydanticValidationError, match="workflow"):
            # ``workflow:`` is a typo for ``workflow_inline:``.
            StepDefinition.model_validate(
                {"id": "s", "type": StepType.SUBWORKFLOW, "workflow": {"name": "child"}}
            )

    def test_workflow_config_rejects_unknown_key(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from agentloom.core.models import WorkflowConfig

        with pytest.raises(PydanticValidationError, match="responses"):
            # Mock responses come from ``responses_file:``; an inline ``responses:`` is refused.
            WorkflowConfig.model_validate({"provider": "mock", "responses": ["a", "b"]})

    @pytest.mark.parametrize(
        ("field", "entry", "typo"),
        [
            ("tools", {"name": "t", "paramters": {}}, "paramters"),
            ("attachments", {"source": "a.png", "media_typ": "image/png"}, "media_typ"),
        ],
    )
    def test_contract_blocks_reject_unknown_keys(
        self, field: str, entry: dict[str, Any], typo: str
    ) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from agentloom.core.models import StepDefinition, StepType

        with pytest.raises(PydanticValidationError, match=typo):
            StepDefinition.model_validate(
                {"id": "s", "type": StepType.LLM_CALL, "prompt": "hi", field: [entry]}
            )

    @pytest.mark.parametrize(
        ("field", "entry"),
        [
            ("tools", {"name": "t", "description": "d", "parameters": {"type": "object"}}),
            (
                "attachments",
                {"type": "image", "source": "a.png", "media_type": "image/png", "fetch": "local"},
            ),
        ],
    )
    def test_contract_blocks_accept_every_known_key(
        self, field: str, entry: dict[str, Any]
    ) -> None:
        from agentloom.core.models import StepDefinition, StepType

        step = StepDefinition.model_validate(
            {"id": "s", "type": StepType.LLM_CALL, "prompt": "hi", field: [entry]}
        )
        assert len(getattr(step, field)) == 1

    def test_the_contract_models_still_ignore_unknown_keys(self) -> None:
        # Stable contracts: only a step's YAML is held to the stricter rule
        from agentloom.core.models import Attachment, ToolDefinition

        assert ToolDefinition.model_validate({"name": "t", "extra": 1}).name == "t"
        assert Attachment.model_validate({"source": "a", "extra": 1}).source == "a"

    def test_built_contract_instances_pass(self) -> None:
        from agentloom.core.models import StepDefinition, StepType, ToolDefinition

        step = StepDefinition(
            id="s", type=StepType.LLM_CALL, prompt="hi", tools=[ToolDefinition(name="t")]
        )
        assert step.tools[0].name == "t"

    def test_workflow_yaml_with_typo_rejected_at_parse(self) -> None:
        yaml = """\
name: typo-test
config:
  provider: mock
  model: mock-model
steps:
  - id: sub
    type: subworkflow
    workflow:
      name: child
"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            # from_yaml wraps Pydantic and parse failures in agentloom's own ``ValidationError``.
            with pytest.raises(ValidationError):
                WorkflowParser.from_yaml(f.name)


def _workflow(*steps: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {"name": "w", "steps": list(steps), **top}


NOTE = {"id": "note", "type": "tool", "tool_name": "t"}


class TestRequiredFieldsByType:
    """A step missing what its type cannot run without fails at load, not when it runs."""

    @pytest.mark.parametrize(
        ("step", "named"),
        [
            ({"type": "router"}, "'conditions' or 'default'"),
            ({"type": "router", "conditions": []}, "'conditions' or 'default'"),
            ({"type": "tool"}, "'tool_name'"),
            ({"type": "subworkflow"}, "'workflow_path' or 'workflow_inline'"),
            (
                {"type": "subworkflow", "workflow_inline": {}},
                "'workflow_path' or 'workflow_inline'",
            ),
            ({"type": "embed"}, "'inputs'"),
            ({"type": "llm_call"}, "'prompt' or 'conversation'"),
            ({"type": "llm_call", "prompt": ""}, "'prompt' or 'conversation'"),
        ],
    )
    def test_refused_at_load(self, step: dict[str, Any], named: str) -> None:
        with pytest.raises(ValidationError, match=named):
            WorkflowParser.from_dict(_workflow({"id": "s", **step}))

    @pytest.mark.parametrize(
        "step",
        [
            {"type": "router", "default": "note"},
            {"type": "router", "conditions": [{"expression": "state.x", "target": "note"}]},
            {"type": "tool", "tool_name": "t"},
            {"type": "embed", "inputs": "state.docs"},
            {"type": "llm_call", "prompt": "p"},
            {"type": "llm_call", "conversation": "state.chat"},
            {"type": "approval_gate"},
        ],
    )
    def test_one_of_the_alternatives_is_enough(self, step: dict[str, Any]) -> None:
        assert WorkflowParser.from_dict(_workflow({"id": "s", **step}, NOTE)).get_step("s")

    def test_inline_subworkflow_is_enough(self) -> None:
        inline = {"name": "c", "steps": [{"id": "c", "type": "llm_call", "prompt": "p"}]}
        step = {"id": "s", "type": "subworkflow", "workflow_inline": inline}
        assert WorkflowParser.from_dict(_workflow(step)).steps[0].workflow_inline == inline

    def test_a_field_of_another_type_does_not_count(self) -> None:
        with pytest.raises(ValidationError, match="'tool_name'"):
            WorkflowParser.from_dict(_workflow({"id": "s", "type": "tool", "prompt": "p"}))


class TestNestedUnknownKeys:
    """A misspelled key in a nested block stops the load and is named."""

    @pytest.mark.parametrize(
        ("typo", "workflow"),
        [
            ("max_retry", _workflow({**NOTE, "retry": {"max_retry": 1}})),
            ("allowed_domain", _workflow(NOTE, config={"sandbox": {"allowed_domain": ["a.b"]}})),
            (
                "header",
                _workflow(
                    {"id": "g", "type": "approval_gate", "notify": {"url": "u", "header": {}}}
                ),
            ),
            (
                "enable",
                _workflow(
                    {"id": "s", "type": "llm_call", "prompt": "p", "thinking": {"enable": True}}
                ),
            ),
            (
                "priority",
                _workflow(
                    NOTE,
                    {
                        "id": "r",
                        "type": "router",
                        "conditions": [{"expression": "x", "target": "note", "priority": 1}],
                    },
                ),
            ),
            (
                "kind",
                _workflow(
                    {
                        "id": "s",
                        "type": "llm_call",
                        "prompt": "p",
                        "tool_choice": {"name": "t", "kind": "function"},
                    }
                ),
            ),
            (
                "strictt",
                _workflow(
                    {
                        "id": "s",
                        "type": "llm_call",
                        "prompt": "p",
                        "response_schema": {"type": "json_object", "strictt": True},
                    }
                ),
            ),
            ("redacted", _workflow(NOTE, state_schema={"api_key": {"redacted": True}})),
        ],
        ids=[
            "retry",
            "sandbox",
            "notify",
            "thinking",
            "condition",
            "tool_choice",
            "schema",
            "state",
        ],
    )
    def test_refused_and_named(self, typo: str, workflow: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match=typo):
            WorkflowParser.from_dict(workflow)

    def test_a_dumped_workflow_still_loads(self) -> None:
        # What a checkpoint stores: every field of every nested model, ``schema_`` by field name.
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        step = {
            "id": "s",
            "type": "llm_call",
            "prompt": "p",
            "response_schema": {"type": "json_schema", "schema": schema},
            "thinking": {"enabled": True},
            "tool_choice": "auto",
        }
        wf = WorkflowParser.from_dict(_workflow(step, state_schema={"k": {"redact": True}}))
        assert WorkflowDefinition.model_validate(wf.model_dump()) == wf


class TestFromYamlSource:
    """``from_yaml`` tells a path from YAML text."""

    @pytest.mark.parametrize("missing", ["workflwo.yaml", "some/dir/missing.yml", "no-such-file"])
    def test_a_missing_path_is_reported_as_missing(
        self, missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValidationError, match=f"file not found: {missing}"):
            WorkflowParser.from_yaml(missing)

    def test_a_path_object_is_a_path_even_with_a_colon(self, tmp_path: Path) -> None:
        missing = tmp_path / "a:b.yaml"
        with pytest.raises(ValidationError, match="file not found"):
            WorkflowParser.from_yaml(missing)

    def test_a_directory_is_not_a_workflow_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="file not found"):
            WorkflowParser.from_yaml(tmp_path)

    def test_an_existing_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "w.yaml"
        path.write_text(SIMPLE_YAML, encoding="utf-8")
        assert WorkflowParser.from_yaml(str(path)).name == "yaml-test"

    def test_a_line_over_the_path_limit_is_yaml_text(self) -> None:
        name = "n" * 300
        text = f"{{name: {name}, steps: [{{id: a, type: llm_call, prompt: hi}}]}}"
        assert WorkflowParser.from_yaml(text).name == name

    def test_a_long_scalar_is_yaml_text_not_a_missing_file(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            WorkflowParser.from_yaml("n" * 5000)

    @pytest.mark.parametrize("text", ["", "   \n", "just\nthree\nwords\n"])
    def test_text_that_is_no_path_keeps_the_mapping_error(self, text: str) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            WorkflowParser.from_yaml(text)


def _loaded(config: dict[str, Any], step: dict[str, Any] | None = None) -> WorkflowDefinition:
    llm_call = {"id": "s", "type": "llm_call", "prompt": "hi", **(step or {})}
    return WorkflowDefinition.model_validate({"name": "w", "config": config, "steps": [llm_call]})


class TestDefaultModelFollowsTheProvider:
    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-haiku-4-5-20251001"),
            ("google", "gemini-2.5-flash"),
            ("mock", "gpt-4o-mini"),
            ("unknown-name", "gpt-4o-mini"),
        ],
    )
    def test_without_a_model_the_provider_gives_it(self, provider: str, model: str) -> None:
        assert _loaded({"provider": provider}).config.model == model

    def test_no_config_at_all_is_openai(self) -> None:
        assert _loaded({}).config.model == "gpt-4o-mini"

    def test_a_named_model_wins(self) -> None:
        assert _loaded({"provider": "anthropic", "model": "claude-x"}).config.model == "claude-x"

    @pytest.mark.parametrize(
        "config",
        [{"provider": "ollama"}, {"provider": "generic_openai", "base_url": "http://x"}],
    )
    def test_a_provider_without_a_default_must_name_one(self, config: dict[str, Any]) -> None:
        with pytest.raises(PydanticValidationError, match=r"no default model.*config\.model"):
            _loaded(config)

    def test_a_provider_without_a_default_loads_with_a_model(self) -> None:
        assert _loaded({"provider": "ollama", "model": "phi4"}).config.model == "phi4"

    def test_switching_provider_moves_a_model_the_file_did_not_set(self) -> None:
        config = WorkflowConfig()
        config.use_provider("anthropic")
        assert (config.provider, config.model) == ("anthropic", "claude-haiku-4-5-20251001")

    def test_switching_provider_keeps_a_model_the_file_set(self) -> None:
        config = WorkflowConfig(model="gpt-4o")
        config.use_provider("anthropic")
        assert (config.provider, config.model) == ("anthropic", "gpt-4o")

    def test_switching_provider_drops_the_endpoint_of_generic_openai(self) -> None:
        config = WorkflowConfig(
            provider="generic_openai", model="m", base_url="http://x", api_key_env="K"
        )
        config.use_provider("openai")
        assert (config.base_url, config.api_key_env) == (None, None)
        WorkflowConfig.model_validate(config.model_dump())

    def test_switching_to_generic_openai_keeps_the_endpoint(self) -> None:
        config = WorkflowConfig(provider="generic_openai", model="m", base_url="http://x")
        config.use_provider("generic_openai")
        assert config.base_url == "http://x"

    def test_switching_to_a_provider_without_a_default_needs_a_model(self) -> None:
        with pytest.raises(ValueError, match="no default model"):
            _loaded({}).use_provider("ollama")

    def test_switching_is_free_when_every_step_names_its_model(self) -> None:
        workflow = _loaded({}, {"model": "phi4"})
        workflow.use_provider("ollama")
        assert workflow.config.provider == "ollama"

    @pytest.mark.parametrize(
        "config",
        [{"provider": "ollama"}, {"provider": "generic_openai", "base_url": "http://x"}],
    )
    @pytest.mark.parametrize(
        "step",
        [
            {"id": "s", "type": "llm_call", "prompt": "hi", "model": "phi4"},
            {"id": "s", "type": "embed", "inputs": "state.d", "model": "phi4"},
            {"id": "s", "type": "agent", "prompt": "hi", "agent": {"name": "a", "model": "phi4"}},
            {"id": "s", "type": "tool", "tool_name": "t"},
        ],
    )
    def test_no_step_on_the_config_model_needs_no_default(
        self, config: dict[str, Any], step: dict[str, Any]
    ) -> None:
        data = {"name": "w", "config": config, "steps": [step]}
        assert WorkflowDefinition.model_validate(data).config.provider == config["provider"]

    @pytest.mark.parametrize(
        "step",
        [
            {"id": "s", "type": "embed", "inputs": "state.d"},
            {"id": "s", "type": "agent", "prompt": "hi", "agent": {"name": "a"}},
        ],
    )
    def test_a_step_on_the_config_model_still_needs_one(self, step: dict[str, Any]) -> None:
        data = {"name": "w", "config": {"provider": "ollama"}, "steps": [step]}
        with pytest.raises(PydanticValidationError, match="no default model"):
            WorkflowDefinition.model_validate(data)

    def test_a_dumped_config_keeps_its_model(self) -> None:
        dumped = WorkflowConfig(provider="anthropic").model_dump()
        assert WorkflowConfig.model_validate(dumped).model == "claude-haiku-4-5-20251001"


class TestAnthropicThinkingBudget:
    THINKING: ClassVar[dict[str, Any]] = {"enabled": True, "budget_tokens": 4000}

    @pytest.mark.parametrize("max_tokens", [4000, 3999])
    def test_max_tokens_at_or_below_the_budget_fails_at_load(self, max_tokens: int) -> None:
        with pytest.raises(PydanticValidationError, match="max_tokens.*budget_tokens"):
            _loaded(
                {"model": "claude-sonnet-4-5"},
                {"max_tokens": max_tokens, "thinking": self.THINKING},
            )

    def test_a_step_that_names_the_model_counts(self) -> None:
        with pytest.raises(PydanticValidationError, match="max_tokens"):
            _loaded(
                {"model": "gpt-4o"},
                {"model": "claude-x", "max_tokens": 100, "thinking": self.THINKING},
            )

    def test_above_the_budget_loads(self) -> None:
        _loaded({"model": "claude-sonnet-4-5"}, {"max_tokens": 4001, "thinking": self.THINKING})

    def test_unset_max_tokens_loads(self) -> None:
        _loaded({"model": "claude-sonnet-4-5"}, {"thinking": self.THINKING})

    @pytest.mark.parametrize("thinking", [{"enabled": True}, {"enabled": True, "level": "high"}])
    def test_no_budget_on_a_claude_model_fails_at_load(self, thinking: dict[str, Any]) -> None:
        with pytest.raises(PydanticValidationError, match="needs budget_tokens"):
            _loaded({"model": "claude-sonnet-4-5"}, {"thinking": thinking})

    def test_no_budget_on_the_default_model_of_anthropic_fails_at_load(self) -> None:
        with pytest.raises(PydanticValidationError, match="needs budget_tokens"):
            _loaded({"provider": "anthropic"}, {"thinking": {"enabled": True, "level": "high"}})

    def test_no_budget_elsewhere_loads(self) -> None:
        _loaded({"model": "o3"}, {"thinking": {"enabled": True, "level": "high"}})

    def test_no_budget_with_thinking_turned_off_loads(self) -> None:
        _loaded({"model": "claude-sonnet-4-5"}, {"thinking": {"enabled": False}})

    def test_another_provider_is_not_held_to_it(self) -> None:
        _loaded({"model": "gemini-2.5-flash"}, {"max_tokens": 100, "thinking": self.THINKING})

    def test_thinking_turned_off_is_not_held_to_it(self) -> None:
        _loaded(
            {"model": "claude-sonnet-4-5"},
            {"max_tokens": 100, "thinking": {"enabled": False, "budget_tokens": 4000}},
        )
