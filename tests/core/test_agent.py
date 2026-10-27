"""Tests for agent definitions and their loading (#120)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentloom.core.agent import load_agent
from agentloom.core.models import AgentDefinition, AgentTool, StepDefinition
from agentloom.core.parser import WorkflowParser
from agentloom.exceptions import ValidationError

SUPPORT = """
name: support
description: Resolves billing questions.
model: gpt-4o-mini
system_prompt: You are a support agent.
tools:
  - http_request
  - name: escalate
    agent: agents/specialist.yaml
memory:
  token_budget: 8000
  trim_policy: summarize_oldest
loop_policy:
  max_iterations: 4
  on_max_iterations: warn_and_return
output_schema:
  type: json_object
constraints:
  max_cost_usd: 0.5
  max_latency_ms: 30000
  forbidden_patterns: ["As a language model"]
"""

SPECIALIST = """
name: specialist
description: Answers what support cannot.
"""


@pytest.fixture
def agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An ``agents/`` directory with the two definitions, as the working directory's."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(SUPPORT)
    (tmp_path / "agents" / "specialist.yaml").write_text(SPECIALIST)
    return tmp_path / "agents"


def _step(agent: object, **fields: object) -> dict[str, object]:
    return {"id": "s", "type": "agent", "agent": agent, "prompt": "hi", **fields}


def _parse(*steps: dict[str, object]) -> list[StepDefinition]:
    return WorkflowParser.from_dict({"name": "w", "steps": list(steps)}).steps


class TestLoading:
    def test_agent_loads_from_yaml(self, agents: Path) -> None:
        agent = load_agent("agents/support.yaml")
        assert agent.name == "support"
        assert agent.tool_names() == ["http_request", "escalate"]
        nested = agent.tools[1]
        assert isinstance(nested, AgentTool)
        assert nested.agent.name == "specialist"
        assert nested.description is None
        assert agent.memory is not None
        assert (agent.memory.token_budget, agent.memory.trim_policy) == (8000, "summarize_oldest")
        assert agent.loop_policy.max_iterations == 4
        assert agent.loop_policy.on_max_iterations == "warn_and_return"
        assert agent.output_schema is not None and agent.output_schema.type == "json_object"
        assert agent.constraints.max_cost_usd == 0.5
        assert agent.constraints.forbidden_patterns == ["As a language model"]

    def test_defaults(self) -> None:
        agent = AgentDefinition(name="bare")
        assert agent.tools == [] and agent.memory is None and agent.output_schema is None
        assert (agent.loop_policy.max_iterations, agent.loop_policy.on_max_iterations) == (
            5,
            "error",
        )
        assert agent.constraints.max_cost_usd is None
        assert agent.constraints.max_latency_ms is None

    def test_step_takes_a_path_or_the_definition_inline(self, agents: Path) -> None:
        by_path, inline = _parse(
            _step("agents/support.yaml"),
            {
                **_step(
                    {
                        "name": "inline",
                        "tools": [{"name": "ask", "agent": "agents/specialist.yaml"}],
                    }
                ),
                "id": "t",
            },
        )
        assert by_path.agent == load_agent(agents / "support.yaml")
        assert inline.agent is not None
        assert inline.agent.name == "inline"
        assert inline.agent.tools == [
            AgentTool(
                name="ask",
                agent=AgentDefinition(
                    name="specialist", description="Answers what support cannot."
                ),
            )
        ]

    def test_missing_file(self, agents: Path) -> None:
        with pytest.raises(ValidationError, match="agent file 'agents/nope.yaml' not found"):
            _parse(_step("agents/nope.yaml"))

    def test_invalid_yaml(self, agents: Path) -> None:
        (agents / "broken.yaml").write_text("name: [unclosed\n")
        with pytest.raises(ValueError, match="agent file 'agents/broken.yaml': invalid YAML"):
            load_agent("agents/broken.yaml")

    def test_not_a_mapping(self, agents: Path) -> None:
        (agents / "list.yaml").write_text("- name: a\n")
        with pytest.raises(ValueError, match="agent file 'agents/list.yaml' must hold a mapping"):
            load_agent("agents/list.yaml")

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            ("unknown: 1\n", "unknown"),
            ("loop_policy: {max_iterations: 0}\n", "max_iterations"),
            ("loop_policy: {on_max_iterations: retry}\n", "on_max_iterations"),
            ("memory: {trim_policy: keep_all}\n", "trim_policy"),
            ("constraints: {max_cost_usd: 0}\n", "max_cost_usd"),
            ("constraints: {max_latency_ms: 0}\n", "max_latency_ms"),
            ("tools: [3]\n", "tools.0.agent"),
        ],
    )
    def test_invalid_values_name_the_file(self, agents: Path, extra: str, match: str) -> None:
        (agents / "bad.yaml").write_text("name: bad\n" + extra)
        with pytest.raises(ValueError, match=match) as raised:
            load_agent("agents/bad.yaml")
        assert str(raised.value).startswith("agent file 'agents/bad.yaml':")

    def test_agent_definition_cycle_rejected(self, agents: Path) -> None:
        (agents / "a.yaml").write_text("name: a\ntools:\n  - {name: to_b, agent: agents/b.yaml}\n")
        (agents / "b.yaml").write_text("name: b\ntools:\n  - {name: to_a, agent: agents/a.yaml}\n")
        cycle = "agent definitions form a cycle: agents/a.yaml -> agents/b.yaml -> agents/a.yaml"
        with pytest.raises(ValueError, match=cycle):
            load_agent("agents/a.yaml")
        with pytest.raises(ValidationError, match=cycle):
            _parse(_step({"name": "top", "tools": [{"name": "to_a", "agent": "agents/a.yaml"}]}))

    def test_self_reference_is_a_cycle(self, agents: Path) -> None:
        (agents / "self.yaml").write_text(
            "name: s\ntools:\n  - {name: me, agent: agents/self.yaml}\n"
        )
        with pytest.raises(ValueError, match="cycle: agents/self.yaml -> agents/self.yaml"):
            load_agent("agents/self.yaml")

    def test_the_same_agent_twice_is_not_a_cycle(self, agents: Path) -> None:
        (agents / "two.yaml").write_text(
            "name: two\ntools:\n"
            "  - {name: one, agent: agents/specialist.yaml}\n"
            "  - {name: other, agent: agents/specialist.yaml}\n"
        )
        assert load_agent("agents/two.yaml").tool_names() == ["one", "other"]


class TestToolNames:
    def test_duplicate_tool_names_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"duplicate tool names \['ask'\]"):
            AgentDefinition(
                name="a",
                tools=["ask", "other", {"name": "ask", "agent": {"name": "b"}}],
            )

    def test_output_tool_name_reserved_with_an_output_schema(self) -> None:
        with pytest.raises(ValueError, match="'final_result' is reserved"):
            AgentDefinition(name="a", tools=["final_result"], output_schema={"type": "json_object"})

    def test_output_tool_name_free_without_an_output_schema(self) -> None:
        assert AgentDefinition(name="a", tools=["final_result"]).tool_names() == ["final_result"]


class TestAgentStepFields:
    def test_agent_step_requires_an_agent(self) -> None:
        with pytest.raises(ValidationError, match="Agent step 's' requires an 'agent'"):
            _parse({"id": "s", "type": "agent", "prompt": "hi"})

    def test_agent_only_on_agent_steps(self) -> None:
        with pytest.raises(ValidationError, match="'agent' applies only to agent steps"):
            _parse({"id": "s", "type": "llm_call", "prompt": "hi", "agent": {"name": "a"}})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model", "gpt-4o"),
            ("system_prompt", "x"),
            ("temperature", 0.0),
            ("max_tokens", 10),
            ("thinking", {"enabled": True}),
            ("tools", []),
            ("tool_choice", "required"),
            ("max_tool_iterations", 2),
            ("response_schema", {"type": "json_object"}),
            ("speaker", "x"),
        ],
    )
    def test_agent_settings_cannot_be_set_on_the_step(self, field: str, value: object) -> None:
        with pytest.raises(
            ValidationError, match=rf"\['{field}'\] come from the agent's definition"
        ):
            _parse(_step({"name": "a"}, **{field: value}))

    def test_step_fields_of_its_own_are_accepted(self) -> None:
        (step,) = _parse(
            _step(
                {"name": "a"},
                conversation="state.chat",
                output="reply",
                timeout=5.0,
                retry={"max_retries": 0},
                depends_on=[],
            )
        )
        assert (step.conversation, step.output, step.timeout) == ("state.chat", "reply", 5.0)
