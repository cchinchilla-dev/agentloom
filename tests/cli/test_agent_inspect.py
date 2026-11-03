"""``agentloom agent inspect`` shows an agent as it loads (#120)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agentloom.cli.main import app
from agentloom.providers.pricing import load_pricing
from agentloom.steps.agent import AGENT_TOOL_PARAMETERS
from tests.cli.test_replay_cut import runner

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _from_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """Example 41's agents name their files and schema module from the repository root."""
    monkeypatch.chdir(REPO)


def _inspect(*argv: str) -> str:
    result = runner.invoke(app, ["agent", "inspect", *argv])
    assert result.exit_code == 0, result.output
    return result.output


def test_inspect_shows_the_agent_and_the_agents_it_calls() -> None:
    shown = yaml.safe_load(_inspect("examples/agents/support.yaml"))

    assert shown["name"] == "support_desk"
    read, refunds = shown["tools"]
    assert (read["name"], read["builtin"]) == ("file_read", True)
    assert read["parameters"]["required"] == ["path"]
    assert refunds["parameters"] == AGENT_TOOL_PARAMETERS
    assert refunds["agent"]["name"] == "refunds"
    assert refunds["agent"]["loop_policy"] == {"max_iterations": 1, "on_max_iterations": "error"}
    assert shown["memory"] == {"token_budget": 4000, "trim_policy": "drop_pairs"}
    assert shown["output_schema"]["json_schema"]["required"] == ["answer", "refund"]
    assert shown["constraints"]["forbidden_patterns"] == ["as an AI language model"]


def test_a_system_prompt_reads_as_written() -> None:
    assert "system_prompt: |\n  You answer billing questions" in _inspect(
        "examples/agents/support.yaml"
    )


def test_json_holds_what_yaml_shows() -> None:
    path = "examples/agents/support.yaml"
    assert json.loads(_inspect(path, "--json")) == yaml.safe_load(_inspect(path))


def test_the_model_comes_with_its_prices(tmp_path: Path) -> None:
    agent = tmp_path / "priced.yaml"
    agent.write_text(
        "name: priced\n"
        "model: claude-haiku-4-5\n"
        "tools:\n"
        "  - lookup\n"
        "  - name: helper\n"
        "    agent: {name: helper, model: not-a-priced-model}\n",
        encoding="utf-8",
    )
    shown = json.loads(_inspect(str(agent), "--json"))

    price = load_pricing()["claude-haiku-4-5"]
    assert shown["pricing"] == {
        "input_per_1k": price.input_cost_per_1k,
        "output_per_1k": price.output_cost_per_1k,
    }
    lookup, helper = shown["tools"]
    assert lookup == {"name": "lookup", "builtin": False, "description": None, "parameters": None}
    assert helper["agent"]["pricing"] is None


def test_a_broken_agent_fails(tmp_path: Path) -> None:
    agent = tmp_path / "broken.yaml"
    agent.write_text("name: broken\nunknown: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["agent", "inspect", str(agent)])

    assert result.exit_code == 1
    assert "Error:" in result.output and "unknown" in result.output
