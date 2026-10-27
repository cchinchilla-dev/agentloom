"""Example 41 runs offline from its recording and verifies (#120)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentloom.cli.main import app
from tests.cli.test_replay_cut import runner

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """From the repository root, as the example says to run it, and with no keys."""
    monkeypatch.chdir(REPO)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_agent_replays_offline() -> None:
    result = runner.invoke(app, ["run", "examples/41_agent.yaml", "--lite", "--json"])

    assert result.exit_code == 0, result.output
    run = json.loads(result.output[result.output.index("{") :])
    assert run["status"] == "success"
    assert run["final_state"]["reply"]["refund"] == "eligible"
    assert run["final_state"]["follow_up"]["refund"] == "not_requested"
    chat = run["final_state"]["chat"]["messages"]
    assert [(m["role"], m["name"]) for m in chat] == [
        ("user", "support_desk"),
        ("assistant", "support_desk"),
    ] * 2
    assert run["total_cost_usd"] == pytest.approx(0.006493)


def test_verify_determinism_passes_on_the_agent_example() -> None:
    argv = ["verify-determinism", "examples/41_agent.yaml", "-r", "recordings/agent.json"]
    result = runner.invoke(app, argv)

    assert result.exit_code == 0, result.output
    assert "status success, 15 events" in result.output
