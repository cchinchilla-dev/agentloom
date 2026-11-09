"""The offline examples replay from their recordings; 41 passes the determinism check."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
        ("user", None),
        ("assistant", "support_desk"),
    ] * 2
    assert run["total_cost_usd"] == pytest.approx(0.006462)


def test_verify_determinism_passes_on_the_agent_example() -> None:
    argv = ["verify-determinism", "examples/41_agent.yaml", "-r", "recordings/agent.json"]
    result = runner.invoke(app, argv)

    assert result.exit_code == 0, result.output
    assert "status success, 18 events" in result.output


HASHED = [
    ("36_structured_output", "structured_output"),
    ("37_embeddings", "embeddings"),
    ("38_chatbot_conversation", "chatbot_conversation"),
    ("39_multi_agent_conversation", "multi_agent_conversation"),
    ("40_fault_replay", "fault_replay"),
    ("41_agent", "agent"),
]


@pytest.mark.parametrize(("example", "recording"), HASHED)
def test_example_replays_strictly(example: str, recording: str) -> None:
    argv = ["replay", f"examples/{example}.yaml", "-r", f"recordings/{recording}.json"]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(("example", "recording"), HASHED)
def test_every_turn_of_the_example_recording_checks_its_request(
    example: str, recording: str
) -> None:
    data = json.loads((REPO / f"recordings/{recording}.json").read_text(encoding="utf-8"))
    turns = [
        t
        for k, v in data.items()
        if not k.startswith("_")
        for t in (v if isinstance(v, list) else [v])
    ]
    assert turns
    assert all(len(turn["request_hash"]) == 64 for turn in turns)


def test_tool_calling_example_replays_strictly() -> None:
    # Its second turn carries a live tool result, so that recording has no hash to check.
    argv = ["replay", "examples/35_tool_calling.yaml", "-r", "recordings/tool_calling.json"]
    with patch("agentloom.steps._tools._execute", AsyncMock(return_value=("{}", True))):
        result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
