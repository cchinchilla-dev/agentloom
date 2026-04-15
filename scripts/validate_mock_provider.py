"""Functional validation of MockProvider + RecordingProvider.

Exercises the full record → replay round-trip end-to-end:

1. Record: wrap a deterministic fake provider with RecordingProvider, run a
   2-step workflow, verify the JSON file is written with both step entries.
2. Replay: load the recorded JSON into MockProvider, run the same workflow,
   verify outputs and token/cost figures match the recording exactly.
3. Miss: run against an unrelated prompt, verify the default_response path.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import anyio

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import TokenUsage, WorkflowStatus
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import RecordingProvider


class DeterministicProvider(BaseProvider):
    name = "det"

    async def complete(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> ProviderResponse:
        # Echo the last user content — deterministic so record/replay is comparable.
        last = messages[-1].get("content", "")
        return ProviderResponse(
            content=f"det::{last}",
            model=model,
            provider="det",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
            cost_usd=0.0001,
            finish_reason="stop",
        )

    def supports_model(self, model: str) -> bool:
        return True


def _workflow(provider_name: str) -> WorkflowDefinition:
    return WorkflowDefinition(
        name="mock-validation",
        config=WorkflowConfig(provider=provider_name, model="test-model"),
        state={"question": "what is 2+2"},
        steps=[
            StepDefinition(
                id="draft",
                type=StepType.LLM_CALL,
                prompt="Draft: {state.question}",
                output="draft_text",
            ),
            StepDefinition(
                id="polish",
                type=StepType.LLM_CALL,
                depends_on=["draft"],
                prompt="Polish: {state.draft_text}",
                output="polished",
            ),
        ],
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        recording = Path(td) / "recording.json"

        print("[1/3] Recording from DeterministicProvider...")
        gw = ProviderGateway()
        gw.register(RecordingProvider(DeterministicProvider(), recording), priority=0)
        engine = WorkflowEngine(workflow=_workflow("det"), provider_gateway=gw)
        r = await engine.run()
        await gw.close()
        assert r.status == WorkflowStatus.SUCCESS, f"record run failed: {r.status}"
        assert recording.exists(), "recording file not created"
        data = json.loads(recording.read_text())
        assert len(data) == 2, f"expected 2 recorded entries, got {len(data)}: {list(data)}"
        for key, entry in data.items():
            assert entry["content"].startswith("det::"), entry
            assert entry["usage"]["total_tokens"] == 12
            assert "latency_ms" in entry
        draft_out = r.final_state["draft_text"]
        polished_out = r.final_state["polished"]
        print(f"  OK: recorded 2 entries → {recording.name}")

        print("[2/3] Replaying via MockProvider...")
        gw2 = ProviderGateway()
        gw2.register(MockProvider(responses_file=recording), priority=0)
        engine2 = WorkflowEngine(workflow=_workflow("mock"), provider_gateway=gw2)
        r2 = await engine2.run()
        await gw2.close()
        assert r2.status == WorkflowStatus.SUCCESS, f"replay failed: {r2.status}"
        assert r2.final_state["draft_text"] == draft_out, "replay diverged on draft"
        assert r2.final_state["polished"] == polished_out, "replay diverged on polish"
        assert r2.total_tokens == r.total_tokens, (
            f"token total drift: {r.total_tokens} → {r2.total_tokens}"
        )
        print(f"  OK: outputs + tokens match record (tokens={r2.total_tokens})")

        print("[3/3] Verifying default_response on unknown prompt...")
        mock = MockProvider(responses_file=recording, default_response="FALLBACK")
        resp = await mock.complete(
            messages=[{"role": "user", "content": "completely-unseen-input"}],
            model="test-model",
        )
        assert resp.content == "FALLBACK", resp.content
        assert resp.usage.total_tokens == 0
        print("  OK: unknown prompt → default_response")

    print("\nAll MockProvider validations passed!")


if __name__ == "__main__":
    anyio.run(main)
