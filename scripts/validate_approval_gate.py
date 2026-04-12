#!/usr/bin/env python3
"""Functional validation of the approval gate step type.

Exercises the full flow programmatically:
  1. Run a workflow with an approval gate → verify it pauses
  2. Verify checkpoint saved with status=paused, paused_step_id=approve
  3. Resume with decision=approved → verify workflow completes
  4. Run again → pause → resume with decision=rejected → verify rejected

Can be run standalone:
    uv run python scripts/validate_approval_gate.py
Or inside Docker:
    docker run --rm agentloom:dev python scripts/validate_approval_gate.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import StepStatus, WorkflowStatus
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.gateway import ProviderGateway


class FakeProvider(BaseProvider):
    """Minimal provider that returns a fixed response."""

    name = "fake"

    async def complete(self, messages, model, **kwargs):  # noqa: ANN001,ANN003
        return ProviderResponse(content="fake-output", model=model, provider="fake")

    async def stream(self, *a, **kw):  # noqa: ANN002,ANN003
        raise NotImplementedError

    def supports_model(self, model: str) -> bool:
        return True


def _gateway() -> ProviderGateway:
    gw = ProviderGateway()
    gw.register(FakeProvider(), priority=0)
    return gw


def _workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="approval-validate",
        config=WorkflowConfig(provider="fake", model="fake"),
        state={"input": "hello"},
        steps=[
            StepDefinition(
                id="draft",
                type=StepType.LLM_CALL,
                prompt="Draft: {state.input}",
                output="draft_text",
            ),
            StepDefinition(
                id="approve",
                type=StepType.APPROVAL_GATE,
                depends_on=["draft"],
                output="decision",
            ),
            StepDefinition(
                id="send",
                type=StepType.LLM_CALL,
                depends_on=["approve"],
                prompt="Send ({state.decision}): {state.draft_text}",
                output="result",
            ),
        ],
    )


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    sys.exit(1)


async def main() -> None:
    with tempfile.TemporaryDirectory() as cp_dir:
        checkpointer = FileCheckpointer(checkpoint_dir=Path(cp_dir))

        # ── Test 1: Run → should pause at approval gate ────────────
        print("\n[1/5] Running workflow (should pause at approval gate)...")
        engine = WorkflowEngine(
            workflow=_workflow(),
            provider_gateway=_gateway(),
            checkpointer=checkpointer,
            run_id="approval-validate",
        )
        result = await engine.run()

        if result.status == WorkflowStatus.PAUSED:
            _ok(f"Workflow paused (status={result.status.value})")
        else:
            _fail(f"Expected PAUSED, got {result.status.value}")

        if result.step_results["draft"].status == StepStatus.SUCCESS:
            _ok("draft step completed before pause")
        else:
            _fail("draft step should be SUCCESS")

        if result.step_results["approve"].status == StepStatus.PAUSED:
            _ok("approve step has PAUSED status")
        else:
            _fail(f"approve step should be PAUSED, got {result.step_results['approve'].status}")

        # ── Test 2: Verify checkpoint ──────────────────────────────
        print("\n[2/5] Verifying checkpoint...")
        loaded = await checkpointer.load("approval-validate")

        if loaded.status == "paused" and loaded.paused_step_id == "approve":
            _ok("Checkpoint: status=paused, paused_step_id=approve")
        else:
            _fail(f"Checkpoint: status={loaded.status}, paused_step_id={loaded.paused_step_id}")

        if "draft" in loaded.completed_steps and "approve" not in loaded.completed_steps:
            _ok("Completed steps correct (draft in, approve out)")
        else:
            _fail(f"Completed steps: {loaded.completed_steps}")

        # ── Test 3: Resume with --approve ──────────────────────────
        print("\n[3/5] Resuming with decision=approved...")
        data = await checkpointer.load("approval-validate")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=data,
            checkpointer=checkpointer,
            provider_gateway=_gateway(),
            approval_decisions={"approve": "approved"},
        )
        result2 = await resumed.run()

        if result2.status == WorkflowStatus.SUCCESS:
            _ok("Workflow completed after approval")
        else:
            _fail(f"Expected SUCCESS, got {result2.status.value}")

        if result2.final_state.get("decision") == "approved":
            _ok("Decision stored in state: approved")
        else:
            _fail(f"Decision in state: {result2.final_state.get('decision')}")

        if result2.step_results["send"].status == StepStatus.SUCCESS:
            _ok("send step completed after approval")
        else:
            _fail("send step should be SUCCESS")

        # ── Test 4: Run again → pause → resume with --reject ──────
        print("\n[4/5] Running again and resuming with decision=rejected...")
        engine2 = WorkflowEngine(
            workflow=_workflow(),
            provider_gateway=_gateway(),
            checkpointer=checkpointer,
            run_id="approval-reject",
        )
        await engine2.run()

        data2 = await checkpointer.load("approval-reject")
        resumed2 = await WorkflowEngine.from_checkpoint(
            checkpoint_data=data2,
            checkpointer=checkpointer,
            provider_gateway=_gateway(),
            approval_decisions={"approve": "rejected"},
        )
        result3 = await resumed2.run()

        if result3.status == WorkflowStatus.SUCCESS:
            _ok("Workflow completed after rejection")
        else:
            _fail(f"Expected SUCCESS, got {result3.status.value}")

        if result3.final_state.get("decision") == "rejected":
            _ok("Decision stored in state: rejected")
        else:
            _fail(f"Decision in state: {result3.final_state.get('decision')}")

        # ── Test 5: Final checkpoint status ────────────────────────
        print("\n[5/5] Verifying final checkpoints...")
        final1 = await checkpointer.load("approval-validate")
        final2 = await checkpointer.load("approval-reject")

        if final1.status == "success" and final2.status == "success":
            _ok("Both checkpoints updated to success")
        else:
            _fail(f"Checkpoint statuses: {final1.status}, {final2.status}")

        if final1.paused_step_id is None and final2.paused_step_id is None:
            _ok("No paused_step_id in final checkpoints")
        else:
            _fail("paused_step_id should be cleared")

        print("\n" + "=" * 50)
        print("  All approval gate validations passed!")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    import anyio

    anyio.run(main)
