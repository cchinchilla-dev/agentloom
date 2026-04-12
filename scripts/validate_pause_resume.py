#!/usr/bin/env python3
"""Functional validation of the pause/resume mechanism.

Exercises the full flow programmatically:
  1. Run a 3-step workflow that pauses at step_b
  2. Verify checkpoint saved with status=paused
  3. Resume from the checkpoint
  4. Verify workflow completes successfully

Can be run standalone:
    uv run python scripts/validate_pause_resume.py
Or inside Docker:
    docker run --rm agentloom:dev python scripts/validate_pause_resume.py
"""

from __future__ import annotations

import asyncio
import json
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
from agentloom.core.results import StepResult, StepStatus, WorkflowStatus
from agentloom.exceptions import PauseRequestedError
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.base import BaseStep, StepContext
from agentloom.steps.registry import StepRegistry, create_default_registry


# --- Helpers ----------------------------------------------------------------


class FakeProvider(BaseProvider):
    """Minimal provider that returns a fixed response."""

    name = "fake"

    async def complete(self, messages, model, **kwargs):  # noqa: ANN001,ANN003
        return ProviderResponse(
            content="fake-output", model=model, provider="fake"
        )

    async def stream(self, *a, **kw):  # noqa: ANN002,ANN003
        raise NotImplementedError

    def supports_model(self, model: str) -> bool:
        return True


class PausingStep(BaseStep):
    """LLM step that pauses on step_b, delegates otherwise."""

    async def execute(self, context: StepContext) -> StepResult:
        if context.step_definition.id == "step_b":
            raise PauseRequestedError("step_b")
        from agentloom.steps.llm_call import LLMCallStep

        return await LLMCallStep().execute(context)


def _gateway() -> ProviderGateway:
    gw = ProviderGateway()
    gw.register(FakeProvider(), priority=0)
    return gw


def _registry_pausing() -> StepRegistry:
    reg = create_default_registry()
    reg.register(StepType.LLM_CALL, PausingStep)
    return reg


def _workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="pause-validate",
        config=WorkflowConfig(provider="fake", model="fake"),
        state={"input": "hello"},
        steps=[
            StepDefinition(
                id="step_a",
                type=StepType.LLM_CALL,
                prompt="A: {state.input}",
                output="result_a",
            ),
            StepDefinition(
                id="step_b",
                type=StepType.LLM_CALL,
                depends_on=["step_a"],
                prompt="B: {state.result_a}",
                output="result_b",
            ),
            StepDefinition(
                id="step_c",
                type=StepType.LLM_CALL,
                depends_on=["step_b"],
                prompt="C: {state.result_b}",
                output="result_c",
            ),
        ],
    )


# --- Validation -------------------------------------------------------------


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    sys.exit(1)


async def main() -> None:
    with tempfile.TemporaryDirectory() as cp_dir:
        checkpointer = FileCheckpointer(checkpoint_dir=Path(cp_dir))

        # ── Step 1: Run → should pause ──────────────────────────────
        print("\n[1/4] Running workflow (should pause at step_b)...")
        engine = WorkflowEngine(
            workflow=_workflow(),
            provider_gateway=_gateway(),
            step_registry=_registry_pausing(),
            checkpointer=checkpointer,
            run_id="validate-pause",
        )
        result = await engine.run()

        if result.status == WorkflowStatus.PAUSED:
            _ok(f"Workflow paused (status={result.status.value})")
        else:
            _fail(f"Expected PAUSED, got {result.status.value}")

        if result.step_results["step_a"].status == StepStatus.SUCCESS:
            _ok("step_a completed before pause")
        else:
            _fail("step_a should be SUCCESS")

        if result.step_results["step_b"].status == StepStatus.PAUSED:
            _ok("step_b has PAUSED status")
        else:
            _fail(f"step_b should be PAUSED, got {result.step_results['step_b'].status}")

        # ── Step 2: Verify checkpoint on disk ───────────────────────
        print("\n[2/4] Verifying checkpoint file...")
        cp_files = list(Path(cp_dir).glob("*.json"))
        if len(cp_files) == 1:
            _ok(f"Checkpoint file: {cp_files[0].name}")
        else:
            _fail(f"Expected 1 checkpoint file, found {len(cp_files)}")

        checkpoint = json.loads(cp_files[0].read_text())
        if checkpoint["status"] == "paused":
            _ok("Checkpoint status = paused")
        else:
            _fail(f"Checkpoint status = {checkpoint['status']}")

        if checkpoint["paused_step_id"] == "step_b":
            _ok("paused_step_id = step_b")
        else:
            _fail(f"paused_step_id = {checkpoint['paused_step_id']}")

        if "step_a" in checkpoint["completed_steps"]:
            _ok("step_a in completed_steps")
        else:
            _fail("step_a not in completed_steps")

        if "step_b" not in checkpoint["completed_steps"]:
            _ok("step_b NOT in completed_steps (correct)")
        else:
            _fail("step_b should not be in completed_steps")

        # ── Step 3: List runs ───────────────────────────────────────
        print("\n[3/4] Listing checkpoint runs...")
        runs = await checkpointer.list_runs()
        if len(runs) == 1 and runs[0].run_id == "validate-pause":
            _ok(f"Found 1 run: {runs[0].run_id} (status={runs[0].status})")
        else:
            _fail(f"Expected 1 run, found {len(runs)}")

        # ── Step 4: Resume → should complete ────────────────────────
        print("\n[4/4] Resuming workflow...")
        checkpoint_data = await checkpointer.load("validate-pause")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint_data,
            checkpointer=checkpointer,
            provider_gateway=_gateway(),
            # Default registry — no pausing, step_b will run normally
        )
        resume_result = await resumed.run()

        if resume_result.status == WorkflowStatus.SUCCESS:
            _ok(f"Resumed workflow completed (status={resume_result.status.value})")
        else:
            _fail(f"Expected SUCCESS, got {resume_result.status.value}")

        if resume_result.step_results["step_b"].status == StepStatus.SUCCESS:
            _ok("step_b now SUCCESS after resume")
        else:
            _fail(f"step_b status = {resume_result.step_results['step_b'].status}")

        if resume_result.step_results["step_c"].status == StepStatus.SUCCESS:
            _ok("step_c completed after resume")
        else:
            _fail(f"step_c status = {resume_result.step_results['step_c'].status}")

        # Verify final checkpoint updated
        final = await checkpointer.load("validate-pause")
        if final.status == "success":
            _ok("Final checkpoint status = success")
        else:
            _fail(f"Final checkpoint status = {final.status}")

        print("\n══════════════════════════════════════════════════")
        print("  All pause/resume validations passed!")
        print("══════════════════════════════════════════════════\n")


if __name__ == "__main__":
    asyncio.run(main())
