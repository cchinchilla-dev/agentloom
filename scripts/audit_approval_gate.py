#!/usr/bin/env python3
"""
Approval Gate Implementation Audit for AgentLoom (Issue #41).

Validates the approval gate step type across all layers:
  1. Source contracts  — enum, model fields, executor, registry, engine, CLI
  2. Test coverage     — required test classes and scenarios
  3. Integration       — validation script, K8s manifest, example workflow
  4. Runtime           — exercises pause → approve and pause → reject cycles

Run:
    python scripts/audit_approval_gate.py          # from repo root
    python scripts/audit_approval_gate.py -v       # verbose (show passing checks)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "agentloom"
TESTS = ROOT / "tests"

_pass_count = 0
_fail_count = 0
_skip_count = 0
_verbose = False


def _green(t: str) -> str:
    return f"\033[92m{t}\033[0m"


def _red(t: str) -> str:
    return f"\033[91m{t}\033[0m"


def _yellow(t: str) -> str:
    return f"\033[93m{t}\033[0m"


def _bold(t: str) -> str:
    return f"\033[1m{t}\033[0m"


def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {_bold(title)}")
    print(f"{'─' * 60}")


def _pass(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    if _verbose:
        print(f"  {_green('✓')} {msg}")


def _fail(msg: str, detail: str = "") -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {_red('✗')} {msg}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"      {line}")


def _skip(msg: str) -> None:
    global _skip_count
    _skip_count += 1
    print(f"  {_yellow('○')} {msg}")


def _read(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def _contains(text: str, *needles: str) -> bool:
    return all(n in text for n in needles)


# Phase 1: Source contracts
def audit_step_type_enum() -> None:
    _header("StepType enum")

    src = _read(SRC / "core" / "models.py")

    if 'APPROVAL_GATE = "approval_gate"' in src:
        _pass("APPROVAL_GATE value in StepType enum")
    else:
        _fail("APPROVAL_GATE missing from StepType enum")


def audit_step_definition_fields() -> None:
    _header("StepDefinition approval gate fields")

    src = _read(SRC / "core" / "models.py")

    if "timeout_seconds" in src and "int | None" in src:
        _pass("timeout_seconds field defined (int | None)")
    else:
        _fail("timeout_seconds field missing on StepDefinition")

    if "on_timeout" in src:
        _pass("on_timeout field defined")
    else:
        _fail("on_timeout field missing on StepDefinition")

    if '"approve"' in src and '"reject"' in src:
        _pass('on_timeout accepts "approve" and "reject"')
    else:
        _fail("on_timeout should be Literal['approve', 'reject']")


def audit_executor() -> None:
    _header("ApprovalGateStep executor")

    path = SRC / "steps" / "approval_gate.py"
    src = _read(path)

    if not src:
        _fail("src/agentloom/steps/approval_gate.py does not exist")
        return

    _pass("approval_gate.py exists")

    if "class ApprovalGateStep" in src and "BaseStep" in src:
        _pass("ApprovalGateStep inherits from BaseStep")
    else:
        _fail("ApprovalGateStep class or BaseStep inheritance missing")

    if "async def execute" in src:
        _pass("execute() method defined")
    else:
        _fail("execute() method missing")

    if "_approval." in src and "state_manager.get" in src:
        _pass("Reads decision from _approval.{step_id} in state")
    else:
        _fail("Should read decision from _approval.{step_id}")

    if "PauseRequestedError" in src:
        _pass("Raises PauseRequestedError when no decision")
    else:
        _fail("Should raise PauseRequestedError on first execution")

    if "StepError" in src:
        _pass("Raises StepError on invalid decision")
    else:
        _fail("Should validate decision value")

    if '"approved"' in src and '"rejected"' in src:
        _pass('Accepts "approved" and "rejected" decisions')
    else:
        _fail("Should accept approved/rejected decisions")

    if "APPROVAL REQUIRED" in src and "stderr" in src:
        _pass("Prints instructions to stderr on pause")
    else:
        _fail("Should print resume instructions to stderr")

    if "step.output" in src and "state_manager.set" in src:
        _pass("Stores decision in output state variable")
    else:
        _fail("Should store decision via step.output")


def audit_registry() -> None:
    _header("Step registry")

    src = _read(SRC / "steps" / "registry.py")

    if "ApprovalGateStep" in src:
        _pass("ApprovalGateStep imported in registry")
    else:
        _fail("ApprovalGateStep not imported in registry.py")

    if "APPROVAL_GATE" in src:
        _pass("APPROVAL_GATE registered in create_default_registry()")
    else:
        _fail("APPROVAL_GATE not registered")


def audit_engine() -> None:
    _header("Engine from_checkpoint() approval support")

    src = _read(SRC / "core" / "engine.py")

    if "approval_decisions" in src:
        _pass("approval_decisions parameter on from_checkpoint()")
    else:
        _fail("approval_decisions parameter missing from from_checkpoint()")
        return

    if "dict[str, str]" in src and "approval_decisions" in src:
        _pass("approval_decisions typed as dict[str, str]")
    else:
        _fail("approval_decisions should be dict[str, str]")

    if "_approval." in src and "state_manager.set" in src:
        _pass("Injects decisions into state as _approval.{step_id}")
    else:
        _fail("Should inject decisions into state via _approval.{step_id}")


def audit_cli() -> None:
    _header("CLI resume --approve/--reject")

    src = _read(SRC / "cli" / "resume.py")

    if "--approve" in src:
        _pass("--approve flag defined")
    else:
        _fail("--approve flag missing from resume command")

    if "--reject" in src:
        _pass("--reject flag defined")
    else:
        _fail("--reject flag missing from resume command")

    if "approve" in src and "reject" in src and "Cannot use" in src:
        _pass("Mutual exclusion validated (--approve + --reject)")
    else:
        _fail("Should validate mutual exclusion of --approve and --reject")

    if "approval_decisions" in src:
        _pass("Builds approval_decisions dict from flags")
    else:
        _fail("Should build approval_decisions and pass to from_checkpoint()")

    if "paused_step_id" in src and "decision" in src:
        _pass("Maps decision to paused_step_id from checkpoint")
    else:
        _fail("Should use checkpoint.paused_step_id as decision key")


# Phase 2: Test coverage
def audit_unit_tests() -> None:
    _header("Unit tests (steps/test_approval_gate.py)")

    src = _read(TESTS / "steps" / "test_approval_gate.py")

    if not src:
        _fail("tests/steps/test_approval_gate.py does not exist")
        return

    _pass("test_approval_gate.py exists")

    required = [
        "test_pauses_without_decision",
        "test_returns_approved",
        "test_returns_rejected",
        "test_invalid_decision_raises",
        "test_stores_output_in_state",
    ]
    found = [t for t in required if f"def {t}" in src]
    missing = [t for t in required if t not in found]

    if len(found) == len(required):
        _pass(f"All {len(required)} required unit test methods present")
    else:
        detail = "\n".join(f"Missing: {m}" for m in missing)
        _fail(f"{len(found)}/{len(required)} unit tests found", detail)


def audit_integration_tests() -> None:
    _header("Integration tests (core/test_engine_approval.py)")

    src = _read(TESTS / "core" / "test_engine_approval.py")

    if not src:
        _fail("tests/core/test_engine_approval.py does not exist")
        return

    _pass("test_engine_approval.py exists")

    required_classes = [
        ("TestApprovalGatePause", "pause behaviour"),
        ("TestApprovalGateResume", "resume with decisions"),
    ]
    for cls, desc in required_classes:
        if f"class {cls}" in src:
            _pass(f"{cls} — {desc}")
        else:
            _fail(f"Missing test class: {cls} ({desc})")

    required_tests = [
        "test_approval_gate_pauses_workflow",
        "test_approval_gate_saves_checkpoint",
        "test_resume_with_approve",
        "test_resume_with_reject",
        "test_downstream_reads_decision",
        "test_resume_skips_completed_steps",
        "test_resume_without_decision_pauses_again",
    ]
    found = [t for t in required_tests if f"def {t}" in src]
    missing = [t for t in required_tests if t not in found]

    if len(found) == len(required_tests):
        _pass(f"All {len(required_tests)} integration test methods present")
    else:
        detail = "\n".join(f"Missing: {m}" for m in missing)
        _fail(f"{len(found)}/{len(required_tests)} integration tests found", detail)


def audit_cli_tests() -> None:
    _header("CLI tests (cli/test_resume.py)")

    src = _read(TESTS / "cli" / "test_resume.py")

    if "TestResumeApproval" in src:
        _pass("TestResumeApproval class present")
    else:
        _fail("TestResumeApproval class missing in test_resume.py")

    cli_tests = [
        "test_resume_approve_flag",
        "test_resume_reject_flag",
        "test_approve_reject_mutual_exclusion",
    ]
    found = [t for t in cli_tests if f"def {t}" in src]
    missing = [t for t in cli_tests if t not in found]

    if len(found) == len(cli_tests):
        _pass(f"All {len(cli_tests)} CLI test methods present")
    else:
        detail = "\n".join(f"Missing: {m}" for m in missing)
        _fail(f"{len(found)}/{len(cli_tests)} CLI tests found", detail)


# Phase 3: Integration artefacts
def audit_artefacts() -> None:
    _header("Integration artefacts")

    # Validation script
    script = ROOT / "scripts" / "validate_approval_gate.py"
    script_src = _read(script)
    if script_src:
        _pass("scripts/validate_approval_gate.py exists")
    else:
        _fail("Validation script missing")

    if script_src and "anyio.run" in script_src:
        _pass("Validation script uses anyio.run")
    elif script_src:
        _fail("Validation script should use anyio.run")

    if script_src and "asyncio.run" not in script_src:
        _pass("No asyncio.run in validation script")
    elif script_src:
        _fail("Validation script should not use asyncio.run")

    for keyword in ("approved", "rejected", "checkpoint", "resume"):
        if script_src and keyword in script_src.lower():
            _pass(f"Validation script covers: {keyword}")
        elif script_src:
            _fail(f"Validation script should cover: {keyword}")

    # K8s manifest
    k8s = ROOT / "deploy" / "k8s" / "examples" / "approval-gate-job.yaml"
    k8s_src = _read(k8s)
    if k8s_src:
        _pass("K8s approval-gate-job.yaml exists")
    else:
        _fail("K8s smoke job manifest missing")

    if k8s_src and "ConfigMap" in k8s_src and "Job" in k8s_src:
        _pass("K8s manifest has ConfigMap + Job")
    elif k8s_src:
        _fail("K8s manifest should define ConfigMap and Job")

    if k8s_src and "anyio" in k8s_src:
        _pass("K8s script uses anyio")
    elif k8s_src:
        _fail("K8s script should use anyio")

    # Example workflow
    example = ROOT / "examples" / "29_approval_gate.yaml"
    example_src = _read(example)
    if example_src:
        _pass("examples/29_approval_gate.yaml exists")
    else:
        _fail("Example workflow missing")

    if example_src and "approval_gate" in example_src:
        _pass("Example uses approval_gate step type")
    elif example_src:
        _fail("Example should use approval_gate step type")

    if example_src and "timeout_seconds" in example_src:
        _pass("Example includes timeout_seconds field")
    elif example_src:
        _fail("Example should demonstrate timeout_seconds")

    if example_src and "on_timeout" in example_src:
        _pass("Example includes on_timeout field")
    elif example_src:
        _fail("Example should demonstrate on_timeout")


# Phase 4: Runtime validation
def audit_runtime() -> None:
    _header("Runtime validation (live approval gate cycles)")

    try:
        import anyio

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
    except ImportError as e:
        _skip(f"Cannot import agentloom: {e}")
        return

    class _FakeProvider(BaseProvider):
        name = "fake"

        async def complete(self, messages, model, **kwargs):
            return ProviderResponse(content="fake", model=model, provider="fake")

        async def stream(self, *a, **kw):
            raise NotImplementedError

        def supports_model(self, model: str) -> bool:
            return True

    def _gw() -> ProviderGateway:
        gw = ProviderGateway()
        gw.register(_FakeProvider(), priority=0)
        return gw

    workflow = WorkflowDefinition(
        name="audit-approval",
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
                prompt="Send: {state.draft_text}",
                output="result",
            ),
        ],
    )

    async def _run_audit() -> None:
        with tempfile.TemporaryDirectory() as cp_dir:
            ckpt = FileCheckpointer(checkpoint_dir=Path(cp_dir))

            # Pause at approval gate
            engine = WorkflowEngine(
                workflow=workflow,
                provider_gateway=_gw(),
                checkpointer=ckpt,
                run_id="audit-approval",
            )
            r1 = await engine.run()

            if r1.status == WorkflowStatus.PAUSED:
                _pass("Runtime: workflow paused at approval gate")
            else:
                _fail(f"Runtime: expected PAUSED, got {r1.status.value}")
                return

            step_d = r1.step_results.get("draft")
            if step_d and step_d.status == StepStatus.SUCCESS:
                _pass("Runtime: draft completed before gate")
            else:
                _fail("Runtime: draft should be SUCCESS")

            step_a = r1.step_results.get("approve")
            if step_a and step_a.status == StepStatus.PAUSED:
                _pass("Runtime: approve step has PAUSED status")
            else:
                _fail("Runtime: approve should be PAUSED")

            # Verify checkpoint
            loaded = await ckpt.load("audit-approval")
            if loaded.status == "paused" and loaded.paused_step_id == "approve":
                _pass("Runtime: checkpoint paused at approve")
            else:
                _fail(
                    f"Runtime: checkpoint status={loaded.status},"
                    f" paused_step_id={loaded.paused_step_id}"
                )

            # Resume with approval
            data = await ckpt.load("audit-approval")
            resumed = await WorkflowEngine.from_checkpoint(
                checkpoint_data=data,
                checkpointer=ckpt,
                provider_gateway=_gw(),
                approval_decisions={"approve": "approved"},
            )
            r2 = await resumed.run()

            if r2.status == WorkflowStatus.SUCCESS:
                _pass("Runtime: workflow completed after approval")
            else:
                _fail(f"Runtime: expected SUCCESS, got {r2.status.value}")

            if r2.final_state.get("decision") == "approved":
                _pass("Runtime: decision=approved in final state")
            else:
                _fail(f"Runtime: decision={r2.final_state.get('decision')}")

            if r2.step_results.get("send", None):
                if r2.step_results["send"].status == StepStatus.SUCCESS:
                    _pass("Runtime: send step completed after approval")
                else:
                    _fail("Runtime: send should be SUCCESS")
            else:
                _fail("Runtime: send step missing from results")

            # Resume with rejection
            engine2 = WorkflowEngine(
                workflow=workflow,
                provider_gateway=_gw(),
                checkpointer=ckpt,
                run_id="audit-reject",
            )
            await engine2.run()

            data2 = await ckpt.load("audit-reject")
            resumed2 = await WorkflowEngine.from_checkpoint(
                checkpoint_data=data2,
                checkpointer=ckpt,
                provider_gateway=_gw(),
                approval_decisions={"approve": "rejected"},
            )
            r3 = await resumed2.run()

            if r3.status == WorkflowStatus.SUCCESS:
                _pass("Runtime: workflow completed after rejection")
            else:
                _fail(f"Runtime: expected SUCCESS after reject, got {r3.status.value}")

            if r3.final_state.get("decision") == "rejected":
                _pass("Runtime: decision=rejected in final state")
            else:
                _fail(f"Runtime: decision={r3.final_state.get('decision')}")

            # Final checkpoint status
            final = await ckpt.load("audit-approval")
            if final.status == "success" and final.paused_step_id is None:
                _pass("Runtime: final checkpoint is success, no paused_step_id")
            else:
                _fail(f"Runtime: final status={final.status}")

    anyio.run(_run_audit)


# Main
def main() -> int:
    global _verbose

    parser = argparse.ArgumentParser(description="Audit approval gate implementation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show passing checks")
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Skip runtime validation",
    )
    args = parser.parse_args()
    _verbose = args.verbose

    print(_bold("\n  AgentLoom — Approval Gate Audit (Issue #41)"))
    print(f"  {'=' * 48}")

    # Phase 1
    audit_step_type_enum()
    audit_step_definition_fields()
    audit_executor()
    audit_registry()
    audit_engine()
    audit_cli()

    # Phase 2
    audit_unit_tests()
    audit_integration_tests()
    audit_cli_tests()

    # Phase 3
    audit_artefacts()

    # Phase 4
    if not args.skip_runtime:
        audit_runtime()
    else:
        _skip("Runtime validation skipped (--skip-runtime)")

    # Summary
    total = _pass_count + _fail_count + _skip_count
    print(f"\n{'=' * 60}")
    print(
        f"  {_bold('Results')}: {_green(f'{_pass_count} passed')}  "
        f"{_red(f'{_fail_count} failed') if _fail_count else f'{_fail_count} failed'}  "
        f"{_yellow(f'{_skip_count} skipped') if _skip_count else f'{_skip_count} skipped'}  "
        f"({total} total)"
    )
    print(f"{'=' * 60}\n")

    return 1 if _fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
