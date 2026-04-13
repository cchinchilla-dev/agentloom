"""Functional validation of webhook notifications on approval gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WebhookConfig,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowStatus
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.registry import create_default_registry


class FakeProvider(BaseProvider):
    name = "fake"

    async def complete(self, messages: list, model: str, **kwargs: object) -> ProviderResponse:
        return ProviderResponse(content="fake-output", model=model, provider="fake")

    async def stream(self, *a: object, **kw: object) -> None:
        raise NotImplementedError

    def supports_model(self, model: str) -> bool:
        return True


def _gw() -> ProviderGateway:
    gw = ProviderGateway()
    gw.register(FakeProvider(), priority=0)
    return gw


def _wf(webhook_url: str | None = None) -> WorkflowDefinition:
    notify = None
    if webhook_url:
        notify = WebhookConfig(url=webhook_url, timeout=5.0)
    return WorkflowDefinition(
        name="webhook-validation",
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
                id="gate",
                type=StepType.APPROVAL_GATE,
                depends_on=["step_a"],
                output="decision",
                notify=notify,
            ),
            StepDefinition(
                id="step_b",
                type=StepType.LLM_CALL,
                depends_on=["gate"],
                prompt="B: {state.result_a}",
                output="result_b",
            ),
        ],
    )


async def main() -> None:
    import anyio

    # --- Phase 1: Webhook capture server ---
    received_webhooks: list[dict] = []

    async def _capture_webhook(stream: anyio.abc.SocketStream) -> None:
        data = await stream.receive(8192)
        text = data.decode("utf-8", errors="replace")
        body_start = text.find("\r\n\r\n")
        if body_start > 0:
            body = text[body_start + 4 :]
            try:
                received_webhooks.append(json.loads(body))
            except json.JSONDecodeError:
                received_webhooks.append({"raw": body})
        response = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Length: 2\r\n"
            "Connection: close\r\n"
            "\r\n"
            "OK"
        )
        await stream.send(response.encode())
        await stream.aclose()

    # Start capture server on a random port
    listener = await anyio.create_tcp_listener(local_host="127.0.0.1", local_port=0)
    port = listener.extra(anyio.abc.SocketAttribute.local_port)
    webhook_url = f"http://127.0.0.1:{port}/webhook"

    with tempfile.TemporaryDirectory() as cp_dir:
        ckpt = FileCheckpointer(checkpoint_dir=Path(cp_dir))

        # --- Phase 2: Run workflow (should pause and send webhook) ---
        print("[1/5] Running workflow with webhook (should pause)...")

        async with anyio.create_task_group() as tg:

            async def _run_and_cancel() -> None:
                eng = WorkflowEngine(
                    workflow=_wf(webhook_url),
                    provider_gateway=_gw(),
                    step_registry=create_default_registry(),
                    checkpointer=ckpt,
                    run_id="wh-test",
                )
                r = await eng.run()
                assert r.status == WorkflowStatus.PAUSED, f"Expected PAUSED, got {r.status}"
                print("  OK: workflow paused")
                # Stop the listener
                tg.cancel_scope.cancel()

            tg.start_soon(listener.serve, _capture_webhook)
            tg.start_soon(_run_and_cancel)

        # --- Phase 3: Verify webhook received ---
        print("[2/5] Verifying webhook payload...")
        assert len(received_webhooks) > 0, "No webhook received"
        wh = received_webhooks[0]
        assert wh.get("run_id") == "wh-test", f"Wrong run_id: {wh}"
        assert wh.get("step_id") == "gate", f"Wrong step_id: {wh}"
        assert wh.get("workflow_name") == "webhook-validation", f"Wrong workflow_name: {wh}"
        assert wh.get("status") == "awaiting_approval", f"Wrong status: {wh}"
        print(f"  OK: webhook payload valid — {wh}")

        # --- Phase 4: Verify checkpoint ---
        print("[3/5] Verifying checkpoint...")
        loaded = await ckpt.load("wh-test")
        assert loaded.status == "paused"
        assert loaded.paused_step_id == "gate"
        assert "step_a" in loaded.completed_steps
        print("  OK: checkpoint valid")

        # --- Phase 5: Resume with approval ---
        print("[4/5] Resuming with approval...")
        data = await ckpt.load("wh-test")
        resumed = await WorkflowEngine.from_checkpoint(
            checkpoint_data=data,
            checkpointer=ckpt,
            provider_gateway=_gw(),
            approval_decisions={"gate": "approved"},
        )
        r2 = await resumed.run()
        assert r2.status == WorkflowStatus.SUCCESS, f"Expected SUCCESS, got {r2.status}"
        assert r2.final_state.get("decision") == "approved"
        print("  OK: resumed with approval")

        # --- Phase 6: No webhook without config ---
        print("[5/5] Verifying no webhook when notify=None...")
        received_webhooks.clear()
        eng2 = WorkflowEngine(
            workflow=_wf(webhook_url=None),
            provider_gateway=_gw(),
            step_registry=create_default_registry(),
            checkpointer=ckpt,
            run_id="wh-none",
        )
        r3 = await eng2.run()
        assert r3.status == WorkflowStatus.PAUSED
        assert len(received_webhooks) == 0, "Unexpected webhook sent"
        print("  OK: no webhook sent when notify is None")

    print("\nAll webhook validations passed!")


import anyio

anyio.run(main)
