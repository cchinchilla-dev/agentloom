"""Integration tests for approval gate + webhook notification."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from agentloom.core.models import StepDefinition, StepType, WebhookConfig
from agentloom.core.state import StateManager
from agentloom.exceptions import PauseRequestedError
from agentloom.steps.approval_gate import ApprovalGateStep
from agentloom.steps.base import StepContext


def _gate_step(*, notify: WebhookConfig | None = None) -> StepDefinition:
    return StepDefinition(
        id="approve_draft",
        type=StepType.APPROVAL_GATE,
        output="decision",
        notify=notify,
    )


def _context(
    step: StepDefinition, state: dict | None = None, observer: object | None = None
) -> StepContext:
    return StepContext(
        step_definition=step,
        state_manager=StateManager(initial_state=state or {}),
        run_id="run-42",
        workflow_name="review-wf",
        observer=observer,
    )


class TestApprovalGateWebhook:
    @respx.mock
    @pytest.mark.anyio()
    async def test_sends_webhook_on_pause(self) -> None:
        route = respx.post("https://hooks.example.com/notify").mock(
            return_value=httpx.Response(200)
        )
        step = _gate_step(notify=WebhookConfig(url="https://hooks.example.com/notify"))
        ctx = _context(step, state={"topic": "launch"})

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(ctx)

        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["run_id"] == "run-42"
        assert body["step_id"] == "approve_draft"
        assert body["workflow_name"] == "review-wf"
        assert body["status"] == "awaiting_approval"

    @pytest.mark.anyio()
    async def test_no_webhook_when_none(self) -> None:
        step = _gate_step(notify=None)
        ctx = _context(step)

        with patch("agentloom.webhooks.sender.send_webhook", new_callable=AsyncMock) as mock_send:
            with pytest.raises(PauseRequestedError):
                await ApprovalGateStep().execute(ctx)
            mock_send.assert_not_called()

    @respx.mock
    @pytest.mark.anyio()
    async def test_webhook_template_rendered(self) -> None:
        route = respx.post("https://hooks.example.com/notify").mock(
            return_value=httpx.Response(200)
        )
        step = _gate_step(
            notify=WebhookConfig(
                url="https://hooks.example.com/notify",
                body_template='{{"msg": "Approve {state.topic}?"}}',
            )
        )
        ctx = _context(step, state={"topic": "budget review"})

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(ctx)

        body = json.loads(route.calls[0].request.content)
        assert body["msg"] == "Approve budget review?"

    @pytest.mark.anyio()
    async def test_observer_notified_pending_on_pause(self) -> None:
        observer = MagicMock()
        step = _gate_step(notify=None)
        ctx = _context(step, observer=observer)

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(ctx)

        observer.on_approval_gate.assert_called_once_with("approve_draft", "review-wf", "pending")

    @pytest.mark.anyio()
    async def test_observer_notified_on_resume(self) -> None:
        observer = MagicMock()
        step = _gate_step(notify=None)
        ctx = _context(
            step,
            state={"_approval": {"approve_draft": "approved"}},
            observer=observer,
        )

        result = await ApprovalGateStep().execute(ctx)

        assert result.output == "approved"
        observer.on_approval_gate.assert_called_once_with("approve_draft", "review-wf", "approved")

    @respx.mock
    @pytest.mark.anyio()
    async def test_webhook_failure_still_pauses(self) -> None:
        """Webhook failure must not prevent the step from pausing."""
        respx.post("https://hooks.example.com/notify").mock(return_value=httpx.Response(500))
        step = _gate_step(notify=WebhookConfig(url="https://hooks.example.com/notify", timeout=0.5))
        ctx = _context(step)

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(ctx)
