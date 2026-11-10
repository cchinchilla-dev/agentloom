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
    async def test_callback_base_url_puts_the_decision_links_in_the_body(
        self, public_dns: str
    ) -> None:
        route = respx.post(f"https://{public_dns}/notify").mock(return_value=httpx.Response(200))
        step = _gate_step(
            notify=WebhookConfig(
                url="https://hooks.example.com/notify",
                callback_base_url="https://callbacks.example.com/",
            )
        )

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(_context(step))

        body = json.loads(route.calls[0].request.content)
        assert body["approve_url"] == "https://callbacks.example.com/approve/run-42"
        assert body["reject_url"] == "https://callbacks.example.com/reject/run-42"

    @respx.mock
    @pytest.mark.anyio()
    async def test_without_callback_base_url_the_body_has_no_links(self, public_dns: str) -> None:
        route = respx.post(f"https://{public_dns}/notify").mock(return_value=httpx.Response(200))
        step = _gate_step(notify=WebhookConfig(url="https://hooks.example.com/notify"))

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(_context(step))

        body = json.loads(route.calls[0].request.content)
        assert "approve_url" not in body
        assert "reject_url" not in body

    @respx.mock
    @pytest.mark.anyio()
    async def test_sends_webhook_on_pause(self, public_dns: str) -> None:
        route = respx.post(f"https://{public_dns}/notify").mock(return_value=httpx.Response(200))
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
    async def test_webhook_template_rendered(self, public_dns: str) -> None:
        route = respx.post(f"https://{public_dns}/notify").mock(return_value=httpx.Response(200))
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
    async def test_a_hook_that_raises_leaves_the_pause_alone(self) -> None:
        observer = MagicMock()
        observer.on_approval_gate.side_effect = RuntimeError("hook bug")

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(_context(_gate_step(notify=None), observer=observer))

    @pytest.mark.anyio()
    async def test_a_hook_that_raises_leaves_the_decision_alone(self) -> None:
        observer = MagicMock()
        observer.on_approval_gate.side_effect = RuntimeError("hook bug")
        ctx = _context(
            _gate_step(notify=None),
            state={"_approval": {"approve_draft": "approved"}},
            observer=observer,
        )

        result = await ApprovalGateStep().execute(ctx)

        assert result.output == "approved"

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
    async def test_webhook_failure_still_pauses(self, public_dns: str) -> None:
        respx.post(f"https://{public_dns}/notify").mock(return_value=httpx.Response(500))
        step = _gate_step(notify=WebhookConfig(url="https://hooks.example.com/notify", timeout=0.5))
        ctx = _context(step)

        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(ctx)
