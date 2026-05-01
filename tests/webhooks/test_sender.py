"""Tests for webhook sender module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from agentloom.core.models import WebhookConfig
from agentloom.webhooks.sender import WebhookContext, _build_payload, send_webhook


@pytest.fixture()
def wh_context() -> WebhookContext:
    return WebhookContext(
        run_id="run-123",
        step_id="approve_draft",
        workflow_name="email-review",
        state={"topic": "budget report", "draft_text": "Dear team..."},
    )


class TestBuildPayload:
    def test_default_payload(self, wh_context: WebhookContext) -> None:
        config = WebhookConfig(url="https://hooks.example.com/wh")
        raw = _build_payload(config, wh_context)
        payload = json.loads(raw)
        assert payload["run_id"] == "run-123"
        assert payload["step_id"] == "approve_draft"
        assert payload["workflow_name"] == "email-review"
        assert payload["status"] == "awaiting_approval"

    def test_default_payload_with_callback_urls(self, wh_context: WebhookContext) -> None:
        ctx = WebhookContext(
            run_id="run-123",
            step_id="approve_draft",
            workflow_name="email-review",
            state={},
            callback_base_url="http://localhost:8642",
        )
        config = WebhookConfig(url="https://hooks.example.com/wh")
        payload = json.loads(_build_payload(config, ctx))
        assert payload["approve_url"] == "http://localhost:8642/approve/run-123"
        assert payload["reject_url"] == "http://localhost:8642/reject/run-123"

    def test_template_rendering(self, wh_context: WebhookContext) -> None:
        config = WebhookConfig(
            url="https://hooks.example.com/wh",
            body_template='{{"text": "Review {state.topic} for run {run_id}"}}',
        )
        raw = _build_payload(config, wh_context)
        payload = json.loads(raw)
        assert payload["text"] == "Review budget report for run run-123"

    def test_template_missing_var_preserved(self, wh_context: WebhookContext) -> None:
        config = WebhookConfig(
            url="https://hooks.example.com/wh",
            body_template='{{"text": "{missing_var}"}}',
        )
        raw = _build_payload(config, wh_context)
        assert "{missing_var}" in raw


class TestSendWebhook:
    @respx.mock
    @pytest.mark.anyio()
    async def test_sends_post_request(self, wh_context: WebhookContext) -> None:
        route = respx.post("https://hooks.example.com/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(url="https://hooks.example.com/wh")
        await send_webhook(config, wh_context)
        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["run_id"] == "run-123"

    @respx.mock
    @pytest.mark.anyio()
    async def test_custom_headers(self, wh_context: WebhookContext) -> None:
        route = respx.post("https://hooks.example.com/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(
            url="https://hooks.example.com/wh",
            headers={"Authorization": "Bearer tok-123"},
        )
        await send_webhook(config, wh_context)
        assert route.calls[0].request.headers["Authorization"] == "Bearer tok-123"

    @respx.mock
    @pytest.mark.anyio()
    async def test_failure_does_not_raise(self, wh_context: WebhookContext) -> None:
        respx.post("https://hooks.example.com/wh").mock(return_value=httpx.Response(500))
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        # Should not raise even on HTTP 500
        await send_webhook(config, wh_context)

    @respx.mock
    @pytest.mark.anyio()
    async def test_retry_on_transient_error(self, wh_context: WebhookContext) -> None:
        route = respx.post("https://hooks.example.com/wh").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(200),
            ]
        )
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        # Exercise the full retry schedule (2s + 4s backoff) by lifting the
        # deadline above the 5s default.
        await send_webhook(config, wh_context, deadline_s=30.0)
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.anyio()
    async def test_all_retries_exhausted(self, wh_context: WebhookContext) -> None:
        route = respx.post("https://hooks.example.com/wh").mock(return_value=httpx.Response(500))
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        await send_webhook(config, wh_context, deadline_s=30.0)
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.anyio()
    async def test_observer_notified_on_success(self, wh_context: WebhookContext) -> None:
        respx.post("https://hooks.example.com/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(url="https://hooks.example.com/wh")
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer)
        observer.on_webhook_delivery.assert_called_once()
        args = observer.on_webhook_delivery.call_args[0]
        assert args[0] == "approve_draft"
        assert args[1] == "email-review"
        assert args[2] == "success"
        assert args[3] > 0  # latency

    @respx.mock
    @pytest.mark.anyio()
    async def test_observer_notified_on_failure(self, wh_context: WebhookContext) -> None:
        respx.post("https://hooks.example.com/wh").mock(return_value=httpx.Response(500))
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=0.5)
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer, deadline_s=30.0)
        observer.on_webhook_delivery.assert_called_once()
        args = observer.on_webhook_delivery.call_args[0]
        assert args[0] == "approve_draft"
        assert args[1] == "email-review"
        assert args[2] == "failed"

    @respx.mock
    @pytest.mark.anyio()
    async def test_deadline_enforced(self, wh_context: WebhookContext) -> None:
        """A slow endpoint must not block the caller past ``deadline_s``."""
        respx.post("https://hooks.example.com/wh").mock(return_value=httpx.Response(500))
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        observer = MagicMock()
        # 1s deadline vs. 2s+4s retry schedule → should timeout fast.
        await send_webhook(config, wh_context, observer=observer, deadline_s=1.0)
        observer.on_webhook_delivery.assert_called_once()
        args = observer.on_webhook_delivery.call_args[0]
        assert args[2] == "timeout"
