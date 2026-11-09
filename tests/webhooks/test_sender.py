"""Tests for webhook sender module."""

from __future__ import annotations

import json
import socket
from typing import Any
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

    def test_a_nested_gates_links_name_the_root_run_and_the_gate(self) -> None:
        ctx = WebhookContext(
            run_id="run-123.sub.inner",
            step_id="approve_draft",
            workflow_name="child",
            callback_base_url="http://localhost:8642/",
        )
        config = WebhookConfig(url="https://hooks.example.com/wh")
        payload = json.loads(_build_payload(config, ctx))
        link = "/run-123?gate=sub.inner.approve_draft"
        assert payload["approve_url"] == f"http://localhost:8642/approve{link}"
        assert payload["reject_url"] == f"http://localhost:8642/reject{link}"

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
    async def test_sends_post_request(self, wh_context: WebhookContext, public_dns: str) -> None:
        route = respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(url="https://hooks.example.com/wh")
        await send_webhook(config, wh_context)
        assert route.called
        request = route.calls[0].request
        body = json.loads(request.content)
        assert body["run_id"] == "run-123"
        # Sent to the checked address, under the URL's name
        assert request.headers["host"] == "hooks.example.com"
        assert request.extensions["sni_hostname"] == "hooks.example.com"

    @respx.mock
    @pytest.mark.anyio()
    async def test_custom_headers(self, wh_context: WebhookContext, public_dns: str) -> None:
        route = respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(
            url="https://hooks.example.com/wh",
            headers={"Authorization": "Bearer tok-123"},
        )
        await send_webhook(config, wh_context)
        assert route.calls[0].request.headers["Authorization"] == "Bearer tok-123"

    @respx.mock
    @pytest.mark.anyio()
    async def test_failure_does_not_raise(
        self, wh_context: WebhookContext, public_dns: str
    ) -> None:
        respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(500))
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        await send_webhook(config, wh_context)

    @respx.mock
    @pytest.mark.anyio()
    async def test_retry_on_transient_error(
        self, wh_context: WebhookContext, public_dns: str
    ) -> None:
        route = respx.post(f"https://{public_dns}/wh").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(200),
            ]
        )
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        # A deadline above the 5s default lets the full 2s + 4s backoff schedule run.
        await send_webhook(config, wh_context, deadline_s=30.0)
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.anyio()
    async def test_all_retries_exhausted(self, wh_context: WebhookContext, public_dns: str) -> None:
        route = respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(500))
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        await send_webhook(config, wh_context, deadline_s=30.0)
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.anyio()
    async def test_observer_notified_on_success(
        self, wh_context: WebhookContext, public_dns: str
    ) -> None:
        respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(200))
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
    async def test_observer_notified_on_failure(
        self, wh_context: WebhookContext, public_dns: str
    ) -> None:
        respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(500))
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
    async def test_observer_without_the_hook_changes_nothing(
        self, wh_context: WebhookContext, public_dns: str
    ) -> None:
        route = respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(url="https://hooks.example.com/wh")
        await send_webhook(config, wh_context, observer=object())  # type: ignore[arg-type]
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.anyio()
    async def test_a_hook_that_raises_does_not_repeat_the_post(
        self, wh_context: WebhookContext, public_dns: str
    ) -> None:
        route = respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(url="https://hooks.example.com/wh")
        observer = MagicMock()
        observer.on_webhook_delivery.side_effect = RuntimeError("hook bug")
        await send_webhook(config, wh_context, observer=observer)
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.anyio()
    async def test_deadline_enforced(self, wh_context: WebhookContext, public_dns: str) -> None:
        """The retry backoff stops at ``deadline_s``; the delivery is reported as a timeout."""
        respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(500))
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        observer = MagicMock()
        # The 1s deadline falls inside the 2s backoff after the first 500.
        await send_webhook(config, wh_context, observer=observer, deadline_s=1.0)
        observer.on_webhook_delivery.assert_called_once()
        args = observer.on_webhook_delivery.call_args[0]
        assert args[2] == "timeout"

    @respx.mock
    @pytest.mark.anyio()
    async def test_every_attempt_goes_to_the_checked_address(
        self, wh_context: WebhookContext, public_dns: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A later lookup answering with loopback (DNS rebinding) is never made."""
        lookups: list[str] = []

        def _rebinding(host: Any, port: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
            address = public_dns if not lookups else "127.0.0.1"
            lookups.append(address)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _rebinding)
        route = respx.post(f"https://{public_dns}/wh").mock(
            side_effect=[httpx.Response(500), httpx.Response(200)]
        )
        config = WebhookConfig(url="https://hooks.example.com/wh", timeout=1.0)
        await send_webhook(config, wh_context, deadline_s=30.0)
        assert route.call_count == 2
        assert lookups == [public_dns]

    @respx.mock
    @pytest.mark.anyio()
    async def test_unresolvable_host_is_blocked(
        self, wh_context: WebhookContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no address to check, nothing is sent."""

        def _fail(*args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
            raise socket.gaierror("no such host")

        monkeypatch.setattr(socket, "getaddrinfo", _fail)
        route = respx.post(url__regex=r".*").mock(return_value=httpx.Response(200))
        config = WebhookConfig(url="https://hooks.example.com/wh")
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer)
        assert not route.called
        assert observer.on_webhook_delivery.call_args[0][2] == "sandbox_blocked"


class TestSandboxGate:
    """The sandbox gates the POST so a notify URL cannot reach internal hosts (SSRF)."""

    @respx.mock
    @pytest.mark.anyio()
    async def test_disabled_sandbox_blocks_loopback_by_default(
        self, wh_context: WebhookContext
    ) -> None:
        config = WebhookConfig(url="http://127.0.0.1:18642/captured")
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer)
        observer.on_webhook_delivery.assert_called_once()
        assert observer.on_webhook_delivery.call_args[0][2] == "sandbox_blocked"

    @respx.mock
    @pytest.mark.anyio()
    async def test_disabled_sandbox_blocks_link_local_metadata(
        self, wh_context: WebhookContext
    ) -> None:
        config = WebhookConfig(
            url="http://169.254.169.254/latest/meta-data/iam/security-credentials/role"
        )
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer)
        observer.on_webhook_delivery.assert_called_once()
        assert observer.on_webhook_delivery.call_args[0][2] == "sandbox_blocked"

    @respx.mock
    @pytest.mark.anyio()
    async def test_enabled_sandbox_enforces_allowed_domains(
        self, wh_context: WebhookContext
    ) -> None:
        from agentloom.tools.sandbox import ToolSandbox

        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allowed_domains=["api.openai.com"],
            allowed_schemes=["https"],
        )
        config = WebhookConfig(
            url="http://169.254.169.254/latest/meta-data/iam/security-credentials/role"
        )
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer, sandbox=sandbox)
        observer.on_webhook_delivery.assert_called_once()
        assert observer.on_webhook_delivery.call_args[0][2] == "sandbox_blocked"

    @respx.mock
    @pytest.mark.anyio()
    async def test_public_https_passes_through(
        self, wh_context: WebhookContext, public_dns: str
    ) -> None:
        respx.post(f"https://{public_dns}/wh").mock(return_value=httpx.Response(200))
        config = WebhookConfig(url="https://hooks.example.com/wh")
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer, deadline_s=10.0)
        observer.on_webhook_delivery.assert_called_once()
        assert observer.on_webhook_delivery.call_args[0][2] == "success"

    @respx.mock
    @pytest.mark.anyio()
    async def test_opt_in_unblocks_internal(self, wh_context: WebhookContext) -> None:
        from agentloom.tools.sandbox import ToolSandbox

        respx.post("http://127.0.0.1:18642/captured").mock(return_value=httpx.Response(200))
        sandbox = ToolSandbox(enabled=False, allow_internal_webhook_targets=True)
        config = WebhookConfig(url="http://127.0.0.1:18642/captured")
        observer = MagicMock()
        await send_webhook(config, wh_context, observer=observer, sandbox=sandbox, deadline_s=10.0)
        observer.on_webhook_delivery.assert_called_once()
        assert observer.on_webhook_delivery.call_args[0][2] == "success"
