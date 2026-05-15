"""Async webhook delivery for approval gate notifications."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from agentloom.core.models import WebhookConfig
from agentloom.core.templates import SafeFormatDict, build_template_vars
from agentloom.exceptions import SandboxViolationError
from agentloom.tools.sandbox import ToolSandbox, default_deny_webhook_target


@runtime_checkable
class WebhookObserver(Protocol):
    """Minimal observer interface for webhook delivery events."""

    def on_webhook_delivery(
        self, step_id: str, workflow_name: str, status: str, latency_s: float
    ) -> None: ...


logger = logging.getLogger("agentloom.webhooks")

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


@dataclass(frozen=True)
class WebhookContext:
    """Contextual data sent alongside the webhook payload."""

    run_id: str
    step_id: str
    workflow_name: str
    state: dict[str, Any] = field(default_factory=dict)
    callback_base_url: str = ""


def _build_payload(config: WebhookConfig, context: WebhookContext) -> str:
    """Render the webhook payload as a JSON string.

    If ``config.body_template`` is set, template variables are resolved
    from the workflow state.  Otherwise a default payload is generated.
    """
    if config.body_template:
        template_vars = build_template_vars(context.state)
        template_vars["run_id"] = context.run_id
        template_vars["step_id"] = context.step_id
        template_vars["workflow_name"] = context.workflow_name
        return config.body_template.format_map(SafeFormatDict(template_vars))

    payload: dict[str, Any] = {
        "run_id": context.run_id,
        "step_id": context.step_id,
        "workflow_name": context.workflow_name,
        "status": "awaiting_approval",
    }
    if context.callback_base_url:
        base = context.callback_base_url.rstrip("/")
        payload["approve_url"] = f"{base}/approve/{context.run_id}"
        payload["reject_url"] = f"{base}/reject/{context.run_id}"
    return json.dumps(payload)


_DEFAULT_DEADLINE_S = 5.0


async def send_webhook(
    config: WebhookConfig,
    context: WebhookContext,
    observer: WebhookObserver | None = None,
    *,
    deadline_s: float = _DEFAULT_DEADLINE_S,
    sandbox: ToolSandbox | None = None,
) -> None:
    """POST a webhook notification with best-effort retry, deadline-bounded.

    The total retry window is capped by *deadline_s* (default 5 s) so a
    webhook storm can never block a workflow for the 28 s of the raw retry
    schedule. Never raises — timeouts and errors are logged so the calling
    step can still pause even if the webhook endpoint is misbehaving.

    When *sandbox* is provided, ``ToolSandbox.validate_webhook_url`` gates the
    destination — same allowlist as ``validate_network`` when the sandbox is
    enabled, default deny-list (loopback / link-local / RFC 1918) when it is
    not. A blocked URL is logged and emitted as a
    ``status="sandbox_blocked"`` observer breadcrumb; the workflow's pause
    or completion is unaffected.
    """
    import time

    import anyio

    blocked_reason: str | None = None
    if sandbox is not None:
        try:
            sandbox.validate_webhook_url(config.url)
        except SandboxViolationError as exc:
            blocked_reason = str(exc)
    else:
        blocked_reason = default_deny_webhook_target(config.url)

    if blocked_reason is not None:
        logger.warning(
            "Webhook delivery to %s blocked by sandbox: %s (step=%s, run=%s)",
            config.url,
            blocked_reason,
            context.step_id,
            context.run_id,
        )
        if observer:
            observer.on_webhook_delivery(
                context.step_id, context.workflow_name, "sandbox_blocked", 0.0
            )
        return

    async def _inner() -> None:
        payload = _build_payload(config, context)
        headers = {"Content-Type": "application/json", **config.headers}
        t0 = time.monotonic()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    resp = await client.post(config.url, content=payload, headers=headers)
                    resp.raise_for_status()
                latency = time.monotonic() - t0
                logger.info(
                    "Webhook delivered to %s (step=%s, run=%s)",
                    config.url,
                    context.step_id,
                    context.run_id,
                )
                if observer:
                    observer.on_webhook_delivery(
                        context.step_id, context.workflow_name, "success", latency
                    )
                return
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    backoff = _BACKOFF_BASE**attempt
                    logger.warning(
                        "Webhook attempt %d/%d failed for %s: %s — retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        config.url,
                        exc,
                        backoff,
                    )
                    logger.debug("Webhook retry traceback", exc_info=True)
                    await anyio.sleep(backoff)
                else:
                    latency = time.monotonic() - t0
                    logger.warning(
                        "Webhook delivery failed after %d attempts for %s: %s",
                        _MAX_RETRIES,
                        config.url,
                        exc,
                    )
                    logger.debug("Webhook final failure traceback", exc_info=True)
                    if observer:
                        observer.on_webhook_delivery(
                            context.step_id, context.workflow_name, "failed", latency
                        )

    t_outer = time.monotonic()
    try:
        with anyio.fail_after(deadline_s):
            await _inner()
    except TimeoutError:
        latency = time.monotonic() - t_outer
        logger.warning(
            "Webhook delivery to %s exceeded deadline of %.1fs (step=%s, run=%s)",
            config.url,
            deadline_s,
            context.step_id,
            context.run_id,
        )
        if observer:
            observer.on_webhook_delivery(context.step_id, context.workflow_name, "timeout", latency)
