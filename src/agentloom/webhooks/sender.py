"""Async webhook delivery for approval gate notifications."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from agentloom.core.models import WebhookConfig
from agentloom.core.templates import SafeFormatDict, build_template_vars

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


async def send_webhook(
    config: WebhookConfig, context: WebhookContext, observer: Any | None = None
) -> None:
    """POST a webhook notification with best-effort retry.

    Never raises — errors are logged so the calling step can still pause
    without being blocked by webhook failures.
    """
    import time

    import anyio

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
                hook = getattr(observer, "on_webhook_delivery", None)
                if hook:
                    hook(context.step_id, context.workflow_name, "success", latency)
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
                    hook = getattr(observer, "on_webhook_delivery", None)
                    if hook:
                        hook(context.step_id, context.workflow_name, "failed", latency)
