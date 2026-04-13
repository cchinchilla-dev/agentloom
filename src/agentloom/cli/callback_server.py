"""CLI command: lightweight HTTP callback server for approval gates."""

from __future__ import annotations

import json
import logging
from typing import Any

import anyio
import typer

logger = logging.getLogger("agentloom.callback")


def callback_server(
    checkpoint_dir: str = typer.Option(
        ".agentloom/checkpoints", "--checkpoint-dir", help="Checkpoint storage directory."
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8642, "--port", help="Bind port."),
    lite: bool = typer.Option(False, "--lite", help="Run in lite mode (no observability)."),
) -> None:
    """Start an HTTP server that accepts approve/reject callbacks."""
    anyio.run(_serve, checkpoint_dir, host, port, lite)


async def _serve(checkpoint_dir: str, host: str, port: int, lite: bool) -> None:
    listener = await anyio.create_tcp_listener(local_host=host, local_port=port)
    typer.echo(f"Callback server listening on {host}:{port}")
    typer.echo("  POST /webhook           — receive webhook notifications")
    typer.echo("  POST /approve/<run_id>  — approve a paused workflow")
    typer.echo("  POST /reject/<run_id>   — reject a paused workflow")
    typer.echo("  GET  /pending           — list paused runs")

    async with anyio.create_task_group() as tg:

        async def _handle_conn(stream: anyio.abc.SocketStream) -> None:
            try:
                await _handle_request(stream, checkpoint_dir, lite)
            except Exception:
                logger.warning("Error handling request", exc_info=True)
            finally:
                await stream.aclose()

        await listener.serve(_handle_conn, task_group=tg)


async def _handle_request(stream: anyio.abc.SocketStream, checkpoint_dir: str, lite: bool) -> None:
    """Parse a raw HTTP/1.1 request and route it."""
    data = await stream.receive(8192)
    if not data:
        return

    # Keep reading until we have the full headers + body
    raw = data
    while b"\r\n\r\n" not in raw:
        chunk = await stream.receive(8192)
        if not chunk:
            break
        raw += chunk

    text = raw.decode("utf-8", errors="replace")
    header_end = text.find("\r\n\r\n")
    if header_end < 0:
        await _send_response(stream, 400, {"error": "bad request"})
        return

    header_section = text[:header_end]
    body_so_far = raw[header_end + 4 :]

    # Parse Content-Length and read remaining body bytes if needed
    content_length = 0
    for line in header_section.split("\r\n")[1:]:
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
            break

    while len(body_so_far) < content_length:
        chunk = await stream.receive(8192)
        if not chunk:
            break
        body_so_far += chunk

    body = body_so_far.decode("utf-8", errors="replace") if body_so_far else ""

    request_line = header_section.split("\r\n", 1)[0]
    parts = request_line.split(" ")
    if len(parts) < 2:
        await _send_response(stream, 400, {"error": "bad request"})
        return

    method, path = parts[0], parts[1]

    if method == "POST" and path == "/webhook":
        await _handle_webhook(stream, body)
    elif method == "GET" and path == "/pending":
        await _handle_pending(stream, checkpoint_dir)
    elif method == "POST" and path.startswith("/approve/"):
        run_id = path[len("/approve/") :]
        await _handle_decision(stream, checkpoint_dir, lite, run_id, "approved")
    elif method == "POST" and path.startswith("/reject/"):
        run_id = path[len("/reject/") :]
        await _handle_decision(stream, checkpoint_dir, lite, run_id, "rejected")
    else:
        await _send_response(stream, 404, {"error": "not found"})


async def _handle_webhook(stream: anyio.abc.SocketStream, body: str) -> None:
    """Receive and log an incoming webhook notification."""
    try:
        payload = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        payload = {"raw": body}
    logger.info("Webhook received: %s", json.dumps(payload, indent=2))
    typer.echo(f"\n[WEBHOOK] Notification received: {json.dumps(payload)}")
    await _send_response(stream, 200, {"status": "received"})


async def _handle_pending(stream: anyio.abc.SocketStream, checkpoint_dir: str) -> None:
    from agentloom.checkpointing.file import FileCheckpointer

    checkpointer = FileCheckpointer(checkpoint_dir=checkpoint_dir)
    all_runs = await checkpointer.list_runs()
    paused = [
        {
            "run_id": r.run_id,
            "workflow_name": r.workflow_name,
            "paused_step_id": r.paused_step_id,
            "updated_at": r.updated_at,
        }
        for r in all_runs
        if r.status == "paused"
    ]
    await _send_response(stream, 200, {"paused_runs": paused})


async def _handle_decision(
    stream: anyio.abc.SocketStream,
    checkpoint_dir: str,
    lite: bool,
    run_id: str,
    decision: str,
) -> None:
    from agentloom.checkpointing.file import FileCheckpointer
    from agentloom.cli.run import _setup_observer, _setup_providers
    from agentloom.core.engine import WorkflowEngine
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.tools.builtins import register_builtins
    from agentloom.tools.registry import ToolRegistry
    from agentloom.tools.sandbox import ToolSandbox

    checkpointer = FileCheckpointer(checkpoint_dir=checkpoint_dir)
    try:
        checkpoint_data = await checkpointer.load(run_id)
    except KeyError:
        await _send_response(stream, 404, {"error": f"no checkpoint for run '{run_id}'"})
        return

    if checkpoint_data.status != "paused":
        await _send_response(
            stream,
            409,
            {"error": f"run '{run_id}' is not paused (status={checkpoint_data.status})"},
        )
        return

    paused_step = checkpoint_data.paused_step_id
    if not paused_step:
        await _send_response(stream, 409, {"error": "checkpoint has no paused step id"})
        return

    approval_decisions = {paused_step: decision}

    engine = await WorkflowEngine.from_checkpoint(
        checkpoint_data=checkpoint_data,
        checkpointer=checkpointer,
        approval_decisions=approval_decisions,
    )

    gateway = ProviderGateway()
    _setup_providers(gateway, engine.workflow.config.provider)
    engine.provider_gateway = gateway

    sandbox_cfg = engine.workflow.config.sandbox
    sandbox = ToolSandbox(
        enabled=sandbox_cfg.enabled,
        allowed_commands=sandbox_cfg.allowed_commands,
        allowed_paths=sandbox_cfg.allowed_paths,
        allow_network=sandbox_cfg.allow_network,
        readable_paths=sandbox_cfg.readable_paths,
        writable_paths=sandbox_cfg.writable_paths,
        allowed_domains=sandbox_cfg.allowed_domains,
        max_write_bytes=sandbox_cfg.max_write_bytes,
    )
    tool_registry = ToolRegistry()
    register_builtins(tool_registry, sandbox=sandbox)
    engine.tool_registry = tool_registry

    observer = _setup_observer(lite)
    engine.observer = observer

    # Respond immediately, then execute in background
    await _send_response(
        stream,
        202,
        {
            "run_id": run_id,
            "decision": decision,
            "step_id": paused_step,
            "message": f"Workflow resuming with decision '{decision}'",
        },
    )

    result = await engine.run()

    if observer:
        observer.shutdown()
    await gateway.close()

    logger.info(
        "Callback resume completed: run=%s decision=%s status=%s",
        run_id,
        decision,
        result.status.value,
    )


async def _send_response(stream: anyio.abc.SocketStream, status: int, body: dict[str, Any]) -> None:
    """Write a minimal HTTP/1.1 JSON response."""
    phrases = {200: "OK", 202: "Accepted", 400: "Bad Request", 404: "Not Found", 409: "Conflict"}
    phrase = phrases.get(status, "Unknown")
    payload = json.dumps(body).encode()
    header = (
        f"HTTP/1.1 {status} {phrase}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    await stream.send(header.encode() + payload)
