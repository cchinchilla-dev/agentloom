"""CLI command: lightweight HTTP callback server for approval gates."""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
from pathlib import Path
from typing import Any

import anyio
import typer

logger = logging.getLogger("agentloom.callback")

# Bounds on what one request may make the server buffer
_MAX_HEADER_BYTES = 64 * 1024
_MAX_BODY_BYTES = 1024 * 1024


def callback_server(
    checkpoint_dir: str = typer.Option(
        ".agentloom/checkpoints", "--checkpoint-dir", help="Checkpoint storage directory."
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8642, "--port", help="Bind port."),
    lite: bool = typer.Option(False, "--lite", help="Run in lite mode (no observability)."),
    token: str | None = typer.Option(
        None,
        "--token",
        help=(
            "Optional shared secret. When set, POST /approve, POST /reject and "
            "GET /pending require a matching `X-AgentLoom-Token` header — defence "
            "in depth for a server reachable beyond loopback."
        ),
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Config file with a providers: block, checked at start. Default: "
            "$AGENTLOOM_CONFIG, then ./agentloom.yaml when it exists."
        ),
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Start an HTTP server that accepts approve/reject callbacks."""
    config_path = _checked_config_file(config_file)
    anyio.run(_serve, checkpoint_dir, host, port, lite, token, config_path)  # pragma: no cover


def _checked_config_file(explicit: Path | None) -> Path | None:
    """The config file, loaded once now: a broken one stops the server before it listens."""
    from agentloom.cli.run import _config_file
    from agentloom.config import load_config
    from agentloom.exceptions import ConfigError

    path = _config_file(explicit)
    if path is not None:
        try:
            load_config(path)
        except ConfigError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from e
    return path


async def _serve(  # pragma: no cover
    checkpoint_dir: str,
    host: str,
    port: int,
    lite: bool,
    token: str | None = None,
    config_path: Path | None = None,
) -> None:
    listener = await anyio.create_tcp_listener(local_host=host, local_port=port)
    # Resolve actual port (may differ from requested when port=0)
    actual_port = port
    for lnr in listener.listeners:
        sock = lnr.extra(anyio.abc.SocketAttribute.raw_socket)
        actual_port = sock.getsockname()[1]
        break
    typer.echo(f"Callback server listening on {host}:{actual_port}")
    typer.echo("  POST /webhook           — receive webhook notifications")
    typer.echo("  POST /approve/<run_id>  — approve a paused workflow")
    typer.echo("  POST /reject/<run_id>   — reject a paused workflow")
    typer.echo("  GET  /pending           — list paused runs")
    warning = _exposure_warning(host, token)
    if warning:
        typer.echo(warning, err=True)

    async with anyio.create_task_group() as tg:

        async def _handle_conn(stream: anyio.abc.SocketStream) -> None:
            try:
                await _handle_request(stream, checkpoint_dir, lite, token, config_path)
            except Exception:
                logger.warning("Error handling request", exc_info=True)
            finally:
                await stream.aclose()

        await listener.serve(_handle_conn, task_group=tg)


def _exposure_warning(host: str, token: str | None) -> str | None:
    """A warning when *host* is reachable beyond loopback and no token guards the server."""
    if token is not None:
        return None
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if loopback:
        return None
    return (
        f"Warning: listening on {host} without --token: anyone who reaches this "
        f"port can approve or reject a paused run."
    )


def _token_matches(provided: str | None, expected: str | None) -> bool:
    """``True`` when no token is required or *provided* matches it, compared in constant time."""
    if expected is None:
        return True
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


def _parse_headers(header_section: str) -> dict[str, str]:
    """Parse the HTTP header block into a lowercase-keyed dict.

    Header names are case-insensitive per RFC 9110, so lowercasing the
    keys lets callers look up ``content-type`` / ``x-agentloom-token``
    without worrying about the sender's casing.
    """
    headers: dict[str, str] = {}
    for line in header_section.split("\r\n")[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    return headers


async def _handle_request(
    stream: anyio.abc.SocketStream,
    checkpoint_dir: str,
    lite: bool,
    token: str | None = None,
    config_path: Path | None = None,
) -> None:
    """Parse a raw HTTP/1.1 request and route it."""
    data = await stream.receive(8192)
    if not data:
        return

    # Headers first; the body is read below, up to Content-Length
    raw = data
    while b"\r\n\r\n" not in raw and len(raw) <= _MAX_HEADER_BYTES:
        chunk = await stream.receive(8192)
        if not chunk:
            break
        raw += chunk

    end = raw.find(b"\r\n\r\n")
    if end > _MAX_HEADER_BYTES or (end < 0 and len(raw) > _MAX_HEADER_BYTES):
        await _send_response(stream, 431, {"error": "request headers too large"})
        return

    text = raw.decode("utf-8", errors="replace")
    header_end = text.find("\r\n\r\n")
    if header_end < 0:
        await _send_response(stream, 400, {"error": "bad request"})
        return

    header_section = text[:header_end]
    body_so_far = raw[header_end + 4 :]

    content_length = 0
    for line in header_section.split("\r\n")[1:]:
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except ValueError:
                await _send_response(stream, 400, {"error": "bad request"})
                return
            if content_length < 0:
                await _send_response(stream, 400, {"error": "bad request"})
                return
            if content_length > _MAX_BODY_BYTES:
                await _send_response(stream, 413, {"error": "request body too large"})
                return
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
    headers = _parse_headers(header_section)

    if method == "POST" and path == "/webhook":
        await _handle_webhook(stream, body, headers.get("content-type"))
    elif method == "GET" and path == "/pending":
        if not _token_matches(headers.get("x-agentloom-token"), token):
            await _send_response(stream, 401, {"error": "invalid or missing X-AgentLoom-Token"})
            return
        await _handle_pending(stream, checkpoint_dir)
    elif method == "POST" and path.startswith("/approve/"):
        run_id = path[len("/approve/") :]
        await _handle_decision(
            stream,
            checkpoint_dir,
            lite,
            run_id,
            "approved",
            provided_token=headers.get("x-agentloom-token"),
            expected_token=token,
            config_path=config_path,
        )
    elif method == "POST" and path.startswith("/reject/"):
        run_id = path[len("/reject/") :]
        await _handle_decision(
            stream,
            checkpoint_dir,
            lite,
            run_id,
            "rejected",
            provided_token=headers.get("x-agentloom-token"),
            expected_token=token,
            config_path=config_path,
        )
    else:
        await _send_response(stream, 404, {"error": "not found"})


async def _handle_webhook(
    stream: anyio.abc.SocketStream, body: str, content_type: str | None
) -> None:
    """Receive, validate, and log an incoming webhook notification.

    A non-empty body that is not a JSON object gets ``400``, so the sender
    sees the failure; a missing ``Content-Type`` is accepted (the server is a
    notification sink) but logs a warning.
    """
    if content_type is None:
        logger.warning("Webhook POST /webhook received with no Content-Type header")
    stripped = body.strip()
    if not stripped:
        # An empty body is a bare ping — accept it without complaint.
        logger.info("Webhook received: (empty body)")
        await _send_response(stream, 200, {"status": "received"})
        return
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as e:
        logger.warning("Malformed webhook body: %s", e)
        await _send_response(stream, 400, {"error": f"invalid JSON: {e}"})
        return
    if not isinstance(payload, dict):
        await _send_response(stream, 400, {"error": "body must be a JSON object"})
        return
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
    *,
    provided_token: str | None = None,
    expected_token: str | None = None,
    config_path: Path | None = None,
) -> None:
    from agentloom.checkpointing.file import FileCheckpointer
    from agentloom.cli.run import _setup_generic_openai, _setup_observer, _setup_providers
    from agentloom.core.engine import WorkflowEngine
    from agentloom.exceptions import ConfigError, ProviderError
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.tools.builtins import register_builtins
    from agentloom.tools.registry import ToolRegistry
    from agentloom.tools.sandbox import ToolSandbox

    # Without --token approve/reject are open; _serve warns when bound beyond loopback
    if not _token_matches(provided_token, expected_token):
        await _send_response(stream, 401, {"error": "invalid or missing X-AgentLoom-Token"})
        return

    checkpointer = FileCheckpointer(checkpoint_dir=checkpoint_dir)
    try:
        checkpoint_data = await checkpointer.load(run_id)
    except KeyError:
        await _send_response(stream, 404, {"error": f"no checkpoint for run '{run_id}'"})
        return
    except ValueError as e:
        # FileCheckpointer rejects a run_id that would escape the checkpoint directory
        await _send_response(stream, 400, {"error": f"invalid run id: {e}"})
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
    try:
        _setup_generic_openai(gateway, engine.workflow.config)
        _setup_providers(gateway, engine.workflow.config.provider, config_path)
    except (ProviderError, ConfigError) as e:
        await _send_response(stream, 500, {"error": str(e)})
        return
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
        allowed_schemes=sandbox_cfg.allowed_schemes,
        max_write_bytes=sandbox_cfg.max_write_bytes,
        danger_opt_in=sandbox_cfg.danger_opt_in,
        allow_internal_webhook_targets=sandbox_cfg.allow_internal_webhook_targets,
    )
    tool_registry = ToolRegistry()
    register_builtins(tool_registry, sandbox=sandbox)
    engine.tool_registry = tool_registry

    observer = _setup_observer(lite)
    engine.observer = observer

    # Respond with 202, then run the engine (blocking this connection)
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
        observer.shutdown()  # pragma: no cover
    await gateway.close()

    logger.info(
        "Callback resume completed: run=%s decision=%s status=%s",
        run_id,
        decision,
        result.status.value,
    )


async def _send_response(stream: anyio.abc.SocketStream, status: int, body: dict[str, Any]) -> None:
    """Write a minimal HTTP/1.1 JSON response."""
    phrases = {
        200: "OK",
        202: "Accepted",
        400: "Bad Request",
        401: "Unauthorized",
        404: "Not Found",
        409: "Conflict",
        413: "Content Too Large",
        431: "Request Header Fields Too Large",
        500: "Internal Server Error",
    }
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
