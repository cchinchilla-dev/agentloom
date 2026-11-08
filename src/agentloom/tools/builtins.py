"""Built-in tools available to all workflows."""

from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path
from typing import Any

import anyio
import httpx

from agentloom.tools.base import BaseTool
from agentloom.tools.sandbox import ToolSandbox

# Credential-bearing header names: the result reaches the model, state, checkpoints, recordings
_CREDENTIAL_HEADER_PARTS = ("authorization", "cookie", "token", "secret", "key")


class HttpRequestTool(BaseTool):
    """Tool that makes HTTP requests (GET, POST, PUT, DELETE)."""

    name = "http_request"
    description = "Make an HTTP request to a URL and return the response body."
    parameters_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to request"},
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "default": "GET",
            },
            "headers": {"type": "object", "default": {}},
            "body": {"type": "string", "default": ""},
            "timeout": {"type": "number", "default": 30},
        },
        "required": ["url"],
    }

    def __init__(self, sandbox: ToolSandbox | None = None) -> None:
        self._sandbox = sandbox or ToolSandbox()

    async def execute(self, **kwargs: Any) -> Any:
        url = kwargs["url"]
        method = kwargs.get("method", "GET")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body", "")
        timeout = kwargs.get("timeout", 30)

        # The internal-host check resolves the name, which blocks: off the event loop
        await anyio.to_thread.run_sync(self._sandbox.validate_network, url)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body if body else None,
            )
            return {
                "status_code": response.status_code,
                "headers": {
                    name: value
                    for name, value in response.headers.items()
                    if not any(part in name.lower() for part in _CREDENTIAL_HEADER_PARTS)
                },
                "body": response.text,
            }


class ShellCommandTool(BaseTool):
    """Tool that executes a shell command with a timeout."""

    name = "shell_command"
    description = "Execute a shell command and return stdout/stderr."
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "default": 30},
            "cwd": {"type": "string", "default": "."},
        },
        "required": ["command"],
    }

    def __init__(self, sandbox: ToolSandbox | None = None) -> None:
        self._sandbox = sandbox or ToolSandbox()

    async def execute(self, **kwargs: Any) -> Any:
        command = kwargs["command"]
        cwd = kwargs.get("cwd", ".")
        timeout = kwargs.get("timeout", 30)

        self._sandbox.validate_path(cwd, tool_name="shell_command")
        self._sandbox.validate_command(command, cwd=cwd)

        try:
            returncode, stdout, stderr = await _run_shell(command, cwd, timeout)
        except TimeoutError:
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }


async def _run_shell(command: str, cwd: str, timeout: float) -> tuple[int | None, bytes, bytes]:
    """Run *command* under ``sh -c``; past *timeout* or on cancellation, kill all it started.

    Killing only the shell would leave its children running, so the shell
    leads a new process group (a session, on POSIX) and the group is killed.
    """
    output: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}

    async def drain(stream: Any, name: str) -> None:
        async for chunk in stream:
            output[name].extend(chunk)

    async with await anyio.open_process(
        ["sh", "-c", command], cwd=cwd, start_new_session=True
    ) as process:
        try:
            with anyio.fail_after(timeout):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(drain, process.stdout, "stdout")
                    tg.start_soon(drain, process.stderr, "stderr")
                    await process.wait()
        except BaseException:
            killpg = getattr(os, "killpg", None)
            if killpg is None:  # pragma: no cover — Windows has no process groups to kill
                process.kill()
            else:
                with contextlib.suppress(ProcessLookupError):
                    killpg(process.pid, signal.SIGKILL)
            raise
    return process.returncode, bytes(output["stdout"]), bytes(output["stderr"])


class FileReadTool(BaseTool):
    """Tool that reads a file and returns its contents."""

    name = "file_read"
    description = "Read the contents of a file."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "encoding": {"type": "string", "default": "utf-8"},
        },
        "required": ["path"],
    }

    def __init__(self, sandbox: ToolSandbox | None = None) -> None:
        self._sandbox = sandbox or ToolSandbox()

    async def execute(self, **kwargs: Any) -> Any:
        path = kwargs["path"]
        encoding = kwargs.get("encoding", "utf-8")

        self._sandbox.validate_path(path, writable=False, tool_name="file_read")

        return Path(path).read_text(encoding=encoding)


class FileWriteTool(BaseTool):
    """Tool that writes content to a file."""

    name = "file_write"
    description = "Write content to a file."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
            "encoding": {"type": "string", "default": "utf-8"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, sandbox: ToolSandbox | None = None) -> None:
        self._sandbox = sandbox or ToolSandbox()

    async def execute(self, **kwargs: Any) -> Any:
        path_str = kwargs["path"]
        content = kwargs["content"]
        encoding = kwargs.get("encoding", "utf-8")

        self._sandbox.validate_path(path_str, writable=True, tool_name="file_write")
        self._sandbox.validate_write_size(len(content.encode(encoding)), tool_name="file_write")

        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return {"written": len(content), "path": str(path)}


def register_builtins(registry: Any, sandbox: ToolSandbox | None = None) -> None:
    """Register all built-in tools with a ToolRegistry.

    With *sandbox*, the tools validate each operation before executing.
    """
    registry.register(HttpRequestTool(sandbox=sandbox))
    registry.register(ShellCommandTool(sandbox=sandbox))
    registry.register(FileReadTool(sandbox=sandbox))
    registry.register(FileWriteTool(sandbox=sandbox))
