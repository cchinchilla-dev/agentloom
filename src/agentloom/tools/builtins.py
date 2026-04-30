"""Built-in tools available to all workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import httpx

from agentloom.tools.base import BaseTool
from agentloom.tools.sandbox import ToolSandbox


class HttpRequestTool(BaseTool):
    """Makes HTTP requests (GET/POST)."""

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

        self._sandbox.validate_network(url)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body if body else None,
            )
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text,
            }


class ShellCommandTool(BaseTool):
    """Executes a shell command with timeout."""

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

        self._sandbox.validate_path(cwd, tool_name="shell_command")
        self._sandbox.validate_command(command, cwd=cwd)

        try:
            result = await anyio.run_process(
                ["sh", "-c", command],
                cwd=cwd,
                check=False,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout.decode(errors="replace"),
                "stderr": result.stderr.decode(errors="replace"),
            }
        except TimeoutError:
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}


class FileReadTool(BaseTool):
    """Reads a file and returns its contents."""

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
    """Writes content to a file."""

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

    Args:
        registry: ToolRegistry instance.
        sandbox: Optional sandbox policy. When provided, built-in tools
            validate operations before executing.
    """
    registry.register(HttpRequestTool(sandbox=sandbox))
    registry.register(ShellCommandTool(sandbox=sandbox))
    registry.register(FileReadTool(sandbox=sandbox))
    registry.register(FileWriteTool(sandbox=sandbox))
