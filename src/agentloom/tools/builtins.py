"""Built-in tools available to all workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import httpx

from agentloom.tools.base import BaseTool


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

    async def execute(self, **kwargs: Any) -> Any:
        url = kwargs["url"]
        method = kwargs.get("method", "GET")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body", "")
        timeout = kwargs.get("timeout", 30)

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

    async def execute(self, **kwargs: Any) -> Any:
        # FIXME: no sandboxing — fine for trusted workflows, not for untrusted input
        command = kwargs["command"]
        cwd = kwargs.get("cwd", ".")

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

    async def execute(self, **kwargs: Any) -> Any:
        path = Path(kwargs["path"])
        encoding = kwargs.get("encoding", "utf-8")
        return path.read_text(encoding=encoding)


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

    async def execute(self, **kwargs: Any) -> Any:
        path = Path(kwargs["path"])
        content = kwargs["content"]
        encoding = kwargs.get("encoding", "utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return {"written": len(content), "path": str(path)}


def register_builtins(registry: Any) -> None:
    """Register all built-in tools with a ToolRegistry."""
    registry.register(HttpRequestTool())
    registry.register(ShellCommandTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
