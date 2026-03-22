"""Tests for built-in tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
import respx

from agentloom.tools.builtins import (
    FileReadTool,
    FileWriteTool,
    HttpRequestTool,
    ShellCommandTool,
    register_builtins,
)
from agentloom.tools.registry import ToolRegistry


class TestHttpRequestTool:
    @respx.mock
    async def test_get_request(self) -> None:
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(200, text='{"ok": true}')
        )
        tool = HttpRequestTool()
        result = await tool.execute(url="https://example.com/api")
        assert result["status_code"] == 200
        assert '{"ok": true}' in result["body"]

    @respx.mock
    async def test_post_request(self) -> None:
        respx.post("https://example.com/api").mock(return_value=httpx.Response(201, text="created"))
        tool = HttpRequestTool()
        result = await tool.execute(
            url="https://example.com/api",
            method="POST",
            body='{"data": 1}',
        )
        assert result["status_code"] == 201


class TestShellCommandTool:
    async def test_echo(self) -> None:
        tool = ShellCommandTool()
        result = await tool.execute(command="echo hello")
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    async def test_stderr(self) -> None:
        tool = ShellCommandTool()
        result = await tool.execute(command="echo error >&2")
        assert "error" in result["stderr"]

    async def test_nonzero_exit(self) -> None:
        tool = ShellCommandTool()
        result = await tool.execute(command="exit 42")
        assert result["returncode"] == 42


class TestFileReadTool:
    async def test_read_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            tool = FileReadTool()
            result = await tool.execute(path=f.name)
        assert result == "hello world"


class TestFileWriteTool:
    async def test_write_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.txt")
            tool = FileWriteTool()
            result = await tool.execute(path=path, content="written content")
            assert result["written"] == 15
            assert Path(path).read_text() == "written content"

    async def test_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "file.txt")
            tool = FileWriteTool()
            await tool.execute(path=path, content="nested")
            assert Path(path).read_text() == "nested"


class TestRegisterBuiltins:
    def test_registers_all_tools(self) -> None:
        registry = ToolRegistry()
        register_builtins(registry)
        tools = registry.list()
        names = [t.name if hasattr(t, "name") else t["name"] for t in tools]
        assert "http_request" in names
        assert "shell_command" in names
        assert "file_read" in names
        assert "file_write" in names
