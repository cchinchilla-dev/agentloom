"""Tests for built-in tools."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
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
    async def test_response_headers_that_can_carry_credentials_are_dropped(self) -> None:
        respx.get("https://example.com/login").mock(
            return_value=httpx.Response(
                200,
                text="ok",
                headers=[
                    ("Content-Type", "text/plain"),
                    ("Set-Cookie", "session=abc"),
                    ("Set-Cookie", "csrf=def"),
                    ("Authorization", "Bearer ghi"),
                    ("Proxy-Authorization", "Basic jkl"),
                    ("X-Amz-Security-Token", "mno"),
                    ("X-Client-Secret", "pqr"),
                    ("X-Api-Key", "stu"),
                    ("X-Request-Id", "req-1"),
                ],
            )
        )
        result = await HttpRequestTool().execute(url="https://example.com/login")
        assert result["headers"] == {
            "content-type": "text/plain",
            "x-request-id": "req-1",
            "content-length": "2",
        }

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

    async def test_timeout_stops_a_long_command(self) -> None:
        start = time.monotonic()
        result = await ShellCommandTool().execute(command="sleep 30", timeout=0.2)
        assert result == {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        assert time.monotonic() - start < 10

    @pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX")
    async def test_timeout_kills_the_shell_process_group(self, monkeypatch: Any) -> None:
        killed: list[tuple[int, int]] = []
        real_killpg = os.killpg

        def spy(pid: int, sig: int) -> None:
            # The shell must lead its own group, or killing the group misses its children
            assert os.getpgid(pid) == pid
            killed.append((pid, sig))
            real_killpg(pid, sig)

        monkeypatch.setattr("agentloom.tools.builtins.os.killpg", spy)
        result = await ShellCommandTool().execute(command="sleep 30 & sleep 30", timeout=0.2)
        assert result["stderr"] == "Command timed out"
        assert [sig for _, sig in killed] == [signal.SIGKILL]

    @pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX")
    async def test_cancelling_the_step_kills_the_process_group(self, monkeypatch: Any) -> None:
        killed: list[int] = []
        real_killpg = os.killpg

        def spy(pid: int, sig: int) -> None:
            killed.append(sig)
            real_killpg(pid, sig)

        # Cancel once the shell runs: a fixed delay can end before a slow runner spawns it.
        started = anyio.Event()
        real_open_process = anyio.open_process

        async def open_process(*args: Any, **kwargs: Any) -> Any:
            process = await real_open_process(*args, **kwargs)
            started.set()
            return process

        monkeypatch.setattr("agentloom.tools.builtins.os.killpg", spy)
        monkeypatch.setattr("agentloom.tools.builtins.anyio.open_process", open_process)
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                partial(ShellCommandTool().execute, command="sleep 30 & sleep 30", timeout=60)
            )
            await started.wait()
            tg.cancel_scope.cancel()
        assert killed == [signal.SIGKILL]


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
