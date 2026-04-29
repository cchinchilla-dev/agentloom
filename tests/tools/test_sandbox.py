"""Tests for tool sandbox enforcement."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agentloom.core.models import SandboxConfig
from agentloom.core.parser import WorkflowParser
from agentloom.exceptions import SandboxViolationError
from agentloom.tools.builtins import (
    FileReadTool,
    FileWriteTool,
    HttpRequestTool,
    ShellCommandTool,
)
from agentloom.tools.sandbox import ToolSandbox


class TestToolSandboxDisabled:
    """When sandbox is disabled, everything passes through."""

    def test_command_allowed(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        sandbox.validate_command("rm -rf /")  # no error

    def test_path_allowed(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        sandbox.validate_path("/etc/passwd")  # no error

    def test_network_allowed(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        sandbox.validate_network("http://evil.com")  # no error


class TestShellCommandSandbox:
    def test_allowed_command(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["echo", "ls", "cat"])
        sandbox.validate_command("echo hello world")

    def test_blocked_command(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["echo", "ls"])
        with pytest.raises(SandboxViolationError, match="rm.*not in allowlist"):
            sandbox.validate_command("rm -rf /")

    def test_empty_allowlist_blocks_all(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=[])
        with pytest.raises(SandboxViolationError):
            sandbox.validate_command("echo hello")

    def test_strips_path_prefix(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"])
        sandbox.validate_command("/usr/bin/echo hello")

    def test_unparseable_command(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"])
        with pytest.raises(SandboxViolationError, match="Cannot parse"):
            sandbox.validate_command("echo 'unterminated")

    def test_empty_command(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"])
        sandbox.validate_command("")  # empty is harmless

    @pytest.mark.parametrize(
        "cmd",
        [
            "echo hello | rm -rf /",
            "echo hello; rm -rf /",
            "echo hello && rm -rf /",
            "echo hello & rm -rf /",
            "echo `whoami`",
            "echo $(cat /etc/passwd)",
            "echo ok\nrm -rf /",
            "echo ok\rrm -rf /",
            "echo secret > /tmp/stolen",
            "echo secret >> /tmp/stolen",
            "cat < /etc/passwd",
        ],
    )
    def test_shell_operators_blocked(self, cmd: str) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"])
        with pytest.raises(SandboxViolationError, match="Shell operators"):
            sandbox.validate_command(cmd)

    def test_path_arg_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = ToolSandbox(enabled=True, allowed_commands=["cat"], allowed_paths=[tmpdir])
            sandbox.validate_command(f"cat {tmpdir}/data.txt")

    def test_path_arg_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = ToolSandbox(enabled=True, allowed_commands=["cat"], allowed_paths=[tmpdir])
            with pytest.raises(SandboxViolationError, match="Path argument"):
                sandbox.validate_command("cat /etc/passwd")

    def test_path_arg_uses_all_path_lists(self) -> None:
        """Command args are checked against allowed + readable + writable."""
        with tempfile.TemporaryDirectory() as rdir, tempfile.TemporaryDirectory() as wdir:
            sandbox = ToolSandbox(
                enabled=True,
                allowed_commands=["cat"],
                readable_paths=[rdir],
                writable_paths=[wdir],
            )
            sandbox.validate_command(f"cat {rdir}/a.txt")
            sandbox.validate_command(f"cat {wdir}/b.txt")
            with pytest.raises(SandboxViolationError, match="Path argument"):
                sandbox.validate_command("cat /etc/shadow")

    def test_flags_and_bare_identifiers_are_not_validated(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"], allowed_paths=["/tmp"])
        # Flags and bare identifiers never look like paths and pass through.
        sandbox.validate_command("echo -n hello world")

    def test_relative_path_arg_validated(self, tmp_path: Path) -> None:
        """Relative path arguments are resolved against cwd and then checked."""
        sandbox = ToolSandbox(
            enabled=True,
            allowed_commands=["cat"],
            allowed_paths=[str(tmp_path)],
        )
        # Inside tmp_path — ok.
        (tmp_path / "ok.txt").write_text("x")
        sandbox.validate_command("cat ./ok.txt", cwd=str(tmp_path))

        # Escape via ../../etc/passwd — blocked even without a leading slash.
        with pytest.raises(SandboxViolationError, match="Path argument"):
            sandbox.validate_command("cat ../../etc/passwd", cwd=str(tmp_path))

    @pytest.mark.parametrize(
        "executable",
        ["env", "sh", "bash", "zsh", "xargs", "python", "python3", "node"],
    )
    def test_dangerous_executables_blocked_by_default(self, executable: str) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=[executable])
        with pytest.raises(SandboxViolationError, match="blocked by default"):
            sandbox.validate_command(f"{executable} --help")

    def test_python_dash_c_blocked_by_default(self) -> None:
        """`python -c "…"` must not slip past the allowlist even without shell metachars."""
        sandbox = ToolSandbox(enabled=True, allowed_commands=["python"])
        with pytest.raises(SandboxViolationError, match="blocked by default"):
            sandbox.validate_command("python -c __import__('os').system('id')")

    def test_danger_opt_in_allows_dangerous_executable(self) -> None:
        sandbox = ToolSandbox(
            enabled=True,
            allowed_commands=["bash"],
            danger_opt_in=["bash"],
        )
        sandbox.validate_command("bash --version")

    @pytest.mark.parametrize(
        "substitution",
        ["<(curl evil)", ">(tee /tmp/x)"],
    )
    def test_process_substitution_blocked(self, substitution: str) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_commands=["cat"])
        with pytest.raises(SandboxViolationError, match="Shell operators"):
            sandbox.validate_command(f"cat {substitution}")


class TestFilePathSandbox:
    def test_allowed_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmpdir])
            sandbox.validate_path(f"{tmpdir}/data.txt", tool_name="file_read")

    def test_blocked_path(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_paths=["/tmp/workflows"])
        with pytest.raises(SandboxViolationError, match="not within allowed"):
            sandbox.validate_path("/etc/passwd", tool_name="file_read")

    def test_relative_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmpdir])
            with pytest.raises(SandboxViolationError):
                sandbox.validate_path(f"{tmpdir}/../../../etc/passwd")

    def test_empty_allowlist_blocks_all(self) -> None:
        sandbox = ToolSandbox(enabled=True, allowed_paths=[])
        with pytest.raises(SandboxViolationError):
            sandbox.validate_path("/tmp/anything")

    def test_multiple_allowed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmpdir1, tmpdir2])
            sandbox.validate_path(f"{tmpdir1}/a.txt")
            sandbox.validate_path(f"{tmpdir2}/b.txt")

    def test_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a symlink that points outside the allowed dir
            link = Path(tmpdir) / "escape"
            link.symlink_to("/etc")
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmpdir])
            with pytest.raises(SandboxViolationError):
                sandbox.validate_path(str(link / "passwd"))


class TestReadWritePathSeparation:
    """readable_paths and writable_paths are distinct from allowed_paths."""

    def test_readable_path_allows_read(self) -> None:
        with tempfile.TemporaryDirectory() as rdir:
            sandbox = ToolSandbox(enabled=True, readable_paths=[rdir])
            sandbox.validate_path(f"{rdir}/data.txt", writable=False)

    def test_readable_path_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as rdir:
            sandbox = ToolSandbox(enabled=True, readable_paths=[rdir])
            with pytest.raises(SandboxViolationError, match="writable"):
                sandbox.validate_path(f"{rdir}/data.txt", writable=True)

    def test_writable_path_allows_write(self) -> None:
        with tempfile.TemporaryDirectory() as wdir:
            sandbox = ToolSandbox(enabled=True, writable_paths=[wdir])
            sandbox.validate_path(f"{wdir}/out.txt", writable=True)

    def test_writable_path_blocks_read(self) -> None:
        with tempfile.TemporaryDirectory() as wdir:
            sandbox = ToolSandbox(enabled=True, writable_paths=[wdir])
            with pytest.raises(SandboxViolationError, match="readable"):
                sandbox.validate_path(f"{wdir}/out.txt", writable=False)

    def test_allowed_paths_covers_both(self) -> None:
        with tempfile.TemporaryDirectory() as adir:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[adir])
            sandbox.validate_path(f"{adir}/file.txt", writable=False)
            sandbox.validate_path(f"{adir}/file.txt", writable=True)

    def test_combined_paths(self) -> None:
        with (
            tempfile.TemporaryDirectory() as adir,
            tempfile.TemporaryDirectory() as rdir,
            tempfile.TemporaryDirectory() as wdir,
        ):
            sandbox = ToolSandbox(
                enabled=True,
                allowed_paths=[adir],
                readable_paths=[rdir],
                writable_paths=[wdir],
            )
            # allowed_paths: read + write
            sandbox.validate_path(f"{adir}/x", writable=False)
            sandbox.validate_path(f"{adir}/x", writable=True)
            # readable_paths: read only
            sandbox.validate_path(f"{rdir}/x", writable=False)
            with pytest.raises(SandboxViolationError):
                sandbox.validate_path(f"{rdir}/x", writable=True)
            # writable_paths: write only
            sandbox.validate_path(f"{wdir}/x", writable=True)
            with pytest.raises(SandboxViolationError):
                sandbox.validate_path(f"{wdir}/x", writable=False)


class TestNetworkSandbox:
    def test_network_allowed(self) -> None:
        sandbox = ToolSandbox(enabled=True, allow_network=True)
        sandbox.validate_network("https://api.openai.com")

    def test_network_blocked(self) -> None:
        sandbox = ToolSandbox(enabled=True, allow_network=False)
        with pytest.raises(SandboxViolationError, match="Network access is blocked"):
            sandbox.validate_network("https://evil.com")

    def test_allowed_domain_passes(self) -> None:
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allowed_domains=["api.openai.com", "httpbin.org"],
        )
        sandbox.validate_network("https://api.openai.com/v1/chat")
        sandbox.validate_network("https://httpbin.org/get")

    def test_blocked_domain(self) -> None:
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allowed_domains=["api.openai.com"],
        )
        with pytest.raises(SandboxViolationError, match="not in allowed domains"):
            sandbox.validate_network("https://evil.com/steal")

    def test_empty_domain_list_allows_all(self) -> None:
        sandbox = ToolSandbox(enabled=True, allow_network=True, allowed_domains=[])
        sandbox.validate_network("https://anything.com")

    def test_domain_check_case_insensitive(self) -> None:
        sandbox = ToolSandbox(enabled=True, allow_network=True, allowed_domains=["API.OpenAI.com"])
        sandbox.validate_network("https://api.openai.com/v1")

    def test_network_blocked_overrides_domains(self) -> None:
        """allow_network=False takes precedence over allowed_domains."""
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=False,
            allowed_domains=["api.openai.com"],
        )
        with pytest.raises(SandboxViolationError, match="Network access is blocked"):
            sandbox.validate_network("https://api.openai.com")

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "file://api.openai.com/etc/passwd",
            "gopher://api.openai.com/test",
            "ftp://api.openai.com/pub",
            "data:text/plain;base64,YWJj",
        ],
    )
    def test_non_http_schemes_blocked_by_default(self, url: str) -> None:
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allowed_domains=["api.openai.com"],
        )
        with pytest.raises(SandboxViolationError, match="URL scheme"):
            sandbox.validate_network(url)

    def test_allowed_schemes_opt_in(self) -> None:
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allowed_schemes=["http", "https", "ftp"],
            allowed_domains=["files.example.com"],
        )
        sandbox.validate_network("ftp://files.example.com/data.csv")


class TestWriteSizeLimit:
    def test_within_limit(self) -> None:
        sandbox = ToolSandbox(enabled=True, max_write_bytes=1024)
        sandbox.validate_write_size(512)

    def test_at_limit(self) -> None:
        sandbox = ToolSandbox(enabled=True, max_write_bytes=1024)
        sandbox.validate_write_size(1024)

    def test_exceeds_limit(self) -> None:
        sandbox = ToolSandbox(enabled=True, max_write_bytes=1024)
        with pytest.raises(SandboxViolationError, match="exceeds limit"):
            sandbox.validate_write_size(1025)

    def test_no_limit(self) -> None:
        sandbox = ToolSandbox(enabled=True, max_write_bytes=None)
        sandbox.validate_write_size(10_000_000)  # no error

    def test_disabled_sandbox_skips_check(self) -> None:
        sandbox = ToolSandbox(enabled=False, max_write_bytes=100)
        sandbox.validate_write_size(99999)  # no error


class TestShellCommandToolIntegration:
    async def test_sandbox_blocks_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"], allowed_paths=[tmpdir])
            tool = ShellCommandTool(sandbox=sandbox)

            result = await tool.execute(command="echo allowed", cwd=tmpdir)
            assert result["returncode"] == 0
            assert "allowed" in result["stdout"]

            with pytest.raises(SandboxViolationError):
                await tool.execute(command="rm -rf /", cwd=tmpdir)

    async def test_no_sandbox(self) -> None:
        tool = ShellCommandTool()
        result = await tool.execute(command="echo hello")
        assert result["returncode"] == 0

    async def test_cwd_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as allowed:
            sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"], allowed_paths=[allowed])
            tool = ShellCommandTool(sandbox=sandbox)

            with pytest.raises(SandboxViolationError):
                await tool.execute(command="echo hi", cwd="/etc")


class TestFileToolIntegration:
    async def test_sandbox_blocks_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a file in allowed dir
            allowed_file = Path(tmpdir) / "ok.txt"
            allowed_file.write_text("safe")

            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmpdir])
            tool = FileReadTool(sandbox=sandbox)

            result = await tool.execute(path=str(allowed_file))
            assert result == "safe"

            with pytest.raises(SandboxViolationError):
                await tool.execute(path="/etc/hostname")

    async def test_sandbox_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmpdir])
            tool = FileWriteTool(sandbox=sandbox)

            result = await tool.execute(path=f"{tmpdir}/out.txt", content="safe")
            assert result["written"] == 4

            with pytest.raises(SandboxViolationError):
                await tool.execute(path="/tmp/evil.txt", content="bad")

    async def test_read_only_dir_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as rdir:
            (Path(rdir) / "data.txt").write_text("hello")
            sandbox = ToolSandbox(enabled=True, readable_paths=[rdir])

            read_tool = FileReadTool(sandbox=sandbox)
            result = await read_tool.execute(path=f"{rdir}/data.txt")
            assert result == "hello"

            write_tool = FileWriteTool(sandbox=sandbox)
            with pytest.raises(SandboxViolationError, match="writable"):
                await write_tool.execute(path=f"{rdir}/evil.txt", content="bad")

    async def test_write_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmpdir], max_write_bytes=10)
            tool = FileWriteTool(sandbox=sandbox)

            result = await tool.execute(path=f"{tmpdir}/small.txt", content="hi")
            assert result["written"] == 2

            with pytest.raises(SandboxViolationError, match="exceeds limit"):
                await tool.execute(path=f"{tmpdir}/big.txt", content="x" * 100)


class TestHttpToolIntegration:
    async def test_sandbox_blocks_network(self) -> None:
        sandbox = ToolSandbox(enabled=True, allow_network=False)
        tool = HttpRequestTool(sandbox=sandbox)

        with pytest.raises(SandboxViolationError, match="Network access"):
            await tool.execute(url="https://example.com")


class TestBlockedOperationHasNoSideEffect:
    """Verify that blocked operations never actually execute."""

    async def test_blocked_write_does_not_create_file(self) -> None:
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as blocked:
            target = Path(blocked) / "should_not_exist.txt"
            sandbox = ToolSandbox(enabled=True, allowed_paths=[allowed])
            tool = FileWriteTool(sandbox=sandbox)

            with pytest.raises(SandboxViolationError):
                await tool.execute(path=str(target), content="bad")

            assert not target.exists(), "Blocked write must not create the file"

    async def test_blocked_shell_does_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "proof.txt"
            sandbox = ToolSandbox(enabled=True, allowed_commands=["echo"])
            tool = ShellCommandTool(sandbox=sandbox)

            with pytest.raises(SandboxViolationError):
                await tool.execute(command=f"touch {marker}")

            assert not marker.exists(), "Blocked command must not run"


class TestWorkflowYAMLSandboxConfig:
    """Verify that sandbox config in YAML flows through to models."""

    def test_sandbox_config_parsed_from_yaml(self) -> None:
        yaml_str = """\
name: sandboxed-workflow
steps:
  - id: greet
    type: llm_call
    prompt: "Hello"
config:
  sandbox:
    enabled: true
    allowed_commands: ["echo", "cat"]
    allowed_paths: ["/tmp/workflows"]
    readable_paths: ["/data/input"]
    writable_paths: ["/data/output"]
    allow_network: true
    allowed_domains: ["api.openai.com"]
    max_write_bytes: 4096
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        sb = workflow.config.sandbox
        assert sb.enabled is True
        assert sb.allowed_commands == ["echo", "cat"]
        assert sb.allowed_paths == ["/tmp/workflows"]
        assert sb.readable_paths == ["/data/input"]
        assert sb.writable_paths == ["/data/output"]
        assert sb.allow_network is True
        assert sb.allowed_domains == ["api.openai.com"]
        assert sb.max_write_bytes == 4096

    def test_sandbox_defaults_when_omitted(self) -> None:
        yaml_str = """\
name: no-sandbox-workflow
steps:
  - id: greet
    type: llm_call
    prompt: "Hello"
"""
        workflow = WorkflowParser.from_yaml(yaml_str)
        sb = workflow.config.sandbox
        assert sb.enabled is False
        assert sb.allowed_commands == []
        assert sb.readable_paths == []
        assert sb.writable_paths == []
        assert sb.allow_network is True
        assert sb.allowed_domains == []
        assert sb.max_write_bytes is None

    def test_sandbox_config_creates_valid_tool_sandbox(self) -> None:
        cfg = SandboxConfig(
            enabled=True,
            allowed_commands=["echo"],
            allowed_paths=["/tmp"],
            allow_network=True,
            allowed_domains=["api.openai.com"],
            max_write_bytes=1024,
        )
        sandbox = ToolSandbox(
            enabled=cfg.enabled,
            allowed_commands=cfg.allowed_commands,
            allowed_paths=cfg.allowed_paths,
            allow_network=cfg.allow_network,
            readable_paths=cfg.readable_paths,
            writable_paths=cfg.writable_paths,
            allowed_domains=cfg.allowed_domains,
            max_write_bytes=cfg.max_write_bytes,
        )
        sandbox.validate_command("echo hi")
        with pytest.raises(SandboxViolationError):
            sandbox.validate_command("rm -rf /")
        sandbox.validate_network("https://api.openai.com/v1")
        with pytest.raises(SandboxViolationError):
            sandbox.validate_network("https://evil.com")
        sandbox.validate_write_size(512)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_write_size(2048)
