"""Tests for tool sandbox enforcement."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

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
from agentloom.tools.sandbox import ToolSandbox, _looks_like_path


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


class TestLooksLikePath:
    """Edge cases of the path-shape heuristic used by argument validation."""

    def test_empty_token_is_not_a_path(self) -> None:
        assert _looks_like_path("") is False

    def test_flag_token_is_not_a_path(self) -> None:
        assert _looks_like_path("-n") is False
        assert _looks_like_path("--flag") is False

    def test_bare_identifier_is_not_a_path(self) -> None:
        assert _looks_like_path("hello") is False

    def test_dot_and_dotdot_are_paths(self) -> None:
        assert _looks_like_path(".") is True
        assert _looks_like_path("..") is True

    def test_tokens_with_slash_are_paths(self) -> None:
        assert _looks_like_path("foo/bar") is True
        assert _looks_like_path("/abs/path") is True
        assert _looks_like_path("./rel") is True


class TestValidatePathSurfacesViolationOnUnresolvable:
    """Null-byte and other ``Path.resolve()`` failures raise ``SandboxViolationError``.

    Previously the raw ``ValueError`` / ``OSError`` leaked through, so
    callers that caught only ``SandboxViolationError`` missed the case
    and surfaced it as an internal error instead of a sandbox refusal.
    """

    def test_null_byte_path_raises_sandbox_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmp])
            with pytest.raises(SandboxViolationError) as excinfo:
                sandbox.validate_path("/tmp/x\x00/etc/passwd")
            assert "Cannot resolve path" in str(excinfo.value)

    def test_disabled_sandbox_does_not_validate_null_byte(self) -> None:
        # When the sandbox is off the validate call is a no-op; we don't
        # synthesize a violation for the disabled case.
        sandbox = ToolSandbox(enabled=False)
        sandbox.validate_path("/tmp/x\x00/etc/passwd")

    def test_runtime_error_wraps_to_sandbox_violation(self, monkeypatch: Any) -> None:
        # ``Path.resolve()`` raises ``RuntimeError`` on symlink loops on
        # Python < 3.13. Python 3.13 rewrote ``pathlib`` and no longer
        # raises in non-strict mode, so the symlink-loop trigger isn't
        # portable as a black-box test. Patch the ``Path`` symbol in
        # ``sandbox`` AFTER constructing the sandbox (so ``__init__``
        # processes ``allowed_paths`` with the real Path), then exercise
        # the defensive ``RuntimeError`` branch in ``validate_path``.

        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmp])

            class _BoomPath:
                def __init__(self, _path: object) -> None:
                    pass

                def resolve(self) -> _BoomPath:
                    raise RuntimeError("Symlink loop from '...'")

            monkeypatch.setattr("agentloom.tools.sandbox.Path", _BoomPath)
            with pytest.raises(SandboxViolationError):
                sandbox.validate_path("/tmp/whatever")

    @pytest.mark.parametrize("bad_input", [None, 42, b"/tmp/bytes"])
    def test_non_string_input_wraps_to_sandbox_violation(self, bad_input: object) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ToolSandbox(enabled=True, allowed_paths=[tmp])
            with pytest.raises(SandboxViolationError):
                sandbox.validate_path(bad_input)  # type: ignore[arg-type]


class TestValidateWebhookUrl:
    """Webhook destination gate.

    Enabled sandbox applies the standard allowlist (schemes, domains).
    Disabled sandbox applies the default deny-list (loopback, link-local,
    RFC 1918, non-http(s)) so workflows that omit a sandbox config still
    can't ship state to internal services. The opt-in flag
    ``allow_internal_webhook_targets`` exists for in-cluster notifications.
    """

    def test_enabled_sandbox_enforces_allowed_domains(self) -> None:
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allowed_domains=["api.openai.com"],
            allowed_schemes=["https"],
        )
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url(
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/role"
            )

    def test_enabled_sandbox_rejects_loopback_even_when_allowed_domains_empty(
        self,
    ) -> None:
        sandbox = ToolSandbox(enabled=True, allow_network=True)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://127.0.0.1:8080/hook")

    def test_disabled_sandbox_blocks_loopback_by_default(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://127.0.0.1:8080/hook")

    def test_disabled_sandbox_blocks_link_local_metadata(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url(
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/role"
            )

    def test_disabled_sandbox_blocks_rfc1918(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        for url in (
            "http://10.0.0.1/x",
            "http://172.16.0.5/x",
            "http://192.168.1.1/x",
        ):
            with pytest.raises(SandboxViolationError):
                sandbox.validate_webhook_url(url)

    def test_disabled_sandbox_blocks_non_http_schemes(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        for url in ("file:///etc/passwd", "gopher://example.com/x", "dict://x/y"):
            with pytest.raises(SandboxViolationError):
                sandbox.validate_webhook_url(url)

    def test_disabled_sandbox_allows_public_https(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        sandbox.validate_webhook_url("https://hooks.slack.com/services/T00/B00/abc")

    def test_opt_in_unblocks_internal(self) -> None:
        sandbox = ToolSandbox(enabled=False, allow_internal_webhook_targets=True)
        sandbox.validate_webhook_url("http://127.0.0.1:8080/hook")

    def test_bare_localhost_name_blocked(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://localhost:8080/x")

    @pytest.mark.parametrize(
        "url",
        [
            "http://[::ffff:169.254.169.254]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://[::ffff:192.168.1.1]/",
        ],
    )
    def test_ipv4_mapped_ipv6_blocked(self, url: str) -> None:
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://0.0.0.0:6379/",
            "http://[::]:8080/",
        ],
    )
    def test_unspecified_addresses_blocked(self, url: str) -> None:
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url(url)

    def test_trailing_dot_hostname_normalised(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://127.0.0.1./")
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://localhost./")

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "data:text/html,whatever",
            "javascript:alert(1)",
            "gopher://example.com/x",
            "ftp://example.com/x",
        ],
    )
    def test_opt_in_does_not_disable_scheme_deny(self, url: str) -> None:
        sandbox = ToolSandbox(enabled=False, allow_internal_webhook_targets=True)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url(url)

    def test_opt_in_allows_legitimate_internal_target(self) -> None:
        sandbox = ToolSandbox(enabled=False, allow_internal_webhook_targets=True)
        sandbox.validate_webhook_url("http://127.0.0.1:8081/notify")

    def test_enabled_sandbox_with_opt_in_still_blocks_non_allowed_scheme(self) -> None:
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allow_internal_webhook_targets=True,
            allowed_schemes=["https"],
        )
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://127.0.0.1/hook")

    def test_percent_encoded_hostname_decoded_before_check(self) -> None:
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://%6c%6f%63%61%6c%68%6f%73%74:8080/x")

    def test_aaaa_only_dns_caught_via_getaddrinfo(self, monkeypatch: Any) -> None:
        import socket as _socket

        def fake_getaddrinfo(host: str, *args: object, **kwargs: object):
            return [
                (
                    _socket.AF_INET6,
                    _socket.SOCK_STREAM,
                    0,
                    "",
                    ("::1", 0, 0, 0),
                ),
            ]

        monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
        sandbox = ToolSandbox(enabled=False)
        with pytest.raises(SandboxViolationError):
            sandbox.validate_webhook_url("http://attacker-aaaa-only.example/x")


class TestValidateCommandFlagEmbeddedPath:
    """``--key=value`` / ``of=path`` style flags carry paths that the
    original ``_looks_like_path`` skipped entirely.
    """

    def test_double_dash_flag_with_path_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ToolSandbox(
                enabled=True,
                allowed_commands=["tee"],
                allowed_paths=[tmp],
            )
            with pytest.raises(SandboxViolationError):
                sandbox.validate_command("tee --output=/etc/passwd")

    def test_dd_style_flag_with_path_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ToolSandbox(
                enabled=True,
                allowed_commands=["dd"],
                allowed_paths=[tmp],
            )
            with pytest.raises(SandboxViolationError):
                sandbox.validate_command("dd of=/etc/passwd")

    def test_flag_with_path_inside_allowlist_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = ToolSandbox(
                enabled=True,
                allowed_commands=["tee"],
                allowed_paths=[tmp],
            )
            sandbox.validate_command(f"tee --output={tmp}/log")


class TestSandboxHelperEdges:
    """Edges of the helper functions that the main tests don't exercise."""

    def test_default_deny_target_rejects_url_without_hostname(self) -> None:
        from agentloom.tools.sandbox import default_deny_webhook_target

        reason = default_deny_webhook_target("http:///path")
        assert reason is not None and "no hostname" in reason

    def test_default_deny_target_allows_public_https(self) -> None:
        from agentloom.tools.sandbox import default_deny_webhook_target

        # Hits the ``return None`` tail when nothing is wrong.
        assert default_deny_webhook_target("https://hooks.example.com/x") is None

    def test_host_is_internal_empty_returns_false(self) -> None:
        from agentloom.tools.sandbox import _host_is_internal

        assert _host_is_internal("") is False

    def test_host_is_internal_dot_only_returns_false(self) -> None:
        from agentloom.tools.sandbox import _host_is_internal

        # ``.`` strips to empty after ``rstrip('.')``.
        assert _host_is_internal(".") is False

    def test_host_is_internal_unresolvable_returns_false(self) -> None:
        from agentloom.tools.sandbox import _host_is_internal

        # ``getaddrinfo`` raises for a clearly bogus name; the helper
        # falls back to ``False`` so the sandbox doesn't over-block.
        assert _host_is_internal("nonexistent.invalid.example.test") is False

    def test_ip_is_internal_cgnat_via_explicit_deny_list(self) -> None:
        # 100.64/10 isn't covered by ``ipaddress.is_private`` on Python
        # 3.12, so the explicit ``_DEFAULT_DENY_NETWORKS`` fallback is
        # what catches it.
        import ipaddress

        from agentloom.tools.sandbox import _ip_is_internal

        assert _ip_is_internal(ipaddress.ip_address("100.64.0.1")) is True

    def test_extract_path_candidates_flag_with_empty_value(self) -> None:
        from agentloom.tools.sandbox import _extract_path_candidates

        # ``--key=`` (empty value) and ``--key=bare`` (no slash) yield
        # nothing — only path-shaped values are extracted.
        assert _extract_path_candidates("--output=") == []
        assert _extract_path_candidates("--name=alice") == []

    def test_extract_path_candidates_bare_dotdot_returns_self(self) -> None:
        from agentloom.tools.sandbox import _extract_path_candidates

        assert _extract_path_candidates("..") == [".."]

    def test_extract_path_candidates_empty_token(self) -> None:
        from agentloom.tools.sandbox import _extract_path_candidates

        assert _extract_path_candidates("") == []


class TestValidateWebhookUrlEnabledSandboxBranches:
    """Cover the enabled-sandbox path where ``allow_internal_webhook_targets``
    is False and the internal-host check fires after ``validate_network``."""

    def test_enabled_sandbox_with_opt_in_skips_internal_check(self) -> None:
        # Opt-in branch: validate_network passes, opt-in returns early
        # before the internal-host gate.
        sandbox = ToolSandbox(
            enabled=True,
            allow_network=True,
            allow_internal_webhook_targets=True,
        )
        sandbox.validate_webhook_url("http://127.0.0.1/hook")

    def test_enabled_sandbox_url_without_hostname_passes_internal_check(
        self,
    ) -> None:
        # URL with no hostname (``http://``) skips the internal-host
        # block since there's nothing to classify — the upstream
        # ``validate_network`` already accepted it.
        sandbox = ToolSandbox(enabled=True, allow_network=True)
        # ``http:///path`` has empty hostname; passes the internal check.
        sandbox.validate_webhook_url("http:///path")

    def test_default_deny_target_rejects_non_http_scheme(self) -> None:
        # ``default_deny_webhook_target`` is also called directly by
        # ``send_webhook`` when no sandbox is provided. Cover the
        # scheme-deny branch via the direct entry point.
        from agentloom.tools.sandbox import default_deny_webhook_target

        reason = default_deny_webhook_target("ftp://example.com/x")
        assert reason is not None and "scheme 'ftp'" in reason

    def test_host_is_internal_resolves_ipv4_record(self, monkeypatch: Any) -> None:
        # AAAA-only case is already covered; pin the IPv4 path too so a
        # future refactor doesn't quietly skip ``AF_INET`` records.
        import socket as _socket

        def fake_getaddrinfo(host: str, *args: object, **kwargs: object):
            return [
                (
                    _socket.AF_INET,
                    _socket.SOCK_STREAM,
                    0,
                    "",
                    ("127.0.0.1", 0),
                ),
            ]

        monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
        from agentloom.tools.sandbox import _host_is_internal

        assert _host_is_internal("host-with-a-record.example") is True

    def test_host_is_internal_returns_false_on_getaddrinfo_error(self, monkeypatch: Any) -> None:
        # Defence: ``getaddrinfo`` raises ``OSError`` for unresolvable
        # hostnames; the helper falls back to ``False`` so the sandbox
        # doesn't over-block public names the resolver couldn't reach.
        def boom(*_args: object, **_kwargs: object):
            raise OSError("no such host")

        monkeypatch.setattr("socket.getaddrinfo", boom)
        from agentloom.tools.sandbox import _host_is_internal

        assert _host_is_internal("unresolvable.test") is False

    def test_host_is_internal_skips_unknown_socket_family(self, monkeypatch: Any) -> None:
        # An exotic address family (Unix, IPX, ...) skips classification
        # rather than crashing.
        import socket as _socket

        def fake_getaddrinfo(host: str, *args: object, **kwargs: object):
            return [
                (_socket.AF_UNIX, _socket.SOCK_STREAM, 0, "", ("/tmp/sock",)),
            ]

        monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
        from agentloom.tools.sandbox import _host_is_internal

        assert _host_is_internal("exotic.example") is False

    def test_host_is_internal_skips_malformed_sockaddr(self, monkeypatch: Any) -> None:
        # ``ipaddress.ip_address`` raising on a malformed sockaddr is
        # caught and the loop moves on.
        import socket as _socket

        def fake_getaddrinfo(host: str, *args: object, **kwargs: object):
            return [
                (_socket.AF_INET, _socket.SOCK_STREAM, 0, "", ("not-an-ip", 0)),
            ]

        monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
        from agentloom.tools.sandbox import _host_is_internal

        assert _host_is_internal("malformed.example") is False
