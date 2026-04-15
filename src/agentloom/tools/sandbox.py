"""Sandbox enforcement for built-in tools.

Validates shell commands against an allowlist and restricts
file operations to allowed paths. When sandbox is disabled,
all operations pass through without validation.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import urlparse

from agentloom.exceptions import SandboxViolationError

# Shell metacharacters that can chain, redirect, or inject commands.
# Checked BEFORE shlex parsing (raw string) to catch operators
# that shlex treats as literal tokens.  Includes \n/\r which act
# as command separators in sh -c.
_SHELL_OPERATOR_RE = re.compile(r"[|;&`<>\n\r]|\$\(|>>")


class ToolSandbox:
    """Validates tool operations against a sandbox policy.

    Args:
        enabled: Whether sandbox enforcement is active.
        allowed_commands: Shell command prefixes that are permitted
            (e.g., ``["echo", "cat", "ls"]``).  An empty list with
            sandbox enabled means NO commands are allowed.
        allowed_paths: Directory prefixes for **both** read and write
            file operations (e.g., ``["/tmp/workflows"]``).  An empty
            list with sandbox enabled means NO file access is allowed.
        readable_paths: Additional directories allowed for **read-only**
            access.  Combined with *allowed_paths* when validating reads.
        writable_paths: Additional directories allowed for **write-only**
            access.  Combined with *allowed_paths* when validating writes.
        allow_network: Whether HTTP requests are permitted.
        allowed_domains: When *allow_network* is ``True``, restrict
            requests to these domains (e.g., ``["api.openai.com"]``).
            An empty list means all domains are permitted.
        max_write_bytes: Maximum size in bytes for a single file write.
            ``None`` means unlimited.
    """

    def __init__(
        self,
        enabled: bool = False,
        allowed_commands: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        allow_network: bool = True,
        *,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_write_bytes: int | None = None,
    ) -> None:
        self.enabled = enabled
        self._allowed_commands = set(allowed_commands or [])
        self._allowed_paths = [Path(p).resolve() for p in (allowed_paths or [])]
        self._readable_paths = [Path(p).resolve() for p in (readable_paths or [])]
        self._writable_paths = [Path(p).resolve() for p in (writable_paths or [])]
        self._allow_network = allow_network
        self._allowed_domains = {d.lower() for d in (allowed_domains or [])}
        self._max_write_bytes = max_write_bytes

    def _paths_for_read(self) -> list[Path]:
        return self._allowed_paths + self._readable_paths

    def _paths_for_write(self) -> list[Path]:
        return self._allowed_paths + self._writable_paths

    def _all_paths(self) -> list[Path]:
        return self._allowed_paths + self._readable_paths + self._writable_paths

    @staticmethod
    def _is_within(resolved: Path, allowed: list[Path]) -> bool:
        """Return ``True`` if *resolved* is inside any of *allowed*."""
        for prefix in allowed:
            try:
                resolved.relative_to(prefix)
                return True
            except ValueError:
                continue
        return False

    def validate_command(self, command: str) -> None:
        """Validate a shell command against the allowlist.

        Blocks shell operators (``|``, ``;``, ``&``, `` ` ``, ``$()``),
        checks the executable against the allowlist, and validates any
        absolute-path arguments against allowed directories.

        Raises:
            SandboxViolationError: If the command is not allowed.
        """
        if not self.enabled:
            return

        if _SHELL_OPERATOR_RE.search(command):
            raise SandboxViolationError(
                "shell_command",
                f"Shell operators are not allowed in sandboxed commands: {command!r}",
            )

        try:
            tokens = shlex.split(command)
        except ValueError:
            raise SandboxViolationError("shell_command", f"Cannot parse command: {command!r}")

        if not tokens:
            return

        executable = tokens[0]

        executable = Path(executable).name

        if executable not in self._allowed_commands:
            raise SandboxViolationError(
                "shell_command",
                f"Command {executable!r} not in allowlist. "
                f"Allowed: {sorted(self._allowed_commands) or '(none)'}",
            )

        all_paths = self._all_paths()
        if all_paths:
            for token in tokens[1:]:
                if token.startswith("/"):
                    resolved = Path(token).resolve()
                    if not self._is_within(resolved, all_paths):
                        raise SandboxViolationError(
                            "shell_command",
                            f"Path argument {str(resolved)!r} not within allowed directories",
                        )

    def validate_path(self, path: str, *, writable: bool = False, tool_name: str = "file") -> None:
        """Validate a file path is within allowed directories.

        Resolves symlinks and relative paths before checking.  When
        *writable* is ``True`` the path is checked against
        ``allowed_paths + writable_paths``; otherwise against
        ``allowed_paths + readable_paths``.

        Raises:
            SandboxViolationError: If the path is outside allowed directories.
        """
        if not self.enabled:
            return

        resolved = Path(path).resolve()
        paths = self._paths_for_write() if writable else self._paths_for_read()

        if not self._is_within(resolved, paths):
            label = "writable" if writable else "readable"
            raise SandboxViolationError(
                tool_name,
                f"Path {str(resolved)!r} not within allowed {label} directories. "
                f"Allowed: {[str(p) for p in paths] or '(none)'}",
            )

    def validate_network(self, url: str) -> None:
        """Validate that network access is permitted.

        When *allow_network* is ``True`` and *allowed_domains* is
        non-empty, the request domain must be in the allowlist.

        Raises:
            SandboxViolationError: If network access is blocked.
        """
        if not self.enabled:
            return

        if not self._allow_network:
            raise SandboxViolationError(
                "http_request", "Network access is blocked by sandbox policy"
            )

        if self._allowed_domains:
            hostname = (urlparse(url).hostname or "").lower()
            if hostname not in self._allowed_domains:
                raise SandboxViolationError(
                    "http_request",
                    f"Domain {hostname!r} not in allowed domains. "
                    f"Allowed: {sorted(self._allowed_domains)}",
                )

    def validate_write_size(self, size: int, tool_name: str = "file_write") -> None:
        """Validate that a write payload does not exceed the size limit.

        Raises:
            SandboxViolationError: If *size* exceeds *max_write_bytes*.
        """
        if not self.enabled:
            return

        if self._max_write_bytes is not None and size > self._max_write_bytes:
            raise SandboxViolationError(
                tool_name,
                f"Write size {size} bytes exceeds limit of {self._max_write_bytes} bytes",
            )
