"""Sandbox enforcement for built-in tools.

Validates shell commands against an allowlist and restricts
file operations to allowed paths. When sandbox is disabled,
all operations pass through without validation.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
import socket
from pathlib import Path
from urllib.parse import urlparse

from agentloom.exceptions import SandboxViolationError

# Shell metacharacters that can chain, redirect, or inject commands.
# Checked BEFORE shlex parsing (raw string) to catch operators
# that shlex treats as literal tokens. Includes \n/\r which act as command
# separators in sh -c, and process substitutions ``<(...)`` / ``>(...)``.
_SHELL_OPERATOR_RE = re.compile(r"[|;&`\n\r]|\$\(|<\(|>\(|[<>]")

# Executables that can execute arbitrary code from their arguments and
# therefore defeat the command allowlist if permitted. Rejected by default
# even when listed in ``allowed_commands``; callers must opt in explicitly
# via ``danger_opt_in``.
_DANGEROUS_EXECUTABLES = frozenset(
    {
        "env",
        "sh",
        "bash",
        "zsh",
        "fish",
        "ksh",
        "dash",
        "xargs",
        "python",
        "python3",
        "node",
        "perl",
        "ruby",
        "php",
        "lua",
        "awk",
        "eval",
        "source",
        ".",
        "exec",
        "nc",
        "ncat",
        "socat",
        "ssh",
    }
)

# Network schemes permitted by default. Anything else (file://, gopher://,
# ftp://, data://, dict://) must be explicitly opted in via
# ``allowed_schemes``.
_DEFAULT_ALLOWED_SCHEMES = frozenset({"http", "https"})


# Networks denied by default for outbound webhook delivery when the workflow
# does not declare an explicit sandbox. Covers loopback, link-local (AWS /
# Azure / GCP metadata service all live in 169.254.169.254), RFC 1918, and the
# carrier-grade NAT range. Bypassable per workflow via
# ``SandboxConfig.allow_internal_webhook_targets``.
_DEFAULT_DENY_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
)


def _host_is_internal(hostname: str) -> bool:
    """Return ``True`` when *hostname* resolves to an internal address.

    Best-effort: skips DNS resolution and only inspects the literal token
    so a workflow can't smuggle ``169.254.169.254`` past the deny-list by
    using ``http://metadata.aws/`` (the destination still needs DNS resolution
    at HTTP time, but the literal host is what users typically configure).
    A hostname that looks like a name but resolves locally via ``/etc/hosts``
    is also caught when DNS lookup is permitted.
    """
    if not hostname:
        return False
    lower = hostname.lower()
    if lower == "localhost" or lower.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(lower)
    except ValueError:
        try:
            resolved = socket.gethostbyname(lower)
            ip = ipaddress.ip_address(resolved)
        except (OSError, ValueError):
            return False
    for net in _DEFAULT_DENY_NETWORKS:
        if ip.version != net.version:
            continue
        if ip in net:
            return True
    return False


def default_deny_webhook_target(url: str) -> str | None:
    """Return a reason if *url* should be blocked by the default deny-list.

    Used by ``send_webhook`` when the workflow has no explicit
    ``ToolSandbox`` — keeps loopback / metadata / RFC 1918 hosts off-limits
    so the SSRF surface stays closed by default. Returns ``None`` when the
    URL is acceptable.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _DEFAULT_ALLOWED_SCHEMES:
        return f"URL scheme {scheme!r} is not allowed for webhook delivery"
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return f"URL {url!r} has no hostname"
    if _host_is_internal(hostname):
        return (
            f"Hostname {hostname!r} resolves to an internal address; "
            f"set sandbox.allow_internal_webhook_targets=true to opt in"
        )
    return None


def _looks_like_path(token: str) -> bool:
    """Heuristic: does *token* look like a path argument?

    We treat any token containing ``/`` or matching ``.``/``..`` as a path.
    Flag-style arguments (``-n``, ``--flag``) and bare identifiers
    (``hello``) are not validated.
    """
    if not token:
        return False
    if token.startswith("-"):
        return False
    if token in (".", ".."):
        return True
    return "/" in token


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
        allowed_schemes: URL schemes permitted in ``validate_network``.
            Defaults to ``{"http", "https"}``. Opt in to ``file``,
            ``ftp``, etc. only when the workflow genuinely requires them.
        max_write_bytes: Maximum size in bytes for a single file write.
            ``None`` means unlimited.
        danger_opt_in: Explicit list of otherwise-dangerous executables
            (``bash``, ``python``, ``xargs`` …) that the workflow accepts
            the risk of. Without this, placing such names in
            ``allowed_commands`` has no effect.
        command_cwd: Directory against which relative path arguments are
            resolved during ``validate_command``. Defaults to the current
            working directory at validation time.
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
        allowed_schemes: list[str] | None = None,
        max_write_bytes: int | None = None,
        danger_opt_in: list[str] | None = None,
        command_cwd: str | None = None,
        allow_internal_webhook_targets: bool = False,
    ) -> None:
        self.enabled = enabled
        self._allowed_commands = set(allowed_commands or [])
        self._allowed_paths = [Path(p).resolve() for p in (allowed_paths or [])]
        self._readable_paths = [Path(p).resolve() for p in (readable_paths or [])]
        self._writable_paths = [Path(p).resolve() for p in (writable_paths or [])]
        self._allow_network = allow_network
        self._allowed_domains = {d.lower() for d in (allowed_domains or [])}
        self._allowed_schemes = (
            {s.lower() for s in allowed_schemes}
            if allowed_schemes
            else set(_DEFAULT_ALLOWED_SCHEMES)
        )
        self._max_write_bytes = max_write_bytes
        self._danger_opt_in = {e.lower() for e in (danger_opt_in or [])}
        self._command_cwd = Path(command_cwd).resolve() if command_cwd else None
        self.allow_internal_webhook_targets = allow_internal_webhook_targets

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

    def validate_command(self, command: str, *, cwd: str | None = None) -> None:
        """Validate a shell command against the allowlist.

        Blocks shell operators (``|``, ``;``, ``&``, `` ` ``, ``$()``,
        redirections, process substitution), rejects dangerous meta-executables
        (``env``, ``bash``, ``python -c`` …) unless explicitly opted in,
        checks the executable against the allowlist, and resolves every
        path-shaped argument (absolute or relative to *cwd*) against allowed
        directories.

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

        executable = Path(tokens[0]).name

        if executable not in self._allowed_commands:
            raise SandboxViolationError(
                "shell_command",
                f"Command {executable!r} not in allowlist. "
                f"Allowed: {sorted(self._allowed_commands) or '(none)'}",
            )

        if (
            executable.lower() in _DANGEROUS_EXECUTABLES
            and executable.lower() not in self._danger_opt_in
        ):
            raise SandboxViolationError(
                "shell_command",
                f"Executable {executable!r} can run arbitrary code from its "
                f"arguments and is blocked by default. Opt in explicitly via "
                f"ToolSandbox(danger_opt_in=[{executable!r}]) if the risk is "
                f"acceptable.",
            )

        all_paths = self._all_paths()
        if all_paths:
            base = self._command_cwd or (Path(cwd).resolve() if cwd else Path.cwd())
            for token in tokens[1:]:
                if not _looks_like_path(token):
                    continue
                candidate = Path(token)
                resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
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

        Note: the check is best-effort with respect to TOCTOU. A tool that
        opens the file long after ``validate_path`` returns should re-check
        the opened file descriptor's real path before trusting it.

        Raises:
            SandboxViolationError: If the path is outside allowed directories,
                or if it cannot be resolved (null bytes, oversized components,
                OS-level rejection — wrapped here so callers only need to
                handle one exception class).
        """
        if not self.enabled:
            return

        try:
            resolved = Path(path).resolve()
        except (ValueError, OSError) as exc:
            raise SandboxViolationError(
                tool_name,
                f"Cannot resolve path {path!r}: {exc}",
            ) from exc

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

        Rejects non-``http``/``https`` schemes by default (``file://``,
        ``gopher://``, ``ftp://``, ``data:`` …) regardless of host, so
        that an allowlisted hostname cannot be reused to fetch local
        files. When *allow_network* is ``True`` and *allowed_domains*
        is non-empty, the request domain must be in the allowlist.

        Raises:
            SandboxViolationError: If network access is blocked.
        """
        if not self.enabled:
            return

        if not self._allow_network:
            raise SandboxViolationError(
                "http_request", "Network access is blocked by sandbox policy"
            )

        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in self._allowed_schemes:
            raise SandboxViolationError(
                "http_request",
                f"URL scheme {scheme!r} is not allowed. Allowed: {sorted(self._allowed_schemes)}",
            )

        if self._allowed_domains:
            hostname = (parsed.hostname or "").lower()
            if hostname not in self._allowed_domains:
                raise SandboxViolationError(
                    "http_request",
                    f"Domain {hostname!r} not in allowed domains. "
                    f"Allowed: {sorted(self._allowed_domains)}",
                )

    def validate_webhook_url(self, url: str) -> None:
        """Validate a webhook destination URL.

        When the sandbox is enabled, applies the same allowlist as
        ``validate_network`` (schemes, domains). When the sandbox is
        disabled, applies the default deny-list so loopback / link-local /
        RFC 1918 / non-http(s) destinations are blocked unless the workflow
        explicitly opts in via ``allow_internal_webhook_targets``.

        Raises:
            SandboxViolationError: If the destination is blocked.
        """
        if self.enabled:
            self.validate_network(url)
            if self.allow_internal_webhook_targets:
                return
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
            if hostname and _host_is_internal(hostname):
                raise SandboxViolationError(
                    "webhook",
                    f"Webhook destination {hostname!r} resolves to an internal "
                    f"address; set sandbox.allow_internal_webhook_targets=true "
                    f"to opt in",
                )
            return

        # Sandbox disabled — apply the default deny-list unless the workflow
        # explicitly authorises internal targets.
        if self.allow_internal_webhook_targets:
            return
        reason = default_deny_webhook_target(url)
        if reason is not None:
            raise SandboxViolationError("webhook", reason)

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
