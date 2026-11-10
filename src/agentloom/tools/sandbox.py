"""Sandbox enforcement for built-in tools.

Validates shell commands against an allowlist, restricts file operations
to allowed paths and gates network access. When the sandbox is disabled,
operations pass through unchecked, except webhook destinations, which keep
the default deny-list.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import re
import shlex
import shutil
import socket
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import httpx

from agentloom.exceptions import SandboxViolationError

# Shell operators that chain, redirect or substitute, matched on the raw string before shlex
# (which keeps them as literal tokens); \n and \r separate commands under sh -c
_SHELL_OPERATOR_RE = re.compile(r"[|;&`\n\r]|\$\(|<\(|>\(|[<>]")

# sh expands these after the check, so a path could leave the lists; bash as sh expands {a,b} too
# A ``~`` counts only where sh expands it: inside a word it is a Windows short name (RUNNER~1)
_SHELL_EXPANSION_RE = re.compile(r"[$*?\[]|(?:^|[\s=:'\"])~|\{[^{}]*(?:,|\.\.)[^{}]*\}")

# All a sandboxed command gets: the rest of the environment holds provider keys
_COMMAND_ENV_VARS = ("PATH", "HOME", "LANG", "LC_ALL", "TZ")

# Run arbitrary code from their arguments: refused even when allowlisted, unless in danger_opt_in
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
        "find",
        "nc",
        "ncat",
        "socat",
        "ssh",
    }
)

# Anything else (file://, gopher://, ftp://, data://) needs an explicit allowed_schemes opt-in
_DEFAULT_ALLOWED_SCHEMES = frozenset({"http", "https"})


# Refused by this module's internal-host checks; 169.254.169.254 is the cloud metadata service
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


def _ip_is_internal(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Classify *ip* against the internal-host categories.

    An IPv4-mapped IPv6 form (``::ffff:127.0.0.1``) is unwrapped first, as it
    matches no IPv4 network; ``ipaddress``'s flags then cover the unspecified
    addresses (``0.0.0.0`` / ``::``) and the reserved and multicast ranges.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    ):
        return True
    for net in _DEFAULT_DENY_NETWORKS:
        if ip.version != net.version:
            continue
        if ip in net:
            return True
    return False


def _check_host(hostname: str) -> tuple[bool, tuple[str, ...]]:
    """Classify *hostname* and return the addresses it names.

    Returns whether the host is internal and, when it is not, the addresses
    to connect to, so the name is resolved once (:func:`send_pinned`): a
    literal IP itself, or every ``getaddrinfo`` answer; none when the name
    does not resolve.

    Inspects the literal host first so a workflow can't smuggle
    ``169.254.169.254`` past the deny-list by using
    ``http://metadata.aws/``. Falls back to ``socket.getaddrinfo`` so a
    hostname that resolves to an internal IPv4 or IPv6 address (via DNS,
    ``/etc/hosts``, or a split-horizon AAAA record) is caught;
    ``gethostbyname`` returns only the first IPv4 result and misses an
    AAAA-only loopback record.

    Trailing dot is normalized — most resolvers strip it before
    delivering the request, so it must not bypass the gate. Percent-
    encoded characters and IDN homographs are decoded so a workflow
    cannot write ``http://%6c%6f%63%61%6c%68%6f%73%74/`` or use a
    Cyrillic look-alike (``http://lоcalhost/``) to slip past the literal
    string check.
    """
    if not hostname:
        return False, ()
    try:
        from urllib.parse import unquote

        decoded = unquote(hostname)
    except Exception:  # pragma: no cover — unquote is total in practice
        decoded = hostname
    with contextlib.suppress(UnicodeError, UnicodeDecodeError):
        decoded = decoded.encode("idna").decode("ascii")
    lower = decoded.lower().rstrip(".")
    if not lower:
        return False, ()
    if lower == "localhost" or lower.endswith(".localhost"):
        return True, ()
    try:
        ip = ipaddress.ip_address(lower)
    except ValueError:
        pass
    else:
        return _ip_is_internal(ip), (str(ip),)
    try:
        records = socket.getaddrinfo(
            lower,
            None,
            type=socket.SOCK_STREAM,
        )
    except (OSError, ValueError):
        return False, ()
    addresses: list[str] = []
    for family, _, _, _, sockaddr in records:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        try:
            ip_addr = ipaddress.ip_address(sockaddr[0])
        except (ValueError, IndexError):
            continue
        if _ip_is_internal(ip_addr):
            return True, ()
        if str(ip_addr) not in addresses:
            addresses.append(str(ip_addr))
    return False, tuple(addresses)


def _unresolved(hostname: str) -> str:
    return f"Host {hostname!r} does not resolve, so its address cannot be checked"


def _default_deny(url: str) -> tuple[str | None, tuple[str, ...]]:
    """Return why the default deny-list blocks *url*, or the addresses to connect to."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _DEFAULT_ALLOWED_SCHEMES:
        return f"URL scheme {scheme!r} is not allowed for webhook delivery", ()
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return f"URL {url!r} has no hostname", ()
    internal, addresses = _check_host(hostname)
    if internal:
        return (
            f"Hostname {hostname!r} resolves to an internal address; "
            f"set sandbox.allow_internal_webhook_targets=true to opt in"
        ), ()
    if not addresses:
        return _unresolved(hostname), ()
    return None, addresses


def default_deny_webhook_target(url: str) -> str | None:
    """Return a reason if *url* should be blocked by the default deny-list.

    Applies to webhooks when the workflow has no enabled ``ToolSandbox``
    — keeps loopback / metadata / RFC 1918 hosts off-limits so the SSRF
    surface stays closed by default, and refuses a host that does not
    resolve, whose address cannot be checked. Returns ``None`` when the
    URL is acceptable.
    """
    return _default_deny(url)[0]


def _proxy_configured(scheme: str) -> bool:
    # The source httpx reads its proxies from when trust_env is on, as it is by default
    proxies = urllib.request.getproxies()
    return bool(proxies.get(scheme) or proxies.get("all"))


def _pin(request: httpx.Request, address: str) -> httpx.Request:
    host = request.url.raw_host.decode("ascii")
    return httpx.Request(
        request.method,
        request.url.copy_with(host=address),
        headers=request.headers,
        stream=request.stream,
        extensions={**request.extensions, "sni_hostname": host},
    )


async def send_pinned(
    client: httpx.AsyncClient,
    request: httpx.Request,
    addresses: Sequence[str],
    *,
    stream: bool = False,
) -> httpx.Response:
    """Send *request* to the addresses the sandbox checked, not to a new lookup.

    A second lookup could get a different answer than the check did (DNS
    rebinding) and reach an internal host. The request keeps its host in the
    ``Host`` header and the TLS server name, so the certificate is checked
    against it; the addresses are tried in order while the connection fails.
    A request with no addresses, to a literal IP, or that goes through a
    proxy from the environment is sent as it is: the proxy resolves the name.

    The client's connection pool is keyed by address, so a client should send
    to one host: a second name pinned to the same address would reuse the TLS
    connection verified for the first.
    """
    literal = True
    try:
        ipaddress.ip_address(request.url.host)
    except ValueError:
        literal = False
    if not addresses or literal or _proxy_configured(request.url.scheme):
        return await client.send(request, stream=stream)
    for address in addresses[:-1]:
        with contextlib.suppress(httpx.ConnectError, httpx.ConnectTimeout):
            return await client.send(_pin(request, address), stream=stream)
    return await client.send(_pin(request, addresses[-1]), stream=stream)


def _extract_path_candidates(token: str) -> list[str]:
    """Extract every path-shaped fragment from a command argument.

    GNU-style utilities accept paths inside flag values: ``tee
    --output=/etc/passwd``, ``dd of=/dev/sda``, ``cp -t /etc/ src``.
    """
    if not token:
        return []
    if "=" in token:
        _, _, value = token.partition("=")
        if value and (value in (".", "..") or "/" in value):
            return [value]
    if token.startswith("-"):
        # A short flag's value can follow it directly: ``grep -f/etc/shadow``
        attached = re.search(r"\.{0,2}/", token)
        return [token[attached.start() :]] if attached else []
    if token in (".", ".."):
        return [token]
    if "/" in token:
        return [token]
    return []


def _looks_like_path(token: str) -> bool:
    """Return ``True`` if :func:`_extract_path_candidates` finds a path in *token*."""
    return bool(_extract_path_candidates(token))


class ToolSandbox:
    """Sandbox policy that validates tool operations.

    Args:
        enabled: Whether sandbox enforcement is active.
        allowed_commands: Executable names that are permitted
            (e.g., ``["echo", "cat", "ls"]``).  An empty list with
            sandbox enabled means no commands are allowed.
        allowed_paths: Directory prefixes for **both** read and write
            file operations (e.g., ``["/tmp/workflows"]``).  An empty
            list with sandbox enabled means no file access is allowed.
        readable_paths: Additional directories allowed for **read-only**
            access.  Combined with *allowed_paths* when validating reads.
        writable_paths: Additional directories allowed for **write-only**
            access.  Combined with *allowed_paths* when validating writes.
        allow_network: Whether HTTP requests are permitted.
        allowed_domains: When *allow_network* is ``True``, restrict
            requests to these domains (e.g., ``["api.openai.com"]``).
            An empty list permits every domain but internal hosts.
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
        allow_internal_webhook_targets: Let webhook destinations reach
            internal hosts (see ``validate_webhook_url``).
        deny_internal_hosts: Refuse internal hosts in ``validate_network``
            even when *allowed_domains* lists them.
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
        deny_internal_hosts: bool = False,
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
        self._deny_internal_hosts = deny_internal_hosts

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
        redirections, process substitution) and anything the shell would
        expand (``$``, ``~``, globs, brace lists), quoted or not; rejects
        dangerous meta-executables (``env``, ``bash``, ``python -c`` …) unless
        explicitly opted in, checks the executable against the allowlist (a
        path must be what the allowlisted name resolves to), and resolves
        every path-shaped argument (absolute or relative to *cwd*) against
        allowed directories.

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

        if _SHELL_EXPANSION_RE.search(command):
            raise SandboxViolationError(
                "shell_command",
                f"Shell expansion is not allowed in sandboxed commands: {command!r}",
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

        base = self._command_cwd or (Path(cwd).resolve() if cwd else Path.cwd())

        if tokens[0] != executable:
            found = shutil.which(executable)
            if found is None or (base / tokens[0]).resolve() != Path(found).resolve():
                raise SandboxViolationError(
                    "shell_command",
                    f"Command {tokens[0]!r} is not what {executable!r} resolves to: "
                    f"{found or '(not found)'}",
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
            for token in tokens[1:]:
                # A bare name has no path shape, but a link of that name can point anywhere
                links = [
                    n for n in (token, token.partition("=")[2]) if n and os.path.islink(base / n)
                ]
                for candidate_str in _extract_path_candidates(token) or links:
                    candidate = Path(candidate_str)
                    resolved = (
                        candidate if candidate.is_absolute() else base / candidate
                    ).resolve()
                    if not self._is_within(resolved, all_paths):
                        raise SandboxViolationError(
                            "shell_command",
                            f"Path argument {str(resolved)!r} not within allowed directories",
                        )

    def command_env(self) -> dict[str, str] | None:
        """The environment for a shell command: the minimal set, or ``None`` to inherit it all."""
        if not self.enabled:
            return None
        return {name: os.environ[name] for name in _COMMAND_ENV_VARS if name in os.environ}

    def validate_path(self, path: str, *, writable: bool = False, tool_name: str = "file") -> None:
        """Validate a file path is within allowed directories.

        Resolves symlinks and relative paths before checking.  When
        *writable* is ``True`` the path is checked against
        ``allowed_paths + writable_paths``; otherwise against
        ``allowed_paths + readable_paths``.

        The check is best-effort with respect to TOCTOU. A tool that
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
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            # Null bytes, ENAMETOOLONG/EACCES, symlink loops, non-str input: one class for callers
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

    def validate_network(self, url: str) -> tuple[str, ...]:
        """Validate that network access is permitted.

        Rejects non-``http``/``https`` schemes by default (``file://``,
        ``gopher://``, ``ftp://``, ``data:`` …) regardless of host, so
        that an allowlisted hostname cannot be reused to fetch local
        files. When *allow_network* is ``True`` and *allowed_domains*
        is non-empty, the request domain must be in the allowlist; when
        it is empty, a host that resolves to an internal address
        (loopback, link-local, RFC 1918), or does not resolve, is
        refused, so naming it in *allowed_domains* is the way to reach it,
        unless *deny_internal_hosts* is set.

        Returns:
            The addresses the internal-host check resolved the host to,
            for :func:`send_pinned`; empty when no check ran.

        Raises:
            SandboxViolationError: If network access is blocked.
        """
        if not self.enabled:
            return ()

        self._validate_scheme_and_domain(url)
        if self._allowed_domains and not self._deny_internal_hosts:
            return ()
        hostname = (urlparse(url).hostname or "").lower()
        internal, addresses = _check_host(hostname)
        if internal:
            advice = (
                "the sandbox refuses internal hosts, listed or not"
                if self._deny_internal_hosts
                else "list it in sandbox.allowed_domains to allow it"
            )
            raise SandboxViolationError(
                "http_request", f"Host {hostname!r} resolves to an internal address; {advice}"
            )
        if not addresses:
            raise SandboxViolationError("http_request", _unresolved(hostname))
        return addresses

    def _validate_scheme_and_domain(self, url: str) -> None:
        """The checks *validate_network* shares with the webhook gate.

        The webhook gate applies its own internal-host rule, with its own opt-in.
        """
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

    def validate_webhook_url(self, url: str) -> tuple[str, ...]:
        """Validate a webhook destination URL.

        Two independent gates apply, in order:

        1. **Scheme gate** — always-on. Non-``http``/``https`` schemes are
           refused regardless of opt-in flags.
        2. **Internal-host gate** — fires unless the workflow explicitly
           opts out via ``allow_internal_webhook_targets``. Blocks
           loopback, link-local (incl. cloud metadata at
           ``169.254.169.254``), RFC 1918, CGNAT, IPv4-mapped IPv6 forms,
           and the unspecified addresses ``0.0.0.0`` / ``::``, and a host
           that does not resolve, whose address cannot be checked.

        Returns:
            The addresses the internal-host gate resolved the host to, for
            :func:`send_pinned`; empty when the gate did not run.

        Raises:
            SandboxViolationError: If the destination is blocked.
        """
        # Always http/https: a wider allowed_schemes (for attachments) must not widen webhooks
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in _DEFAULT_ALLOWED_SCHEMES:
            raise SandboxViolationError(
                "webhook",
                f"URL scheme {scheme!r} is not allowed for webhook delivery",
            )

        if self.enabled:
            # Enforces allowed_domains; its scheme check cannot reopen what the gate above refused
            self._validate_scheme_and_domain(url)
            if self.allow_internal_webhook_targets:
                return ()
            hostname = (parsed.hostname or "").lower()
            if not hostname:
                return ()
            internal, addresses = _check_host(hostname)
            if internal:
                raise SandboxViolationError(
                    "webhook",
                    f"Webhook destination {hostname!r} resolves to an internal "
                    f"address; set sandbox.allow_internal_webhook_targets=true "
                    f"to opt in",
                )
            if not addresses:
                raise SandboxViolationError("webhook", _unresolved(hostname))
            return addresses

        if self.allow_internal_webhook_targets:
            return ()
        reason, addresses = _default_deny(url)
        if reason is not None:
            raise SandboxViolationError("webhook", reason)
        return addresses

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
