"""Multi-modal content types and attachment resolution."""

from __future__ import annotations

import base64
import ipaddress
import logging
import mimetypes
import socket
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import anyio
import httpx
from pydantic import BaseModel

from agentloom.core.models import Attachment, SandboxConfig

logger = logging.getLogger("agentloom.multimodal")

# Maximum attachment size after download/read (20 MB).
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

# ---------------------------------------------------------------------------
# Content block types — the internal format that flows through the gateway
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """A text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """An image content block with base64-encoded data."""

    type: Literal["image"] = "image"
    data: str  # base64-encoded
    media_type: str  # e.g. "image/jpeg"


class ImageURLBlock(BaseModel):
    """An image URL block for provider-side fetching (fetch: provider)."""

    type: Literal["image_url"] = "image_url"
    url: str
    media_type: str


class DocumentBlock(BaseModel):
    """A document content block (PDF)."""

    type: Literal["document"] = "document"
    data: str  # base64-encoded
    media_type: str  # e.g. "application/pdf"


class AudioBlock(BaseModel):
    """An audio content block."""

    type: Literal["audio"] = "audio"
    data: str  # base64-encoded
    media_type: str  # e.g. "audio/wav"


ContentBlock = TextBlock | ImageBlock | ImageURLBlock | DocumentBlock | AudioBlock


# ---------------------------------------------------------------------------
# Media type detection
# ---------------------------------------------------------------------------

_EXTENSION_MEDIA_TYPES: dict[str, str] = {
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    # Documents
    ".pdf": "application/pdf",
    # Audio
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}

_MEDIA_TYPE_DEFAULTS: dict[str, str] = {
    "image": "image/png",
    "pdf": "application/pdf",
    "audio": "audio/wav",
}


def detect_media_type(source: str, attachment_type: str = "image") -> str:
    """Auto-detect media type from a file path or URL.

    Checks the explicit extension table first for consistent cross-platform
    results, then falls back to ``mimetypes.guess_type()``.
    """
    lower = source.lower()
    for ext, mt in _EXTENSION_MEDIA_TYPES.items():
        if lower.endswith(ext):
            return mt
    guessed, _ = mimetypes.guess_type(source)
    if guessed:
        return guessed
    return _MEDIA_TYPE_DEFAULTS.get(attachment_type, "application/octet-stream")


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------


def _is_url(source: str) -> bool:
    """Return True if *source* looks like an HTTP(S) URL (case-insensitive)."""
    return source.lower().startswith(("http://", "https://"))


def _is_base64(source: str) -> bool:
    """Heuristic: raw base64 strings are long and decode successfully.

    Excludes URLs and obvious file paths (starting with ``/``, ``~``, ``.``,
    or containing ``\\``).  Note: ``/`` is a valid base64 character so we
    only check for it at position 0 (absolute path indicator).
    """
    if _is_url(source):
        return False
    if source[:1] in ("/", "~", ".") or "\\" in source:
        return False
    if len(source) > 64:
        try:
            base64.b64decode(source, validate=True)
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Security: SSRF protection + sandbox validation
# ---------------------------------------------------------------------------

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if *ip* is private, reserved, or in a blocked range."""
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS)


async def _resolve_and_validate_host(hostname: str) -> list[tuple[Any, ...]]:
    """Resolve hostname and validate all IPs are public.

    Returns the ``getaddrinfo`` results on success.
    Raises ``PermissionError`` if any resolved IP is private/reserved
    or if DNS resolution fails (fail-closed).
    """
    try:
        results = await anyio.getaddrinfo(hostname, None)
    except (socket.gaierror, ValueError, OSError):
        msg = f"DNS resolution failed for '{hostname}' (SSRF fail-closed)"
        raise PermissionError(msg)
    if not results:
        msg = f"DNS returned no results for '{hostname}' (SSRF fail-closed)"
        raise PermissionError(msg)
    for _family, _, _, _, sockaddr in results:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_private_ip(ip):
            msg = f"Hostname '{hostname}' resolves to private IP {ip} (SSRF blocked)"
            raise PermissionError(msg)
    return results


async def _validate_url_sandbox(url: str, sandbox: SandboxConfig) -> None:
    """Raise if the URL is blocked by sandbox policy or SSRF rules."""
    if not sandbox.allow_network:
        msg = f"URL fetching blocked by sandbox (allow_network=false): {url}"
        raise PermissionError(msg)

    hostname = urlparse(url).hostname or ""

    if sandbox.allowed_domains and hostname not in sandbox.allowed_domains:
        msg = (
            f"URL domain '{hostname}' not in sandbox allowed_domains: "
            f"{sandbox.allowed_domains}"
        )
        raise PermissionError(msg)

    await _resolve_and_validate_host(hostname)


def _validate_provider_url(url: str, sandbox: SandboxConfig) -> None:
    """Validate a URL for provider passthrough mode.

    Even in passthrough mode we enforce scheme and domain restrictions
    to prevent leaking internal URLs to external providers.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        msg = f"Invalid URL scheme for provider passthrough: {parsed.scheme!r}"
        raise PermissionError(msg)

    hostname = parsed.hostname or ""

    # Block obviously-internal hostnames (IP literals, localhost)
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_private_ip(ip):
            msg = f"Cannot send private IP URL to provider: {url}"
            raise PermissionError(msg)
    except ValueError:
        pass  # Not an IP literal — hostname is fine

    if hostname in ("localhost", "localhost.localdomain"):
        msg = f"Cannot send localhost URL to provider: {url}"
        raise PermissionError(msg)

    if sandbox.allowed_domains and hostname not in sandbox.allowed_domains:
        msg = (
            f"URL domain '{hostname}' not in sandbox allowed_domains "
            f"(fetch: provider): {sandbox.allowed_domains}"
        )
        raise PermissionError(msg)


def _validate_file_sandbox(file_path: str, sandbox: SandboxConfig) -> None:
    """Raise if the file path is outside the sandbox allowed directories.

    Uses the resolved (symlink-followed) path for the check AND for the
    subsequent read to prevent TOCTOU symlink races.
    """
    if not sandbox.enabled:
        return
    allowed = sandbox.readable_paths or sandbox.allowed_paths
    if not allowed:
        return
    resolved = str(Path(file_path).resolve())
    for allowed_dir in allowed:
        if resolved.startswith(str(Path(allowed_dir).resolve())):
            return
    msg = f"File path '{file_path}' not in sandbox readable_paths: {allowed}"
    raise PermissionError(msg)


def _check_size(data: bytes, source: str) -> None:
    """Raise if attachment data exceeds the size limit."""
    if len(data) > MAX_ATTACHMENT_BYTES:
        mb = len(data) / (1024 * 1024)
        limit_mb = MAX_ATTACHMENT_BYTES / (1024 * 1024)
        msg = f"Attachment '{source}' is {mb:.1f} MB, exceeds limit of {limit_mb:.0f} MB"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Fetching / reading
# ---------------------------------------------------------------------------


async def _validate_redirect_target(response: httpx.Response) -> None:
    """Event hook: validate each redirect target against SSRF rules.

    Called by httpx before following a redirect.  Raises if the redirect
    target points to a private/reserved IP address.
    """
    if response.is_redirect and response.has_redirect_location:
        location = response.headers.get("location", "")
        parsed = urlparse(location)
        hostname = parsed.hostname
        if hostname:
            try:
                ip = ipaddress.ip_address(hostname)
                if _is_private_ip(ip):
                    msg = f"Redirect to private IP blocked: {location}"
                    raise PermissionError(msg)
            except ValueError:
                pass  # Hostname, not IP literal — DNS will be checked on connect
            if hostname in ("localhost", "localhost.localdomain"):
                msg = f"Redirect to localhost blocked: {location}"
                raise PermissionError(msg)


async def _fetch_url(url: str) -> tuple[bytes, str | None]:
    """Fetch URL content with streaming size enforcement.

    Reads the response in chunks and aborts if the cumulative size
    exceeds :data:`MAX_ATTACHMENT_BYTES`, preventing OOM from large responses.
    Redirects are validated against SSRF rules before following.
    """
    async with httpx.AsyncClient(
        timeout=30.0,
        max_redirects=5,
        event_hooks={"response": [_validate_redirect_target]},
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_ATTACHMENT_BYTES:
                    limit_mb = MAX_ATTACHMENT_BYTES / (1024 * 1024)
                    msg = f"Attachment '{url}' exceeds {limit_mb:.0f} MB during download"
                    raise ValueError(msg)
                chunks.append(chunk)
            data = b"".join(chunks)
        content_type = response.headers.get("content-type", "").split(";")[0].strip() or None
        return data, content_type


# ---------------------------------------------------------------------------
# Block builders per attachment type
# ---------------------------------------------------------------------------


def _make_block(
    attachment_type: str,
    data: str,
    media_type: str,
) -> ContentBlock:
    """Create the appropriate ContentBlock for the attachment type."""
    if attachment_type == "image":
        return ImageBlock(data=data, media_type=media_type)
    if attachment_type == "pdf":
        return DocumentBlock(data=data, media_type=media_type)
    if attachment_type == "audio":
        return AudioBlock(data=data, media_type=media_type)
    msg = f"Unsupported attachment type: {attachment_type!r}"
    raise ValueError(msg)


def _make_url_block(
    attachment_type: str,
    url: str,
    media_type: str,
) -> ContentBlock:
    """Create a URL-passthrough block.  Only images support this today."""
    if attachment_type == "image":
        return ImageURLBlock(url=url, media_type=media_type)
    msg = f"URL passthrough (fetch: provider) is only supported for images, not {attachment_type!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Single-attachment resolver
# ---------------------------------------------------------------------------


async def _resolve_single(
    attachment: Attachment,
    sandbox: SandboxConfig,
) -> ContentBlock:
    """Resolve a single attachment to a ContentBlock."""
    source = attachment.source
    if not source or not source.strip():
        msg = "Attachment source must not be empty"
        raise ValueError(msg)
    media_type = attachment.media_type or detect_media_type(source, attachment.type)

    if _is_url(source):
        if attachment.fetch == "provider":
            _validate_provider_url(source, sandbox)
            return _make_url_block(attachment.type, source, media_type)

        # fetch: local (default) — download and base64-encode
        await _validate_url_sandbox(source, sandbox)
        raw_bytes, content_type = await _fetch_url(source)
        if not attachment.media_type and content_type:
            media_type = content_type
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return _make_block(attachment.type, encoded, media_type)

    if _is_base64(source):
        return _make_block(attachment.type, source, media_type)

    # Treat as local file path — resolve symlinks first, then validate and read
    resolved = str(Path(source).resolve())
    _validate_file_sandbox(source, sandbox)
    path = anyio.Path(resolved)
    raw_bytes = await path.read_bytes()
    _check_size(raw_bytes, source)
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return _make_block(attachment.type, encoded, media_type)


async def resolve_attachments(
    attachments: list[Attachment],
    sandbox: SandboxConfig | None = None,
) -> list[ContentBlock]:
    """Resolve a list of attachments to content blocks.

    Security controls applied:

    * **SSRF protection** — URLs that resolve to private/reserved IP ranges
      (RFC 1918, loopback, link-local) are blocked.  Redirects are validated
      before following.  DNS failures are fail-closed.
    * **Provider passthrough** — ``fetch: provider`` mode still validates
      the URL against private IPs and ``allowed_domains``.
    * **Sandbox domain filtering** — when ``allowed_domains`` is non-empty,
      only those domains can be fetched.
    * **File path sandboxing** — when ``sandbox.enabled`` is True, file paths
      are validated against ``readable_paths`` / ``allowed_paths``.  Symlinks
      are resolved before validation to prevent traversal.
    * **Size limit** — downloads are streamed and aborted if they exceed
      :data:`MAX_ATTACHMENT_BYTES` (20 MB), preventing OOM.
    """
    cfg = sandbox or SandboxConfig()
    blocks: list[ContentBlock] = []
    for att in attachments:
        block = await _resolve_single(att, cfg)
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def build_multimodal_content(
    text: str,
    blocks: list[ContentBlock],
) -> str | list[ContentBlock]:
    """Build message content from text and resolved attachment blocks.

    Returns a plain string when *blocks* is empty (backward compatible).
    Otherwise returns a list starting with a :class:`TextBlock` followed
    by the attachment blocks.
    """
    if not blocks:
        return text
    return [TextBlock(text=text), *blocks]


def estimate_content_tokens(content: object) -> int:
    """Estimate token count for message content (str or list of blocks).

    Used by the gateway rate-limiter.  Rough heuristic: ~4 chars per text
    token; ~85 tokens per image; ~200 tokens per document page (rough);
    ~100 tokens per audio clip (rough).
    """
    if isinstance(content, str):
        return len(content) // 4
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, TextBlock):
                total += len(block.text) // 4
            elif isinstance(block, (ImageBlock, ImageURLBlock)):
                total += 85
            elif isinstance(block, DocumentBlock):
                total += 200
            elif isinstance(block, AudioBlock):
                total += 100
        return total
    return 0


def extract_text_content(content: Any) -> str:
    """Extract text from message content (str or list of blocks).

    Useful for looking up mock responses or logging.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
        return " ".join(parts)
    return ""
