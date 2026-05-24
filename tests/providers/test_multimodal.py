"""Tests for multi-modal content types and attachment resolution."""

from __future__ import annotations

import base64
import tempfile

import httpx
import pytest
import respx

from agentloom.core.models import Attachment, SandboxConfig
from agentloom.providers.multimodal import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    ImageURLBlock,
    TextBlock,
    build_multimodal_content,
    detect_media_type,
    estimate_content_tokens,
    extract_text_content,
    resolve_attachments,
)

# A tiny 1x1 red PNG for file-based tests.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
)


class TestDetectMediaType:
    def test_jpeg(self) -> None:
        assert detect_media_type("photo.jpg") == "image/jpeg"
        assert detect_media_type("/path/to/photo.jpeg") == "image/jpeg"

    def test_png(self) -> None:
        assert detect_media_type("image.png") == "image/png"

    def test_webp(self) -> None:
        assert detect_media_type("photo.webp") == "image/webp"

    def test_gif(self) -> None:
        assert detect_media_type("anim.gif") == "image/gif"

    def test_url_with_extension(self) -> None:
        assert detect_media_type("https://example.com/photo.jpg") == "image/jpeg"

    def test_unknown_defaults_to_png(self) -> None:
        assert detect_media_type("data") == "image/png"
        assert detect_media_type("https://example.com/image") == "image/png"

    def test_pdf_detection(self) -> None:
        assert detect_media_type("report.pdf", "pdf") == "application/pdf"

    def test_audio_detection(self) -> None:
        assert detect_media_type("clip.wav", "audio") == "audio/wav"
        assert detect_media_type("clip.mp3", "audio") == "audio/mpeg"

    def test_unknown_pdf_fallback(self) -> None:
        assert detect_media_type("document", "pdf") == "application/pdf"

    def test_unknown_audio_fallback(self) -> None:
        assert detect_media_type("data", "audio") == "audio/wav"


class TestBuildMultimodalContent:
    def test_no_blocks_returns_string(self) -> None:
        result = build_multimodal_content("hello", [])
        assert result == "hello"

    def test_with_blocks_returns_list(self) -> None:
        blocks: list[ContentBlock] = [ImageBlock(data="abc", media_type="image/png")]
        result = build_multimodal_content("describe", blocks)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], TextBlock)
        assert result[0].text == "describe"
        assert isinstance(result[1], ImageBlock)


class TestEstimateContentTokens:
    def test_string_content(self) -> None:
        assert estimate_content_tokens("hello world!") == len("hello world!") // 4

    def test_multimodal_content(self) -> None:
        blocks = [TextBlock(text="describe"), ImageBlock(data="abc", media_type="image/png")]
        tokens = estimate_content_tokens(blocks)
        assert tokens == len("describe") // 4 + 85

    def test_empty(self) -> None:
        assert estimate_content_tokens("") == 0
        assert estimate_content_tokens([]) == 0

    def test_document_block(self) -> None:
        blocks: list[ContentBlock] = [DocumentBlock(data="abc", media_type="application/pdf")]
        assert estimate_content_tokens(blocks) == 200

    def test_audio_block(self) -> None:
        blocks: list[ContentBlock] = [AudioBlock(data="abc", media_type="audio/wav")]
        assert estimate_content_tokens(blocks) == 100


class TestExtractTextContent:
    def test_string(self) -> None:
        assert extract_text_content("hello") == "hello"

    def test_list_with_text_blocks(self) -> None:
        blocks = [TextBlock(text="describe"), ImageBlock(data="abc", media_type="image/png")]
        assert extract_text_content(blocks) == "describe"

    def test_non_string_non_list(self) -> None:
        assert extract_text_content(42) == ""


class TestResolveAttachments:
    async def test_local_file_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_TINY_PNG)
            f.flush()
            path = f.name

        att = Attachment(type="image", source=path)
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media_type == "image/png"
        decoded = base64.b64decode(blocks[0].data)
        assert decoded == _TINY_PNG

    async def test_raw_base64(self) -> None:
        raw_b64 = base64.b64encode(_TINY_PNG).decode("ascii")
        att = Attachment(type="image", source=raw_b64, media_type="image/png")
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].data == raw_b64

    @respx.mock
    async def test_url_fetch_local(self) -> None:
        respx.get("https://example.com/image.jpg").mock(
            return_value=httpx.Response(
                200, content=_TINY_PNG, headers={"content-type": "image/jpeg"}
            )
        )
        att = Attachment(type="image", source="https://example.com/image.jpg")
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media_type == "image/jpeg"

    async def test_url_fetch_provider_returns_url_block(self) -> None:
        att = Attachment(
            type="image",
            source="https://example.com/image.jpg",
            fetch="provider",
        )
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageURLBlock)
        assert blocks[0].url == "https://example.com/image.jpg"

    async def test_pdf_attachment(self) -> None:
        pdf_data = b"%PDF-1.4 fake"
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_data)
            f.flush()
            path = f.name

        att = Attachment(type="pdf", source=path)
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert isinstance(blocks[0], DocumentBlock)
        assert blocks[0].media_type == "application/pdf"

    async def test_audio_attachment(self) -> None:
        audio_data = b"RIFF" + b"\x00" * 40  # fake WAV header
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            f.flush()
            path = f.name

        att = Attachment(type="audio", source=path)
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert isinstance(blocks[0], AudioBlock)

    async def test_url_passthrough_only_for_images(self) -> None:
        att = Attachment(type="pdf", source="https://example.com/doc.pdf", fetch="provider")
        with pytest.raises(ValueError, match="only supported for images"):
            await resolve_attachments([att])


class TestSandboxRestrictions:
    async def test_url_blocked_by_allow_network_false(self) -> None:
        sandbox = SandboxConfig(allow_network=False)
        att = Attachment(type="image", source="https://example.com/image.jpg")
        with pytest.raises(PermissionError, match="allow_network=false"):
            await resolve_attachments([att], sandbox=sandbox)

    async def test_url_blocked_by_allowed_domains(self) -> None:
        sandbox = SandboxConfig(allowed_domains=["trusted.com"])
        att = Attachment(type="image", source="https://evil.com/image.jpg")
        with pytest.raises(PermissionError, match="allowed_domains"):
            await resolve_attachments([att], sandbox=sandbox)

    @respx.mock
    async def test_url_allowed_by_matching_domain(self) -> None:
        respx.get("https://trusted.com/image.png").mock(
            return_value=httpx.Response(200, content=_TINY_PNG)
        )
        sandbox = SandboxConfig(allowed_domains=["trusted.com"])
        att = Attachment(type="image", source="https://trusted.com/image.png")
        blocks = await resolve_attachments([att], sandbox=sandbox)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)

    async def test_fetch_provider_validates_private_ip(self) -> None:
        """URL passthrough still blocks private IP URLs."""
        att = Attachment(
            type="image",
            source="https://192.168.1.1/image.jpg",
            fetch="provider",
        )
        with pytest.raises(PermissionError, match="private IP"):
            await resolve_attachments([att])

    async def test_fetch_provider_allows_public_url(self) -> None:
        """URL passthrough allows public URLs."""
        att = Attachment(
            type="image",
            source="https://example.com/image.jpg",
            fetch="provider",
        )
        blocks = await resolve_attachments([att])
        assert isinstance(blocks[0], ImageURLBlock)

    async def test_file_path_blocked_by_sandbox(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_TINY_PNG)
            f.flush()
            path = f.name

        sandbox = SandboxConfig(enabled=True, readable_paths=["/other/dir"])
        att = Attachment(type="image", source=path)
        with pytest.raises(PermissionError, match="readable_paths"):
            await resolve_attachments([att], sandbox=sandbox)

    async def test_file_path_allowed_by_sandbox(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as f:
            f.write(_TINY_PNG)
            f.flush()
            path = f.name

        sandbox = SandboxConfig(enabled=True, readable_paths=["/tmp"])
        att = Attachment(type="image", source=path)
        blocks = await resolve_attachments([att], sandbox=sandbox)
        assert isinstance(blocks[0], ImageBlock)


class TestSSRFProtection:
    async def test_localhost_blocked(self) -> None:
        att = Attachment(type="image", source="https://localhost/image.png")
        with pytest.raises(PermissionError, match="private IP"):
            await resolve_attachments([att])

    async def test_127_0_0_1_blocked(self) -> None:
        att = Attachment(type="image", source="https://127.0.0.1/image.png")
        with pytest.raises(PermissionError, match="private IP"):
            await resolve_attachments([att])


class TestURLClassification:
    def test_case_insensitive_url(self) -> None:
        from agentloom.providers.multimodal import _is_url

        assert _is_url("HTTP://example.com/img.png") is True
        assert _is_url("hTtPs://example.com/img.png") is True
        assert _is_url("https://example.com/img.png") is True

    def test_is_base64_excludes_absolute_paths(self) -> None:
        from agentloom.providers.multimodal import _is_base64

        assert _is_base64("/etc/passwd") is False
        assert _is_base64("./relative/path") is False
        assert _is_base64("~/home/file") is False
        assert _is_base64("C:\\windows\\path") is False

    async def test_redirect_to_localhost_blocked(self) -> None:
        from agentloom.providers.multimodal import _validate_redirect_target

        response = httpx.Response(
            302,
            headers={"location": "http://127.0.0.1:8080/secret"},
            request=httpx.Request("GET", "https://external.com/redir"),
        )
        with pytest.raises(PermissionError, match="private IP"):
            await _validate_redirect_target(response)


class TestSizeLimit:
    async def test_oversized_file_rejected(self) -> None:
        from agentloom.providers.multimodal import MAX_ATTACHMENT_BYTES

        # Create a file just over the limit
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x00" * (MAX_ATTACHMENT_BYTES + 1))
            f.flush()
            path = f.name

        att = Attachment(type="image", source=path)
        with pytest.raises(ValueError, match="exceeds limit"):
            await resolve_attachments([att])


class TestEmptySource:
    async def test_empty_string_raises(self) -> None:
        att = Attachment(type="image", source="")
        with pytest.raises(ValueError, match="must not be empty"):
            await resolve_attachments([att])

    async def test_whitespace_only_raises(self) -> None:
        att = Attachment(type="image", source="   ")
        with pytest.raises(ValueError, match="must not be empty"):
            await resolve_attachments([att])


class TestAttachmentResolutionErrorWrapping:
    """Deterministic shape/policy refusals (unsupported type, size
    limit, empty source) get wrapped in ``AttachmentResolutionError``
    so the resilience layer sees a non-retryable marker.
    ``AttachmentResolutionError`` subclasses ``ValueError`` so pre-0.5.0
    ``except ValueError`` clauses keep working. ``PermissionError``
    (sandbox blocks) and ``httpx.HTTPError`` (network failures) are
    intentionally NOT wrapped so the existing rules continue to apply."""

    async def test_size_limit_raises_attachment_resolution_error(self) -> None:
        import tempfile

        from agentloom.exceptions import AttachmentResolutionError
        from agentloom.providers.multimodal import MAX_ATTACHMENT_BYTES

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x00" * (MAX_ATTACHMENT_BYTES + 1))
            f.flush()
            path = f.name
        att = Attachment(type="image", source=path)
        with pytest.raises(AttachmentResolutionError) as excinfo:
            await resolve_attachments([att])
        # Non-retryable contract: the resilience layer must skip retries.
        assert excinfo.value.is_retryable is False
        # Backwards compat: still a ValueError.
        assert isinstance(excinfo.value, ValueError)

    async def test_empty_source_raises_attachment_resolution_error(self) -> None:
        from agentloom.exceptions import AttachmentResolutionError

        att = Attachment(type="image", source="")
        with pytest.raises(AttachmentResolutionError) as excinfo:
            await resolve_attachments([att])
        assert excinfo.value.is_retryable is False
        # Source defaults to ``<unknown>`` when empty.
        assert excinfo.value.source == "<unknown>"

    async def test_missing_local_file_wrapped(self) -> None:
        """A local attachment path that does not exist raises
        ``FileNotFoundError`` from ``read_bytes()``. It is a permanent
        failure — the file will not appear on a retry — so it must be
        wrapped in ``AttachmentResolutionError`` to carry the
        non-retryable marker."""
        from agentloom.exceptions import AttachmentResolutionError

        att = Attachment(type="image", source="/nonexistent/path/does-not-exist.png")
        with pytest.raises(AttachmentResolutionError) as excinfo:
            await resolve_attachments([att])
        assert excinfo.value.is_retryable is False

    async def test_permission_error_not_wrapped(self) -> None:
        """Sandbox blocks raise ``PermissionError`` directly — they're
        already terminal at the call site, no need for the wrapper. The
        existing ``pytest.raises(PermissionError)`` tests around sandbox
        rejection continue to work without changes."""
        from agentloom.core.models import SandboxConfig

        sandbox = SandboxConfig(allow_network=False)
        att = Attachment(type="image", source="https://example.com/image.jpg")
        with pytest.raises(PermissionError):
            await resolve_attachments([att], sandbox=sandbox)

    async def test_already_classified_error_propagates_without_rewrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``AttachmentResolutionError`` raised inside ``_resolve_single``
        is propagated as-is — the dedicated ``except`` arm stops the
        generic ``ValueError`` handler from re-wrapping it into a
        double-prefixed ``Attachment 'X': Attachment 'X': ...`` message."""
        from agentloom.exceptions import AttachmentResolutionError
        from agentloom.providers import multimodal

        original = AttachmentResolutionError("doc.png", "already classified")

        async def _raise_classified(att: object, cfg: object) -> object:
            raise original

        monkeypatch.setattr(multimodal, "_resolve_single", _raise_classified)
        att = Attachment(type="image", source="doc.png")
        with pytest.raises(AttachmentResolutionError) as excinfo:
            await resolve_attachments([att])
        # Same object surfaces — not re-wrapped.
        assert excinfo.value is original


class TestOpenAIAudioFormats:
    async def test_unsupported_audio_format_raises(self) -> None:
        from agentloom.exceptions import ProviderError
        from agentloom.providers.multimodal import AudioBlock, TextBlock
        from agentloom.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="only supports WAV and MP3"):
            provider._format_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            TextBlock(text="transcribe"),
                            AudioBlock(data="abc", media_type="audio/ogg"),
                        ],
                    }
                ]
            )
        await provider.close()

    def test_wav_format_detected(self) -> None:
        from agentloom.providers.multimodal import AudioBlock, TextBlock
        from agentloom.providers.openai import OpenAIProvider

        result = OpenAIProvider._format_messages(
            [
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="transcribe"),
                        AudioBlock(data="abc", media_type="audio/wav"),
                    ],
                }
            ]
        )
        audio_part = result[0]["content"][1]
        assert audio_part["input_audio"]["format"] == "wav"

    def test_mp3_format_detected(self) -> None:
        from agentloom.providers.multimodal import AudioBlock, TextBlock
        from agentloom.providers.openai import OpenAIProvider

        result = OpenAIProvider._format_messages(
            [
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="transcribe"),
                        AudioBlock(data="abc", media_type="audio/mpeg"),
                    ],
                }
            ]
        )
        audio_part = result[0]["content"][1]
        assert audio_part["input_audio"]["format"] == "mp3"


class TestDataURLAttachments:
    """F71: ``data:`` URLs inline their payload — the resolver decodes
    them instead of treating the whole string as a filesystem path and
    surfacing a misleading FileNotFoundError."""

    # 1x1 transparent PNG.
    _PNG_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAA"
        "AAYAAjCB0C8AAAAASUVORK5CYII="
    )

    async def test_data_url_base64_decoded_as_inline(self) -> None:
        from agentloom.providers.multimodal import ImageBlock

        att = Attachment(type="image", source=f"data:image/png;base64,{self._PNG_B64}")
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].media_type == "image/png"
        assert blocks[0].data == self._PNG_B64

    async def test_data_url_invalid_base64_raises_resolution_error(self) -> None:
        from agentloom.exceptions import AttachmentResolutionError

        att = Attachment(type="image", source="data:image/png;base64,!!!not-base64!!!")
        with pytest.raises(AttachmentResolutionError, match="invalid base64"):
            await resolve_attachments([att])

    async def test_data_url_malformed_raises_resolution_error(self) -> None:
        from agentloom.exceptions import AttachmentResolutionError

        # No comma — not a well-formed data: URL.
        att = Attachment(type="image", source="data:not-a-valid-url")
        with pytest.raises(AttachmentResolutionError, match="malformed data: URL"):
            await resolve_attachments([att])

    async def test_data_url_unaffected_by_sandbox(self) -> None:
        from agentloom.core.models import SandboxConfig

        # A strict sandbox with a domain allowlist must NOT block a
        # data: URL — it makes no network call and opens no file.
        sandbox = SandboxConfig(enabled=True, allowed_domains=["api.openai.com"])
        att = Attachment(type="image", source=f"data:image/png;base64,{self._PNG_B64}")
        blocks = await resolve_attachments([att], sandbox=sandbox)
        assert len(blocks) == 1

    async def test_data_url_with_charset_param(self) -> None:
        import base64 as _b64

        att = Attachment(type="image", source="data:text/plain;charset=utf-8,Hello")
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        # The ``;charset=`` parameter is parsed and skipped — the payload
        # after the comma decodes to the literal text.
        assert _b64.b64decode(blocks[0].data) == b"Hello"

    async def test_data_url_without_base64_marker(self) -> None:
        # Non-base64 data: URL — percent-decoded text.
        att = Attachment(type="image", source="data:text/plain,Hello%20World")
        blocks = await resolve_attachments([att])
        import base64 as _b64

        assert _b64.b64decode(blocks[0].data) == b"Hello World"

    async def test_data_url_scheme_is_case_insensitive(self) -> None:
        # RFC 3986 §3.1: URI schemes are case-insensitive. ``DATA:`` (or
        # mixed-case ``Data:``) must be recognised — pre-fix the
        # ``startswith`` check rejected anything but lowercase ``data:``
        # and fell through to file-path handling.
        att = Attachment(type="image", source=f"DATA:image/png;base64,{self._PNG_B64}")
        blocks = await resolve_attachments([att])
        assert len(blocks) == 1
        assert blocks[0].data == self._PNG_B64

    async def test_data_url_non_base64_preserves_binary_bytes(self) -> None:
        # Non-base64 data: URLs can carry binary bytes through ``%xx``
        # escapes. Decoding via ``unquote(...).encode("utf-8")`` would
        # corrupt anything that isn't valid UTF-8; ``unquote_to_bytes``
        # preserves the raw sequence. A single 0x80 byte is the canonical
        # not-a-UTF-8-start regression case.
        import base64 as _b64

        att = Attachment(type="image", source="data:application/octet-stream,%80%81%FF")
        blocks = await resolve_attachments([att])
        # Round-trips exactly to the three raw bytes 0x80 0x81 0xFF.
        assert _b64.b64decode(blocks[0].data) == b"\x80\x81\xff"


class TestSymlinkRejectionMessage:
    """F70: a symlink inside readable_paths pointing OUT of it must
    produce an error naming BOTH the symlink path and the resolved
    target — pre-0.5.0 only one was named, misdirecting the operator."""

    async def test_symlink_error_names_resolved_target(self, tmp_path) -> None:
        import os

        from agentloom.core.models import SandboxConfig

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("classified")
        link = allowed / "sneak.txt"
        os.symlink(outside, link)

        sandbox = SandboxConfig(enabled=True, readable_paths=[str(allowed)])
        att = Attachment(type="image", source=str(link))
        with pytest.raises(PermissionError) as excinfo:
            await resolve_attachments([att], sandbox=sandbox)
        msg = str(excinfo.value)
        # Both the symlink path AND the resolved target appear.
        assert str(link) in msg
        assert "resolves to" in msg
