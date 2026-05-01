"""Tests for Google Gemini provider adapter."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agentloom.exceptions import ProviderError
from agentloom.providers.google import GoogleProvider

MOCK_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Gemini says hi"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 8,
        "candidatesTokenCount": 4,
        "totalTokenCount": 12,
    },
}


class TestGoogleProvider:
    @respx.mock
    async def test_successful_completion(self) -> None:
        respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = GoogleProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
        )
        assert result.content == "Gemini says hi"
        assert result.provider == "google"
        assert result.usage.total_tokens == 12
        await provider.close()

    @respx.mock
    async def test_google_missing_thoughts_token_count_defaults_to_zero(self) -> None:
        # The base mock response does not include ``thoughtsTokenCount``.
        # The adapter must default the field to 0 (rather than raising or
        # propagating ``None``) — Gemini omits the field on non-thinking
        # models and intermittently on ``gemini-3-flash-preview`` even
        # when thinking is enabled.
        respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = GoogleProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
        )
        assert result.usage.reasoning_tokens == 0
        assert result.usage.billable_completion_tokens == result.usage.completion_tokens
        assert result.reasoning_content is None
        await provider.close()

    @respx.mock
    async def test_google_thoughts_token_count_populates_reasoning_tokens(self) -> None:
        # When Gemini surfaces ``thoughtsTokenCount`` (Gemini 2.5+ thinking
        # models), it must land in ``TokenUsage.reasoning_tokens`` and feed
        # cost calculation at the output rate.
        thinking_response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "answer"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 200,
                "totalTokenCount": 215,
            },
        }
        respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=thinking_response)
        )
        provider = GoogleProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
        )
        assert result.usage.reasoning_tokens == 200
        assert result.usage.billable_completion_tokens == 205  # 5 + 200
        await provider.close()

    @respx.mock
    async def test_google_thinking_config_passed_to_payload(self) -> None:
        # ``thinking_config`` must be translated into Gemini's wire format
        # under ``generationConfig.thinkingConfig`` with the per-field
        # rename. Snake_case at the public API → camelCase on the wire.
        from agentloom.core.models import ThinkingConfig

        route = respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = GoogleProvider(api_key="test-key")
        cfg = ThinkingConfig(enabled=True, budget_tokens=4096, level="high", capture_reasoning=True)
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            thinking_config=cfg,
        )
        sent = json.loads(route.calls.last.request.content)
        assert sent["generationConfig"]["thinkingConfig"] == {
            "thinkingBudget": 4096,
            "thinkingLevel": "high",
            "includeThoughts": True,
        }
        await provider.close()

    @respx.mock
    async def test_google_includes_thoughts_writes_reasoning_content(self) -> None:
        # When ``includeThoughts=true`` is set, Gemini returns thought
        # summaries as separate parts marked ``thought=true``. The adapter
        # must split those into ``reasoning_content`` and keep them out of
        # the visible ``content``.
        response_with_thoughts = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me think about this carefully.", "thought": True},
                            {"text": "The answer is 42."},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 7,
                "totalTokenCount": 22,
            },
        }
        respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=response_with_thoughts)
        )
        provider = GoogleProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
        )
        assert result.content == "The answer is 42."
        assert result.reasoning_content == "Let me think about this carefully."
        await provider.close()

    @respx.mock
    async def test_system_instruction(self) -> None:
        route = respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = GoogleProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "hi"},
            ],
            model="gemini-2.5-flash",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        assert "systemInstruction" in body
        assert body["systemInstruction"]["parts"][0]["text"] == "Be concise"
        await provider.close()

    @respx.mock
    async def test_api_error_raises(self) -> None:
        respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(404, text="model not found")
        )
        provider = GoogleProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="404"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="gemini-2.5-flash",
            )
        await provider.close()

    @respx.mock
    async def test_multimodal_image_block_formatting(self) -> None:
        route = respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import ImageBlock, TextBlock

        provider = GoogleProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="Describe this image"),
                        ImageBlock(data="abc123", media_type="image/jpeg"),
                    ],
                }
            ],
            model="gemini-2.5-flash",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        parts = body["contents"][0]["parts"]
        assert parts[0] == {"text": "Describe this image"}
        assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
        assert parts[1]["inline_data"]["data"] == "abc123"
        await provider.close()

    async def test_multimodal_url_passthrough_raises(self) -> None:
        from agentloom.providers.multimodal import ImageURLBlock, TextBlock

        provider = GoogleProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="does not support URL passthrough"):
            await provider.complete(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            TextBlock(text="Describe"),
                            ImageURLBlock(
                                url="https://example.com/img.jpg", media_type="image/jpeg"
                            ),
                        ],
                    }
                ],
                model="gemini-2.5-flash",
            )
        await provider.close()

    @respx.mock
    async def test_multimodal_document_block_formatting(self) -> None:
        route = respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import DocumentBlock, TextBlock

        provider = GoogleProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="Summarize"),
                        DocumentBlock(data="pdf_b64", media_type="application/pdf"),
                    ],
                }
            ],
            model="gemini-2.5-flash",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        parts = body["contents"][0]["parts"]
        assert parts[1]["inline_data"]["mime_type"] == "application/pdf"
        assert parts[1]["inline_data"]["data"] == "pdf_b64"
        await provider.close()

    @respx.mock
    async def test_multimodal_audio_block_formatting(self) -> None:
        route = respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import AudioBlock, TextBlock

        provider = GoogleProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="Transcribe"),
                        AudioBlock(data="audio_b64", media_type="audio/wav"),
                    ],
                }
            ],
            model="gemini-2.5-flash",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        parts = body["contents"][0]["parts"]
        assert parts[1]["inline_data"]["mime_type"] == "audio/wav"
        assert parts[1]["inline_data"]["data"] == "audio_b64"
        await provider.close()

    @respx.mock
    async def test_streaming_yields_chunks(self) -> None:
        sse = (
            'data: {"candidates":[{"content":{"parts":[{"text":"Gemini"}],"role":"model"}}]}\n\n'
            'data: {"candidates":[{"content":{"parts":[{"text":" says"}],"role":"model"}}]}\n\n'
            'data: {"candidates":[{"content":{"parts":[{"text":" hi"}],"role":"model"},'
            '"finishReason":"STOP"}],'
            '"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":4,"totalTokenCount":12}}\n\n'
        )
        respx.post(url__regex=r".*/models/gemini.*streamGenerateContent.*").mock(
            return_value=httpx.Response(200, content=sse.encode())
        )
        provider = GoogleProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}], model="gemini-2.5-flash"
        )
        chunks = [chunk async for chunk in sr]
        assert chunks == ["Gemini", " says", " hi"]
        assert sr.content == "Gemini says hi"
        assert sr.usage.total_tokens == 12
        assert sr.finish_reason == "STOP"
        await provider.close()

    @respx.mock
    async def test_streaming_api_error(self) -> None:
        respx.post(url__regex=r".*/models/gemini.*streamGenerateContent.*").mock(
            return_value=httpx.Response(404, text="model not found")
        )
        provider = GoogleProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}], model="gemini-2.5-flash"
        )
        with pytest.raises(ProviderError, match="404"):
            async for _ in sr:
                pass
        await provider.close()

    @respx.mock
    async def test_streaming_warns_on_missing_usage_metadata(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """Stream completing without ``usageMetadata`` must log a warning so
        zero-cost reports surface instead of being silently swallowed."""
        import logging

        sse = (
            'data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}]}\n\n'
            'data: {"candidates":[{"content":{"parts":[{"text":" there"}],"role":"model"},'
            '"finishReason":"STOP"}]}\n\n'  # No usageMetadata anywhere.
        )
        respx.post(url__regex=r".*/models/gemini.*streamGenerateContent.*").mock(
            return_value=httpx.Response(200, content=sse.encode())
        )
        provider = GoogleProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}], model="gemini-2.5-flash"
        )
        with caplog.at_level(logging.WARNING, logger="agentloom.providers.google"):
            _ = [c async for c in sr]

        assert sr.usage.total_tokens == 0
        assert any("usageMetadata" in r.message for r in caplog.records)
        await provider.close()

    def test_supports_gemini_models(self) -> None:
        p = GoogleProvider(api_key="k")
        assert p.supports_model("gemini-2.5-flash")
        assert p.supports_model("gemini-3.1-pro")
        assert not p.supports_model("gpt-4o-mini")

    @respx.mock
    async def test_generation_config_built_from_temperature_max_tokens_extras(self) -> None:
        """Verify temperature, maxOutputTokens, and allowlisted extras (top_p,
        seed, response_mime_type) all land in `generationConfig`, and tools
        plus safety_settings sit at the top level."""
        captured: dict[str, object] = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json=MOCK_RESPONSE)

        respx.post(url__regex=r".*/models/gemini.*").mock(side_effect=_capture)
        provider = GoogleProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            temperature=0.7,
            max_tokens=128,
            top_p=0.9,
            seed=42,
            response_mime_type="application/json",
            tools=[{"function_declarations": []}],
            safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
        )
        gc = captured["generationConfig"]
        assert gc["temperature"] == 0.7
        assert gc["maxOutputTokens"] == 128
        # Gemini wire format is camelCase; the adapter translates from the
        # snake_case public API to the right field names.
        assert gc["topP"] == 0.9
        assert gc["seed"] == 42
        assert gc["responseMimeType"] == "application/json"
        assert "tools" in captured
        assert "safetySettings" in captured

