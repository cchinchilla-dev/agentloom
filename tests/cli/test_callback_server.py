"""Tests for the callback server CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from agentloom.checkpointing.base import CheckpointData
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)


def _make_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="cb-test",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"input": "hello"},
        steps=[
            StepDefinition(
                id="step_a",
                type=StepType.LLM_CALL,
                prompt="Do: {state.input}",
                output="result_a",
            ),
            StepDefinition(
                id="gate",
                type=StepType.APPROVAL_GATE,
                depends_on=["step_a"],
                output="decision",
            ),
            StepDefinition(
                id="step_b",
                type=StepType.LLM_CALL,
                depends_on=["gate"],
                prompt="Continue: {state.result_a}",
                output="result_b",
            ),
        ],
    )


def _write_checkpoint(
    cp_dir: Path,
    run_id: str,
    *,
    status: str = "paused",
    paused_step_id: str | None = "gate",
) -> None:
    wf = _make_workflow()
    data = CheckpointData(
        workflow_name=wf.name,
        run_id=run_id,
        workflow_definition=wf.model_dump(),
        state={"input": "hello", "result_a": "processed"},
        step_results={},
        completed_steps=["step_a"],
        status=status,
        paused_step_id=paused_step_id,
        created_at="2026-04-13T10:00:00+00:00",
        updated_at="2026-04-13T10:00:01+00:00",
    )
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / f"{run_id}.json").write_text(data.model_dump_json(indent=2))


async def _request(port: int, method: str, path: str) -> tuple[int, dict]:
    """Send a raw HTTP request to the callback server."""
    async with httpx.AsyncClient() as client:
        if method == "GET":
            resp = await client.get(f"http://127.0.0.1:{port}{path}", timeout=10)
        else:
            resp = await client.post(f"http://127.0.0.1:{port}{path}", timeout=10)
    return resp.status_code, resp.json()


class TestCallbackServer:
    @pytest.mark.anyio()
    async def test_pending_lists_paused_runs(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-list1")
        _write_checkpoint(tmp_path, "run-list2", status="failed", paused_step_id=None)

        from agentloom.cli.callback_server import _handle_pending

        # Use a mock stream to capture response
        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        await _handle_pending(FakeStream(), str(tmp_path))  # type: ignore[arg-type]
        assert len(responses) == 1
        status, body = responses[0]
        assert status == 200
        paused = body["paused_runs"]
        run_ids = [r["run_id"] for r in paused]
        assert "run-list1" in run_ids
        assert "run-list2" not in run_ids

    @pytest.mark.anyio()
    async def test_approve_endpoint(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-approve")

        from agentloom.cli.callback_server import _handle_decision

        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                await _handle_decision(
                    FakeStream(),
                    str(tmp_path),
                    True,
                    "run-approve",
                    "approved",  # type: ignore[arg-type]
                )

        assert len(responses) == 1
        status, body = responses[0]
        assert status == 202
        assert body["decision"] == "approved"

    @pytest.mark.anyio()
    async def test_reject_endpoint(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-reject")

        from agentloom.cli.callback_server import _handle_decision

        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                await _handle_decision(
                    FakeStream(),
                    str(tmp_path),
                    True,
                    "run-reject",
                    "rejected",  # type: ignore[arg-type]
                )

        assert len(responses) == 1
        status, body = responses[0]
        assert status == 202
        assert body["decision"] == "rejected"

    @pytest.mark.anyio()
    async def test_webhook_endpoint(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        body = json.dumps({"run_id": "abc", "step_id": "gate", "status": "awaiting_approval"})
        await _handle_webhook(FakeStream(), body)  # type: ignore[arg-type]

        assert len(responses) == 1
        status, resp_body = responses[0]
        assert status == 200
        assert resp_body["status"] == "received"

    @pytest.mark.anyio()
    async def test_unknown_run_id_404(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_decision

        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        await _handle_decision(
            FakeStream(),
            str(tmp_path),
            True,
            "nonexistent",
            "approved",  # type: ignore[arg-type]
        )

        assert len(responses) == 1
        status, body = responses[0]
        assert status == 404
        assert "no checkpoint" in body["error"]

    @pytest.mark.anyio()
    async def test_webhook_invalid_json(self) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        await _handle_webhook(FakeStream(), "not valid json {{{")  # type: ignore[arg-type]
        assert responses[0][0] == 200

    @pytest.mark.anyio()
    async def test_decision_not_paused_409(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-done", status="success", paused_step_id=None)

        from agentloom.cli.callback_server import _handle_decision

        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        await _handle_decision(
            FakeStream(),
            str(tmp_path),
            True,
            "run-done",
            "approved",  # type: ignore[arg-type]
        )
        assert responses[0][0] == 409
        assert "not paused" in responses[0][1]["error"]

    @pytest.mark.anyio()
    async def test_decision_no_paused_step_409(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-nostep", status="paused", paused_step_id=None)

        from agentloom.cli.callback_server import _handle_decision

        responses: list[tuple[int, dict]] = []

        class FakeStream:
            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                body = json.loads(text[body_start:])
                status = int(text.split(" ")[1])
                responses.append((status, body))

        await _handle_decision(
            FakeStream(),
            str(tmp_path),
            True,
            "run-nostep",
            "approved",  # type: ignore[arg-type]
        )
        assert responses[0][0] == 409
        assert "no paused step" in responses[0][1]["error"]


class TestCallbackServerHTTP:
    """Integration tests that exercise the full HTTP routing layer."""

    @pytest.mark.anyio()
    async def test_full_http_webhook(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        class FakeStream:
            def __init__(self, raw: bytes) -> None:
                self._raw = raw
                self._sent: list[bytes] = []

            async def receive(self, n: int) -> bytes:
                data, self._raw = self._raw, b""
                return data

            async def send(self, data: bytes) -> None:
                self._sent.append(data)

        body = json.dumps({"run_id": "r1", "status": "pending"})
        raw = (
            f"POST /webhook HTTP/1.1\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"\r\n"
            f"{body}"
        ).encode()

        stream = FakeStream(raw)
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        resp_text = b"".join(stream._sent).decode()
        assert "200 OK" in resp_text

    @pytest.mark.anyio()
    async def test_full_http_pending(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-http-p1")

        from agentloom.cli.callback_server import _handle_request

        class FakeStream:
            def __init__(self, raw: bytes) -> None:
                self._raw = raw
                self._sent: list[bytes] = []

            async def receive(self, n: int) -> bytes:
                data, self._raw = self._raw, b""
                return data

            async def send(self, data: bytes) -> None:
                self._sent.append(data)

        raw = b"GET /pending HTTP/1.1\r\nHost: localhost\r\n\r\n"
        stream = FakeStream(raw)
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        resp_text = b"".join(stream._sent).decode()
        assert "200 OK" in resp_text
        assert "run-http-p1" in resp_text

    @pytest.mark.anyio()
    async def test_full_http_404(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        class FakeStream:
            def __init__(self, raw: bytes) -> None:
                self._raw = raw
                self._sent: list[bytes] = []

            async def receive(self, n: int) -> bytes:
                data, self._raw = self._raw, b""
                return data

            async def send(self, data: bytes) -> None:
                self._sent.append(data)

        raw = b"GET /nonexistent HTTP/1.1\r\nHost: localhost\r\n\r\n"
        stream = FakeStream(raw)
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        resp_text = b"".join(stream._sent).decode()
        assert "404 Not Found" in resp_text

    @pytest.mark.anyio()
    async def test_full_http_approve(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-http-a1")

        from agentloom.cli.callback_server import _handle_request

        class FakeStream:
            def __init__(self, raw: bytes) -> None:
                self._raw = raw
                self._sent: list[bytes] = []

            async def receive(self, n: int) -> bytes:
                data, self._raw = self._raw, b""
                return data

            async def send(self, data: bytes) -> None:
                self._sent.append(data)

        raw = b"POST /approve/run-http-a1 HTTP/1.1\r\nHost: localhost\r\n\r\n"
        stream = FakeStream(raw)

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]

        resp_text = b"".join(stream._sent).decode()
        assert "202 Accepted" in resp_text

    @pytest.mark.anyio()
    async def test_full_http_reject(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-http-r1")

        from agentloom.cli.callback_server import _handle_request

        class FakeStream:
            def __init__(self, raw: bytes) -> None:
                self._raw = raw
                self._sent: list[bytes] = []

            async def receive(self, n: int) -> bytes:
                data, self._raw = self._raw, b""
                return data

            async def send(self, data: bytes) -> None:
                self._sent.append(data)

        raw = b"POST /reject/run-http-r1 HTTP/1.1\r\nHost: localhost\r\n\r\n"
        stream = FakeStream(raw)

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]

        resp_text = b"".join(stream._sent).decode()
        assert "202 Accepted" in resp_text


class TestCallbackServerEdgeCases:
    """Edge cases for the HTTP parsing layer."""

    @staticmethod
    def _make_chunk_stream(chunks: list[bytes]):
        class FakeStream:
            def __init__(self) -> None:
                self._chunks = list(chunks)
                self._sent: list[bytes] = []

            async def receive(self, n: int) -> bytes:
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

            async def send(self, data: bytes) -> None:
                self._sent.append(data)

        return FakeStream()

    @pytest.mark.anyio()
    async def test_empty_receive_returns_silently(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        stream = self._make_chunk_stream([b""])
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert stream._sent == []

    @pytest.mark.anyio()
    async def test_streamed_headers(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        _write_checkpoint(tmp_path, "run-chunk")
        chunks = [b"GET /pend", b"ing HTTP/1.1\r\n", b"Host: x\r\n\r\n"]
        stream = self._make_chunk_stream(chunks)
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert b"200 OK" in b"".join(stream._sent)

    @pytest.mark.anyio()
    async def test_streamed_body(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        body = json.dumps({"run_id": "r1"})
        headers = f"POST /webhook HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n"
        stream = self._make_chunk_stream([headers.encode(), body[:3].encode(), body[3:].encode()])
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert b"200 OK" in b"".join(stream._sent)

    @pytest.mark.anyio()
    async def test_no_double_crlf_400(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        stream = self._make_chunk_stream([b"GET /x HTTP/1.1\r\nHost: y\r\n"])
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert b"400 Bad Request" in b"".join(stream._sent)

    @pytest.mark.anyio()
    async def test_invalid_content_length_400(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        raw = b"POST /webhook HTTP/1.1\r\nContent-Length: not-a-number\r\n\r\n"
        stream = self._make_chunk_stream([raw])
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert b"400 Bad Request" in b"".join(stream._sent)

    @pytest.mark.anyio()
    async def test_negative_content_length_400(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        raw = b"POST /webhook HTTP/1.1\r\nContent-Length: -5\r\n\r\n"
        stream = self._make_chunk_stream([raw])
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert b"400 Bad Request" in b"".join(stream._sent)

    @pytest.mark.anyio()
    async def test_malformed_request_line_400(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        raw = b"GARBAGE\r\nHost: x\r\n\r\n"
        stream = self._make_chunk_stream([raw])
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert b"400 Bad Request" in b"".join(stream._sent)
