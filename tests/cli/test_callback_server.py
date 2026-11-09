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
    workflow: WorkflowDefinition | None = None,
) -> None:
    wf = workflow or _make_workflow()
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

            def _wire(gw: object, default: str, config_path: object = None) -> None:
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

            def _wire(gw: object, default: str, config_path: object = None) -> None:
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
        await _handle_webhook(FakeStream(), body, "application/json")  # type: ignore[arg-type]

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

        await _handle_webhook(FakeStream(), "not valid json {{{", "application/json")  # type: ignore[arg-type]
        assert responses[0][0] == 400
        assert "invalid JSON" in responses[0][1]["error"]

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

    @pytest.mark.anyio()
    async def test_decision_generic_openai_key_not_set_500(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key variable missing on the server answers 500, not a dropped connection."""
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        endpoint = WorkflowConfig(
            provider="generic_openai",
            base_url="http://vllm.local:8000/v1",
            api_key_env="VLLM_API_KEY",
        )
        wf = _make_workflow().model_copy(update={"config": endpoint})
        _write_checkpoint(tmp_path, "run-generic", workflow=wf)

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
            "run-generic",
            "approved",  # type: ignore[arg-type]
        )
        assert len(responses) == 1
        assert responses[0][0] == 500
        assert "'VLLM_API_KEY' is not set" in responses[0][1]["error"]
        saved = json.loads((tmp_path / "run-generic.json").read_text())
        assert saved["status"] == "paused"


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

            def _wire(gw: object, default: str, config_path: object = None) -> None:
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

            def _wire(gw: object, default: str, config_path: object = None) -> None:
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

    @pytest.mark.anyio()
    async def test_client_disconnect_mid_body_does_not_hang(self, tmp_path: Path) -> None:
        # The body stops 25 bytes short of Content-Length; EOF must end the read loop.
        from agentloom.cli.callback_server import _handle_request

        headers = b"POST /webhook HTTP/1.1\r\nContent-Length: 30\r\n\r\n"
        stream = self._make_chunk_stream([headers, b"hello"])
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        # The truncated body is still routed and rejected as invalid JSON.
        sent = b"".join(stream._sent)
        assert sent, "handler must respond even on truncated body"
        assert b"400 Bad Request" in sent


class TestWebhookBodyValidation:
    """POST /webhook rejects malformed or non-object JSON; a missing Content-Type only warns."""

    @staticmethod
    def _capture_stream():
        class FakeStream:
            def __init__(self) -> None:
                self.responses: list[tuple[int, dict]] = []

            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                self.responses.append((int(text.split(" ")[1]), json.loads(text[body_start:])))

        return FakeStream()

    @pytest.mark.anyio()
    async def test_valid_json_accepted(self) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        stream = self._capture_stream()
        await _handle_webhook(stream, json.dumps({"run_id": "r"}), "application/json")  # type: ignore[arg-type]
        assert stream.responses[0][0] == 200

    @pytest.mark.anyio()
    async def test_invalid_json_returns_400(self) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        stream = self._capture_stream()
        await _handle_webhook(stream, "{not json", "application/json")  # type: ignore[arg-type]
        assert stream.responses[0][0] == 400
        assert "invalid JSON" in stream.responses[0][1]["error"]

    @pytest.mark.anyio()
    async def test_non_object_body_returns_400(self) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        stream = self._capture_stream()
        await _handle_webhook(stream, "[1, 2, 3]", "application/json")  # type: ignore[arg-type]
        assert stream.responses[0][0] == 400
        assert "JSON object" in stream.responses[0][1]["error"]

    @pytest.mark.anyio()
    async def test_missing_content_type_warns_but_accepts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        stream = self._capture_stream()
        with caplog.at_level("WARNING"):
            await _handle_webhook(stream, json.dumps({"x": 1}), None)  # type: ignore[arg-type]
        assert stream.responses[0][0] == 200
        assert any("Content-Type" in rec.message for rec in caplog.records)

    @pytest.mark.anyio()
    async def test_empty_body_accepted_as_ping(self) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        stream = self._capture_stream()
        await _handle_webhook(stream, "", "application/json")  # type: ignore[arg-type]
        assert stream.responses[0][0] == 200
        assert stream.responses[0][1]["status"] == "received"

    @pytest.mark.anyio()
    async def test_whitespace_only_body_accepted_as_ping(self) -> None:
        from agentloom.cli.callback_server import _handle_webhook

        stream = self._capture_stream()
        await _handle_webhook(stream, "   \n\t  ", "application/json")  # type: ignore[arg-type]
        assert stream.responses[0][0] == 200


class TestCallbackPathTraversal:
    """Approve/reject guards: a path-traversal run_id answers 400, a missing token 401."""

    @staticmethod
    def _capture_stream():
        class FakeStream:
            def __init__(self) -> None:
                self.responses: list[tuple[int, dict]] = []

            async def send(self, data: bytes) -> None:
                text = data.decode()
                body_start = text.index("\r\n\r\n") + 4
                self.responses.append((int(text.split(" ")[1]), json.loads(text[body_start:])))

        return FakeStream()

    @pytest.mark.anyio()
    async def test_path_traversal_run_id_returns_400(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_decision

        stream = self._capture_stream()
        await _handle_decision(
            stream,
            str(tmp_path),
            True,
            "../../etc/passwd",
            "approved",  # type: ignore[arg-type]
        )
        assert stream.responses[0][0] == 400
        assert "invalid run id" in stream.responses[0][1]["error"]

    @pytest.mark.anyio()
    async def test_token_auth_rejects_missing_header(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-tok", status="paused", paused_step_id="gate")

        from agentloom.cli.callback_server import _handle_decision

        stream = self._capture_stream()
        await _handle_decision(
            stream,
            str(tmp_path),
            True,
            "run-tok",
            "approved",  # type: ignore[arg-type]
            provided_token=None,
            expected_token="s3cret",
        )
        assert stream.responses[0][0] == 401

    @pytest.mark.anyio()
    async def test_token_auth_accepts_matching_header(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-tok2", status="success", paused_step_id=None)

        from agentloom.cli.callback_server import _handle_decision

        stream = self._capture_stream()
        # The 409 (run not paused) shows the request got past the token check.
        await _handle_decision(
            stream,
            str(tmp_path),
            True,
            "run-tok2",
            "approved",  # type: ignore[arg-type]
            provided_token="s3cret",
            expected_token="s3cret",
        )
        assert stream.responses[0][0] == 409


class TestCallbackServerLimits:
    """GET /pending sits behind the token; one request cannot make the server buffer unbounded."""

    @staticmethod
    def _chunk_stream(chunks: list[bytes]):
        class FakeStream:
            def __init__(self) -> None:
                self.chunks = list(chunks)
                self.sent = b""

            async def receive(self, n: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

            async def send(self, data: bytes) -> None:
                self.sent += data

        return FakeStream()

    @pytest.mark.anyio()
    @pytest.mark.parametrize(
        ("header", "status"),
        [("", b"401 Unauthorized"), ("X-AgentLoom-Token: nope\r\n", b"401 Unauthorized")],
    )
    async def test_pending_without_the_token_is_refused(
        self, tmp_path: Path, header: str, status: bytes
    ) -> None:
        from agentloom.cli.callback_server import _handle_request

        _write_checkpoint(tmp_path, "run-secret")
        stream = self._chunk_stream([f"GET /pending HTTP/1.1\r\n{header}\r\n".encode()])
        await _handle_request(stream, str(tmp_path), True, "s3cret")  # type: ignore[arg-type]
        assert stream.sent.startswith(b"HTTP/1.1 " + status)
        assert b"run-secret" not in stream.sent

    @pytest.mark.anyio()
    async def test_pending_with_the_token_lists_runs(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        _write_checkpoint(tmp_path, "run-secret")
        raw = b"GET /pending HTTP/1.1\r\nX-AgentLoom-Token: s3cret\r\n\r\n"
        stream = self._chunk_stream([raw])
        await _handle_request(stream, str(tmp_path), True, "s3cret")  # type: ignore[arg-type]
        assert stream.sent.startswith(b"HTTP/1.1 200 OK")
        assert b"run-secret" in stream.sent

    @pytest.mark.anyio()
    async def test_endless_headers_stop_at_the_cap(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        chunks = [b"GET /pending HTTP/1.1\r\nX-Pad: "] + [b"a" * 8192] * 20
        stream = self._chunk_stream(chunks)
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert stream.sent.startswith(b"HTTP/1.1 431 Request Header Fields Too Large")
        assert stream.chunks, "the server kept reading past the header cap"

    @pytest.mark.anyio()
    async def test_a_body_over_the_cap_is_refused_unread(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_request

        headers = b"POST /webhook HTTP/1.1\r\nContent-Length: 2000000\r\n\r\n"
        stream = self._chunk_stream([headers] + [b"{" * 8192] * 5)
        await _handle_request(stream, str(tmp_path), True)  # type: ignore[arg-type]
        assert stream.sent.startswith(b"HTTP/1.1 413 Content Too Large")
        assert len(stream.chunks) == 5

    @pytest.mark.parametrize(
        ("host", "token", "warns"),
        [
            ("0.0.0.0", None, True),
            ("callback.internal", None, True),
            ("0.0.0.0", "s3cret", False),
            ("127.0.0.1", None, False),
            ("::1", None, False),
            ("localhost", None, False),
        ],
    )
    def test_exposure_warning(self, host: str, token: str | None, warns: bool) -> None:
        from agentloom.cli.callback_server import _exposure_warning

        assert (_exposure_warning(host, token) is not None) is warns

    def test_token_comparison_accepts_any_text(self) -> None:
        from agentloom.cli.callback_server import _token_matches

        assert _token_matches("s3cret", "s3cret")
        assert not _token_matches("señal", "s3cret")
        assert not _token_matches(None, "s3cret")
        assert _token_matches(None, None)


class TestConfigFile:
    """The server reads its config file once at start and hands the path to each resume."""

    class _Responses:
        def __init__(self) -> None:
            self.sent: list[tuple[int, dict]] = []

        async def send(self, data: bytes) -> None:
            text = data.decode()
            self.sent.append(
                (int(text.split(" ")[1]), json.loads(text[text.index("\r\n\r\n") + 4 :]))
            )

    def test_a_valid_file_is_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentloom.cli.callback_server import _checked_config_file

        monkeypatch.delenv("AGENTLOOM_CONFIG", raising=False)
        config = tmp_path / "agentloom.yaml"
        config.write_text("providers: [{name: openai}]\n")
        assert _checked_config_file(config) == config.resolve()

    def test_the_command_refuses_a_broken_file_before_listening(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from agentloom.cli.main import app

        config = tmp_path / "agentloom.yaml"
        config.write_text("providers: [{name: openai, api_key: YOUR_KEY_HERE}]\n")
        with patch("agentloom.cli.callback_server.anyio.run") as serve:
            result = CliRunner().invoke(app, ["callback-server", "--config", str(config)])
        assert result.exit_code == 1
        assert "set OPENAI_API_KEY instead" in result.output
        serve.assert_not_called()

    def test_a_broken_file_stops_the_server_before_it_listens(self, tmp_path: Path) -> None:
        import typer

        from agentloom.cli.callback_server import _checked_config_file

        config = tmp_path / "agentloom.yaml"
        config.write_text("providers: [{name: opneai}]\n")
        with pytest.raises(typer.Exit) as exc_info:
            _checked_config_file(config)
        assert exc_info.value.exit_code == 1

    @pytest.mark.anyio()
    async def test_the_path_reaches_provider_setup(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_decision
        from tests.conftest import MockProvider

        _write_checkpoint(tmp_path, "run-config")
        config = tmp_path / "agentloom.yaml"
        seen: list[object] = []

        def _wire(gw: object, default: str, config_path: object = None) -> None:
            seen.append(config_path)
            gw.register(MockProvider(), priority=0)  # type: ignore[attr-defined]

        stream = self._Responses()
        with (
            patch("agentloom.cli.run._setup_providers", side_effect=_wire),
            patch("agentloom.cli.run._setup_observer", return_value=None),
        ):
            await _handle_decision(
                stream,  # type: ignore[arg-type]
                str(tmp_path),
                True,
                "run-config",
                "approved",
                config_path=config,
            )
        assert seen == [config]
        assert stream.sent[0][0] == 202

    @pytest.mark.anyio()
    @pytest.mark.parametrize("decision", ["approve", "reject"])
    async def test_a_request_passes_the_path_on(self, tmp_path: Path, decision: str) -> None:
        from agentloom.cli.callback_server import _handle_request
        from tests.conftest import MockProvider

        _write_checkpoint(tmp_path, "run-config-http")
        config = tmp_path / "agentloom.yaml"
        seen: list[object] = []

        class _Stream:
            def __init__(self) -> None:
                self._raw = f"POST /{decision}/run-config-http HTTP/1.1\r\nHost: x\r\n\r\n".encode()

            async def receive(self, n: int) -> bytes:
                data, self._raw = self._raw, b""
                return data

            async def send(self, data: bytes) -> None:
                pass

        def _wire(gw: object, default: str, config_path: object = None) -> None:
            seen.append(config_path)
            gw.register(MockProvider(), priority=0)  # type: ignore[attr-defined]

        with (
            patch("agentloom.cli.run._setup_providers", side_effect=_wire),
            patch("agentloom.cli.run._setup_observer", return_value=None),
        ):
            await _handle_request(_Stream(), str(tmp_path), True, None, config)  # type: ignore[arg-type]
        assert seen == [config]

    @pytest.mark.anyio()
    async def test_a_config_error_answers_500(self, tmp_path: Path) -> None:
        from agentloom.cli.callback_server import _handle_decision
        from agentloom.exceptions import ConfigError

        _write_checkpoint(tmp_path, "run-config-bad")
        stream = self._Responses()
        with (
            patch(
                "agentloom.cli.run._setup_providers",
                side_effect=ConfigError("config file 'x': providers must be a list"),
            ),
            patch("agentloom.cli.run._setup_observer", return_value=None),
        ):
            await _handle_decision(
                stream,  # type: ignore[arg-type]
                str(tmp_path),
                True,
                "run-config-bad",
                "approved",
            )
        assert stream.sent == [(500, {"error": "config file 'x': providers must be a list"})]
