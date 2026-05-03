"""Tests for the run-history writer and loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import (
    StepResult,
    StepStatus,
    TokenUsage,
    WorkflowResult,
    WorkflowStatus,
)
from agentloom.history.writer import (
    RunHistoryWriter,
    build_record,
    load_records,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_workflow(name: str = "wf-test") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={},
        steps=[StepDefinition(id="s1", type=StepType.LLM_CALL, prompt="hi")],
    )


def _make_result(workflow: WorkflowDefinition, *, cost: float = 0.05) -> WorkflowResult:
    step = StepResult(
        step_id="s1",
        status=StepStatus.SUCCESS,
        duration_ms=120.0,
        cost_usd=cost,
        token_usage=TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
        provider="openai",
        model="gpt-4o-mini",
    )
    return WorkflowResult(
        workflow_name=workflow.name,
        status=WorkflowStatus.SUCCESS,
        step_results={"s1": step},
        total_duration_ms=150.0,
        total_tokens=12,
        total_cost_usd=cost,
    )


class TestBuildRecord:
    def test_record_includes_required_fields(self) -> None:
        wf = _make_workflow()
        record = build_record(_make_result(wf), wf, run_id="run-abc")

        assert record["run_id"] == "run-abc"
        assert record["workflow_name"] == "wf-test"
        assert record["status"] == "success"
        assert record["providers_used"] == ["openai/gpt-4o-mini"]
        assert record["total_cost_usd"] == 0.05
        assert record["total_tokens"] == 12
        assert record["steps_executed"] == 1
        # Schema version pinned so consumers can branch on it.
        assert record["_schema_version"] == 1

    def test_workflow_hash_stable_for_equal_workflows(self) -> None:
        wf1 = _make_workflow("hash-test")
        wf2 = _make_workflow("hash-test")
        r1 = build_record(_make_result(wf1), wf1, run_id="r1")
        r2 = build_record(_make_result(wf2), wf2, run_id="r2")
        assert r1["workflow_hash"] == r2["workflow_hash"]

    def test_workflow_hash_differs_for_different_workflows(self) -> None:
        wf1 = _make_workflow("a")
        wf2 = _make_workflow("b")
        r1 = build_record(_make_result(wf1), wf1, run_id="r1")
        r2 = build_record(_make_result(wf2), wf2, run_id="r2")
        assert r1["workflow_hash"] != r2["workflow_hash"]

    def test_unknown_providers_list_empty(self) -> None:
        wf = _make_workflow()
        result = WorkflowResult(
            workflow_name=wf.name,
            status=WorkflowStatus.SUCCESS,
            step_results={},
        )
        record = build_record(result, wf, run_id="r")
        assert record["providers_used"] == []
        assert record["steps_executed"] == 0


class TestRunHistoryWriter:
    async def test_record_writes_json_file(self, tmp_path: Path) -> None:
        writer = RunHistoryWriter(runs_dir=tmp_path)
        wf = _make_workflow()
        path = await writer.record(_make_result(wf), wf, run_id="run-42")
        assert path is not None
        assert path.parent == tmp_path

        data = json.loads(path.read_text())
        assert data["run_id"] == "run-42"
        assert data["workflow_name"] == "wf-test"

    async def test_anonymous_run_id_still_writes(self, tmp_path: Path) -> None:
        writer = RunHistoryWriter(runs_dir=tmp_path)
        wf = _make_workflow()
        path = await writer.record(_make_result(wf), wf, run_id="")
        assert path is not None
        # File name carries a timestamp fallback.
        assert path.name.startswith("anon-")

    async def test_atomic_write_no_tempfile_leftover(self, tmp_path: Path) -> None:
        # Writer routes through ``write_text(<tmp>)`` + ``os.replace`` so a
        # crash mid-write leaves either the previous file or no file —
        # never a half-written one. After a successful record() the
        # directory must contain only the final ``.json``, no ``.json.tmp``
        # leftover (would confuse ``load_records``).
        writer = RunHistoryWriter(runs_dir=tmp_path)
        wf = _make_workflow()
        await writer.record(_make_result(wf), wf, run_id="atomic-1")
        files = sorted(p.name for p in tmp_path.iterdir())
        assert files == ["atomic-1.json"]

    async def test_env_var_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(RunHistoryWriter.ENV_VAR, str(tmp_path))
        writer = RunHistoryWriter()
        assert writer.runs_dir == tmp_path

    async def test_write_failure_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = RunHistoryWriter(runs_dir=tmp_path)

        def _boom() -> Path:
            raise OSError("disk full")

        # Force the internal write to fail; record() must swallow it.
        async def _fake_run_sync(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            fn()  # actually call — _boom raises, caught by record()

        monkeypatch.setattr("anyio.to_thread.run_sync", _fake_run_sync)
        wf = _make_workflow()
        # Should not raise; returns None on failure.
        result = await writer.record(_make_result(wf), wf, run_id="r")
        assert result is None


class TestLoadRecords:
    async def test_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        assert load_records(tmp_path / "absent") == []

    async def test_loads_and_sorts_newest_first(self, tmp_path: Path) -> None:
        writer = RunHistoryWriter(runs_dir=tmp_path)
        wf = _make_workflow()

        # Record two runs; second timestamp is strictly later.
        await writer.record(_make_result(wf, cost=0.01), wf, run_id="older")
        await writer.record(_make_result(wf, cost=0.02), wf, run_id="newer")

        records = load_records(tmp_path)
        assert len(records) == 2
        # Newest first.
        assert records[0]["run_id"] == "newer"
        assert records[1]["run_id"] == "older"

    async def test_skips_unreadable_files(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("{ not valid json")
        writer = RunHistoryWriter(runs_dir=tmp_path)
        wf = _make_workflow()
        await writer.record(_make_result(wf), wf, run_id="good")

        records = load_records(tmp_path)
        # Corrupt file silently skipped; valid one loaded.
        assert [r["run_id"] for r in records] == ["good"]

    async def test_skips_non_dict_top_level(self, tmp_path: Path) -> None:
        # ``load_records`` only yields dict records — a JSON top-level list
        # or string is treated as malformed and silently dropped so a
        # hand-crafted file in the runs dir can't break the CLI listing.
        (tmp_path / "list.json").write_text(json.dumps(["not", "a", "record"]))
        records = load_records(tmp_path)
        assert records == []

    async def test_falls_back_to_mtime_when_timestamp_missing(self, tmp_path: Path) -> None:
        # No ``timestamp`` field → ordering uses the file's mtime so the
        # CLI listing stays usable even on records authored by external
        # tools that didn't follow the schema.
        (tmp_path / "no-ts.json").write_text(json.dumps({"run_id": "no-ts"}))
        records = load_records(tmp_path)
        assert records[0]["run_id"] == "no-ts"

    async def test_falls_back_to_mtime_on_bad_timestamp(self, tmp_path: Path) -> None:
        # Unparseable timestamp shouldn't raise — fall back to mtime.
        (tmp_path / "bad-ts.json").write_text(
            json.dumps({"run_id": "bad-ts", "timestamp": "not-a-date"})
        )
        records = load_records(tmp_path)
        assert records[0]["run_id"] == "bad-ts"


class TestAgentLoomVersionResolution:
    def test_returns_unknown_when_distribution_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # In-repo runs (or vendored installs) where the package metadata
        # isn't registered must still produce a record — version field
        # falls back to ``"unknown"`` rather than raising.
        from importlib.metadata import PackageNotFoundError

        from agentloom.history import writer as writer_mod

        def _raise(_: str) -> str:
            raise PackageNotFoundError("agentloom")

        monkeypatch.setattr(writer_mod, "_pkg_version", _raise)
        assert writer_mod._agentloom_version() == "unknown"

    def test_returns_unknown_when_importlib_metadata_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defensive against the documented ImportError fallback at the top
        # of writer.py — exercised by setting ``_pkg_version`` to ``None``.
        from agentloom.history import writer as writer_mod

        monkeypatch.setattr(writer_mod, "_pkg_version", None)
        assert writer_mod._agentloom_version() == "unknown"


class TestPermissionErrorPath:
    async def test_record_returns_none_on_readonly_filesystem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Containers / CI sometimes mount the workdir read-only; the
        # writer must downgrade ``PermissionError`` to a debug log + None
        # so workflow execution is unaffected by an unwritable history dir.
        writer = RunHistoryWriter(runs_dir=tmp_path)

        async def _fake_run_sync(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise PermissionError("read-only fs")

        monkeypatch.setattr("anyio.to_thread.run_sync", _fake_run_sync)
        wf = _make_workflow()
        result = await writer.record(_make_result(wf), wf, run_id="r")
        assert result is None

    async def test_record_returns_none_on_generic_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The catch-all ``except Exception`` branch must also downgrade to
        # a warning + None — a corrupted disk, an interrupted system call,
        # an unexpected backend bug should never propagate up and break
        # the workflow result handoff. Regression for the observed gap
        # where the previous "fake_run_sync swallows fn()" test silently
        # short-circuited before reaching the exception path.
        writer = RunHistoryWriter(runs_dir=tmp_path)

        async def _fake_run_sync(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("disk full")  # noqa: TRY003 — synthetic test failure

        monkeypatch.setattr("anyio.to_thread.run_sync", _fake_run_sync)
        wf = _make_workflow()
        result = await writer.record(_make_result(wf), wf, run_id="r")
        assert result is None
