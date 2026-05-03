"""Tests for ``agentloom history`` CLI command — focus on the new
date / cost filters added to close issue #77's "filterable by date,
workflow, cost, provider" requirement."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentloom.cli.main import app

runner = CliRunner()


def _write_record(runs_dir: Path, name: str, **fields: object) -> None:
    """Drop a single run-history record under ``runs_dir`` for the CLI to read."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{name}.json").write_text(json.dumps(fields, default=str))


def test_history_filter_by_workflow(tmp_path: Path) -> None:
    _write_record(
        tmp_path, "a", run_id="a", workflow_name="alpha", timestamp="2026-05-01T10:00:00+00:00"
    )
    _write_record(
        tmp_path, "b", run_id="b", workflow_name="beta", timestamp="2026-05-01T11:00:00+00:00"
    )

    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--workflow", "alpha"])
    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" not in result.output


def test_history_filter_by_provider_prefix(tmp_path: Path) -> None:
    _write_record(
        tmp_path,
        "a",
        run_id="a",
        workflow_name="x",
        providers_used=["openai/gpt-4o-mini"],
        timestamp="2026-05-01T10:00:00+00:00",
    )
    _write_record(
        tmp_path,
        "b",
        run_id="b",
        workflow_name="x",
        providers_used=["anthropic/claude-haiku"],
        timestamp="2026-05-01T11:00:00+00:00",
    )

    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--provider", "openai"])
    assert result.exit_code == 0
    assert "a " in result.output  # run id 'a' shown in column
    assert "b " not in result.output


def test_history_filter_by_since_date(tmp_path: Path) -> None:
    # Issue #77 requires filtering by date — ``--since YYYY-MM-DD`` keeps
    # only records timestamped at or after the given UTC midnight.
    _write_record(
        tmp_path, "old", run_id="old", workflow_name="x", timestamp="2026-04-30T23:59:59+00:00"
    )
    _write_record(
        tmp_path, "new", run_id="new", workflow_name="x", timestamp="2026-05-02T10:00:00+00:00"
    )

    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--since", "2026-05-01"])
    assert result.exit_code == 0
    assert "new" in result.output
    assert "old " not in result.output


def test_history_filter_by_until_date(tmp_path: Path) -> None:
    _write_record(
        tmp_path, "early", run_id="early", workflow_name="x", timestamp="2026-04-30T10:00:00+00:00"
    )
    _write_record(
        tmp_path, "late", run_id="late", workflow_name="x", timestamp="2026-05-03T10:00:00+00:00"
    )

    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--until", "2026-05-01"])
    assert result.exit_code == 0
    assert "early" in result.output
    assert "late " not in result.output


def test_history_filter_by_since_and_until_combined(tmp_path: Path) -> None:
    _write_record(
        tmp_path, "a", run_id="a", workflow_name="x", timestamp="2026-04-30T23:59:59+00:00"
    )
    _write_record(
        tmp_path, "b", run_id="b", workflow_name="x", timestamp="2026-05-01T12:00:00+00:00"
    )
    _write_record(
        tmp_path, "c", run_id="c", workflow_name="x", timestamp="2026-05-02T12:00:00+00:00"
    )
    _write_record(
        tmp_path, "d", run_id="d", workflow_name="x", timestamp="2026-05-03T12:00:00+00:00"
    )

    result = runner.invoke(
        app,
        ["history", "--runs-dir", str(tmp_path), "--since", "2026-05-01", "--until", "2026-05-02"],
    )
    assert result.exit_code == 0
    # b and c are inside the window; --until is inclusive of end-of-day
    # via the day boundary at midnight UTC, so c (12:00 on 2026-05-02) is
    # AFTER 2026-05-02T00:00:00Z and gets excluded — that's the same
    # semantics typical of ``find -newermt`` / ``git log --until``.
    assert " b " in result.output
    assert "a " not in result.output
    assert "d " not in result.output


def test_history_filter_by_min_cost(tmp_path: Path) -> None:
    _write_record(
        tmp_path,
        "cheap",
        run_id="cheap",
        workflow_name="x",
        timestamp="2026-05-01T10:00:00+00:00",
        total_cost_usd=0.001,
    )
    _write_record(
        tmp_path,
        "expensive",
        run_id="expensive",
        workflow_name="x",
        timestamp="2026-05-01T11:00:00+00:00",
        total_cost_usd=0.50,
    )

    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--min-cost", "0.10"])
    assert result.exit_code == 0
    assert "expensive" in result.output
    assert "cheap" not in result.output


def test_history_filter_by_max_cost(tmp_path: Path) -> None:
    _write_record(
        tmp_path,
        "cheap",
        run_id="cheap",
        workflow_name="x",
        timestamp="2026-05-01T10:00:00+00:00",
        total_cost_usd=0.001,
    )
    _write_record(
        tmp_path,
        "expensive",
        run_id="expensive",
        workflow_name="x",
        timestamp="2026-05-01T11:00:00+00:00",
        total_cost_usd=0.50,
    )

    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--max-cost", "0.10"])
    assert result.exit_code == 0
    assert "cheap" in result.output
    assert "expensive" not in result.output


def test_history_invalid_date_emits_clean_error(tmp_path: Path) -> None:
    _write_record(
        tmp_path, "a", run_id="a", workflow_name="x", timestamp="2026-05-01T10:00:00+00:00"
    )
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--since", "yesterday"])
    assert result.exit_code != 0
    # typer.BadParameter surfaces a usage error on stderr; just ensure the
    # message mentions the offending value rather than a raw traceback.
    assert "yesterday" in (result.output + (result.stderr or ""))


def test_history_no_records_message(tmp_path: Path) -> None:
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No run records found" in result.output


def test_history_json_output_filtered(tmp_path: Path) -> None:
    _write_record(
        tmp_path,
        "a",
        run_id="a",
        workflow_name="x",
        timestamp="2026-05-02T10:00:00+00:00",
        total_cost_usd=0.5,
    )
    _write_record(
        tmp_path,
        "b",
        run_id="b",
        workflow_name="x",
        timestamp="2026-05-02T11:00:00+00:00",
        total_cost_usd=0.001,
    )
    result = runner.invoke(
        app, ["history", "--runs-dir", str(tmp_path), "--min-cost", "0.10", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload) == 1
    assert payload[0]["run_id"] == "a"
