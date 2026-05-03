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
    # Closed-open window: ``--since 2026-05-01`` is inclusive of midnight
    # UTC on 2026-05-01; ``--until 2026-05-02`` is exclusive of midnight
    # UTC on 2026-05-02. So only ``b`` (12:00 on 2026-05-01) qualifies —
    # ``a`` is before, ``c`` is on/after the upper bound, ``d`` is after.
    # Same semantics as ``git log --until`` / ``find -newermt``.
    assert " b " in result.output
    assert " a " not in result.output
    assert " c " not in result.output
    assert " d " not in result.output


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


def test_history_date_filter_excludes_record_without_timestamp(tmp_path: Path) -> None:
    # A record that lacks a parseable ``timestamp`` cannot satisfy a date
    # predicate — it must be excluded when ``--since`` / ``--until`` is set
    # (rather than being silently included as if it matched).
    _write_record(tmp_path, "no_ts", run_id="no_ts", workflow_name="x")  # no timestamp
    _write_record(
        tmp_path,
        "good",
        run_id="good",
        workflow_name="x",
        timestamp="2026-05-02T10:00:00+00:00",
    )
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--since", "2026-05-01"])
    assert result.exit_code == 0
    assert "good" in result.output
    assert "no_ts" not in result.output


def test_history_date_filter_excludes_record_with_unparseable_timestamp(
    tmp_path: Path,
) -> None:
    _write_record(tmp_path, "bad_ts", run_id="bad_ts", workflow_name="x", timestamp="garbage")
    _write_record(
        tmp_path,
        "good",
        run_id="good",
        workflow_name="x",
        timestamp="2026-05-02T10:00:00+00:00",
    )
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--since", "2026-05-01"])
    assert result.exit_code == 0
    assert "good" in result.output
    assert "bad_ts" not in result.output


def test_history_naive_timestamp_treated_as_utc(tmp_path: Path) -> None:
    # Naive timestamps (no tz suffix) — possible from a record produced
    # by an older writer or external tool — get treated as already-UTC.
    # Documented behavior; covers the ``ts.tzinfo is None`` branch.
    _write_record(
        tmp_path, "naive", run_id="naive", workflow_name="x", timestamp="2026-05-02T10:00:00"
    )
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--since", "2026-05-01"])
    assert result.exit_code == 0
    assert "naive" in result.output


def test_history_full_iso_timestamp_accepted(tmp_path: Path) -> None:
    # ``--since`` accepts full ISO 8601, not only ``YYYY-MM-DD``. Cover
    # the non-date-only branch in ``_parse_date``.
    _write_record(
        tmp_path,
        "early",
        run_id="early",
        workflow_name="x",
        timestamp="2026-05-01T10:00:00+00:00",
    )
    _write_record(
        tmp_path,
        "late",
        run_id="late",
        workflow_name="x",
        timestamp="2026-05-01T15:00:00+00:00",
    )
    result = runner.invoke(
        app,
        [
            "history",
            "--runs-dir",
            str(tmp_path),
            "--since",
            "2026-05-01T12:00:00+00:00",
        ],
    )
    assert result.exit_code == 0
    assert "late" in result.output
    assert "early" not in result.output


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
