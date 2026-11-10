"""Tests for the ``agentloom history`` command and its date, workflow, cost and provider filters."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
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
    # ``--since YYYY-MM-DD`` keeps records timestamped at or after that day's UTC midnight.
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
    # Half-open UTC window: ``--since`` includes its midnight, ``--until`` excludes its own day.
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
    # typer prints BadParameter as a usage error on stderr: it names the value, no traceback.
    assert "yesterday" in (result.output + (result.stderr or ""))


def test_history_no_records_message(tmp_path: Path) -> None:
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No run records found" in result.output


def test_history_date_filter_excludes_record_without_timestamp(tmp_path: Path) -> None:
    # A record without a parseable timestamp cannot satisfy a date filter, so it is left out.
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
    # A timestamp without a tz suffix (written by an external tool) is read as UTC.
    _write_record(
        tmp_path, "naive", run_id="naive", workflow_name="x", timestamp="2026-05-02T10:00:00"
    )
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--since", "2026-05-01"])
    assert result.exit_code == 0
    assert "naive" in result.output


def test_history_no_records_as_json_is_an_empty_list(tmp_path: Path) -> None:
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


@pytest.fixture(params=["America/New_York", "Asia/Tokyo"])
def local_zone(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A local time zone away from UTC, whichever way the machine running the test is."""
    if not hasattr(time, "tzset"):
        pytest.skip("the local zone can't be changed on this platform")
    monkeypatch.setenv("TZ", request.param)
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.usefixtures("local_zone")
@pytest.mark.parametrize(
    ("option", "bound", "shown"),
    [
        ("--since", "2026-05-02T10:00:00", True),
        ("--since", "2026-05-02T10:00:01", False),
        ("--until", "2026-05-02T10:00:01", True),
        ("--until", "2026-05-02T10:00:00", False),
    ],
)
def test_history_bound_without_offset_is_utc_in_any_local_zone(
    tmp_path: Path, option: str, bound: str, shown: bool
) -> None:
    _write_record(
        tmp_path, "r", run_id="the-run", workflow_name="x", timestamp="2026-05-02T10:00:00+00:00"
    )
    result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), option, bound])
    assert result.exit_code == 0
    assert ("the-run" in result.output) is shown


def test_history_bound_with_an_offset_keeps_it(tmp_path: Path) -> None:
    _write_record(
        tmp_path, "r", run_id="the-run", workflow_name="x", timestamp="2026-05-02T10:00:00+00:00"
    )
    result = runner.invoke(
        app, ["history", "--runs-dir", str(tmp_path), "--since", "2026-05-02T11:00:00+02:00"]
    )
    assert "the-run" in result.output


def test_history_full_iso_timestamp_accepted(tmp_path: Path) -> None:
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
