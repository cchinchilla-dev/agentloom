"""CLI command: ``agentloom history`` — list per-run records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import typer

from agentloom.history.writer import RunHistoryWriter, load_records


def _parse_date(value: str | None) -> datetime | None:
    """Parse ``YYYY-MM-DD`` (or full ISO 8601) into a UTC ``datetime``.

    Date-only inputs anchor at midnight UTC so ``--since 2026-05-02`` and
    ``--until 2026-05-02`` form a closed-then-open day window when used
    together, which is the expected operator semantics. Raises
    ``typer.BadParameter`` so the CLI surfaces a clean usage error instead
    of a traceback.
    """
    if value is None:
        return None
    try:
        if len(value) == 10:  # YYYY-MM-DD
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid date '{value}' — expected YYYY-MM-DD or ISO 8601"
        ) from exc


def _record_timestamp(record: dict[str, Any]) -> datetime | None:
    """Coerce the record's ``timestamp`` field to a UTC ``datetime``.

    Returns ``None`` when the field is missing or unparseable so callers
    can decide whether to filter the record out (date filters keep it
    out — undateable records can't satisfy a date predicate) vs. keep it
    (no filter applied → keep).
    """
    ts_raw = record.get("timestamp")
    if not isinstance(ts_raw, str):
        return None
    try:
        ts = datetime.fromisoformat(ts_raw)
    except ValueError:
        return None
    # Normalize to UTC: naive timestamps are treated as already-UTC
    # (matches what ``RunHistoryWriter`` writes via ``datetime.now(UTC)``);
    # tz-aware timestamps in any other zone are converted so ``--since`` /
    # ``--until`` comparisons are unambiguous regardless of source.
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _matches(
    record: dict[str, Any],
    workflow: str | None,
    provider: str | None,
    since: datetime | None,
    until: datetime | None,
    min_cost: float | None,
    max_cost: float | None,
) -> bool:
    if workflow and record.get("workflow_name") != workflow:
        return False
    if provider and not any(p.startswith(f"{provider}/") for p in record.get("providers_used", [])):
        return False
    if since is not None or until is not None:
        ts = _record_timestamp(record)
        if ts is None:
            # An undateable record can't satisfy a date predicate.
            return False
        if since is not None and ts < since:
            return False
        # Closed-open window — ``--until 2026-05-02`` excludes records on
        # that day (anchored at 00:00 UTC). Matches ``git log --until``
        # and ``find -newermt`` operator expectations and keeps
        # ``--since X --until Y`` composable as a half-open range.
        if until is not None and ts >= until:
            return False
    cost = float(record.get("total_cost_usd") or 0.0)
    if min_cost is not None and cost < min_cost:
        return False
    return not (max_cost is not None and cost > max_cost)


def history(
    runs_dir: str = typer.Option(
        RunHistoryWriter.DEFAULT_DIR,
        "--runs-dir",
        envvar=RunHistoryWriter.ENV_VAR,
        help="Directory where run records are stored.",
    ),
    workflow: str | None = typer.Option(None, "--workflow", help="Filter by workflow name."),
    provider: str | None = typer.Option(
        None, "--provider", help="Filter by provider (matches the prefix of providers_used)."
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Keep records with timestamp >= this date (YYYY-MM-DD or ISO 8601).",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        help="Keep records with timestamp <= this date (YYYY-MM-DD or ISO 8601).",
    ),
    min_cost: float | None = typer.Option(
        None, "--min-cost", help="Keep records with total_cost_usd >= this value."
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="Keep records with total_cost_usd <= this value."
    ),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum records to show."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List recorded workflow runs (most recent first)."""
    since_dt = _parse_date(since)
    until_dt = _parse_date(until)
    records = load_records(runs_dir)
    filtered = [
        r
        for r in records
        if _matches(r, workflow, provider, since_dt, until_dt, min_cost, max_cost)
    ][:limit]

    if not filtered:
        typer.echo("No run records found.")
        return

    if output_json:
        typer.echo(json.dumps(filtered, indent=2, default=str))
        return

    # Table output — keep columns stable; downstream grep/awk scripts depend on them.
    typer.echo(
        f"{'TIMESTAMP':<26} {'RUN ID':<20} {'WORKFLOW':<22} "
        f"{'STATUS':<10} {'COST USD':>10} {'DUR MS':>10}"
    )
    typer.echo("-" * 102)
    for rec in filtered:
        ts = (rec.get("timestamp") or "")[:25]
        run_id = (rec.get("run_id") or "")[:19]
        wf_name = (rec.get("workflow_name") or "")[:21]
        status = (rec.get("status") or "")[:9]
        cost = float(rec.get("total_cost_usd") or 0.0)
        dur = float(rec.get("duration_ms") or 0.0)
        typer.echo(f"{ts:<26} {run_id:<20} {wf_name:<22} {status:<10} {cost:>10.4f} {dur:>10.1f}")
