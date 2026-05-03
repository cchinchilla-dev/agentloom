"""Run-history logger — one JSON record per workflow execution.

Records are intentionally small and self-contained so post-hoc debugging
(and the ``agentloom history`` CLI) never requires replaying the whole
workflow. Disk I/O happens in a worker thread so the write does not
block the event loop.

Default location is ``./agentloom_runs/``; callers can override via the
``AGENTLOOM_RUNS_DIR`` env var or the ``runs_dir`` constructor argument.

The file format is ``v1`` — one pretty-printed JSON document per file,
named ``<run_id>.json``. The per-file layout keeps atomic writes and
direct inspection trivial; a future version can bundle into a single
``runs.jsonl`` when volumes grow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from agentloom.core.models import WorkflowDefinition
from agentloom.core.results import StepResult, StepStatus, WorkflowResult

# Allow only safe characters in the file basename so a hostile or buggy
# caller can't pass ``../etc/passwd`` or similar and write outside the
# configured runs dir. Leading ``.`` is rejected so ``.``, ``..`` and
# hidden-file forms can't slip through; the engine's auto-generated
# UUID hex always passes. This guard exists for the public
# ``RunHistoryWriter`` surface.
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")

try:
    from importlib.metadata import (
        PackageNotFoundError,
    )
    from importlib.metadata import (
        version as _pkg_version,
    )
except ImportError:  # pragma: no cover — stdlib on supported versions
    _pkg_version = None  # type: ignore[assignment]
    PackageNotFoundError = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger("agentloom.history")

_RECORD_SCHEMA_VERSION = 1


def _agentloom_version() -> str:
    """Best-effort package version resolution.

    Works both for an installed distribution and for in-repo runs where
    the package metadata may not be available.
    """
    if _pkg_version is None:
        return "unknown"
    try:
        return _pkg_version("agentloom")
    except PackageNotFoundError:
        return "unknown"


def _workflow_hash(workflow: WorkflowDefinition) -> str:
    """Stable SHA-256 of the workflow definition's canonical JSON form.

    Lets a reader correlate multiple runs of the same workflow even if
    the user renames or re-paths the YAML.
    """
    payload = json.dumps(workflow.model_dump(), sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _providers_used(step_results: dict[str, StepResult]) -> list[str]:
    """Distinct ``provider/model`` pairs observed across step results."""
    pairs: set[str] = set()
    for r in step_results.values():
        if r.provider and r.model:
            pairs.add(f"{r.provider}/{r.model}")
    return sorted(pairs)


def build_record(
    result: WorkflowResult,
    workflow: WorkflowDefinition,
    *,
    run_id: str,
    agentloom_version: str | None = None,
) -> dict[str, Any]:
    """Assemble the persistent run record.

    Pure function — no I/O — so tests can assert shape without touching
    disk and callers can serialize it to anywhere (not just the default
    ``./agentloom_runs/`` location).
    """
    return {
        "_schema_version": _RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "agentloom_version": agentloom_version or _agentloom_version(),
        "python_version": platform.python_version(),
        "workflow_name": result.workflow_name,
        "workflow_hash": _workflow_hash(workflow),
        "status": result.status.value,
        "providers_used": _providers_used(result.step_results),
        "total_cost_usd": result.total_cost_usd,
        "total_tokens": result.total_tokens,
        # Count every step that the engine actually attempted (i.e.
        # produced a result for), not only successes. ``"steps_executed"``
        # here matches the issue #77 example field name and reflects total
        # work done, including failures and pauses — useful for spotting
        # workflows that consistently fail past a certain step count.
        # Skipped steps (router branches not taken) are excluded so the
        # number tracks runtime cost rather than DAG size.
        "steps_executed": sum(
            1 for r in result.step_results.values() if r.status != StepStatus.SKIPPED
        ),
        "duration_ms": result.total_duration_ms,
        "error": result.error,
    }


class RunHistoryWriter:
    """Writes one JSON record per workflow execution to ``runs_dir``.

    Missing run_id, empty workflow name, and similar edge cases are
    accepted — the writer never raises to the caller; failures log a
    warning so that a crashed history write cannot prevent a workflow
    from returning its result.
    """

    ENV_VAR = "AGENTLOOM_RUNS_DIR"
    DEFAULT_DIR = "./agentloom_runs"

    def __init__(self, runs_dir: str | Path | None = None) -> None:
        resolved = (
            runs_dir if runs_dir is not None else os.environ.get(self.ENV_VAR, self.DEFAULT_DIR)
        )
        self.runs_dir = Path(resolved)

    def _path_for(self, run_id: str) -> Path:
        # Fall back to a timestamp-based name when the engine ran without
        # an explicit run_id, OR when the supplied id contains characters
        # that would let it escape ``self.runs_dir`` (path separators,
        # ``..``, NUL, etc). Engine-generated UUID hex always passes the
        # safe-name regex; this guard exists because ``RunHistoryWriter``
        # is part of the public API and a hostile / buggy caller could
        # otherwise write outside the configured directory.
        if not run_id or not _SAFE_RUN_ID_RE.match(run_id):
            name = datetime.now(UTC).strftime("anon-%Y%m%dT%H%M%S%f")
        else:
            name = run_id
        return self.runs_dir / f"{name}.json"

    async def record(
        self,
        result: WorkflowResult,
        workflow: WorkflowDefinition,
        *,
        run_id: str,
    ) -> Path | None:
        """Persist a run record. Returns the path on success, None on failure."""
        record = build_record(result, workflow, run_id=run_id)
        target = self._path_for(run_id)

        def _write() -> Path:
            # Atomic write: serialize to a sibling tempfile, then rename
            # over the target. ``os.replace`` is atomic on the same
            # filesystem (POSIX + NTFS), so a crash mid-write leaves
            # either the previous file or no file — never a half-written
            # one. Important because ``load_records`` reads files
            # opportunistically during ``agentloom history``.
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(f"{target.suffix}.tmp")
            tmp.write_text(json.dumps(record, indent=2, default=str))
            os.replace(tmp, target)
            return target

        try:
            return await anyio.to_thread.run_sync(_write)
        except PermissionError:
            # Read-only filesystem (containers, CI) is expected — debug log
            # only, no traceback dump in normal user output.
            logger.debug(
                "Run history dir not writable (%s); skipping record.",
                target.parent,
            )
            return None
        except Exception:
            logger.warning(
                "Failed to write run history record to %s — continuing",
                target,
            )
            logger.debug("Run history error trace", exc_info=True)
            return None


def load_records(runs_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Read every ``*.json`` record under *runs_dir*, newest first.

    Ordering uses the record's ``timestamp`` field when present and falls
    back to file mtime. Files that fail to parse are skipped with a
    warning; a corrupt record should never break the CLI listing.
    """
    base = Path(
        runs_dir
        if runs_dir is not None
        else os.environ.get(RunHistoryWriter.ENV_VAR, RunHistoryWriter.DEFAULT_DIR)
    )
    if not base.is_dir():
        return []

    records: list[tuple[float, dict[str, Any]]] = []
    for path in base.glob("*.json"):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping unreadable run record: %s", path)
            continue
        if not isinstance(doc, dict):
            continue
        ts_raw = doc.get("timestamp")
        try:
            ts = (
                datetime.fromisoformat(ts_raw).timestamp()
                if isinstance(ts_raw, str)
                else path.stat().st_mtime
            )
        except ValueError:
            ts = path.stat().st_mtime
        records.append((ts, doc))

    records.sort(key=lambda entry: entry[0], reverse=True)
    return [doc for _, doc in records]


__all__ = [
    "RunHistoryWriter",
    "build_record",
    "load_records",
]
