"""File-system checkpoint backend — the default."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from functools import partial
from pathlib import Path

import anyio

from agentloom.checkpointing.base import BaseCheckpointer, CheckpointData

logger = logging.getLogger("agentloom.checkpointing")

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class FileCheckpointer(BaseCheckpointer):
    """Store checkpoints as JSON files in a local directory.

    Layout::

        {checkpoint_dir}/
            {run_id}.json
            {run_id}.json
            ...
    """

    def __init__(self, checkpoint_dir: str | Path = ".agentloom/checkpoints") -> None:
        self._dir = Path(checkpoint_dir)

    def _checkpoint_path(self, run_id: str) -> Path:
        """Resolve a safe checkpoint file path, rejecting traversal attempts."""
        if not run_id or not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        path = self._dir / f"{run_id}.json"
        # Belt-and-suspenders: verify the resolved path is inside _dir
        base_dir = self._dir.resolve(strict=False)
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(base_dir):  # pragma: no cover
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return path

    @staticmethod
    def _parse_checkpoint(raw: str, source: str) -> CheckpointData:
        """Parse raw JSON into a CheckpointData, with a clear error on corruption."""
        try:
            return CheckpointData.model_validate_json(raw)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Checkpoint '{source}' is unreadable or corrupted") from exc

    # public API

    async def save(self, data: CheckpointData) -> None:
        # Do the JSON serialization inside the thread too — for workflows
        # carrying large accumulated state the ``model_dump_json`` call can
        # block the event loop for tens of ms per layer.
        def _dump_and_write() -> None:
            payload = data.model_dump_json(indent=2)
            self._write(data.run_id, payload)

        await anyio.to_thread.run_sync(_dump_and_write)

    async def load(self, run_id: str) -> CheckpointData:
        path = self._checkpoint_path(run_id)
        try:
            raw = await anyio.to_thread.run_sync(partial(self._read, path))
        except FileNotFoundError as exc:
            raise KeyError(f"No checkpoint found for run '{run_id}'") from exc
        return self._parse_checkpoint(raw, run_id)

    async def list_runs(self) -> list[CheckpointData]:
        paths = await anyio.to_thread.run_sync(self._glob)
        results: list[CheckpointData] = []
        for p in paths:
            try:
                raw = await anyio.to_thread.run_sync(partial(self._read, p))
                results.append(self._parse_checkpoint(raw, str(p)))
            except (ValueError, OSError) as exc:
                logger.warning("Skipping corrupted checkpoint %s: %s", p, exc)
        return results

    async def delete(self, run_id: str) -> None:
        path = self._checkpoint_path(run_id)
        try:
            await anyio.to_thread.run_sync(partial(self._unlink, path))
        except FileNotFoundError as exc:
            raise KeyError(f"No checkpoint found for run '{run_id}'") from exc

    def _write(self, run_id: str, payload: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self._checkpoint_path(run_id)
        # Atomic write: temp file → fsync → rename
        fd = None
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                prefix=f"{run_id}.",
                dir=str(self._dir),
            )
            os.write(fd, payload.encode())
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(tmp_path, str(target))
            tmp_path = None  # replaced successfully
        finally:  # pragma: no cover — cleanup after crash mid-write
            if fd is not None:
                os.close(fd)
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    @staticmethod
    def _read(path: Path) -> str:
        return path.read_text()

    def _glob(self) -> list[Path]:
        if not self._dir.exists():
            return []
        return sorted(self._dir.glob("*.json"))

    @staticmethod
    def _unlink(path: Path) -> None:
        path.unlink()
