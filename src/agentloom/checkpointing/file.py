"""File-system checkpoint backend — the default."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import anyio

from agentloom.checkpointing.base import BaseCheckpointer, CheckpointData


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

    # -- public API -----------------------------------------------------------

    async def save(self, data: CheckpointData) -> None:
        payload = data.model_dump_json(indent=2)
        await anyio.to_thread.run_sync(partial(self._write, data.run_id, payload))

    async def load(self, run_id: str) -> CheckpointData:
        path = self._dir / f"{run_id}.json"
        try:
            raw = await anyio.to_thread.run_sync(partial(self._read, path))
        except FileNotFoundError as exc:
            raise KeyError(f"No checkpoint found for run '{run_id}'") from exc
        return CheckpointData.model_validate_json(raw)

    async def list_runs(self) -> list[CheckpointData]:
        paths = await anyio.to_thread.run_sync(self._glob)
        results: list[CheckpointData] = []
        for p in paths:
            raw = await anyio.to_thread.run_sync(partial(self._read, p))
            results.append(CheckpointData.model_validate_json(raw))
        return results

    async def delete(self, run_id: str) -> None:
        path = self._dir / f"{run_id}.json"
        try:
            await anyio.to_thread.run_sync(partial(self._unlink, path))
        except FileNotFoundError as exc:
            raise KeyError(f"No checkpoint found for run '{run_id}'") from exc

    # -- sync helpers (run in worker thread) ----------------------------------

    def _write(self, run_id: str, payload: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{run_id}.json").write_text(payload)

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
