"""Tests for the FileCheckpointer backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentloom.checkpointing.base import CheckpointData
from agentloom.checkpointing.file import FileCheckpointer


def _make_checkpoint(
    run_id: str = "abc123",
    workflow_name: str = "test-workflow",
    status: str = "success",
) -> CheckpointData:
    return CheckpointData(
        workflow_name=workflow_name,
        run_id=run_id,
        workflow_definition={"name": workflow_name, "steps": []},
        state={"question": "hello"},
        step_results={
            "step_a": {
                "step_id": "step_a",
                "status": "success",
                "output": "done",
                "duration_ms": 42.0,
            }
        },
        completed_steps=["step_a"],
        status=status,
        created_at="2026-04-12T10:00:00+00:00",
        updated_at="2026-04-12T10:00:01+00:00",
    )


class TestFileCheckpointerSaveLoad:
    """Round-trip save/load tests."""

    async def test_save_creates_file(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        data = _make_checkpoint()
        await cp.save(data)
        assert (tmp_path / "abc123.json").exists()

    async def test_load_returns_saved_data(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        original = _make_checkpoint()
        await cp.save(original)
        loaded = await cp.load("abc123")
        assert loaded.run_id == original.run_id
        assert loaded.workflow_name == original.workflow_name
        assert loaded.state == original.state
        assert loaded.step_results == original.step_results
        assert loaded.completed_steps == original.completed_steps
        assert loaded.status == original.status

    async def test_load_missing_run_raises_key_error(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        with pytest.raises(KeyError, match="No checkpoint found"):
            await cp.load("nonexistent")

    async def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "checkpoints"
        cp = FileCheckpointer(checkpoint_dir=nested)
        await cp.save(_make_checkpoint())
        assert (nested / "abc123.json").exists()

    async def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        await cp.save(_make_checkpoint(status="running"))
        await cp.save(_make_checkpoint(status="success"))
        loaded = await cp.load("abc123")
        assert loaded.status == "success"


class TestFileCheckpointerListRuns:
    """Tests for listing runs."""

    async def test_list_runs_empty(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        assert await cp.list_runs() == []

    async def test_list_runs_returns_all(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        await cp.save(_make_checkpoint(run_id="run1", workflow_name="wf-a"))
        await cp.save(_make_checkpoint(run_id="run2", workflow_name="wf-b"))
        entries = await cp.list_runs()
        assert len(entries) == 2
        run_ids = {e.run_id for e in entries}
        assert run_ids == {"run1", "run2"}

    async def test_list_runs_nonexistent_dir(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path / "nope")
        assert await cp.list_runs() == []


class TestFileCheckpointerDelete:
    """Tests for deleting runs."""

    async def test_delete_removes_file(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        await cp.save(_make_checkpoint())
        await cp.delete("abc123")
        assert not (tmp_path / "abc123.json").exists()

    async def test_delete_missing_raises_key_error(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        with pytest.raises(KeyError, match="No checkpoint found"):
            await cp.delete("nonexistent")


class TestFileCheckpointerSecurity:
    """Tests for directory traversal and input validation."""

    async def test_load_rejects_path_traversal(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid run_id"):
            await cp.load("../../etc/passwd")

    async def test_delete_rejects_path_traversal(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid run_id"):
            await cp.delete("../secret")

    async def test_load_rejects_empty_run_id(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid run_id"):
            await cp.load("")

    async def test_load_rejects_slash_in_run_id(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid run_id"):
            await cp.load("foo/bar")


class TestFileCheckpointerCorruption:
    """Tests for corrupted checkpoint handling."""

    async def test_load_corrupted_file_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not valid json {{{")
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="unreadable or corrupted"):
            await cp.load("bad")

    async def test_list_runs_skips_corrupted_files(self, tmp_path: Path) -> None:
        # Write one valid and one corrupted checkpoint
        cp = FileCheckpointer(checkpoint_dir=tmp_path)
        await cp.save(_make_checkpoint(run_id="good"))
        (tmp_path / "corrupt.json").write_text("broken json")
        entries = await cp.list_runs()
        assert len(entries) == 1
        assert entries[0].run_id == "good"
