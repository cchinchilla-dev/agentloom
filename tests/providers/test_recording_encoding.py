"""Recordings are read as UTF-8 whatever the locale.

agentloom writes recordings as ASCII JSON, but one written by hand or by
another tool can hold UTF-8 text as is. Read with the locale encoding, it
decoded wrong (latin-1) or not at all (ASCII).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

TEXT = "café ☕"

# Every reader of a recording file, run where the locale encoding is ASCII.
_READ_EVERY_WAY = """
import json, sys
from pathlib import Path
import anyio
from agentloom.providers.base import ToolCall
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import merge_into_recording
from agentloom.providers.tool_tape import ToolMode, ToolReplayer

path = Path(sys.argv[1])
served = ToolReplayer(path, ToolMode.REPLAY).serve("s", [ToolCall(id="1", name="t", arguments={})])


async def ask():
    mock = MockProvider(responses_file=path)
    return await mock.complete([{"role": "user", "content": "q"}], "m", step_id="s")


reply = anyio.run(ask)
merge_into_recording(path, lambda merged: None)
kept = json.loads(path.read_bytes())["_tools"]["s"][0]["result"]
print(json.dumps([served[0][0], reply.content, kept]))
"""


def test_recordings_are_read_as_utf8_whatever_the_locale(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    recording = {
        "_version": 2,
        "s": {"content": TEXT, "model": "m", "provider": "mock"},
        "_tools": {"s": [{"name": "t", "arguments": {}, "result": TEXT, "success": True}]},
    }
    path.write_bytes(json.dumps(recording, ensure_ascii=False).encode("utf-8"))
    ascii_locale = {"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    run = subprocess.run(
        [sys.executable, "-c", _READ_EVERY_WAY, str(path)],
        env={**os.environ, **ascii_locale},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    # The tool replayer, the mock provider and the file merge all keep the text as written.
    assert json.loads(run.stdout) == [TEXT, TEXT, TEXT]
