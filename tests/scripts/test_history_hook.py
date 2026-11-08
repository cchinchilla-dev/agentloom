"""The ``no-history-comments`` pre-commit hook: what its pattern rejects and lets through."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_CONFIG = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"


def _pattern() -> re.Pattern[str]:
    hooks = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))["repos"][0]["hooks"]
    hook = next(h for h in hooks if h["id"] == "no-history-comments")
    return re.compile(hook["entry"])


@pytest.mark.parametrize(
    "line",
    [
        "    # Pre-0.5.0 the gateway retried here.",
        "x = 1  # post-0.4.2 behaviour",
        "# Pre-fix the formatter dropped the error type.",
        "# regression for #163",
        "# ---- summarize via the gateway (HIGH-1 fix) ----",
        "    # keeps the prompt hash stable (F28).",
    ],
)
def test_rejects_history_markers(line: str) -> None:
    assert _pattern().search(line)


@pytest.mark.parametrize(
    "line",
    [
        "import foo  # noqa: F401",
        "# Retry until the breaker opens.",
        "version = 'pre-0.5.0'",
        "# A pre-flight check runs first.",
        "# Format the F-string key as is.",
    ],
)
def test_accepts_ordinary_lines(line: str) -> None:
    assert not _pattern().search(line)
