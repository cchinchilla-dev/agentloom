"""Tests for `scripts/check_docs.py`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "check_docs.py"

PAGE = """---
title: "Steps"
description: "What each step type does."
order: {order}
section: "Workflows"
---

# Steps

The step types.

## Router

See [the gate](#approval_gate) and [concepts](concepts.md#layers).

## `approval_gate`
"""

CONCEPTS = """---
title: "Concepts"
description: "The ideas."
order: 2
section: "Start"
---

# Concepts

## Layers
"""


def _load_script():
    """Import the script as a module (it is under scripts/, outside the package tree)."""
    spec = importlib.util.spec_from_file_location("check_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _write_docs(tmp_path: Path, steps: str | None = None) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "steps.md").write_text(steps if steps is not None else PAGE.format(order=5))
    (docs / "concepts.md").write_text(CONCEPTS)
    return docs


def test_passes_on_valid_pages(tmp_path: Path, script) -> None:
    _write_docs(tmp_path)
    code, message = script.check(tmp_path)
    assert code == 0, message
    assert message == "docs OK: 2 pages"


def test_the_repository_docs_pass(script) -> None:
    code, message = script.check(ROOT)
    assert code == 0, message


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("`agent inspect`", "agent-inspect"),
        ("Tools & sandbox", "tools--sandbox"),
        ("`approval_gate`", "approval_gate"),
        ("[CLI](cli.md) options", "cli-options"),
        ("What's next", "whats-next"),
    ],
)
def test_slug_matches_github(script, heading: str, expected: str) -> None:
    assert script.slug(heading) == expected


def test_requires_every_frontmatter_key(tmp_path: Path, script) -> None:
    _write_docs(tmp_path, PAGE.format(order=5).replace('section: "Workflows"\n', ""))
    code, message = script.check(tmp_path)
    assert code == 1
    assert "frontmatter needs 'section' (str)" in message


def test_refuses_a_page_without_frontmatter(tmp_path: Path, script) -> None:
    _write_docs(tmp_path, "# Steps\n")
    code, message = script.check(tmp_path)
    assert code == 1
    assert "docs/steps.md: no frontmatter" in message


def test_refuses_a_long_description(tmp_path: Path, script) -> None:
    _write_docs(tmp_path, PAGE.format(order=5).replace("What each step type does.", "x" * 161))
    code, message = script.check(tmp_path)
    assert code == 1
    assert "161 characters" in message


def test_needs_an_h1_first(tmp_path: Path, script) -> None:
    _write_docs(tmp_path, PAGE.format(order=5).replace("# Steps\n", "## Steps\n"))
    code, message = script.check(tmp_path)
    assert code == 1
    assert "must be an h1" in message


def test_reports_missing_files_and_headings(tmp_path: Path, script) -> None:
    page = PAGE.format(order=5) + "\n[gone](missing.md) and [nowhere](concepts.md#nope)\n"
    _write_docs(tmp_path, page)
    code, message = script.check(tmp_path)
    assert code == 1
    assert "link to a missing file: missing.md" in message
    assert "link to a missing heading: concepts.md#nope" in message


def test_numbers_repeated_headings(tmp_path: Path, script) -> None:
    page = PAGE.format(order=5) + "\n## Router\n\n[second](#router-1)\n"
    _write_docs(tmp_path, page)
    code, message = script.check(tmp_path)
    assert code == 0, message


def test_ignores_code_fences_and_inline_code(tmp_path: Path, script) -> None:
    page = PAGE.format(order=5) + "\n```text\n!!! note\n[x](missing.md)\n```\n\n`[y](missing.md)`\n"
    _write_docs(tmp_path, page)
    code, message = script.check(tmp_path)
    assert code == 0, message


def test_flags_mkdocs_syntax(tmp_path: Path, script) -> None:
    _write_docs(tmp_path, PAGE.format(order=5) + '\n!!! warning "Careful"\n\n=== "Tab"\n')
    code, message = script.check(tmp_path)
    assert code == 1
    assert message.count("MkDocs syntax") == 2


def test_flags_unbalanced_markers(tmp_path: Path, script) -> None:
    _write_docs(tmp_path, PAGE.format(order=5) + "\n<!-- tabs -->\n\n<!-- /cards -->\n")
    code, message = script.check(tmp_path)
    assert code == 1
    assert "unbalanced <!-- cards --> marker" in message


def test_refuses_a_shared_order(tmp_path: Path, script) -> None:
    _write_docs(tmp_path, PAGE.format(order=2))
    code, message = script.check(tmp_path)
    assert code == 1
    assert "pages share order 2: concepts.md, steps.md" in message


def test_main_returns_the_exit_code(tmp_path: Path, script, capsys) -> None:
    _write_docs(tmp_path)
    assert script.main(tmp_path) == 0
    assert "docs OK" in capsys.readouterr().out
