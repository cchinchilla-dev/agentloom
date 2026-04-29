"""Tests for `scripts/check_version_linearity.py`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "check_version_linearity.py"


def _load_script():
    """Import the script as a module (it is under scripts/, outside the package tree)."""
    spec = importlib.util.spec_from_file_location("check_version_linearity", SCRIPT)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _write_repo(tmp_path: Path, *, py_version: str, changelog: str) -> None:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{py_version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(changelog)


def test_version_linearity_passes_on_clean_state(tmp_path: Path, script) -> None:
    _write_repo(
        tmp_path,
        py_version="1.2.3",
        changelog="# Changelog\n\n## [1.2.3] - 2026-04-26\n",
    )
    code, message = script.check(tmp_path)
    assert code == 0
    assert "1.2.3" in message


def test_version_linearity_fails_on_mismatch(tmp_path: Path, script) -> None:
    _write_repo(
        tmp_path,
        py_version="1.2.3",
        changelog="# Changelog\n\n## [1.2.4] - 2026-04-26\n",
    )
    code, message = script.check(tmp_path)
    assert code == 1
    assert "mismatch" in message
    assert "1.2.3" in message and "1.2.4" in message


def test_version_linearity_fails_on_changelog_without_versioned_entry(
    tmp_path: Path, script
) -> None:
    _write_repo(
        tmp_path,
        py_version="1.2.3",
        changelog="# Changelog\n\n## [Unreleased]\n",
    )
    code, message = script.check(tmp_path)
    assert code == 1
    assert "no versioned entry" in message


def test_version_linearity_fails_when_pyproject_missing(tmp_path: Path, script) -> None:
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [1.0.0]\n")
    code, message = script.check(tmp_path)
    assert code == 1
    assert "pyproject.toml not found" in message


def test_version_linearity_fails_when_changelog_missing(tmp_path: Path, script) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n')
    code, message = script.check(tmp_path)
    assert code == 1
    assert "CHANGELOG.md not found" in message


def test_latest_changelog_version_picks_first_versioned_header(script) -> None:
    text = """# Changelog

## [Unreleased]

## [2.0.0] - 2026-05-01

## [1.0.0] - 2026-01-01
"""
    assert script.latest_changelog_version(text) == "2.0.0"


def test_latest_changelog_version_handles_pre_release_suffix(script) -> None:
    text = "## [0.2.0a0] - 2026-04-26\n"
    assert script.latest_changelog_version(text) == "0.2.0a0"


def test_latest_changelog_version_returns_none_for_unreleased_only(script) -> None:
    text = "# Changelog\n\n## [Unreleased]\n"
    assert script.latest_changelog_version(text) is None


def test_check_against_real_repo_passes() -> None:
    """The repo itself must always pass version-linearity (regression guard)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    module = _load_script()
    code, message = module.check(repo_root)
    assert code == 0, message
