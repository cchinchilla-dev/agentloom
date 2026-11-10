"""Tests for `scripts/check_version_linearity.py`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "check_version_linearity.py"


def _load_script():
    """Import the script as a module (it is under scripts/, outside the package tree)."""
    spec = importlib.util.spec_from_file_location("check_version_linearity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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


def test_version_linearity_fails_on_malformed_pyproject(tmp_path: Path, script) -> None:
    (tmp_path / "pyproject.toml").write_text('[project\nversion = "1.0.0"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [1.0.0] - 2026-04-28\n")
    code, message = script.check(tmp_path)
    assert code == 1
    assert "invalid pyproject.toml" in message


def test_version_linearity_fails_when_pyproject_lacks_version(tmp_path: Path, script) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [1.0.0] - 2026-04-28\n")
    code, message = script.check(tmp_path)
    assert code == 1
    assert "[project].version" in message


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
    repo_root = Path(__file__).resolve().parent.parent.parent
    module = _load_script()
    code, message = module.check(repo_root)
    assert code == 0, message


def _write_package(tmp_path: Path, *, version: str, init_version: str) -> None:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "pkg"\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(f"## [{version}] - 2026-10-17\n")
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text(f'__version__ = "{init_version}"\n')


def test_version_linearity_fails_when_init_version_drifts(tmp_path: Path, script) -> None:
    _write_package(tmp_path, version="1.2.3", init_version="1.2.2")
    code, message = script.check(tmp_path)
    assert code == 1
    assert "src/pkg/__init__.py=1.2.2" in message


def test_version_linearity_passes_when_init_version_matches(tmp_path: Path, script) -> None:
    _write_package(tmp_path, version="1.2.3", init_version="1.2.3")
    code, _ = script.check(tmp_path)
    assert code == 0


def test_version_linearity_fails_when_named_package_has_no_version(tmp_path: Path, script) -> None:
    _write_package(tmp_path, version="1.2.3", init_version="1.2.3")
    (tmp_path / "src" / "pkg" / "__init__.py").write_text('"""No version here."""\n')
    code, message = script.check(tmp_path)
    assert code == 1
    assert "src/pkg/__init__.py has no __version__" in message


def _write_chart(tmp_path: Path, line: str) -> None:
    chart = tmp_path / "deploy" / "helm" / "agentloom"
    chart.mkdir(parents=True)
    (chart / "Chart.yaml").write_text(f"apiVersion: v2\nname: agentloom\nversion: 0.2.1\n{line}")


def test_version_linearity_fails_when_the_chart_app_version_differs(tmp_path: Path, script) -> None:
    _write_package(tmp_path, version="1.2.3", init_version="1.2.3")
    _write_chart(tmp_path, 'appVersion: "0.1.2"\n')
    code, message = script.check(tmp_path)
    assert code == 1
    assert "deploy/helm/agentloom/Chart.yaml appVersion=0.1.2" in message


@pytest.mark.parametrize(
    "line", ['appVersion: "1.2.3"\n', "appVersion: '1.2.3'\n", "appVersion: 1.2.3  # the image\n"]
)
def test_version_linearity_accepts_a_quoted_or_bare_chart_app_version(
    tmp_path: Path, script, line: str
) -> None:
    _write_package(tmp_path, version="1.2.3", init_version="1.2.3")
    _write_chart(tmp_path, line)
    code, message = script.check(tmp_path)
    assert code == 0, message


def test_version_linearity_fails_when_the_chart_has_no_app_version(tmp_path: Path, script) -> None:
    _write_package(tmp_path, version="1.2.3", init_version="1.2.3")
    _write_chart(tmp_path, "")
    code, message = script.check(tmp_path)
    assert code == 1
    assert "deploy/helm/agentloom/Chart.yaml has no appVersion" in message


def test_version_linearity_passes_without_a_chart(tmp_path: Path, script) -> None:
    _write_package(tmp_path, version="1.2.3", init_version="1.2.3")
    assert not (tmp_path / "deploy").exists()
    code, _ = script.check(tmp_path)
    assert code == 0


def _write_overlay(tmp_path: Path, body: str) -> None:
    path = tmp_path / "deploy" / "k8s" / "overlays" / "production" / "kustomization.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(body)


def test_version_linearity_fails_when_the_production_image_tag_differs(
    tmp_path: Path, script
) -> None:
    _write_repo(tmp_path, py_version="1.2.3", changelog="## [1.2.3]\n")
    _write_overlay(tmp_path, "images:\n  - name: x\n    newTag: 1.2.2  # Pin to release\n")
    code, message = script.check(tmp_path)
    assert code != 0
    assert "kustomization.yaml newTag=1.2.2" in message


def test_version_linearity_passes_when_the_production_image_tag_matches(
    tmp_path: Path, script
) -> None:
    _write_repo(tmp_path, py_version="1.2.3", changelog="## [1.2.3]\n")
    _write_overlay(tmp_path, 'images:\n  - name: x\n    newTag: "1.2.3"\n')
    assert script.check(tmp_path)[0] == 0


def test_version_linearity_fails_when_the_production_overlay_has_no_tag(
    tmp_path: Path, script
) -> None:
    _write_repo(tmp_path, py_version="1.2.3", changelog="## [1.2.3]\n")
    _write_overlay(tmp_path, "images: []\n")
    code, message = script.check(tmp_path)
    assert code != 0
    assert "has no newTag" in message
