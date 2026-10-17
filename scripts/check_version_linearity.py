"""Fail CI if pyproject.toml, CHANGELOG.md and `__version__` disagree on the version.

Invoked from `.github/workflows/ci.yml` job `version-linearity`. Also runnable
locally as a pre-commit check. Mirrors the same gate used in agentanvil so
release prep stays consistent across the two repos.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

VERSION_HEADER_RE = re.compile(r"^##\s*\[?(\d+\.\d+\.\d+[\w.]*)\]?", re.M)
INIT_VERSION_RE = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.M)


def latest_changelog_version(text: str) -> str | None:
    match = VERSION_HEADER_RE.search(text)
    return match.group(1) if match else None


def package_version(root: Path, project: dict[str, object]) -> tuple[str, str] | None:
    """Return (path, version) for `__version__` in src/<name>/__init__.py, or None if absent."""
    name = project.get("name")
    if not isinstance(name, str):
        return None
    init = root / "src" / name.replace("-", "_") / "__init__.py"
    if not init.exists():
        return None
    match = INIT_VERSION_RE.search(init.read_text())
    return (init.relative_to(root).as_posix(), match.group(1)) if match else None


def check(root: Path) -> tuple[int, str]:
    """Return (exit_code, message). 0 = ok, non-zero = fail."""
    pyproject_path = root / "pyproject.toml"
    changelog_path = root / "CHANGELOG.md"
    if not pyproject_path.exists():
        return 1, f"pyproject.toml not found at {pyproject_path}"
    if not changelog_path.exists():
        return 1, f"CHANGELOG.md not found at {changelog_path}"

    try:
        project = tomllib.loads(pyproject_path.read_text())["project"]
        declared = project["version"]
    except tomllib.TOMLDecodeError as exc:
        return 1, f"invalid pyproject.toml: {exc}"
    except (KeyError, TypeError):
        return 1, "pyproject.toml must define [project].version"

    latest = latest_changelog_version(changelog_path.read_text())

    if latest is None:
        return 1, "CHANGELOG.md has no versioned entry"
    if declared != latest:
        return 1, f"version mismatch: pyproject.toml={declared} vs CHANGELOG.md={latest}"
    init = package_version(root, project)
    if init is not None and init[1] != declared:
        return 1, f"version mismatch: pyproject.toml={declared} vs {init[0]}={init[1]}"
    return 0, f"version-linearity OK: {declared}"


def main(root: Path | None = None) -> int:
    code, message = check(root or Path.cwd())
    stream = sys.stdout if code == 0 else sys.stderr
    print(message, file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
