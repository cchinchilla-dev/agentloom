"""Fail CI if the pages under docs/ would not publish.

Invoked from `.github/workflows/ci.yml` job `docs`. The site that renders these
pages reads each page's frontmatter and resolves its links, so this checks what
it needs: the frontmatter keys, the title heading, links to files and headings
that exist, and no syntax left from the old MkDocs site.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED = {"title": str, "description": str, "order": int, "section": str}
MAX_DESCRIPTION = 160  # what search results show of a description
FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LINK_RE = re.compile(r"!?\[(?:[^\]\\]|\\.)*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
# Admonitions, content tabs and snippets of MkDocs Material, which GitHub and the site show as text
MKDOCS_RE = re.compile(r"^\s*(!!!|\?\?\?\+?|=== \"|--8<--)")
MARKERS = ("tabs", "cards")


def slug(text: str) -> str:
    """The id a heading gets on the site and on GitHub (github-slugger)."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text.lower())
    return text.replace(" ", "-")


def frontmatter(text: str) -> tuple[dict[str, object] | None, str]:
    """Return the `key: value` pairs between the leading `---` lines, and the body after them."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    data: dict[str, object] = {}
    for line in text[4:end].splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            data[key.strip()] = value[1:-1].replace('\\"', '"')
        elif re.fullmatch(r"-?\d+", value):
            data[key.strip()] = int(value)
        else:
            data[key.strip()] = value
    return data, text[end + 5 :]


def prose_lines(body: str) -> list[tuple[int, str]]:
    """Return (line number, text) of the lines outside code fences."""
    lines, fenced = [], False
    for number, line in enumerate(body.splitlines(), start=1):
        if FENCE_RE.match(line):
            fenced = not fenced
        elif not fenced:
            lines.append((number, line))
    return lines


def anchors(body: str) -> set[str]:
    """Return the ids of the headings of a page, numbered on repeats as github-slugger does."""
    seen: dict[str, int] = {}
    found = set()
    for _, line in prose_lines(body):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = slug(match.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return found


def check_page(path: Path, root: Path, bodies: dict[Path, str]) -> list[str]:
    name = path.relative_to(root).as_posix()
    data, body = frontmatter(path.read_text(encoding="utf-8"))
    if data is None:
        return [f"{name}: no frontmatter between --- lines"]
    errors = []
    for key, kind in REQUIRED.items():
        if not isinstance(data.get(key), kind):
            errors.append(f"{name}: frontmatter needs '{key}' ({kind.__name__})")
    description = data.get("description")
    if isinstance(description, str) and len(description) > MAX_DESCRIPTION:
        size = len(description)
        errors.append(f"{name}: description has {size} characters, over {MAX_DESCRIPTION}")
    lines = prose_lines(body)
    # The site takes the h1 and the paragraph after it as the page's heading and lede
    first = next((line for _, line in lines if line.strip()), "")
    if not re.match(r"# \S", first):
        errors.append(f"{name}: the first line after the frontmatter must be an h1 ('# …')")
    open_marker: str | None = None
    for number, line in lines:
        where = f"{name}:{number}"
        if MKDOCS_RE.match(line):
            errors.append(f"{where}: MkDocs syntax, which the site shows as text: {line.strip()}")
        marker = re.fullmatch(r"\s*<!--\s*(/?)(\w+)\s*-->\s*", line)
        if marker and marker.group(2) in MARKERS:
            closing, block = marker.group(1) == "/", marker.group(2)
            if closing and open_marker != block or not closing and open_marker is not None:
                errors.append(f"{where}: unbalanced <!-- {block} --> marker")
            open_marker = None if closing else block
        for target in LINK_RE.findall(INLINE_CODE_RE.sub("", line)):
            errors.extend(check_link(target, path, root, bodies, where))
    if open_marker is not None:
        errors.append(f"{name}: <!-- {open_marker} --> is never closed")
    return errors


def check_link(
    target: str, page: Path, root: Path, bodies: dict[Path, str], where: str
) -> list[str]:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # https:, mailto:
        return []
    file_part, _, anchor = target.partition("#")
    dest = (page.parent / file_part).resolve() if file_part else page.resolve()
    if not dest.is_relative_to(root.resolve()):
        return [f"{where}: link outside the repository: {target}"]
    if not dest.exists():
        return [f"{where}: link to a missing file: {target}"]
    if anchor and dest.suffix == ".md" and dest in bodies and anchor not in anchors(bodies[dest]):
        return [f"{where}: link to a missing heading: {target}"]
    return []


def check(root: Path) -> tuple[int, str]:
    """Return (exit_code, message). 0 = ok, non-zero = fail."""
    pages = sorted((root / "docs").glob("*.md"))
    if not pages:
        return 1, f"no pages under {root / 'docs'}"
    bodies = {page.resolve(): frontmatter(page.read_text(encoding="utf-8"))[1] for page in pages}
    errors = [error for page in pages for error in check_page(page, root, bodies)]
    orders: dict[object, list[str]] = {}
    for page in pages:
        data, _ = frontmatter(page.read_text(encoding="utf-8"))
        if data and isinstance(data.get("order"), int):
            orders.setdefault(data["order"], []).append(page.name)
    for order, names in orders.items():
        if len(names) > 1:
            errors.append(f"docs: pages share order {order}: {', '.join(names)}")
    if errors:
        return 1, "\n".join(errors)
    return 0, f"docs OK: {len(pages)} pages"


def main(root: Path | None = None) -> int:
    code, message = check(root or Path.cwd())
    stream = sys.stdout if code == 0 else sys.stderr
    print(message, file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
