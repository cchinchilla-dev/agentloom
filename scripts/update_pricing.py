"""Refresh the vendor entries of `providers/pricing.yaml` from LiteLLM's public price list.

    python scripts/update_pricing.py [--source FILE] [--table FILE] [--changes FILE]

Prices come from LiteLLM's `model_prices_and_context_window.json` (MIT), fetched unless
`--source` names a local copy. The OpenAI, Anthropic and Gemini entries it lists are updated,
and the text models the table lacks are added at the end of their section. Nothing is removed,
so local models stay as they are, and an entry whose key line ends in `# pinned` keeps its
prices. The table is rewritten as text, so its comments and order survive, and only when a
price changes; `--changes` then gets the changes as Markdown. Run weekly by
`.github/workflows/pricing-update.yml`, which opens a pull request with them.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import urllib.request
from decimal import Decimal
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
TABLE = Path(__file__).resolve().parent.parent / "src" / "agentloom" / "providers" / "pricing.yaml"
# LiteLLM's provider -> the table's section; no Vertex, as the Google adapter calls AI Studio
SECTIONS = {"openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Google Gemini"}
EMBEDDINGS = "Embeddings"
MODES = {"chat", "responses", "embedding"}
# Never added: no text in and out, a moving alias or an experiment, a fine-tune (its id holds a
# colon), or open weights, which a prefix match would price when they run locally
SKIP = re.compile(
    r"audio|realtime|tts|transcribe|search|image|computer-use|robotics|lyria|customtools"
    r"|-latest$|-exp-|experimental|learnlm|^ft:|gemma"
)
DATE = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$")
HEADER = re.compile(r"^(# Model pricing \(USD per 1K tokens\)) — .*$")
SECTION = re.compile(r"^# --- (.+?)(?: \(.*\))? ---")  # "Embeddings (output: 0.0)" -> "Embeddings"
KEY = re.compile(r"^([A-Za-z0-9][^:\s]*):\s*(#.*)?$")
PRICE = re.compile(r"^(  (input|output): )(\S+)\s*$")

Prices = tuple[Decimal, Decimal]
Found = dict[str, tuple[str, Prices]]
Row = tuple[str, str, str]


def vendor_prices(source: dict[str, object]) -> Found:
    """The section and per-1K prices of each text model of the three vendors in *source*."""
    found: Found = {}
    for key, entry in source.items():
        if not isinstance(entry, dict) or entry.get("mode") not in MODES:
            continue
        provider = entry.get("litellm_provider")
        if provider not in SECTIONS:
            continue
        name = key.removeprefix("gemini/") if provider == "gemini" else key
        embedding = entry["mode"] == "embedding"
        if "/" in name or "input_cost_per_token" not in entry:
            continue
        if not embedding and "output_cost_per_token" not in entry:
            continue
        # The base fields are Gemini Pro's <=200K tier; the tiers above it are left out
        prices = (
            _per_1k(entry["input_cost_per_token"]),
            _per_1k(entry.get("output_cost_per_token")),
        )
        if key.startswith("gemini/") or name not in found:  # AI Studio's own entry wins
            found[name] = (EMBEDDINGS if embedding else SECTIONS[str(provider)], prices)
    return found


def update(text: str, found: Found, today: datetime.date) -> tuple[str, str]:
    """*text* with *found*'s prices and the changes as Markdown; *text* itself if none change."""
    lines = text.splitlines()
    changed: list[Row] = []
    ends: dict[str, int] = {}  # section -> the line after its last entry
    seen: set[str] = set()
    section = None
    for i, line in enumerate(lines):
        if match := SECTION.match(line):
            section = match[1]
        elif (match := KEY.match(line)) and section:
            ends[section] = i + 3
            seen.add(match[1])
            if match[1] in found and "pinned" not in (match[2] or ""):
                _refresh(lines, i, match[1], found[match[1]][1], changed)
    new = sorted(n for n in found if n not in seen and _wanted(n, found))
    added: list[Row] = []
    # Bottom section first, so inserting a block leaves the lines of the ones above where they are
    for section in sorted({found[n][0] for n in new}, key=lambda s: -ends.get(s, -1)):
        if section not in ends:
            raise SystemExit(f"pricing table has no '# --- {section} ---' section with entries")
        block = []
        for name in (n for n in new if found[n][0] == section):
            prices = [_fmt(p) for p in found[name][1]]
            block += [f"{name}:", f"  input: {prices[0]}", f"  output: {prices[1]}"]
            added.append((name, prices[0], prices[1]))
        lines[ends[section] : ends[section]] = block
    if not changed and not added:
        return text, ""
    lines[0] = HEADER.sub(rf"\1 — updated {today.isoformat()}.", lines[0])
    return "\n".join(lines) + "\n", _report(changed, sorted(added))


def _refresh(lines: list[str], i: int, key: str, prices: Prices, changed: list[Row]) -> None:
    """Write *prices* into the input and output lines under *key*, at line *i*."""
    cells = []
    for j, field, price in ((i + 1, "input", prices[0]), (i + 2, "output", prices[1])):
        match = PRICE.match(lines[j]) if j < len(lines) else None
        if match is None or match[2] != field:
            raise SystemExit(f"pricing table line {j + 1}: expected '{field}:' under '{key}'")
        if Decimal(match[3]) == price:
            cells.append(match[3])
        else:
            cells.append(f"{match[3]} → {_fmt(price)}")
            lines[j] = match[1] + _fmt(price)
    if any("→" in cell for cell in cells):
        changed.append((key, cells[0], cells[1]))


def _wanted(name: str, found: Found) -> bool:
    """Whether a model the table lacks is added."""
    if SKIP.search(name) or not any(found[name][1]):  # a model the table lacks already costs 0
        return False
    alias = DATE.sub("", name)
    # A dated id priced as its alias is already covered by the alias's prefix match
    return alias == name or alias not in found or found[alias][1] != found[name][1]


def _per_1k(price: float | None) -> Decimal:
    """A per-token price as a per-1K one, exactly (no float rounding)."""
    return Decimal(repr(float(price or 0))) * 1000


def _fmt(price: Decimal) -> str:
    """*price* as the table writes it: `0.0`, or at least five decimals."""
    if price == 0:
        return "0.0"
    decimals = len(format(price.normalize(), "f").partition(".")[2])
    return format(price, f".{max(decimals, 5)}f")


def _report(changed: list[Row], added: list[Row]) -> str:
    """The changes as Markdown, for the pull request."""
    out = [
        "Prices per 1K tokens (USD) from LiteLLM's `model_prices_and_context_window.json`. "
        "Check each against the vendor's pricing page before merging; an entry whose key line "
        "ends in `# pinned` keeps a price set by hand.",
        "",
    ]
    for title, rows in (("Changed", changed), ("Added", added)):
        if rows:
            out += [f"### {title}", "", "| Model | Input | Output |", "|---|---|---|"]
            out += [f"| `{name}` | {i} | {o} |" for name, i, o in rows] + [""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", type=Path, help="a local copy of the price list")
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--changes", type=Path, help="write the changes here as Markdown")
    args = parser.parse_args(argv)
    if args.source:
        raw = args.source.read_text(encoding="utf-8")
    else:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "agentloom-pricing"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    source = json.loads(raw)
    if not isinstance(source, dict):
        raise SystemExit("the price list is not a JSON object")
    text = args.table.read_text(encoding="utf-8")
    new, report = update(text, vendor_prices(source), datetime.date.today())
    if new == text:
        print("pricing table unchanged")
        return 0
    args.table.write_text(new, encoding="utf-8")
    if args.changes:
        args.changes.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
