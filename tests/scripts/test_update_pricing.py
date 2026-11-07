"""Tests for `scripts/update_pricing.py`."""

from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path

import pytest

from agentloom.providers import pricing

SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "update_pricing.py"
TODAY = datetime.date(2026, 11, 7)
TABLE = """\
# Model pricing (USD per 1K tokens) — updated July 2026.

# --- OpenAI ---
gpt-a:
  input: 0.00100
  output: 0.00400

# --- Anthropic ---
# A comment the update keeps.
claude-a:
  input: 0.00300
  output: 0.01500

# --- Google Gemini ---
gemini-a:
  input: 0.00030
  output: 0.00250

# --- Ollama (local — free) ---
qwen3:
  input: 0.0
  output: 0.0

# --- Embeddings (output: 0.0) ---
text-embedding-a:
  input: 0.00002
  output: 0.0
"""
# The newest models of each vendor, checked against their pricing pages on 2026-11-07
CURRENT = [
    "gpt-6.1-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]


def _load_script():
    """Import the script as a module (it is under scripts/, outside the package tree)."""
    spec = importlib.util.spec_from_file_location("update_pricing", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _entry(
    provider: str,
    per_token_in: float,
    per_token_out: float | None = None,
    mode: str = "chat",
    **extra: float,
) -> dict[str, object]:
    """An entry of the price list, which gives prices per token."""
    entry: dict[str, object] = {
        "litellm_provider": provider,
        "mode": mode,
        "input_cost_per_token": per_token_in,
        **extra,
    }
    if per_token_out is not None:
        entry["output_cost_per_token"] = per_token_out
    return entry


def _update(script, source: dict[str, object], table: str = TABLE) -> tuple[str, str]:
    return script.update(table, script.vendor_prices(source), TODAY)


@pytest.mark.parametrize("model", CURRENT)
def test_current_models_are_priced(model: str) -> None:
    table = pricing.load_pricing(str(Path(pricing.__file__).parent / "pricing.yaml"))
    assert model in table
    assert table[model].input_cost_per_1k > 0
    assert table[model].output_cost_per_1k > 0


def test_update_converts_per_token_prices_to_per_1k(script) -> None:
    new, report = _update(
        script,
        {
            "gpt-a": _entry("openai", 5.5e-07, 2.2e-06),
            "text-embedding-a": _entry("openai", 3.75e-08, mode="embedding"),
        },
    )
    assert "gpt-a:\n  input: 0.00055\n  output: 0.00220\n" in new
    assert "text-embedding-a:\n  input: 0.0000375\n  output: 0.0\n" in new
    assert "| `gpt-a` | 0.00100 → 0.00055 | 0.00400 → 0.00220 |" in report
    assert "| `text-embedding-a` | 0.00002 → 0.0000375 | 0.0 |" in report


def test_update_flattens_tiered_gemini_prices(script) -> None:
    new, _ = _update(
        script,
        {
            "gemini/gemini-a": _entry(
                "gemini",
                1.25e-06,
                1e-05,
                input_cost_per_token_above_200k_tokens=2.5e-06,
                output_cost_per_token_above_200k_tokens=1.5e-05,
            ),
            "gemini-a": _entry("gemini", 9e-06, 9e-05),  # AI Studio's prefixed entry wins
            "vertex_ai/gemini-a": _entry("vertex_ai-language-models", 8e-06, 8e-05),
        },
    )
    assert "gemini-a:\n  input: 0.00125\n  output: 0.01000\n" in new


def test_update_keeps_entries_the_source_lacks(script) -> None:
    new, _ = _update(script, {"gpt-a": _entry("openai", 2e-06, 8e-06)})
    assert "gpt-a:\n  input: 0.00200\n  output: 0.00800\n" in new
    assert "claude-a:\n  input: 0.00300\n  output: 0.01500\n" in new
    assert "qwen3:\n  input: 0.0\n  output: 0.0\n" in new
    assert "# A comment the update keeps.\n" in new


def test_update_leaves_an_unchanged_table_untouched(script, tmp_path: Path) -> None:
    source = {
        "gpt-a": _entry("openai", 1e-06, 4e-06),
        "claude-a": _entry("anthropic", 3e-06, 1.5e-05),
        "gemini/gemini-a": _entry("gemini", 3e-07, 2.5e-06),
        "text-embedding-a": _entry("openai", 2e-08, mode="embedding"),
    }
    assert _update(script, source) == (TABLE, "")
    table, changes = tmp_path / "pricing.yaml", tmp_path / "changes.md"
    table.write_text(TABLE, encoding="utf-8")
    (tmp_path / "prices.json").write_text(json.dumps(source), encoding="utf-8")
    argv = ["--source", str(tmp_path / "prices.json"), "--table", str(table)]
    assert script.main([*argv, "--changes", str(changes)]) == 0
    assert table.read_text(encoding="utf-8") == TABLE
    assert not changes.exists()


def test_update_keeps_a_pinned_entry(script) -> None:
    table = TABLE.replace("claude-a:\n", "claude-a:  # pinned: the vendor's page says otherwise\n")
    assert _update(script, {"claude-a": _entry("anthropic", 2e-06, 1e-05)}, table) == (table, "")


def test_update_adds_text_models_at_the_end_of_their_section(script) -> None:
    source = {
        "claude-b": _entry("anthropic", 1e-06, 5e-06),
        "gpt-b": _entry("openai", 5e-07, 2e-06),
        "text-embedding-b": _entry("openai", 1e-07, mode="embedding"),
        "gpt-a": _entry("openai", 1e-06, 4e-06),
        "gpt-a-2026-01-01": _entry("openai", 1e-06, 4e-06),  # its alias's prefix covers it
        "gpt-a-0613": _entry("openai", 3e-06, 6e-06),  # a snapshot priced apart
        "gpt-b-audio-preview": _entry("openai", 1e-06, 1e-06),
        "gpt-b-latest": _entry("openai", 1e-06, 1e-06),
        "ft:gpt-b": _entry("openai", 1e-06, 1e-06),
        "gemini/gemma-b": _entry("gemini", 1e-07, 2e-07),
        "gemini/gemini-free": _entry("gemini", 0.0, 0.0),
        "gpt-picture": _entry("openai", 1e-06, 1e-06, mode="image_generation"),
        "gpt-no-output": _entry("openai", 1e-06),
        "openai/gpt-c": _entry("openai", 1e-06, 1e-06),
        "gemini-vertex": _entry("vertex_ai-language-models", 1e-06, 4e-06),  # not AI Studio
    }
    found = script.vendor_prices(source)
    new, report = script.update(TABLE, found, TODAY)
    assert new.startswith("# Model pricing (USD per 1K tokens) — updated 2026-11-07.\n")
    assert (
        "gpt-a:\n  input: 0.00100\n  output: 0.00400\n"
        "gpt-a-0613:\n  input: 0.00300\n  output: 0.00600\n"
        "gpt-b:\n  input: 0.00050\n  output: 0.00200\n\n# --- Anthropic ---"
    ) in new
    assert (
        "claude-a:\n  input: 0.00300\n  output: 0.01500\n"
        "claude-b:\n  input: 0.00100\n  output: 0.00500\n\n# --- Google Gemini ---"
    ) in new
    assert new.endswith("text-embedding-b:\n  input: 0.00010\n  output: 0.0\n")
    left_out = ["2026-01-01", "audio", "latest", "ft:", "gemma", "gemini-free", "picture"]
    for name in [*left_out, "no-output", "gpt-c", "vertex"]:
        assert name not in new
    assert "| `gpt-b` | 0.00050 | 0.00200 |" in report
    assert "### Changed" not in report
    assert script.update(new, found, TODAY) == (new, "")


def test_update_fails_for_a_new_model_whose_section_the_table_lacks(script) -> None:
    table = TABLE.split("# --- Embeddings")[0]
    with pytest.raises(SystemExit, match="Embeddings"):
        _update(script, {"text-embedding-b": _entry("openai", 1e-07, mode="embedding")}, table)


def test_update_fails_for_an_entry_out_of_shape(script) -> None:
    swapped = "  output: 0.00400\n  input: 0.00100"
    table = TABLE.replace("  input: 0.00100\n  output: 0.00400", swapped)
    with pytest.raises(SystemExit, match="line 5: expected 'input:' under 'gpt-a'"):
        _update(script, {"gpt-a": _entry("openai", 2e-06, 8e-06)}, table)


def test_main_writes_the_table_and_its_changes(script, tmp_path: Path) -> None:
    table, changes = tmp_path / "pricing.yaml", tmp_path / "changes.md"
    table.write_text(TABLE, encoding="utf-8")
    source = tmp_path / "prices.json"
    source.write_text(json.dumps({"gpt-a": _entry("openai", 2e-06, 8e-06)}), encoding="utf-8")
    argv = ["--source", str(source), "--table", str(table), "--changes", str(changes)]
    assert script.main(argv) == 0
    text = table.read_text(encoding="utf-8")
    assert f"— updated {datetime.date.today().isoformat()}." in text.splitlines()[0]
    assert "gpt-a:\n  input: 0.00200\n  output: 0.00800\n" in text
    assert "| `gpt-a` | 0.00100 → 0.00200 | 0.00400 → 0.00800 |" in changes.read_text("utf-8")


def test_main_refuses_a_price_list_that_is_no_object(script, tmp_path: Path) -> None:
    (tmp_path / "prices.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a JSON object"):
        script.main(["--source", str(tmp_path / "prices.json")])
