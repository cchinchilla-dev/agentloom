"""Check the Helm chart's values.schema.json against values.yaml, without the helm binary."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

_CHART_DIR = Path(__file__).resolve().parents[2] / "deploy" / "helm" / "agentloom"
_SCHEMA_PATH = _CHART_DIR / "values.schema.json"
_VALUES_PATH = _CHART_DIR / "values.yaml"


def _schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def _values() -> dict:
    return yaml.safe_load(_VALUES_PATH.read_text())


def test_schema_file_exists() -> None:
    assert _SCHEMA_PATH.is_file(), "values.schema.json must ship with the chart"


def test_schema_is_valid_json_object() -> None:
    schema = _schema()
    assert schema["type"] == "object"
    # Top-level ``additionalProperties: false`` is what rejects typo'd ``--set`` keys.
    assert schema["additionalProperties"] is False


def test_schema_covers_every_values_key() -> None:
    """Every values.yaml key is declared, or ``helm lint`` fails on the chart's own defaults."""
    schema_keys = set(_schema()["properties"].keys())
    values_keys = set(_values().keys())
    missing = values_keys - schema_keys
    assert not missing, f"values.yaml keys missing from schema: {sorted(missing)}"


def test_schema_declares_no_phantom_keys() -> None:
    """Every schema property has a values.yaml key; none drifts ahead of the chart."""
    schema_keys = set(_schema()["properties"].keys())
    values_keys = set(_values().keys())
    phantom = schema_keys - values_keys
    assert not phantom, f"schema declares keys absent from values.yaml: {sorted(phantom)}"


def test_typed_fields_have_explicit_types() -> None:
    # Spot-check that known sub-objects pin field types, so a wrong-typed ``--set`` fails.
    props = _schema()["properties"]
    assert props["schedule"]["properties"]["enabled"]["type"] == "boolean"
    assert props["job"]["properties"]["backoffLimit"]["type"] == "integer"
    assert props["image"]["properties"]["pullPolicy"]["enum"] == [
        "Always",
        "IfNotPresent",
        "Never",
    ]
