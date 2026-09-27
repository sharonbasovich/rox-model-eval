"""Deterministic scorer for structured extraction (C6).

Rox ingests messy text (pasted signatures, CSV rows, forwarded emails) into
strict records. What matters in production:
  * JSON validity and schema validity -- a malformed record breaks the pipeline.
  * Field accuracy against a labelled reference.
  * Fabrication -- a non-null value where the source text supports none. A
    fabricated employee count or funding stage silently poisons the CRM, so it
    is tracked separately from "wrong" and used as an adoption gate.
"""

from __future__ import annotations

import re
from typing import Any

import jsonschema

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .common import parse_json


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        cleaned = re.sub(r"^https?://", "", cleaned)
        cleaned = re.sub(r"^www\.", "", cleaned).rstrip("/")
        return cleaned or None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def score_record(
    output_text: str,
    expected: dict[str, Any],
    json_schema: dict[str, Any] | None,
    pass_threshold: float = 1.0,
) -> ScoreBreakdown:
    parsed = parse_json(output_text)
    if not isinstance(parsed, dict):
        return ScoreBreakdown(
            passed=False,
            score=0.0,
            format_valid=False,
            metrics={"json_valid": 0.0, "schema_valid": 0.0, "field_accuracy": 0.0},
            notes=["output is not a JSON object"],
        )

    notes: list[str] = []
    schema_valid = True
    if json_schema is not None:
        try:
            jsonschema.validate(parsed, json_schema)
        except jsonschema.ValidationError as exc:
            schema_valid = False
            notes.append(f"schema: {exc.message}")

    correct = 0
    fabricated = 0
    for field, want in expected.items():
        got = parsed.get(field)
        if _normalize(got) == _normalize(want):
            correct += 1
            continue
        if want is None:
            if got is not None:
                fabricated += 1
                notes.append(f"fabricated {field}={got!r} (source supports none)")
        else:
            notes.append(f"{field}: expected {want!r}, got {got!r}")

    total_nulls = sum(1 for v in expected.values() if v is None)
    field_accuracy = correct / len(expected) if expected else 1.0
    fabrication_rate = fabricated / total_nulls if total_nulls else 0.0
    score = field_accuracy if schema_valid else field_accuracy * 0.5
    return ScoreBreakdown(
        passed=schema_valid and field_accuracy >= pass_threshold,
        score=round(score, 4),
        format_valid=schema_valid,
        fabrication=round(fabrication_rate, 4),
        metrics={
            "json_valid": 1.0,
            "schema_valid": float(schema_valid),
            "field_accuracy": round(field_accuracy, 4),
        },
        notes=notes,
    )


def score_extraction(suite: Suite, task: Task, output: RunOutput) -> ScoreBreakdown:
    return score_record(output.text, task.expected, suite.json_schema)
