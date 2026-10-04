"""C10 bulk data operations over generated tables (dedupe, reconciliation, rollups, import
validation). Output is one JSON object; every field is compared exactly.

`expected`:
  sets:      {field: [ids]}          compared as sets (F1)
  groups:    {field: [[ids], ...]}   duplicate clusters, compared by pairwise F1
  values:    {field: {key: value}}   numbers within 0.5, strings case-insensitive; share of
                                     keys right over the union of expected and returned keys
  valid_ids: ids present in the input; any other id returned counts as fabricated
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .common import norm, parse_json


def _f1(want: set[str], got: set[str]) -> float:
    if not want and not got:
        return 1.0
    hit = len(want & got)
    return 2 * hit / (len(want) + len(got))


def _pairs(groups: Any) -> set[str]:
    out: set[str] = set()
    for g in groups if isinstance(groups, list) else []:
        if isinstance(g, list):
            out |= {"|".join(sorted(p)) for p in combinations({str(x) for x in g}, 2)}
    return out


def _value_ok(want: Any, got: Any) -> bool:
    if isinstance(want, int | float) and not isinstance(want, bool):
        try:
            return abs(float(str(got).replace(",", "").replace("$", "")) - want) < 0.5
        except ValueError:
            return False
    return norm(want) == norm(got)


def _ids(value: Any) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_ids(v) for v in value)) if value else set()
    return {str(value)}


def score_data_ops(suite: Suite, task: Task, output: RunOutput) -> ScoreBreakdown:
    exp = task.expected
    parsed = parse_json(output.text)
    if not isinstance(parsed, dict):
        return ScoreBreakdown(
            passed=False, score=0.0, format_valid=False, notes=["output is not a JSON object"]
        )
    notes: list[str] = []
    metrics: dict[str, float] = {}
    returned: set[str] = set()

    for field, want in exp.get("sets", {}).items():
        got = _ids(parsed.get(field, []))
        returned |= got
        metrics[field] = round(_f1({str(w) for w in want}, got), 4)
        missing, extra = sorted({str(w) for w in want} - got), sorted(got - {str(w) for w in want})
        if missing or extra:
            notes.append(f"{field}: missing {missing[:6]}, extra {extra[:6]}")
    for field, want in exp.get("groups", {}).items():
        got_pairs, want_pairs = _pairs(parsed.get(field)), _pairs(want)
        returned |= _ids(parsed.get(field, []))
        metrics[field] = round(_f1(want_pairs, got_pairs), 4)
        if got_pairs != want_pairs:
            notes.append(
                f"{field}: {len(want_pairs - got_pairs)} duplicate pairs missed, "
                f"{len(got_pairs - want_pairs)} wrongly merged"
            )
    for field, want in exp.get("values", {}).items():
        raw = parsed.get(field)
        answer: dict[str, Any] = raw if isinstance(raw, dict) else {}
        keys = {str(k) for k in want} | {str(k) for k in answer}
        wrong = sorted(
            k for k in keys if k not in want or k not in answer or not _value_ok(want[k], answer[k])
        )
        metrics[field] = round(1 - len(wrong) / len(keys), 4) if keys else 1.0
        notes += [
            f"{field}[{k}]: expected {want.get(k)!r}, got {answer.get(k)!r}" for k in wrong[:6]
        ]

    valid = {str(v) for v in exp.get("valid_ids", [])}
    fabricated = sorted(returned - valid) if valid else []
    if fabricated:
        notes.append(f"ids not in the input: {fabricated[:6]}")
    parts = list(metrics.values())
    score = sum(parts) / len(parts) if parts else 0.0
    return ScoreBreakdown(
        passed=bool(parts) and all(p == 1.0 for p in parts) and not fabricated,
        score=round(score, 4),
        fabrication=round(len(fabricated) / max(1, len(returned)), 4),
        metrics=metrics,
        notes=notes,
    )
