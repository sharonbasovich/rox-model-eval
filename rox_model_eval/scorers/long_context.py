"""C7 long-context synthesis: find needles, ignore superseded values.

Output contract: a JSON object of named fields. `expected.fields` maps field ->
accepted phrasings; `expected.traps` maps field -> stale/distractor values that
appear earlier in the transcript but were later corrected.
"""

from __future__ import annotations

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .common import mentions, parse_json


def score_long_context(suite: Suite, task: Task, output: RunOutput) -> ScoreBreakdown:
    parsed = parse_json(output.text)
    if not isinstance(parsed, dict):
        return ScoreBreakdown(
            passed=False, score=0.0, format_valid=False, notes=["output is not a JSON object"]
        )
    fields: dict[str, list[str]] = task.expected["fields"]
    traps: dict[str, list[str]] = task.expected.get("traps", {})
    notes: list[str] = []
    correct = 0
    trapped = 0
    for name, accepted in fields.items():
        got = str(parsed.get(name, ""))
        if mentions(got, accepted):
            correct += 1
        elif name in traps and mentions(got, traps[name]):
            trapped += 1
            notes.append(f"{name}: used superseded value {got!r}")
        else:
            notes.append(f"{name}: expected {accepted[0]!r}, got {got!r}")
    accuracy = correct / len(fields)
    return ScoreBreakdown(
        passed=accuracy >= 0.8 and trapped == 0,
        score=round(accuracy, 4),
        metrics={
            "needle_recall": round(accuracy, 4),
            "distractor_rate": round(trapped / len(fields), 4),
        },
        notes=notes,
    )
