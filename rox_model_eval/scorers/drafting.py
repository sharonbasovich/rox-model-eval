"""C2 outbound drafting: hard constraints a sales team actually enforces.

Output contract: {"subject": str, "body": str}. Checked: word budget, subject
length, personalisation facts used, banned phrases (competitor names, spammy
claims, promises legal would reject), a call to action, and no invented numbers.
Tone and persuasiveness go to the optional LLM judge.
"""

from __future__ import annotations

import re

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .common import coverage, flatten_inputs, mentions, parse_json, unsupported_numbers

_CTA = re.compile(r"\?|\b(call|chat|meet|meeting|demo|calendar|time next week|15 minutes)\b", re.I)


def score_drafting(suite: Suite, task: Task, output: RunOutput) -> ScoreBreakdown:
    parsed = parse_json(output.text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("body"), str):
        return ScoreBreakdown(
            passed=False,
            score=0.0,
            format_valid=False,
            notes=["output is not a {subject, body} JSON object"],
        )
    exp = task.expected
    subject = str(parsed.get("subject", ""))
    body = parsed["body"]
    checks: dict[str, bool] = {}
    notes: list[str] = []

    words = len(body.split())
    max_words = int(exp.get("max_words", 150))
    checks["word_budget"] = words <= max_words
    if not checks["word_budget"]:
        notes.append(f"{words} words > budget {max_words}")
    checks["subject"] = 0 < len(subject) <= int(exp.get("max_subject_chars", 60))
    if not checks["subject"]:
        notes.append(f"subject length {len(subject)} out of bounds")
    cov, missing = coverage(f"{subject} {body}", exp.get("must_include", []))
    checks["personalization"] = cov >= 0.99
    notes += [f"missing personalization: {m}" for m in missing]
    banned = [p for p in exp.get("must_not_include", []) if mentions(f"{subject} {body}", p)]
    checks["banned_phrases"] = not banned
    notes += [f"banned phrase: {b!r}" for b in banned]
    if exp.get("require_cta", True):
        checks["cta"] = bool(_CTA.search(body))
        if not checks["cta"]:
            notes.append("no call to action")
    invented = unsupported_numbers(f"{subject} {body}", flatten_inputs(task.inputs))
    notes += [f"unsupported number {n}" for n in invented]

    adherence = sum(checks.values()) / len(checks)
    fabrication = min(1.0, len(invented) / 3)
    score = 0.85 * adherence + 0.15 * (1 - fabrication)
    metrics = {f"constraint_{k}": float(v) for k, v in checks.items()}
    metrics["constraint_adherence"] = round(adherence, 4)
    return ScoreBreakdown(
        passed=all(checks.values()) and not invented,
        score=round(score, 4),
        fabrication=round(fabrication, 4),
        metrics=metrics,
        notes=notes,
    )
