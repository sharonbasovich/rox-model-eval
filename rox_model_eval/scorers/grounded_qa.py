"""C4 grounded CRM Q&A: correct answer, valid citations, abstain when unanswerable.

Output contract: {"answerable": bool, "answer": str, "citations": [record_id]}.
Answering a question the CRM cannot support is scored as fabrication -- that is
the "confident wrong answer in the chat agent" failure.
"""

from __future__ import annotations

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .common import coverage, flatten_inputs, parse_json, unsupported_numbers


def score_grounded_qa(suite: Suite, task: Task, output: RunOutput) -> ScoreBreakdown:
    parsed = parse_json(output.text)
    if not isinstance(parsed, dict) or "answerable" not in parsed:
        return ScoreBreakdown(
            passed=False,
            score=0.0,
            format_valid=False,
            notes=["output is not a {answerable, answer, citations} object"],
        )
    exp = task.expected
    said_answerable = bool(parsed.get("answerable"))
    answer = str(parsed.get("answer", ""))
    citations = [str(c) for c in parsed.get("citations", []) or []]

    if not exp["answerable"]:
        ok = not said_answerable
        return ScoreBreakdown(
            passed=ok,
            score=1.0 if ok else 0.0,
            fabrication=0.0 if ok else 1.0,
            metrics={"abstained": float(ok)},
            notes=[] if ok else ["answered a question the records cannot support"],
        )

    notes: list[str] = []
    if not said_answerable:
        notes.append("abstained on an answerable question")
    valid = set(exp.get("valid_ids", []))
    invalid = [c for c in citations if c not in valid]
    notes += [f"cites nonexistent record {c!r}" for c in invalid]
    required = exp.get("required_citations", [])
    recall = sum(1 for r in required if r in citations) / len(required) if required else 1.0
    cov, missing = coverage(answer, exp.get("answer_mentions", []))
    notes += [f"answer missing: {m}" for m in missing]
    derived = [
        alt
        for group in exp.get("answer_mentions", [])
        for alt in ([group] if isinstance(group, str) else group)
    ]
    invented = unsupported_numbers(answer, flatten_inputs(task.inputs), ignore=derived)
    notes += [f"unsupported number {n}" for n in invented]

    fabrication = min(1.0, (len(invalid) + len(invented)) / max(1, len(citations) + 1))
    score = (0.6 * cov + 0.4 * recall) * (1.0 if said_answerable else 0.0)
    return ScoreBreakdown(
        passed=said_answerable and cov >= 0.99 and recall >= 0.99 and fabrication == 0,
        score=round(score, 4),
        fabrication=round(fabrication, 4),
        metrics={"answer_coverage": round(cov, 4), "citation_recall": round(recall, 4)},
        notes=notes,
    )
