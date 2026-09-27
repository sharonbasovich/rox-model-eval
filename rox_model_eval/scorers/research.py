"""C1 account research: coverage of key facts, citation validity, invented numbers.

Output contract: {"summary": str, "facts": [{"claim": str, "source": "S1"}],
"open_questions": [str]}. A brief that reads well but cites a source that does
not exist, or states a number no source contains, is exactly the failure a rep
will repeat to a prospect -- so both are hard gates.
"""

from __future__ import annotations

from typing import Any

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .common import coverage, flatten_inputs, mentions, parse_json, unsupported_numbers


def score_research(suite: Suite, task: Task, output: RunOutput) -> ScoreBreakdown:
    parsed = parse_json(output.text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        return ScoreBreakdown(
            passed=False,
            score=0.0,
            format_valid=False,
            notes=["output is not a research brief JSON object"],
        )
    exp = task.expected
    valid_ids = set(exp.get("source_ids", []))
    facts: list[Any] = parsed["facts"]
    cited = [f for f in facts if isinstance(f, dict)]
    bad = [f.get("source") for f in cited if f.get("source") not in valid_ids]
    citation_validity = 1 - len(bad) / len(cited) if cited else 0.0

    body = " ".join([str(parsed.get("summary", "")), *(str(f.get("claim", "")) for f in cited)])
    cov, missing = coverage(body, exp.get("must_cover", []))
    forbidden = [p for p in exp.get("forbidden", []) if mentions(body, p)]
    invented = unsupported_numbers(body, flatten_inputs(task.inputs))
    claims_with_numbers = max(1, len(cited))
    fabrication = min(1.0, (len(invented) + len(forbidden)) / claims_with_numbers)

    notes = [f"missing: {m}" for m in missing]
    notes += [f"cites unknown source {b!r}" for b in bad]
    notes += [f"states unsupported claim {f!r}" for f in forbidden]
    notes += [f"unsupported number {n}" for n in invented]
    score = 0.5 * cov + 0.3 * citation_validity + 0.2 * (1 - fabrication)
    return ScoreBreakdown(
        passed=cov >= 0.75 and citation_validity >= 0.99 and fabrication == 0,
        score=round(score, 4),
        fabrication=round(fabrication, 4),
        metrics={"coverage": round(cov, 4), "citation_validity": round(citation_validity, 4)},
        notes=notes,
    )
