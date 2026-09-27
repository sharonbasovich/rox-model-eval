"""C3 insights / prioritisation: rank deals or accounts, justify the top pick.

Output contract: {"ranking": [id, ...], "top_reason": str}. Scored with NDCG
against a labelled ideal order, plus whether the top pick is right (that is what
a rep acts on) and whether its reason mentions the decisive signal.
"""

from __future__ import annotations

import math

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .common import coverage, parse_json


def ndcg(predicted: list[str], ideal: list[str]) -> float:
    rel = {item: len(ideal) - i for i, item in enumerate(ideal)}
    dcg = sum(rel.get(item, 0) / math.log2(i + 2) for i, item in enumerate(predicted))
    idcg = sum(rel[item] / math.log2(i + 2) for i, item in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def score_ranking(suite: Suite, task: Task, output: RunOutput) -> ScoreBreakdown:
    parsed = parse_json(output.text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("ranking"), list):
        return ScoreBreakdown(
            passed=False,
            score=0.0,
            format_valid=False,
            notes=["output is not a {ranking, top_reason} JSON object"],
        )
    ideal = [str(x) for x in task.expected["ideal_order"]]
    predicted = [str(x) for x in parsed["ranking"]]
    notes: list[str] = []
    unknown = [p for p in predicted if p not in ideal]
    missing = [i for i in ideal if i not in predicted]
    if unknown:
        notes.append(f"ranked unknown ids {unknown}")
    if missing:
        notes.append(f"omitted ids {missing}")
    quality = ndcg(predicted, ideal)
    top_ok = bool(predicted) and predicted[0] == ideal[0]
    if not top_ok:
        notes.append(f"top pick {predicted[:1]} != {ideal[0]!r}")
    reason_cov, reason_missing = coverage(
        str(parsed.get("top_reason", "")), task.expected.get("top_reason_mentions", [])
    )
    notes += [f"top_reason missing: {m}" for m in reason_missing]
    score = 0.6 * quality + 0.25 * float(top_ok) + 0.15 * reason_cov
    return ScoreBreakdown(
        passed=top_ok and quality >= 0.9 and not unknown and not missing,
        score=round(score, 4),
        fabrication=round(len(unknown) / max(1, len(predicted)), 4),
        metrics={"ndcg": round(quality, 4), "top1": float(top_ok), "reason": reason_cov},
        notes=notes,
    )
