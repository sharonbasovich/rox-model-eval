"""Turns raw attempts into per-model metrics and an adopt/route/hold verdict."""

from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel

from .config import ModelSpec, Weights
from .types import Attempt


class ModelSummary(BaseModel):
    model_id: str
    capability: str
    attempts: int
    errors: int
    pass_rate: float
    mean_score: float
    score_stdev: float
    json_valid_rate: float
    schema_valid_rate: float
    fabrication_rate: float
    p50_latency_s: float
    p95_latency_s: float
    p50_ttft_s: float | None
    cost_per_task_usd: float
    cost_per_success_usd: float | None
    baseline: bool
    verdict: str = ""
    reasons: list[str] = []


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def summarize(attempts: list[Attempt], specs: dict[str, ModelSpec]) -> list[ModelSummary]:
    grouped: dict[tuple[str, str], list[Attempt]] = defaultdict(list)
    for a in attempts:
        grouped[(a.model_id, a.capability)].append(a)

    summaries = []
    for (model_id, capability), rows in grouped.items():
        n = len(rows)
        passes = sum(r.scores.passed for r in rows)
        scores = [r.scores.score for r in rows]
        latencies = [r.response.timings.total_s for r in rows if not r.response.error]
        ttfts = [r.response.timings.ttft_s for r in rows if r.response.timings.ttft_s is not None]
        total_cost = sum(r.cost_usd for r in rows)
        summaries.append(
            ModelSummary(
                model_id=model_id,
                capability=capability,
                attempts=n,
                errors=sum(1 for r in rows if r.response.error),
                pass_rate=passes / n,
                mean_score=statistics.fmean(scores),
                score_stdev=statistics.pstdev(scores) if n > 1 else 0.0,
                json_valid_rate=sum(r.scores.json_valid for r in rows) / n,
                schema_valid_rate=sum(r.scores.schema_valid for r in rows) / n,
                fabrication_rate=statistics.fmean(r.scores.fabrication_rate for r in rows),
                p50_latency_s=_pct(latencies, 0.5),
                p95_latency_s=_pct(latencies, 0.95),
                p50_ttft_s=_pct(ttfts, 0.5) if ttfts else None,
                cost_per_task_usd=total_cost / n,
                cost_per_success_usd=(total_cost / passes) if passes else None,
                baseline=specs[model_id].baseline,
            )
        )
    return summaries


def recommend(summaries: list[ModelSummary], weights: Weights) -> list[ModelSummary]:
    """Gate on quality + fabrication, then compare cost-per-success with the baseline.

    ADOPT: clears gates and is at least as good as baseline while cheaper or equal.
    ROUTE: clears gates but trails baseline -- a candidate for a cheaper traffic tier.
    HOLD:  fails a gate, or clears gates without beating baseline on cost or quality.
    """
    by_cap: dict[str, list[ModelSummary]] = defaultdict(list)
    for s in summaries:
        by_cap[s.capability].append(s)

    for rows in by_cap.values():
        base = next((r for r in rows if r.baseline), None)
        for s in rows:
            reasons: list[str] = []
            if s.errors == s.attempts:
                s.verdict, s.reasons = "HOLD", ["every call errored (check API key / model id)"]
                continue
            if s.mean_score < weights.quality_gate:
                reasons.append(f"quality {s.mean_score:.2f} < gate {weights.quality_gate:.2f}")
            if s.fabrication_rate > weights.fabrication_gate:
                reasons.append(
                    f"fabrication {s.fabrication_rate:.1%} > gate {weights.fabrication_gate:.0%}"
                )
            if s.baseline:
                s.verdict = "BASELINE"
                s.reasons = ["current production default", *(f"fails gate: {r}" for r in reasons)]
                continue
            if reasons:
                s.verdict, s.reasons = "HOLD", reasons
                continue
            if base is None:
                s.verdict, s.reasons = "ADOPT", ["clears all gates (no baseline to compare)"]
                continue
            cheaper = (
                s.cost_per_success_usd is not None
                and base.cost_per_success_usd is not None
                and s.cost_per_success_usd <= base.cost_per_success_usd
            )
            as_good = s.mean_score >= base.mean_score - 0.01
            if as_good and cheaper:
                s.verdict, s.reasons = "ADOPT", ["matches baseline quality at lower $/success"]
            elif as_good:
                s.verdict, s.reasons = "ROUTE", ["matches baseline quality but costs more"]
            elif cheaper:
                s.verdict, s.reasons = "ROUTE", ["clears gates, cheaper, trails baseline"]
            else:
                s.verdict, s.reasons = "HOLD", ["trails baseline and costs more"]
    return summaries
