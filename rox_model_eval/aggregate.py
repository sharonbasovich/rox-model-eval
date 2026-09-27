"""Turns raw attempts into per-capability metrics, verdicts, a composite fitness score,
a routing table and a cost/quality Pareto frontier."""

from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel, Field

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
    consistency: float
    format_valid_rate: float
    fabrication_rate: float
    safety_violation_rate: float
    judge_mean: float | None = None
    pairwise_win_rate: float | None = None
    p50_latency_s: float
    p95_latency_s: float
    p50_ttft_s: float | None
    cost_per_task_usd: float
    cost_per_success_usd: float | None
    baseline: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    gates_passed: bool = False
    verdict: str = ""
    reasons: list[str] = Field(default_factory=list)


class ModelOverall(BaseModel):
    model_id: str
    baseline: bool
    fitness: float
    weight_coverage: float
    cost_per_task_usd: float
    p95_latency_s: float
    capability_verdicts: dict[str, str]
    verdict: str = ""
    reasons: list[str] = Field(default_factory=list)
    pareto: bool = False


class RouteRow(BaseModel):
    capability: str
    quality_pick: str | None
    value_pick: str | None
    baseline: str | None
    note: str = ""


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
        by_task: dict[str, set[bool]] = defaultdict(set)
        for r in rows:
            by_task[r.task_id].add(r.scores.passed)
        metric_keys = sorted({k for r in rows for k in r.scores.metrics})
        judged = [r.scores.judge_score for r in rows if r.scores.judge_score is not None]
        summaries.append(
            ModelSummary(
                model_id=model_id,
                capability=capability,
                attempts=n,
                errors=sum(1 for r in rows if r.response.error),
                pass_rate=passes / n,
                mean_score=statistics.fmean(scores),
                score_stdev=statistics.pstdev(scores) if n > 1 else 0.0,
                consistency=sum(len(v) == 1 for v in by_task.values()) / len(by_task),
                format_valid_rate=sum(r.scores.format_valid for r in rows) / n,
                fabrication_rate=statistics.fmean(r.scores.fabrication for r in rows),
                safety_violation_rate=sum(r.scores.safety_violation for r in rows) / n,
                judge_mean=statistics.fmean(judged) if judged else None,
                p50_latency_s=_pct(latencies, 0.5),
                p95_latency_s=_pct(latencies, 0.95),
                p50_ttft_s=_pct(ttfts, 0.5) if ttfts else None,
                cost_per_task_usd=total_cost / n,
                cost_per_success_usd=(total_cost / passes) if passes else None,
                baseline=specs[model_id].baseline,
                metrics={
                    k: round(statistics.fmean(r.scores.metrics.get(k, 0.0) for r in rows), 4)
                    for k in metric_keys
                },
            )
        )
    return summaries


def _gate_failures(s: ModelSummary, w: Weights) -> list[str]:
    out = []
    if s.mean_score < w.quality_gate:
        out.append(f"quality {s.mean_score:.2f} < gate {w.quality_gate:.2f}")
    if s.fabrication_rate > w.fabrication_gate:
        out.append(f"fabrication {s.fabrication_rate:.1%} > gate {w.fabrication_gate:.0%}")
    if s.format_valid_rate < w.format_gate:
        out.append(f"format-valid {s.format_valid_rate:.0%} < gate {w.format_gate:.0%}")
    if s.safety_violation_rate > w.safety_gate:
        out.append(f"safety violations {s.safety_violation_rate:.1%} > gate {w.safety_gate:.0%}")
    return out


def recommend(summaries: list[ModelSummary], weights: Weights) -> list[ModelSummary]:
    """Gate on quality, fabrication, format and safety, then compare with the baseline.

    ADOPT: clears gates and is at least as good as baseline while cheaper or equal.
    ROUTE: clears gates but trails baseline or costs more -- a candidate for a traffic tier.
    HOLD:  fails a gate, or trails baseline and costs more.
    """
    by_cap: dict[str, list[ModelSummary]] = defaultdict(list)
    for s in summaries:
        by_cap[s.capability].append(s)

    for rows in by_cap.values():
        base = next((r for r in rows if r.baseline), None)
        for s in rows:
            if s.errors == s.attempts:
                s.gates_passed = False
                s.verdict, s.reasons = "HOLD", ["every call errored (check API key / model id)"]
                if s.baseline:
                    s.verdict = "BASELINE"
                continue
            failures = _gate_failures(s, weights)
            s.gates_passed = not failures
            if s.baseline:
                s.verdict = "BASELINE"
                s.reasons = ["current production default", *(f"fails gate: {r}" for r in failures)]
                continue
            if failures:
                s.verdict, s.reasons = "HOLD", failures
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


def overall(
    summaries: list[ModelSummary], specs: dict[str, ModelSpec], weights: Weights
) -> list[ModelOverall]:
    """Composite Rox Fitness Score per model plus an overall adopt/route/hold call."""
    by_model: dict[str, list[ModelSummary]] = defaultdict(list)
    for s in summaries:
        by_model[s.model_id].append(s)
    total_weight = sum(weights.capabilities.values()) or 1.0

    results: list[ModelOverall] = []
    for model_id, rows in by_model.items():
        ws = [(weights.capabilities.get(r.capability, 0.0) or 0.01, r) for r in rows]
        wsum = sum(w for w, _ in ws)
        results.append(
            ModelOverall(
                model_id=model_id,
                baseline=specs[model_id].baseline,
                fitness=round(sum(w * r.mean_score for w, r in ws) / wsum, 4),
                weight_coverage=round(
                    sum(weights.capabilities.get(r.capability, 0.0) for r in rows) / total_weight,
                    4,
                ),
                cost_per_task_usd=sum(w * r.cost_per_task_usd for w, r in ws) / wsum,
                p95_latency_s=max(r.p95_latency_s for r in rows),
                capability_verdicts={r.capability: r.verdict for r in rows},
            )
        )

    base = next((r for r in results if r.baseline), None)
    for o in results:
        rows = by_model[o.model_id]
        unsafe = [r.capability for r in rows if r.safety_violation_rate > weights.safety_gate]
        if o.baseline:
            o.verdict = "BASELINE"
            o.reasons = ["current production default"]
            o.reasons += [f"safety gate failed on {', '.join(unsafe)}"] if unsafe else []
            continue
        if unsafe:
            o.verdict, o.reasons = "HOLD", [f"safety gate failed on {', '.join(unsafe)}"]
            continue
        verdicts = o.capability_verdicts
        wins = sorted(c for c, v in verdicts.items() if v in {"ADOPT", "ROUTE"})
        heavy_holds = sorted(
            c
            for c, v in verdicts.items()
            if v == "HOLD" and weights.capabilities.get(c, 0.0) >= 0.15
        )
        if base is None:
            beats = True
        else:
            beats = (
                o.fitness >= base.fitness - 0.01 and o.cost_per_task_usd <= base.cost_per_task_usd
            )
        if all(v == "ADOPT" for v in verdicts.values()) or (beats and not heavy_holds):
            o.verdict = "ADOPT"
            o.reasons = [f"fitness {o.fitness:.2f} at ${o.cost_per_task_usd:.5f}/task"]
            if base:
                o.reasons.append(f"baseline {base.fitness:.2f} at ${base.cost_per_task_usd:.5f}")
        elif wins:
            o.verdict, o.reasons = "ROUTE", [f"route {', '.join(wins)}"]
            if heavy_holds:
                o.reasons.append(f"hold on high-traffic {', '.join(heavy_holds)}")
        else:
            o.verdict, o.reasons = "HOLD", ["no capability clears gates vs baseline"]

    for o in results:
        o.pareto = not any(
            other is not o
            and other.fitness >= o.fitness
            and other.cost_per_task_usd <= o.cost_per_task_usd
            and (other.fitness > o.fitness or other.cost_per_task_usd < o.cost_per_task_usd)
            for other in results
        )
    return sorted(results, key=lambda o: -o.fitness)


def routing_table(summaries: list[ModelSummary], weights: Weights) -> list[RouteRow]:
    """Per capability: best-quality model and cheapest model within `route_margin` of it."""
    by_cap: dict[str, list[ModelSummary]] = defaultdict(list)
    for s in summaries:
        by_cap[s.capability].append(s)
    table = []
    for cap in sorted(by_cap):
        rows = by_cap[cap]
        base = next((r.model_id for r in rows if r.baseline), None)
        ok = [r for r in rows if r.gates_passed]
        if not ok:
            table.append(
                RouteRow(
                    capability=cap,
                    quality_pick=None,
                    value_pick=None,
                    baseline=base,
                    note="no model clears gates; keep humans in loop",
                )
            )
            continue
        best = max(ok, key=lambda r: (r.mean_score, -(r.cost_per_success_usd or 0)))
        near = [r for r in ok if r.mean_score >= best.mean_score - weights.route_margin]
        value = min(near, key=lambda r: r.cost_per_success_usd or float("inf"))
        note = ""
        if (
            value.model_id != best.model_id
            and best.cost_per_success_usd
            and value.cost_per_success_usd
        ):
            saving = 1 - value.cost_per_success_usd / best.cost_per_success_usd
            gap = best.mean_score - value.mean_score
            note = f"value pick saves {saving:.0%} per success for -{gap:.2f} quality"
        table.append(
            RouteRow(
                capability=cap,
                quality_pick=best.model_id,
                value_pick=value.model_id,
                baseline=base,
                note=note,
            )
        )
    return table
