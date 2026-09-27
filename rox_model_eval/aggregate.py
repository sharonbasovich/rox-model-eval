"""Turns raw attempts into per-capability metrics, verdicts, a composite fitness score,
a routing table and a cost/quality Pareto frontier."""

from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel, Field

from .config import ModelSpec, Weights
from .types import Attempt

CAPABILITY_LABELS = {
    "c1_research": "Account research briefs",
    "c2_drafting": "Email drafting",
    "c3_insights": "Deal & lead prioritisation",
    "c4_grounded_qa": "CRM Q&A (grounded)",
    "c5_tool_calling": "Agent tool use",
    "c6_extraction": "Record extraction",
    "c7_long_context": "Long call transcripts",
    "c8_safety": "Prompt-injection safety",
}


def label(capability: str) -> str:
    return CAPABILITY_LABELS.get(capability, capability)


def _cheaper_phrase(ratio: float) -> str:
    """`ratio` = candidate cost / baseline cost."""
    if ratio <= 1:
        return f"{1 - ratio:.0%} cheaper"
    return f"{ratio:.1f}x the cost"


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
    mean_latency_s: float = 0.0
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
    latency_per_task_s: float = 0.0
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
                mean_latency_s=statistics.fmean(latencies) if latencies else 0.0,
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
        out.append(f"quality {s.mean_score:.2f} is below the {w.quality_gate:.2f} minimum")
    if s.fabrication_rate > w.fabrication_gate:
        out.append(
            f"invents unsupported facts in {s.fabrication_rate:.1%} of output "
            f"(limit {w.fabrication_gate:.0%})"
        )
    if s.format_valid_rate < w.format_gate:
        out.append(
            f"only {s.format_valid_rate:.0%} of outputs are machine-readable "
            f"(needs {w.format_gate:.0%})"
        )
    if s.safety_violation_rate > w.safety_gate:
        out.append(
            f"safety: obeyed an injected instruction or took a forbidden action in "
            f"{s.safety_violation_rate:.0%} of attempts (limit {w.safety_gate:.0%})"
        )
    return out


def _versus_best(s: ModelSummary, rows: list[ModelSummary]) -> str:
    """Head-to-head sentence for a gate-passing model when no baseline is set."""
    best = max(rows, key=lambda r: r.mean_score)
    if best is s or best.mean_score <= s.mean_score:
        return f"Clears every gate with the top quality score here ({s.mean_score:.2f})."
    cost = ""
    if s.cost_per_success_usd and best.cost_per_success_usd:
        cost = (
            f", {_cheaper_phrase(s.cost_per_success_usd / best.cost_per_success_usd)} "
            "per successful task"
        )
    return (
        f"Clears every gate; quality {s.mean_score:.2f} vs {best.mean_score:.2f} for the top "
        f"scorer ({best.model_id}){cost}."
    )


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
                s.reasons = [
                    "Marked as the baseline in config; the other models are measured against it.",
                    *(f"Fails a gate: {r}." for r in failures),
                ]
                continue
            if failures:
                s.verdict = "HOLD"
                s.reasons = [f"Fails a gate: {r}." for r in failures]
                continue
            if base is None:
                s.verdict, s.reasons = "PASS", [_versus_best(s, rows)]
                continue
            gap = s.mean_score - base.mean_score
            vs = f"quality {s.mean_score:.2f} vs baseline {base.mean_score:.2f}"
            ratio = (
                s.cost_per_success_usd / base.cost_per_success_usd
                if s.cost_per_success_usd is not None and base.cost_per_success_usd
                else None
            )
            cost = f"{_cheaper_phrase(ratio)} per successful task" if ratio is not None else ""
            cheaper = ratio is not None and ratio <= 1
            as_good = gap >= -0.01
            if as_good and cheaper:
                s.verdict = "ADOPT"
                s.reasons = [f"As good as the baseline ({vs}) and {cost}."]
            elif as_good:
                s.verdict = "ROUTE"
                s.reasons = [f"As good as the baseline ({vs}) but {cost}."]
            elif cheaper:
                s.verdict = "ROUTE"
                s.reasons = [
                    f"Clears every gate and is {cost}, but trails the baseline "
                    f"({vs}); suited to high-volume or lower-stakes traffic."
                ]
            else:
                s.verdict = "HOLD"
                s.reasons = [f"Worse than the baseline ({vs}) and {cost}."]
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
                latency_per_task_s=round(sum(w * r.mean_latency_s for w, r in ws) / wsum, 3),
                p95_latency_s=max(r.p95_latency_s for r in rows),
                capability_verdicts={r.capability: r.verdict for r in rows},
            )
        )

    base = next((r for r in results if r.baseline), None)
    top = max(results, key=lambda r: r.fitness, default=None)
    for o in results:
        rows = by_model[o.model_id]
        unsafe = [
            label(r.capability) for r in rows if r.safety_violation_rate > weights.safety_gate
        ]
        if o.baseline:
            o.verdict = "BASELINE"
            o.reasons = [
                "Marked as the baseline in config; the other models are measured against it."
            ]
            if unsafe:
                o.reasons.append(f"It fails the safety gate on {', '.join(unsafe)}.")
            continue
        if unsafe:
            o.verdict = "HOLD"
            o.reasons = [
                f"Fails the safety gate on {', '.join(unsafe)}: it obeyed an injected "
                "instruction or took a forbidden action, which blocks adoption regardless "
                "of quality or cost."
            ]
            continue
        verdicts = o.capability_verdicts
        if base is None and top is not None:
            passed = [c for c, v in verdicts.items() if v == "PASS"]
            failed = sorted(label(c) for c, v in verdicts.items() if v != "PASS")
            o.verdict = "PASS" if not failed else ("PARTIAL" if passed else "HOLD")
            gate = (
                f"Clears every gate on all {len(verdicts)} capabilities."
                if not failed
                else f"Fails a gate on {', '.join(failed)}."
            )
            if o is top:
                o.reasons = [gate, f"Highest Rox Fitness in this run ({o.fitness:.3f})."]
            else:
                cost = o.cost_per_task_usd / top.cost_per_task_usd if top.cost_per_task_usd else 1
                speed = (
                    o.latency_per_task_s / top.latency_per_task_s if top.latency_per_task_s else 1
                )
                faster = f"{1 - speed:.0%} faster" if speed <= 1 else f"{speed - 1:.0%} slower"
                o.reasons = [
                    gate,
                    f"Rox Fitness {o.fitness:.3f} vs {top.fitness:.3f} for {top.model_id}, "
                    f"{_cheaper_phrase(cost)} and {faster} per task.",
                ]
            continue
        wins = sorted(c for c, v in verdicts.items() if v in {"ADOPT", "ROUTE"})
        holds = sorted(c for c, v in verdicts.items() if v == "HOLD")
        heavy_holds = [c for c in holds if weights.capabilities.get(c, 0.0) >= 0.15]
        compare = ""
        beats = True
        if base is not None:
            ratio = o.cost_per_task_usd / base.cost_per_task_usd if base.cost_per_task_usd else 1
            compare = (
                f"Rox Fitness {o.fitness:.3f} vs baseline {base.fitness:.3f}, "
                f"{_cheaper_phrase(ratio)} per task"
            )
            beats = (
                o.fitness >= base.fitness - 0.01 and o.cost_per_task_usd <= base.cost_per_task_usd
            )
        if all(v == "ADOPT" for v in verdicts.values()) or (beats and not heavy_holds):
            o.verdict = "ADOPT"
            o.reasons = [
                f"Can replace the baseline: {compare or f'Rox Fitness {o.fitness:.3f}'}, "
                "and it clears every quality and safety gate."
            ]
        elif wins:
            o.verdict = "ROUTE"
            scope = (
                f"all {len(wins)} capabilities"
                if not holds
                else f"{len(wins)} of {len(verdicts)} capabilities"
            )
            lead = f"Passes every gate on {scope}"
            if not holds and base is not None:
                o.reasons = [
                    f"{lead} and is {compare.split(', ', 1)[1]}, but its Rox Fitness is "
                    f"{o.fitness:.3f} vs {base.fitness:.3f} for the baseline, outside the 0.01 "
                    "that counts as a tie. Route the capabilities the routing plan assigns to "
                    "it rather than replacing the baseline everywhere."
                ]
            else:
                o.reasons = [f"{lead} ({', '.join(label(c) for c in wins)}). {compare}."]
            if heavy_holds:
                o.reasons.append(
                    "Keep the baseline for high-traffic "
                    f"{', '.join(label(c) for c in heavy_holds)}."
                )
        else:
            o.verdict = "HOLD"
            o.reasons = ["No capability clears the gates against the baseline."]

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
        if value.model_id != best.model_id:
            gap = best.mean_score - value.mean_score
            saving = (
                f" and {1 - value.cost_per_success_usd / best.cost_per_success_usd:.0%} "
                "cheaper per successful task"
                if best.cost_per_success_usd and value.cost_per_success_usd
                else ""
            )
            note = (
                f"{value.model_id} is within {gap:.2f} of the best quality "
                f"(margin {weights.route_margin:.2f}){saving}."
            )
        elif len(ok) == 1:
            note = f"Only {value.model_id} clears every gate."
        else:
            cheaper = [
                r
                for r in ok
                if (r.cost_per_success_usd or float("inf")) < (value.cost_per_success_usd or 0)
            ]
            if not cheaper:
                note = f"{value.model_id} has the best quality and is also the cheapest."
            else:
                alt = min(cheaper, key=lambda r: r.cost_per_success_usd or float("inf"))
                note = (
                    f"{value.model_id} has the best quality; the cheaper {alt.model_id} trails "
                    f"by {value.mean_score - alt.mean_score:.2f}, more than the "
                    f"{weights.route_margin:.2f} margin."
                )
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
