"""Run artifacts: attempts.jsonl, summary.json, a Markdown scorecard and a
self-contained HTML scorecard (no external assets) meant to be read without a presenter."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from .aggregate import ModelOverall, ModelSummary, RouteRow, label
from .config import Weights
from .history import Regression
from .types import Attempt


class CapabilityInfo(BaseModel):
    name: str
    description: str = ""
    tasks: int = 0


class RunReport(BaseModel):
    run_id: str
    suites: dict[str, str]
    reps: int
    judge: str | None = None
    overall: list[ModelOverall]
    capabilities: list[ModelSummary]
    routing: list[RouteRow]
    regressions: list[Regression] = Field(default_factory=list)
    simulated: bool = False
    capability_info: dict[str, CapabilityInfo] = Field(default_factory=dict)
    weights: Weights | None = None


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.5f}"


def _opt(v: float | None, fmt: str = "{:.2f}") -> str:
    return "–" if v is None else fmt.format(v)


def _failures(attempts: list[Attempt], cap: str, n: int = 8) -> list[Attempt]:
    return sorted(
        (a for a in attempts if a.capability == cap and not a.scores.passed),
        key=lambda a: (not a.scores.safety_violation, a.scores.score),
    )[:n]


_SIM_NOTE = (
    "Rows from `mock-*` models come from the offline simulator, which exists only to "
    "exercise the harness. They are not measurements of any real model."
)


def render_markdown(report: RunReport, attempts: list[Attempt]) -> str:
    lines = [f"# Rox Model Eval — Scorecard `{report.run_id}`", ""]
    if report.simulated:
        lines += [f"> {_SIM_NOTE}", ""]
    lines += [
        f"Suites: {', '.join(report.suites)} · reps: {report.reps} · judge: {report.judge or 'none'}",
        "",
        "## Summary",
        "",
        *(f"- {line}" for line in bottom_line(report)),
        "",
        "### Reusing this on the next model release",
        "",
        *(f"- {line}" for line in reuse_lines(report)),
        "",
        "## Overall",
        "",
        "| Model | Verdict | Rox Fitness | Weight covered | $/task | Time/task | worst p95 "
        "| Pareto | Why |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for o in report.overall:
        lines.append(
            f"| `{o.model_id}` | **{o.verdict}** | {o.fitness:.3f} | {o.weight_coverage:.0%} "
            f"| {_money(o.cost_per_task_usd)} | {o.latency_per_task_s:.1f}s | {o.p95_latency_s:.2f}s "
            f"| {'yes' if o.pareto else ''} "
            f"| {' '.join(o.reasons)} |"
        )
    lines += [
        "",
        "## Routing table",
        "",
        "| Capability | Best quality | Best value | Baseline | Note |",
        "|---|---|---|---|---|",
    ]
    for r in report.routing:
        lines.append(
            f"| {label(r.capability)} | `{r.quality_pick}` | `{r.value_pick}` | `{r.baseline}` | {r.note} |"
        )
    lines += ["", "## Regressions", ""]
    if not report.regressions:
        lines.append("- none vs previous runs on unchanged suites")
    for g in report.regressions:
        lines.append(
            f"- `{g.model_id}` {g.capability} {g.metric}: {g.previous} → {g.current} (prev run {g.previous_run})"
        )
    lines.append("")

    for cap in sorted({s.capability for s in report.capabilities}):
        rows = sorted(
            (s for s in report.capabilities if s.capability == cap), key=lambda s: -s.mean_score
        )
        lines += [
            f"## {label(cap)} (`{cap}`)",
            "",
            "| Model | Verdict | Score (±sd) | Pass | Consistent | Format | Fabrication | Safety viol. "
            "| Judge | vs baseline | p50 / p95 | $/task | $/success |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in rows:
            lines.append(
                f"| `{s.model_id}` | **{s.verdict}** | {s.mean_score:.2f} ±{s.score_stdev:.2f} "
                f"| {s.pass_rate:.0%} | {s.consistency:.0%} | {s.format_valid_rate:.0%} "
                f"| {s.fabrication_rate:.1%} | {s.safety_violation_rate:.1%} | {_opt(s.judge_mean)} "
                f"| {_opt(s.pairwise_win_rate, '{:.0%}')} "
                f"| {s.p50_latency_s:.2f}s / {s.p95_latency_s:.2f}s "
                f"| {_money(s.cost_per_task_usd)} | {_money(s.cost_per_success_usd)} |"
            )
        lines += ["", "**Why:**", ""]
        lines += [f"- `{s.model_id}`: {' '.join(s.reasons)}" for s in rows]
        lines += ["", "**Worst failures (receipts):**", ""]
        fails = _failures(attempts, cap)
        if not fails:
            lines.append("- none")
        for a in fails:
            detail = "; ".join(a.scores.notes[:3]) or "failed"
            lines.append(f"- `{a.model_id}` · `{a.task_id}` rep {a.rep}: {detail}")
        lines.append("")
    return "\n".join(lines)


_VERDICT_COLOR = {
    "ADOPT": "#15803d",
    "ROUTE": "#b45309",
    "HOLD": "#b91c1c",
    "BASELINE": "#1d4ed8",
    "PASS": "#15803d",
    "PARTIAL": "#b45309",
}

VERDICT_MEANING = {
    "PASS": "Clears every quality, fabrication, format and safety gate (overall: on every "
    "capability). Used when no baseline model is configured.",
    "PARTIAL": "Clears every gate on some capabilities but fails at least one gate elsewhere.",
    "BASELINE": "The model marked `baseline: true` in the config as the one currently in use. "
    "ADOPT and ROUTE are relative to it.",
    "ADOPT": "Good enough to replace the baseline as the default: same or better quality, "
    "same or lower cost, and it passes every gate.",
    "ROUTE": "Passes every gate on some capabilities but is not a full replacement. Send it "
    "the capabilities the routing table assigns to it, usually to save cost.",
    "HOLD": "Do not use yet. It fails a quality, fabrication, format or safety gate, or it is "
    "both worse and more expensive than the baseline.",
}

METRIC_MEANING = [
    (
        "Rox Fitness",
        "Overall quality from 0 to 1: the average task score across the capabilities "
        "tested, using the capability weights below.",
    ),
    (
        "Score",
        "Average task score from 0 to 1 for one capability, from deterministic checks "
        "(required facts, citations, JSON schema, tool sequence) blended with the AI judge "
        "where a rubric exists.",
    ),
    ("Pass", "Share of attempts that met the per-task pass bar."),
    ("Consistent", "Share of tasks that got the same pass/fail result on every repetition."),
    ("Format", "Share of outputs that were machine-readable (valid JSON, valid tool arguments)."),
    (
        "Fabrication",
        "Share of output content not supported by the inputs, e.g. invented "
        "numbers, names or records.",
    ),
    (
        "Safety viol.",
        "Share of attempts where the model obeyed an injected instruction, leaked "
        "confidential notes, or took a write action nobody asked for.",
    ),
    ("Judge", "Rubric score from the AI judge (1 to 5), only for open-ended writing tasks."),
    (
        "vs baseline",
        "How often the judge preferred this model's answer over the baseline's in "
        "a blind side-by-side (50% = tie).",
    ),
    ("$/task", "Average API cost of one task. Multi-step agent tasks include every step."),
    ("$/success", "Cost per task divided by the pass rate: what one correct result costs."),
    ("Time/task", "Average wall-clock seconds to finish one task, all steps included."),
    ("p50 / p95", "Median and 95th-percentile time per task: typical and worst-case waits."),
]


def _heat(v: float) -> str:
    hue = max(0.0, min(1.0, v)) * 120
    return f"hsl({hue:.0f},70%,88%)"


def _frontier(overall: list[ModelOverall], xval: Callable[[ModelOverall], float]) -> set[str]:
    """Models no other model beats on both fitness and the x metric (lower x is better)."""
    return {
        o.model_id
        for o in overall
        if not any(
            other is not o
            and other.fitness >= o.fitness
            and xval(other) <= xval(o)
            and (other.fitness > o.fitness or xval(other) < xval(o))
            for other in overall
        )
    }


def scatter_svg(
    overall: list[ModelOverall],
    xval: Callable[[ModelOverall], float],
    x_label: str,
    tick: Callable[[float], str],
    log_x: bool = False,
    width: int = 520,
    height: int = 340,
) -> str:
    """Metric on x (lower is better) vs Rox Fitness on y; the efficient frontier is dashed."""
    if not overall:
        return ""
    pad_l, pad_r, pad_t, pad_b = 56, 110, 20, 50

    def tx(v: float) -> float:
        return math.log10(max(v, 1e-9)) if log_x else v

    xs = [tx(xval(o)) for o in overall]
    lo, hi = min(xs), max(xs)
    span = hi - lo
    if span < 1e-9:
        span = 1.0 if log_x else max(abs(hi), 1.0)
    lo, hi = lo - span * 0.15, hi + span * 0.15
    if not log_x:
        lo = max(0.0, lo)
    fits = [o.fitness for o in overall]
    ylo, yhi = max(0.0, min(fits) - 0.05), min(1.0, max(fits) + 0.03)
    if yhi - ylo < 1e-6:
        ylo, yhi = 0.0, 1.0
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b

    def x(v: float) -> float:
        return pad_l + (tx(v) - lo) / (hi - lo) * iw

    def y(f: float) -> float:
        return pad_t + ih - (f - ylo) / (yhi - ylo) * ih

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        'xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif" font-size="11">',
        f'<rect x="{pad_l}" y="{pad_t}" width="{iw}" height="{ih}" fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<text x="{pad_l + 6}" y="{pad_t + 14}" fill="#15803d" font-weight="600">'
        "◤ better: higher quality, lower " + ("cost" if log_x else "time") + "</text>",
        f'<text x="{pad_l + iw / 2}" y="{height - 10}" text-anchor="middle">'
        f"{html.escape(x_label)} → (lower is better)</text>",
        f'<text x="14" y="{pad_t + ih / 2}" transform="rotate(-90 14 {pad_t + ih / 2})" '
        'text-anchor="middle">Rox Fitness → (higher is better)</text>',
    ]
    for i in range(5):
        f = ylo + (yhi - ylo) * i / 4
        parts.append(
            f'<line x1="{pad_l}" x2="{pad_l + iw}" y1="{y(f):.1f}" y2="{y(f):.1f}" '
            'stroke="#e2e8f0"/>'
            f'<text x="{pad_l - 6}" y="{y(f) + 4:.1f}" text-anchor="end">{f:.2f}</text>'
        )
    for i in range(4):
        t = lo + (hi - lo) * i / 3
        v = 10**t if log_x else t
        parts.append(
            f'<text x="{x(v):.1f}" y="{pad_t + ih + 16}" text-anchor="middle">'
            f"{html.escape(tick(v))}</text>"
        )
    front_ids = _frontier(overall, xval)
    front = sorted((o for o in overall if o.model_id in front_ids), key=xval)
    if len(front) > 1:
        pts = " ".join(f"{x(xval(o)):.1f},{y(o.fitness):.1f}" for o in front)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#6366f1" stroke-dasharray="4 3"/>'
        )
    for o in overall:
        color = _VERDICT_COLOR.get(o.verdict, "#334155")
        cx, cy = x(xval(o)), y(o.fitness)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{color}">'
            f"<title>{html.escape(o.model_id)}: fitness {o.fitness:.3f}, "
            f"{html.escape(tick(xval(o)))}</title></circle>"
            f'<text x="{cx + 10:.1f}" y="{cy - 4:.1f}" font-weight="600">'
            f"{html.escape(o.model_id)}</text>"
            f'<text x="{cx + 10:.1f}" y="{cy + 10:.1f}" fill="#475569">'
            f"{o.fitness:.3f} · {html.escape(tick(xval(o)))}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _money_tick(v: float) -> str:
    return "$" + f"{float(f'{v:.2g}'):f}".rstrip("0").rstrip(".")


def cost_svg(overall: list[ModelOverall]) -> str:
    return scatter_svg(
        overall, lambda o: o.cost_per_task_usd, "Cost per task, USD (log scale)", _money_tick, True
    )


def time_svg(overall: list[ModelOverall]) -> str:
    return scatter_svg(
        overall, lambda o: o.latency_per_task_s, "Average time per task, seconds", "{:.1f}s".format
    )


def pareto_svg(overall: list[ModelOverall]) -> str:
    return cost_svg(overall)


def _pct_change(new: float, old: float) -> str:
    if not old:
        return "n/a"
    d = new / old - 1
    return f"{abs(d):.0%} {'lower' if d < 0 else 'higher'}"


def _plan(report: RunReport, fallback: str) -> tuple[float, float, float] | None:
    """Weighted ($/task, fitness, s/task) if each capability uses its routing pick."""
    weights = report.weights.capabilities if report.weights else {}
    cell = {(s.model_id, s.capability): s for s in report.capabilities}
    cost = fit = secs = wsum = 0.0
    for r in report.routing:
        pick = cell.get((r.value_pick or fallback, r.capability))
        if pick is None:
            continue
        w = weights.get(r.capability, 0.0) or 0.01
        wsum += w
        cost += w * pick.cost_per_task_usd
        fit += w * pick.mean_score
        secs += w * pick.mean_latency_s
    return (cost / wsum, fit / wsum, secs / wsum) if wsum else None


def bottom_line(report: RunReport) -> list[str]:
    """Plain-English summary sentences that stand on their own."""
    lines: list[str] = []
    if not report.overall:
        return lines
    base = next((o for o in report.overall if o.baseline), None)
    top = max(report.overall, key=lambda o: o.fitness)
    ref = base or top
    candidates = [o for o in report.overall if not o.baseline]
    adopt = [o for o in candidates if o.verdict == "ADOPT"]
    routed: dict[str, list[str]] = {}
    for r in report.routing:
        if r.value_pick:
            routed.setdefault(r.value_pick, []).append(r.capability)

    if base is None:
        lines.append(
            f"Highest quality: {top.model_id} (Rox Fitness {top.fitness:.3f}, "
            f"{_money(top.cost_per_task_usd)} and {top.latency_per_task_s:.1f}s per task)."
        )
        for o in report.overall:
            if o is top:
                continue
            cost = o.cost_per_task_usd / top.cost_per_task_usd if top.cost_per_task_usd else 1
            secs = o.latency_per_task_s / top.latency_per_task_s if top.latency_per_task_s else 1
            lines.append(
                f"{o.model_id}: Rox Fitness {o.fitness:.3f} ({o.fitness / top.fitness:.1%} of "
                f"{top.model_id}'s) at {cost:.0%} of the cost and {secs:.0%} of the time per task."
            )
        if len(routed) > 1:
            parts = [f"{m} for {', '.join(label(c) for c in cs)}" for m, cs in routed.items()]
            lines.append(
                "Per capability, the cheapest model that clears every gate and is close to the "
                f"best quality is: {'; '.join(parts)}."
            )
    elif adopt:
        best = max(adopt, key=lambda o: o.fitness)
        lines.append(
            f"Recommendation: switch from {base.model_id} to {best.model_id}. It matches or "
            f"beats the baseline's quality (Rox Fitness {best.fitness:.3f} vs {base.fitness:.3f}) "
            f"at {_pct_change(best.cost_per_task_usd, base.cost_per_task_usd)} cost per task, "
            "and passes every quality and safety gate."
        )
    else:
        moved = {m: cs for m, cs in routed.items() if m != base.model_id}
        if moved:
            parts = [f"send {', '.join(label(c) for c in cs)} to {m}" for m, cs in moved.items()]
            n = sum(len(cs) for cs in moved.values())
            lines.append(
                f"Recommendation: keep {base.model_id} as the default, and {'; '.join(parts)} "
                f"({n} of {len(report.routing)} capabilities)."
            )
        else:
            lines.append(
                f"Recommendation: keep {base.model_id}. No candidate earns traffic on any "
                "capability yet."
            )

    plan = _plan(report, ref.model_id)
    if plan and len(routed) > 1:
        cost, fit, secs = plan
        who = f"{ref.model_id} {'alone' if base else 'for everything'}"
        lines.append(
            f"On this benchmark that split costs {_money(cost)} per task vs "
            f"{_money(ref.cost_per_task_usd)} for {who} "
            f"({_pct_change(cost, ref.cost_per_task_usd)}), with Rox Fitness {fit:.3f} vs "
            f"{ref.fitness:.3f} and {secs:.1f}s vs {ref.latency_per_task_s:.1f}s per task."
        )
    if base is not None:
        for o in candidates:
            lines.append(f"{o.model_id} ({o.verdict}): {' '.join(o.reasons)}")

    unsafe = sorted(
        {(s.model_id, label(s.capability)) for s in report.capabilities if s.safety_violation_rate}
    )
    if unsafe:
        lines.append(
            "Safety: "
            + "; ".join(f"{m} had violations on {c}" for m, c in unsafe)
            + ". Any violation is an automatic HOLD for that capability."
        )
    else:
        lines.append(
            "Safety: no model obeyed a prompt injection, leaked confidential notes or took an "
            "unrequested action."
        )
    if report.judge and report.judge in {o.model_id for o in report.overall}:
        lines.append(
            f"Judge bias: the AI judge ({report.judge}) is also one of the models tested, so the "
            "judge-scored part of writing tasks may favour it."
        )
    if report.reps < 3:
        lines.append(
            f"Confidence: each task ran {report.reps} time{'s' if report.reps != 1 else ''}. "
            "Treat differences smaller than about 0.02 in fitness as noise until a "
            "3-repetition run confirms them."
        )
    return lines


def reuse_lines(report: RunReport) -> list[str]:
    """How to rerun this benchmark on the next model release, with this run's measured cost."""
    ids = [o.model_id for o in report.overall]
    n_tasks = sum(i.tasks for i in report.capability_info.values())
    per_model: dict[str, tuple[float, float]] = {}
    for s in report.capabilities:
        cost, secs = per_model.get(s.model_id, (0.0, 0.0))
        per_model[s.model_id] = (
            cost + s.cost_per_task_usd * s.attempts,
            secs + s.mean_latency_s * s.attempts,
        )
    lines = [
        "Add the new model to config/models.yaml: one entry with provider, model name and "
        "per-token prices. OpenAI (Chat Completions or Responses), Anthropic and any "
        "OpenAI-compatible endpoint work without code changes.",
        f"Run one command: python -m rox_model_eval run --models <new-model>,{','.join(ids)} "
        "--suite all --reps 3. The same tasks, scorers and gates run unchanged, so results are "
        "directly comparable with earlier releases.",
    ]
    if per_model:
        costs = [c for c, _ in per_model.values()]
        mins = [t / 60 for _, t in per_model.values()]
        tasks = f"all {n_tasks} tasks" if n_tasks else "every task"

        def span(lo: str, hi: str) -> str:
            return lo if lo == hi else f"{lo}-{hi}"

        lines.append(
            f"Cost and time: one pass over {tasks} cost "
            f"{span(f'${min(costs):.2f}', f'${max(costs):.2f}')} per model in this run and took "
            f"about {span(f'{min(mins):.0f}', f'{max(mins):.0f}')} minutes of model time per "
            "model. A 3-repetition run costs about three times that, scaled by the new model's "
            "per-token prices."
        )
    lines.append(
        "Every run is saved to run history. The next scorecard flags any capability where a "
        "model scored lower than on its previous run of identical tasks, which also catches "
        "silent provider-side model updates."
    )
    return lines


_CSS = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f1f5f9;color:#0f172a;
line-height:1.45}
main{max-width:1180px;margin:0 auto;padding:28px}
h1{margin:0 0 4px}h2{margin-top:36px;border-bottom:1px solid #cbd5e1;padding-bottom:4px}
h3{margin:4px 0 8px}.sub{color:#475569;margin-bottom:14px}.lede{color:#334155;margin:4px 0 10px}
.card{background:#fff;border-radius:10px;padding:16px 20px;
box-shadow:0 1px 3px #0002;margin:14px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;
border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}
th{background:#f8fafc;font-weight:600}th[title]{cursor:help;text-decoration:underline dotted}
.v{font-weight:700;color:#fff;border-radius:4px;padding:2px 6px;font-size:11px}
.warn{background:#fef3c7;border:1px solid #f59e0b;padding:10px 14px;border-radius:8px;margin:10px 0}
.bottom{border-left:5px solid #1d4ed8}.bottom li{margin:6px 0}
code{background:#f1f5f9;padding:1px 4px;border-radius:3px}.grid{display:flex;gap:18px;flex-wrap:wrap}
.grid>.card{flex:1 1 440px}.tiles{display:flex;gap:14px;flex-wrap:wrap}
.tile{flex:1 1 260px;background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px #0002}
.tile .nums{display:flex;gap:16px;margin:8px 0}.tile .nums b{display:block;font-size:20px}
.tile .nums span{color:#475569;font-size:12px}.tile p{margin:6px 0 0;font-size:13px;color:#334155}
details summary{cursor:pointer;color:#334155}dl{margin:0}dt{font-weight:600;margin-top:8px}
dd{margin:2px 0 0 0;color:#334155}
"""


def _badge(v: str) -> str:
    return f'<span class="v" style="background:{_VERDICT_COLOR.get(v, "#334155")}">{v}</span>'


def _th(name: str) -> str:
    meaning = dict(METRIC_MEANING).get(name)
    title = f' title="{html.escape(meaning)}"' if meaning else ""
    return f"<th{title}>{html.escape(name)}</th>"


def _glossary(report: RunReport) -> str:
    e = html.escape
    out = [
        "<h2>How to read this scorecard</h2><div class='card'><div class='grid'>",
        "<div style='flex:1 1 420px'><h3>Verdicts</h3><dl>",
    ]
    used = {o.verdict for o in report.overall} | {s.verdict for s in report.capabilities}
    for v, meaning in VERDICT_MEANING.items():
        if v in used:
            out.append(f"<dt>{_badge(v)}</dt><dd>{e(meaning)}</dd>")
    out.append("</dl>")
    w = report.weights
    if w is not None:
        out.append(
            "<h3 style='margin-top:14px'>Gates (pass/fail, checked per capability)</h3><ul>"
            f"<li>Quality score at least {w.quality_gate:.2f}</li>"
            f"<li>Fabrication at most {w.fabrication_gate:.0%} of output</li>"
            f"<li>At least {w.format_gate:.0%} of outputs machine-readable</li>"
            f"<li>At least {w.pass_rate_gate:.0%} of attempts pass every task-specific check</li>"
            f"<li>Safety violations at most {w.safety_gate:.0%} (zero tolerance)</li></ul>"
            "<p class='lede'>A model that fails a gate is HOLD for that capability no matter how "
            f"cheap or fast it is. A cheaper model within {w.route_margin:.2f} of the best "
            "quality becomes the routing pick.</p>"
        )
    out.append("</div><div style='flex:1 1 420px'><h3>Metrics</h3><dl>")
    out += [f"<dt>{e(n)}</dt><dd>{e(m)}</dd>" for n, m in METRIC_MEANING]
    out.append("</dl></div></div>")
    if w is not None:
        out.append(
            "<h3 style='margin-top:14px'>Capabilities and weights</h3>"
            + (
                "<p class='lede'>Weights are equal because no real usage mix is known. Set them "
                "in <code>config/weights.yaml</code> to reflect actual traffic.</p>"
                if len({w.capabilities.get(c, 0) for c in report.suites}) == 1
                else ""
            )
            + "<table><tr><th>Capability</th><th>Weight</th><th>What it tests</th></tr>"
        )
        for cap in sorted(report.suites):
            info = report.capability_info.get(cap)
            out.append(
                f"<tr><td>{e(label(cap))}</td><td>{w.capabilities.get(cap, 0):.1%}</td>"
                f"<td>{e(info.description if info else '')}</td></tr>"
            )
        out.append("</table>")
    out.append("</div>")
    return "".join(out)


_REGRESSION_METRIC = {
    "mean_score": "quality score dropped",
    "fabrication_rate": "fabrication rose",
    "safety_violation_rate": "new safety violations",
    "p95_latency_s": "p95 time per task rose",
}


def render_html(report: RunReport, attempts: list[Attempt]) -> str:
    e = html.escape
    caps = sorted({s.capability for s in report.capabilities})
    models = [o.model_id for o in report.overall]
    cell = {(s.model_id, s.capability): s for s in report.capabilities}
    n_tasks = sum(i.tasks for i in report.capability_info.values())
    scope = (
        f"{len(models)} models · {n_tasks or '?'} Rox-style tasks (synthetic data) across {len(report.suites)} "
        f"capabilities · {report.reps} repetition{'s' if report.reps != 1 else ''} per task · "
        f"AI judge: {report.judge or 'none'}"
    )
    out = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>Rox Model Eval {e(report.run_id)}"
        f"</title><style>{_CSS}</style></head><body><main>",
        "<h1>How frontier models handle Rox workflows</h1>",
        "<div class='lede'>Each model runs the same test tasks, modeled on Rox's product "
        "workflows: account research, email drafting, deal and lead prioritisation, CRM "
        "Q&amp;A, agent tool use, record extraction, long call transcripts and prompt-injection "
        "attacks. Answers are scored automatically and compared on quality, cost, speed and "
        "safety.</div>"
        "<div class='warn'>Where the tasks come from: each capability mirrors a Rox product "
        "surface observed while exploring Rox's web app (chat agent, insights, company and "
        "people enrichment, deals, campaigns, CSV upload). The task data is synthetic, and "
        "capability weights are equal because Rox's real usage mix isn't known. Results show "
        "how models handle Rox-style workflows; they are not measurements of Rox's production "
        "system.</div>",
        f"<div class='sub'>{e(scope)} · run <code>{e(report.run_id)}</code></div>",
    ]
    if report.simulated:
        out.append(f"<div class='warn'>{e(_SIM_NOTE)}</div>")

    out.append("<div class='card bottom'><h3>Summary</h3><ul>")
    out += [f"<li>{e(line)}</li>" for line in bottom_line(report)]
    out.append("</ul><h3>Reusing this on the next model release</h3><ul>")
    out += [f"<li>{e(line)}</li>" for line in reuse_lines(report)]
    out.append("</ul></div><div class='tiles'>")
    for o in report.overall:
        out.append(
            f"<div class='tile'><div><code>{e(o.model_id)}</code> {_badge(o.verdict)}</div>"
            f"<div class='nums'><div><b>{o.fitness:.3f}</b><span>Rox Fitness</span></div>"
            f"<div><b>{_money(o.cost_per_task_usd)}</b><span>per task</span></div>"
            f"<div><b>{o.latency_per_task_s:.1f}s</b><span>per task</span></div></div>"
            f"<p>{e(' '.join(o.reasons))}</p></div>"
        )
    out.append("</div>")

    out.append(
        "<h2>Quality vs cost and speed</h2><p class='lede'>Each dot is a model, coloured by "
        "verdict. The best models sit in the top-left: high Rox Fitness for little money or "
        "time. The dashed line joins models that no other model beats on both axes.</p>"
        "<div class='grid'>"
        f"<div class='card'><h3>Rox Fitness vs cost per task</h3>{cost_svg(report.overall)}</div>"
        f"<div class='card'><h3>Rox Fitness vs time per task</h3>{time_svg(report.overall)}</div>"
        "</div>"
    )

    out.append(
        "<h2>Where each model is good enough</h2><p class='lede'>Quality score per capability "
        "(green = high) with that capability's verdict. Weight = its weight in Rox Fitness.</p>"
        "<div class='card'><table><tr><th>Capability</th><th>Weight</th>"
    )
    wts = report.weights.capabilities if report.weights else {}
    out += [f"<th><code>{e(m)}</code></th>" for m in models]
    out.append("</tr>")
    for cap in caps:
        out.append(f"<tr><td>{e(label(cap))}</td><td>{wts.get(cap, 0):.1%}</td>")
        for m in models:
            s = cell.get((m, cap))
            if s is None:
                out.append("<td>–</td>")
                continue
            out.append(
                f"<td style='background:{_heat(s.mean_score)}'>{s.mean_score:.2f} {_badge(s.verdict)}"
                f"<br><small>pass {s.pass_rate:.0%} · {_money(s.cost_per_task_usd)}/task · "
                f"{s.mean_latency_s:.1f}s</small></td>"
            )
        out.append("</tr>")
    out.append("</table></div>")

    out.append(
        "<h2>Routing plan</h2><p class='lede'>Which model to send each capability to. "
        "<b>Use</b> is the cheapest model that passes every gate and is close enough to the best "
        "quality; <b>Best quality</b> is the top scorer if cost did not matter.</p>"
        "<div class='card'><table><tr><th>Capability</th><th>Use</th><th>Best quality</th>"
        "<th>Why</th></tr>"
    )
    for r in report.routing:
        changed = bool(r.value_pick and r.baseline and r.value_pick != r.baseline)
        out.append(
            f"<tr><td>{e(label(r.capability))}</td><td><code>{e(str(r.value_pick))}</code>"
            f"{' ← change' if changed else ''}</td><td><code>{e(str(r.quality_pick))}</code></td>"
            f"<td>{e(r.note)}</td></tr>"
        )
    out.append(
        "</table></div><h2>Regressions</h2><p class='lede'>Compared with the previous run of the "
        "same model on identical tasks.</p><div class='card'>"
    )
    if not report.regressions:
        out.append("<p>None: no model got worse since its last run on unchanged tasks.</p>")
    else:
        out.append(
            "<table><tr><th>Model</th><th>Capability</th><th>What changed</th><th>Before</th>"
            "<th>Now</th><th>Previous run</th></tr>"
        )
        for g in report.regressions:
            out.append(
                f"<tr><td><code>{e(g.model_id)}</code></td><td>{e(label(g.capability))}</td>"
                f"<td>{e(_REGRESSION_METRIC.get(g.metric, g.metric))}</td>"
                f"<td>{g.previous:.3g}</td><td>{g.current:.3g}</td><td>{e(g.previous_run)}</td></tr>"
            )
        out.append("</table>")
        if report.reps < 3:
            out.append(
                "<p class='sub'>With fewer than 3 repetitions, one unlucky answer can trigger a "
                "flag; confirm with a repeat run.</p>"
            )
    out.append("</div>")

    out.append(_glossary(report))

    out.append(
        "<h2>Capability details</h2><p class='lede'>Full metrics per capability, with the "
        "worst failing answers as receipts. Hover a column header for its definition.</p>"
    )
    for cap in caps:
        rows = sorted(
            (s for s in report.capabilities if s.capability == cap), key=lambda s: -s.mean_score
        )
        info = report.capability_info.get(cap)
        out.append(
            f"<div class='card'><h3>{e(label(cap))} <small><code>{e(cap)}</code></small></h3>"
            + (f"<p class='lede'>{e(info.description)}</p>" if info else "")
            + "<table><tr><th>Model</th><th>Verdict</th>"
            + "".join(
                _th(n)
                for n in (
                    "Score",
                    "Pass",
                    "Consistent",
                    "Format",
                    "Fabrication",
                    "Safety viol.",
                    "Judge",
                    "vs baseline",
                    "Time/task",
                    "p50 / p95",
                    "$/task",
                    "$/success",
                )
            )
            + "<th>Why</th></tr>"
        )
        for s in rows:
            out.append(
                f"<tr><td><code>{e(s.model_id)}</code></td><td>{_badge(s.verdict)}</td>"
                f"<td>{s.mean_score:.2f} ±{s.score_stdev:.2f}</td><td>{s.pass_rate:.0%}</td>"
                f"<td>{s.consistency:.0%}</td><td>{s.format_valid_rate:.0%}</td>"
                f"<td>{s.fabrication_rate:.1%}</td><td>{s.safety_violation_rate:.1%}</td>"
                f"<td>{_opt(s.judge_mean)}</td><td>{_opt(s.pairwise_win_rate, '{:.0%}')}</td>"
                f"<td>{s.mean_latency_s:.1f}s</td>"
                f"<td>{s.p50_latency_s:.1f}s / {s.p95_latency_s:.1f}s</td>"
                f"<td>{_money(s.cost_per_task_usd)}</td><td>{_money(s.cost_per_success_usd)}</td>"
                f"<td>{e(' '.join(s.reasons))}</td></tr>"
            )
        out.append("</table>")
        metric_rows = [s for s in rows if s.metrics]
        if metric_rows:
            out.append("<details><summary>Scorer sub-metrics</summary><table><tr><th>Model</th>")
            keys = sorted({k for s in metric_rows for k in s.metrics})
            out += [f"<th>{e(k)}</th>" for k in keys]
            out.append("</tr>")
            for s in metric_rows:
                out.append(
                    f"<tr><td><code>{e(s.model_id)}</code></td>"
                    + "".join(f"<td>{s.metrics.get(k, 0):.2f}</td>" for k in keys)
                    + "</tr>"
                )
            out.append("</table></details>")
        fails = _failures(attempts, cap)
        out.append(f"<details><summary>Worst failing answers ({len(fails)})</summary><ul>")
        for a in fails:
            out.append(
                f"<li><code>{e(a.model_id)}</code> · task <code>{e(a.task_id)}</code> "
                f"rep {a.rep}: {e('; '.join(a.scores.notes[:3]) or 'failed')}</li>"
            )
        out.append("</ul></details></div>")
    out.append("</main></body></html>")
    return "".join(out)


def write_run(out_dir: Path, report: RunReport, attempts: list[Attempt]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "attempts.jsonl").open("w") as fh:
        for a in attempts:
            fh.write(a.model_dump_json() + "\n")
    (out_dir / "summary.json").write_text(json.dumps(report.model_dump(), indent=2))
    (out_dir / "scorecard.md").write_text(render_markdown(report, attempts))
    html_path = out_dir / "scorecard.html"
    html_path.write_text(render_html(report, attempts))
    return html_path
