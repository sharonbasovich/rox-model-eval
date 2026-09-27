"""Run artifacts: attempts.jsonl, summary.json, a Markdown scorecard and a
self-contained HTML scorecard (no external assets) with a cost/quality chart."""

from __future__ import annotations

import html
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .aggregate import ModelOverall, ModelSummary, RouteRow
from .history import Regression
from .types import Attempt


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
        "## Overall",
        "",
        "| Model | Verdict | Rox Fitness | Weight covered | $/task (weighted) | worst p95 | Pareto | Why |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for o in report.overall:
        lines.append(
            f"| `{o.model_id}` | **{o.verdict}** | {o.fitness:.3f} | {o.weight_coverage:.0%} "
            f"| {_money(o.cost_per_task_usd)} | {o.p95_latency_s:.2f}s | {'yes' if o.pareto else ''} "
            f"| {'; '.join(o.reasons)} |"
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
            f"| {r.capability} | `{r.quality_pick}` | `{r.value_pick}` | `{r.baseline}` | {r.note} |"
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
            f"## {cap}",
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
        lines += [f"- `{s.model_id}`: {'; '.join(s.reasons)}" for s in rows]
        lines += ["", "**Worst failures (receipts):**", ""]
        fails = _failures(attempts, cap)
        if not fails:
            lines.append("- none")
        for a in fails:
            detail = "; ".join(a.scores.notes[:3]) or "failed"
            lines.append(f"- `{a.model_id}` · `{a.task_id}` rep {a.rep}: {detail}")
        lines.append("")
    return "\n".join(lines)


_VERDICT_COLOR = {"ADOPT": "#15803d", "ROUTE": "#b45309", "HOLD": "#b91c1c", "BASELINE": "#1d4ed8"}


def _heat(v: float) -> str:
    hue = max(0.0, min(1.0, v)) * 120
    return f"hsl({hue:.0f},70%,88%)"


def pareto_svg(overall: list[ModelOverall], width: int = 560, height: int = 340) -> str:
    """Cost (log x) vs Rox Fitness (y); frontier models joined by a line."""
    import math

    if not overall:
        return ""
    pad = 50
    costs = [max(o.cost_per_task_usd, 1e-9) for o in overall]
    lo, hi = math.log10(min(costs)), math.log10(max(costs))
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    fits = [o.fitness for o in overall]
    ylo, yhi = max(0.0, min(fits) - 0.1), min(1.0, max(fits) + 0.1)
    if yhi - ylo < 1e-6:
        ylo, yhi = 0.0, 1.0

    def x(c: float) -> float:
        return pad + (math.log10(max(c, 1e-9)) - lo) / (hi - lo) * (width - 2 * pad)

    def y(f: float) -> float:
        return height - pad - (f - ylo) / (yhi - ylo) * (height - 2 * pad)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif" font-size="11">',
        f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" '
        'fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle">$ per task (log scale) →</text>',
        f'<text x="14" y="{height / 2}" transform="rotate(-90 14 {height / 2})" '
        'text-anchor="middle">Rox Fitness →</text>',
    ]
    for f in (ylo, (ylo + yhi) / 2, yhi):
        parts.append(f'<text x="{pad - 6}" y="{y(f) + 4}" text-anchor="end">{f:.2f}</text>')
    for c in (10**lo, 10**hi):
        parts.append(
            f'<text x="{x(c)}" y="{height - pad + 14}" text-anchor="middle">${c:.4g}</text>'
        )
    frontier = sorted((o for o in overall if o.pareto), key=lambda o: o.cost_per_task_usd)
    if len(frontier) > 1:
        pts = " ".join(f"{x(o.cost_per_task_usd):.1f},{y(o.fitness):.1f}" for o in frontier)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#6366f1" stroke-dasharray="4 3"/>'
        )
    for o in overall:
        color = _VERDICT_COLOR.get(o.verdict, "#334155")
        cx, cy = x(o.cost_per_task_usd), y(o.fitness)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{7 if o.pareto else 5}" fill="{color}"/>'
        )
        parts.append(f'<text x="{cx + 9:.1f}" y="{cy - 6:.1f}">{html.escape(o.model_id)}</text>')
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}
main{max-width:1180px;margin:0 auto;padding:28px}
h1{margin:0 0 4px}h2{margin-top:32px;border-bottom:1px solid #cbd5e1;padding-bottom:4px}
.sub{color:#475569;margin-bottom:18px}.card{background:#fff;border-radius:10px;padding:16px 20px;
box-shadow:0 1px 3px #0002;margin:14px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid #e2e8f0;
text-align:left;vertical-align:top}th{background:#f8fafc;font-weight:600}
.v{font-weight:700;color:#fff;border-radius:4px;padding:2px 6px;font-size:11px}
.warn{background:#fef3c7;border:1px solid #f59e0b;padding:10px 14px;border-radius:8px}
code{background:#f1f5f9;padding:1px 4px;border-radius:3px}.grid{display:flex;gap:18px;flex-wrap:wrap}
details summary{cursor:pointer;color:#334155}
"""


def _badge(v: str) -> str:
    return f'<span class="v" style="background:{_VERDICT_COLOR.get(v, "#334155")}">{v}</span>'


def render_html(report: RunReport, attempts: list[Attempt]) -> str:
    e = html.escape
    caps = sorted({s.capability for s in report.capabilities})
    models = [o.model_id for o in report.overall]
    cell = {(s.model_id, s.capability): s for s in report.capabilities}
    out = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>Rox Model Eval {e(report.run_id)}"
        f"</title><style>{_CSS}</style></head><body><main>",
        "<h1>Rox Model Eval — Scorecard</h1>",
        f"<div class='sub'>run <code>{e(report.run_id)}</code> · suites {len(report.suites)} · "
        f"reps {report.reps} · judge {e(report.judge or 'none')}</div>",
    ]
    if report.simulated:
        out.append(f"<div class='warn'>{e(_SIM_NOTE)}</div>")

    out.append(
        "<h2>Overall recommendation</h2><div class='grid'><div class='card' style='flex:1 1 520px'>"
        "<table><tr><th>Model</th><th>Verdict</th><th>Rox Fitness</th><th>Covered</th>"
        "<th>$/task</th><th>worst p95</th><th>Why</th></tr>"
    )
    for o in report.overall:
        out.append(
            f"<tr><td><code>{e(o.model_id)}</code>{' ★' if o.pareto else ''}</td><td>{_badge(o.verdict)}</td>"
            f"<td>{o.fitness:.3f}</td><td>{o.weight_coverage:.0%}</td><td>{_money(o.cost_per_task_usd)}</td>"
            f"<td>{o.p95_latency_s:.2f}s</td><td>{e('; '.join(o.reasons))}</td></tr>"
        )
    out.append(
        "</table><p class='sub'>★ = on the cost/quality Pareto frontier</p></div>"
        f"<div class='card'>{pareto_svg(report.overall)}</div></div>"
    )

    out.append("<h2>Capability heatmap</h2><div class='card'><table><tr><th>Capability</th>")
    out += [f"<th><code>{e(m)}</code></th>" for m in models]
    out.append("</tr>")
    for cap in caps:
        out.append(f"<tr><td>{e(cap)}</td>")
        for m in models:
            s = cell.get((m, cap))
            if s is None:
                out.append("<td>–</td>")
                continue
            out.append(
                f"<td style='background:{_heat(s.mean_score)}'>{s.mean_score:.2f} {_badge(s.verdict)}"
                f"<br><small>pass {s.pass_rate:.0%} · fab {s.fabrication_rate:.0%} · "
                f"{_money(s.cost_per_success_usd)}/succ</small></td>"
            )
        out.append("</tr>")
    out.append("</table></div>")

    out.append(
        "<h2>Routing table</h2><div class='card'><table><tr><th>Capability</th><th>Best quality</th>"
        "<th>Best value</th><th>Baseline</th><th>Note</th></tr>"
    )
    for r in report.routing:
        out.append(
            f"<tr><td>{e(r.capability)}</td><td><code>{e(str(r.quality_pick))}</code></td>"
            f"<td><code>{e(str(r.value_pick))}</code></td><td><code>{e(str(r.baseline))}</code></td>"
            f"<td>{e(r.note)}</td></tr>"
        )
    out.append("</table></div><h2>Regressions</h2><div class='card'>")
    if not report.regressions:
        out.append("<p>None vs previous runs on unchanged suites.</p>")
    else:
        out.append(
            "<table><tr><th>Model</th><th>Capability</th><th>Metric</th><th>Before</th><th>Now</th>"
            "<th>Prev run</th></tr>"
        )
        for g in report.regressions:
            out.append(
                f"<tr><td><code>{e(g.model_id)}</code></td><td>{e(g.capability)}</td><td>{e(g.metric)}</td>"
                f"<td>{g.previous}</td><td>{g.current}</td><td>{e(g.previous_run)}</td></tr>"
            )
        out.append("</table>")
    out.append("</div>")

    for cap in caps:
        rows = sorted(
            (s for s in report.capabilities if s.capability == cap), key=lambda s: -s.mean_score
        )
        out.append(
            f"<h2>{e(cap)}</h2><div class='card'><table><tr><th>Model</th><th>Verdict</th><th>Score ±sd</th>"
            "<th>Pass</th><th>Consistent</th><th>Format</th><th>Fabrication</th><th>Safety viol.</th>"
            "<th>Judge</th><th>vs baseline</th><th>p50/p95</th><th>$/success</th><th>Why</th></tr>"
        )
        for s in rows:
            out.append(
                f"<tr><td><code>{e(s.model_id)}</code></td><td>{_badge(s.verdict)}</td>"
                f"<td>{s.mean_score:.2f} ±{s.score_stdev:.2f}</td><td>{s.pass_rate:.0%}</td>"
                f"<td>{s.consistency:.0%}</td><td>{s.format_valid_rate:.0%}</td><td>{s.fabrication_rate:.1%}</td>"
                f"<td>{s.safety_violation_rate:.1%}</td><td>{_opt(s.judge_mean)}</td>"
                f"<td>{_opt(s.pairwise_win_rate, '{:.0%}')}</td>"
                f"<td>{s.p50_latency_s:.2f}s / {s.p95_latency_s:.2f}s</td><td>{_money(s.cost_per_success_usd)}</td>"
                f"<td>{e('; '.join(s.reasons))}</td></tr>"
            )
        out.append("</table>")
        metric_rows = [s for s in rows if s.metrics]
        if metric_rows:
            out.append("<details><summary>Scorer metrics</summary><table><tr><th>Model</th>")
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
        out.append(f"<details><summary>Worst failures ({len(fails)})</summary><ul>")
        for a in fails:
            out.append(
                f"<li><code>{e(a.model_id)}</code> · <code>{e(a.task_id)}</code> rep {a.rep}: "
                f"{e('; '.join(a.scores.notes[:3]) or 'failed')}</li>"
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
