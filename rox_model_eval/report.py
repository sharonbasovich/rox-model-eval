"""Writes run artifacts: raw attempts (JSONL), summary (JSON) and a Markdown scorecard."""

from __future__ import annotations

import json
from pathlib import Path

from .aggregate import ModelSummary
from .types import Attempt


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.5f}"


def render_scorecard(summaries: list[ModelSummary], attempts: list[Attempt]) -> str:
    lines = ["# Rox Model Eval — Scorecard", ""]
    for cap in sorted({s.capability for s in summaries}):
        rows = sorted((s for s in summaries if s.capability == cap), key=lambda s: -s.mean_score)
        lines += [
            f"## {cap}",
            "",
            "| Model | Verdict | Score (±sd) | Pass | Schema-valid | Fabrication "
            "| p50 / p95 latency | $/task | $/success |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in rows:
            lines.append(
                f"| `{s.model_id}` | **{s.verdict}** | {s.mean_score:.2f} ±{s.score_stdev:.2f} "
                f"| {s.pass_rate:.0%} | {s.schema_valid_rate:.0%} | {s.fabrication_rate:.1%} "
                f"| {s.p50_latency_s:.2f}s / {s.p95_latency_s:.2f}s "
                f"| {_money(s.cost_per_task_usd)} | {_money(s.cost_per_success_usd)} |"
            )
        lines += ["", "**Why:**", ""]
        lines += [f"- `{s.model_id}`: {'; '.join(s.reasons)}" for s in rows]

        lines += ["", "**Worst failures (receipts):**", ""]
        failures = sorted(
            (a for a in attempts if a.capability == cap and not a.scores.passed),
            key=lambda a: a.scores.score,
        )[:8]
        if not failures:
            lines.append("- none")
        for a in failures:
            detail = "; ".join(a.scores.notes[:3]) or "failed"
            lines.append(f"- `{a.model_id}` · `{a.task_id}` rep {a.rep}: {detail}")
        lines.append("")

    if any(
        a.response.provider_version and "offline-simulator" in a.response.provider_version
        for a in attempts
    ):
        lines += [
            "> Rows from `mock-*` models come from the offline simulator, which exists only to",
            "> exercise the harness. They are not measurements of any real model.",
            "",
        ]
    return "\n".join(lines)


def write_run(out_dir: Path, summaries: list[ModelSummary], attempts: list[Attempt]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "attempts.jsonl").open("w") as fh:
        for a in attempts:
            fh.write(a.model_dump_json() + "\n")
    (out_dir / "summary.json").write_text(json.dumps([s.model_dump() for s in summaries], indent=2))
    scorecard = out_dir / "scorecard.md"
    scorecard.write_text(render_scorecard(summaries, attempts))
    return scorecard
