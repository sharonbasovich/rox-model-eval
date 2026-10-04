"""Command line entry point.

rox-eval run --models a,b --suite all [--judge m] [--reps 3]
rox-eval calibrate --judge m
rox-eval history
rox-eval list
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from . import history
from .adapters import build_adapter
from .aggregate import overall, recommend, routing_table, summarize
from .config import ModelSpec, load_models, load_weights
from .judge import calibrate, load_labels
from .report import CapabilityInfo, RunReport, write_run
from .runner import pairwise_vs_baseline, run_suite
from .suites import ALL_SUITES, load_suite, resolve_suites, suite_hash
from .types import Attempt


def _progress(a: Attempt) -> None:
    mark = "." if a.scores.passed else ("E" if a.response.error else "x")
    if a.scores.safety_violation:
        mark = "!"
    print(mark, end="", flush=True)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models-file", default="config/models.yaml")
    p.add_argument("--weights-file", default="config/weights.yaml")
    p.add_argument("--suites-dir", default="suites")


def _pick(specs: dict[str, ModelSpec], ids: list[str]) -> list[ModelSpec] | None:
    unknown = [m for m in ids if m not in specs]
    if unknown:
        print(f"unknown model id(s): {', '.join(unknown)}", file=sys.stderr)
        return None
    return [specs[m] for m in ids]


def _with_effort(spec: ModelSpec, effort: str) -> ModelSpec:
    if spec.adapter != "openai_responses":
        return spec
    return spec.model_copy(update={"params": {**spec.params, "reasoning": {"effort": effort}}})


def cmd_run(args: argparse.Namespace) -> int:
    specs = load_models(args.models_file)
    chosen = _pick(specs, [m.strip() for m in args.models.split(",") if m.strip()])
    judge = _pick(specs, [args.judge]) if args.judge else []
    if chosen is None or judge is None:
        return 2
    if args.reasoning_effort:
        chosen = [_with_effort(s, args.reasoning_effort) for s in chosen]
        judge = [_with_effort(s, args.reasoning_effort) for s in judge]
        specs = {**specs, **{s.id: s for s in chosen + judge}}
    judge_spec = judge[0] if judge else None
    weights = load_weights(args.weights_file)
    names = resolve_suites(args.suite)

    stamp = datetime.now(UTC)
    run_id = f"{stamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:4]}"
    attempts: list[Attempt] = []
    pairwise: dict[str, dict[str, float]] = {}
    hashes: dict[str, str] = {}
    info: dict[str, CapabilityInfo] = {}
    baseline = next((s.id for s in chosen if s.baseline), None)
    for name in names:
        suite = load_suite(args.suites_dir, name)
        hashes[suite.capability] = suite_hash(args.suites_dir, name)
        info[suite.capability] = CapabilityInfo(
            name=suite.name, description=" ".join(suite.description.split()), tasks=len(suite.tasks)
        )
        print(
            f"{suite.capability}: {len(suite.tasks)} tasks x {args.reps} reps "
            f"x {len(chosen)} models ",
            end="",
        )
        rows = run_suite(chosen, suite, args.reps, _progress, judge_spec, args.workers)
        print()
        attempts += rows
        if judge_spec and baseline and suite.judge_rubric and not args.no_pairwise:
            pairwise[suite.capability] = pairwise_vs_baseline(judge_spec, suite, rows, baseline)

    summaries = recommend(summarize(attempts, specs), weights)
    for s in summaries:
        s.pairwise_win_rate = pairwise.get(s.capability, {}).get(s.model_id)
    db = Path(args.history)
    regressions = [] if args.no_history else history.detect(db, run_id, summaries, hashes, weights)
    if not args.no_history:
        history.record(db, run_id, stamp.isoformat(), summaries, hashes)

    report = RunReport(
        run_id=run_id,
        suites=hashes,
        reps=args.reps,
        judge=judge_spec.id if judge_spec else None,
        overall=overall(summaries, specs, weights),
        capabilities=summaries,
        routing=routing_table(summaries, weights),
        regressions=regressions,
        simulated=any(specs[a.model_id].adapter == "mock" for a in attempts),
        capability_info=info,
        weights=weights,
    )
    label = names[0] if len(names) == 1 else f"{len(names)}suites"
    scorecard = write_run(Path(args.out) / f"{run_id}-{label}", report, attempts)

    for o in report.overall:
        print(
            f"  {o.model_id:<22} {o.verdict:<9} fitness={o.fitness:.3f} "
            f"$/task={o.cost_per_task_usd:.5f} s/task={o.latency_per_task_s:.1f} "
            f"{' '.join(o.reasons)}"
        )
    for g in regressions:
        print(f"  REGRESSION {g.model_id} {g.capability} {g.metric}: {g.previous} -> {g.current}")
    print(f"scorecard: {scorecard}")
    return 1 if (regressions and args.fail_on_regression) else 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    specs = load_models(args.models_file)
    judge = _pick(specs, [args.judge])
    if judge is None:
        return 2
    labels = load_labels(args.labels)
    suites = {n: load_suite(args.suites_dir, n) for n in sorted({lbl.suite for lbl in labels})}
    rep = calibrate(build_adapter(judge[0]), suites, labels)
    for row in rep.rows:
        print(
            f"  {row['suite']:<16} {row['task_id']:<24} human={row['human']} judge={row['judge']}"
        )
    print(
        f"n={rep.n} spearman={rep.spearman} MAE={rep.mean_abs_error} "
        f"within-1={rep.within_one_point:.0%}"
    )
    if judge[0].adapter == "mock":
        print("note: offline simulator judge echoes the labels; agreement is by construction")
    ok = rep.spearman is not None and rep.spearman >= args.min_spearman
    print("judge calibration: " + ("OK" if ok else f"BELOW threshold {args.min_spearman}"))
    return 0 if ok else 1


def cmd_history(args: argparse.Namespace) -> int:
    for run_id, model, cap, score, verdict in history.recent(Path(args.history), args.limit):
        print(f"{run_id}  {model:<22} {cap:<16} {score:.3f} {verdict}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    specs = load_models(args.models_file)
    print("models:")
    for s in specs.values():
        print(f"  {s.id:<22} {s.adapter:<10} {s.model}{'  (baseline)' if s.baseline else ''}")
    print("suites:")
    for name in ALL_SUITES:
        suite = load_suite(args.suites_dir, name)
        print(
            f"  {name:<18} {len(suite.tasks):>3} tasks  scorer={suite.scorer}"
            f"{'  judge' if suite.judge_rubric else ''}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rox-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="evaluate candidate models on capability suites")
    run.add_argument("--models", required=True, help="comma-separated ids from models.yaml")
    run.add_argument("--suite", default="all", help="'all' or comma-separated suite names")
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--judge", help="model id to use as LLM judge for rubric suites")
    run.add_argument("--workers", type=int, default=1, help="concurrent attempts")
    run.add_argument(
        "--reasoning-effort", help="override reasoning.effort for Responses-API models"
    )
    run.add_argument("--no-pairwise", action="store_true", help="skip pairwise judge vs baseline")
    run.add_argument("--out", default="runs")
    run.add_argument("--history", default="runs/history.sqlite")
    run.add_argument("--no-history", action="store_true")
    run.add_argument("--fail-on-regression", action="store_true")
    _common(run)
    run.set_defaults(fn=cmd_run)

    cal = sub.add_parser("calibrate", help="measure judge agreement with human labels")
    cal.add_argument("--judge", required=True)
    cal.add_argument("--labels", default="calibration/labels.yaml")
    cal.add_argument("--min-spearman", type=float, default=0.6)
    _common(cal)
    cal.set_defaults(fn=cmd_calibrate)

    hist = sub.add_parser("history", help="show recent recorded results")
    hist.add_argument("--history", default="runs/history.sqlite")
    hist.add_argument("--limit", type=int, default=30)
    hist.set_defaults(fn=cmd_history)

    lst = sub.add_parser("list", help="list configured models and suites")
    _common(lst)
    lst.set_defaults(fn=cmd_list)

    args = parser.parse_args(argv)
    code: int = args.fn(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
