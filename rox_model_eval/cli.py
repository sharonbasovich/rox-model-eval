"""Command line entry point: `rox-eval run --models a,b --suite c6_extraction`."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from .aggregate import recommend, summarize
from .config import load_models, load_weights
from .report import write_run
from .runner import run_suite
from .suites import load_suite
from .types import Attempt


def _progress(a: Attempt) -> None:
    mark = "." if a.scores.passed else ("E" if a.response.error else "x")
    print(mark, end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rox-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="evaluate candidate models on a capability suite")
    run.add_argument("--models", required=True, help="comma-separated ids from models.yaml")
    run.add_argument("--suite", required=True, help="suite directory name under --suites-dir")
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--models-file", default="config/models.yaml")
    run.add_argument("--weights-file", default="config/weights.yaml")
    run.add_argument("--suites-dir", default="suites")
    run.add_argument("--out", default="runs")
    args = parser.parse_args(argv)

    specs = load_models(args.models_file)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in specs]
    if unknown:
        print(f"unknown model id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    suite = load_suite(args.suites_dir, args.suite)
    print(f"{suite.capability}: {len(suite.tasks)} tasks x {args.reps} reps x {len(wanted)} models")
    attempts = run_suite([specs[m] for m in wanted], suite, args.reps, _progress)
    print()

    summaries = recommend(summarize(attempts, specs), load_weights(args.weights_file))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    scorecard = write_run(Path(args.out) / f"{stamp}-{suite.capability}", summaries, attempts)
    for s in sorted(summaries, key=lambda s: -s.mean_score):
        print(
            f"  {s.model_id:<20} {s.verdict:<9} score={s.mean_score:.2f} "
            f"fab={s.fabrication_rate:.1%} $/success={s.cost_per_success_usd}"
        )
    print(f"scorecard: {scorecard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
