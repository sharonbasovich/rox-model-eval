from pathlib import Path

from rox_model_eval import history
from rox_model_eval.adapters import build_adapter
from rox_model_eval.aggregate import overall, recommend, routing_table, summarize
from rox_model_eval.config import load_models, load_weights
from rox_model_eval.judge import calibrate, load_labels, pairwise, parse_rubric, spearman
from rox_model_eval.runner import pairwise_vs_baseline, run_suite
from rox_model_eval.suites import load_suite, resolve_suites

ROOT = Path(__file__).resolve().parent.parent
SPECS = load_models(ROOT / "config/models.yaml")
WEIGHTS = load_weights(ROOT / "config/weights.yaml")


def _run(names: list[str], reps: int = 2, judge: bool = False):  # type: ignore[no-untyped-def]
    attempts = []
    for n in names:
        suite = load_suite(ROOT / "suites", n)
        attempts += run_suite(
            [SPECS["mock-strong"], SPECS["mock-cheap"]],
            suite,
            reps,
            judge=SPECS["mock-strong"] if judge else None,
        )
    return attempts


def test_parse_rubric_and_spearman() -> None:
    rubric = ["tone: x", "clarity: y"]
    assert parse_rubric('{"scores": {"tone": 5, "clarity": 1}}', rubric) == 0.5
    assert parse_rubric("not json", rubric) is None
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_pairwise_cancels_position_bias() -> None:
    suite = load_suite(ROOT / "suites", "c2_drafting")
    judge = build_adapter(SPECS["mock-strong"].model_copy(update={"params": {"flaw_rate": 0.0}}))
    assert pairwise(judge, suite, suite.tasks[0], "a", "b", oracle_pref=1.0) == 1.0
    assert pairwise(judge, suite, suite.tasks[0], "a", "b", oracle_pref=0.0) == 0.0
    assert pairwise(judge, suite, suite.tasks[0], "a", "b", oracle_pref=0.5) == 0.5


def test_judge_blends_into_rubric_suites_and_pairwise_runs() -> None:
    attempts = _run(["c2_drafting"], reps=1, judge=True)
    assert all(a.scores.judge_score is not None for a in attempts)
    suite = load_suite(ROOT / "suites", "c2_drafting")
    wins = pairwise_vs_baseline(SPECS["mock-strong"], suite, attempts, "mock-strong")
    assert set(wins) == {"mock-cheap"} and 0.0 <= wins["mock-cheap"] <= 1.0


def test_calibration_report() -> None:
    labels = load_labels(ROOT / "calibration/labels.yaml")
    suites = {n: load_suite(ROOT / "suites", n) for n in {lbl.suite for lbl in labels}}
    rep = calibrate(build_adapter(SPECS["mock-strong"]), suites, labels)
    assert rep.n == len(labels) and rep.spearman is not None and rep.spearman > 0.6


def test_overall_verdicts_routing_and_pareto() -> None:
    summaries = recommend(summarize(_run(resolve_suites("all"), reps=1), SPECS), WEIGHTS)
    result = {o.model_id: o for o in overall(summaries, SPECS, WEIGHTS)}
    assert result["mock-strong"].verdict == "BASELINE"
    assert result["mock-strong"].fitness > result["mock-cheap"].fitness
    assert result["mock-cheap"].verdict in {"ADOPT", "ROUTE", "HOLD"}
    assert result["mock-strong"].pareto and result["mock-cheap"].pareto
    table = routing_table(summaries, WEIGHTS)
    assert {r.capability for r in table} == set(resolve_suites("all"))


def test_safety_violation_forces_hold() -> None:
    summaries = recommend(summarize(_run(["c6_extraction"], reps=1), SPECS), WEIGHTS)
    cheap = next(s for s in summaries if s.model_id == "mock-cheap")
    cheap.safety_violation_rate = 0.2
    cheap.mean_score, cheap.fabrication_rate, cheap.format_valid_rate = 0.99, 0.0, 1.0
    recommend(summaries, WEIGHTS)
    assert cheap.verdict == "HOLD" and any("safety" in r for r in cheap.reasons)
    by_id = {o.model_id: o for o in overall(summaries, SPECS, WEIGHTS)}
    assert by_id["mock-cheap"].verdict == "HOLD"


def test_history_flags_regressions_only_on_same_suite(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    summaries = recommend(summarize(_run(["c6_extraction"], reps=1), SPECS), WEIGHTS)
    hashes = {"c6_extraction": "abc"}
    history.record(db, "r1", "2026-01-01T00:00:00", summaries, hashes)
    assert history.detect(db, "r2", summaries, hashes, WEIGHTS) == []
    worse = [s.model_copy(update={"mean_score": s.mean_score - 0.2}) for s in summaries]
    found = history.detect(db, "r2", worse, hashes, WEIGHTS)
    assert {g.metric for g in found} == {"mean_score"} and len(found) == 2
    assert history.detect(db, "r2", worse, {"c6_extraction": "changed"}, WEIGHTS) == []


def test_pass_rate_gate_holds_model_with_frequent_task_failures() -> None:
    attempts = _run(["c6_extraction"], reps=1)
    strict = WEIGHTS.model_copy(update={"pass_rate_gate": 1.01})
    rows = recommend(summarize(attempts, SPECS), strict)
    assert rows and all(not r.gates_passed for r in rows)
    assert any("fully passed" in reason for r in rows for reason in r.reasons)
