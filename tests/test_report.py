from pathlib import Path

from rox_model_eval.aggregate import ModelOverall, overall, recommend, routing_table, summarize
from rox_model_eval.config import load_models, load_weights
from rox_model_eval.report import (
    CapabilityInfo,
    RunReport,
    bottom_line,
    cost_svg,
    render_html,
    time_svg,
)
from rox_model_eval.runner import run_suite
from rox_model_eval.suites import load_suite

ROOT = Path(__file__).resolve().parent.parent
SPECS = load_models(ROOT / "config/models.yaml")
WEIGHTS = load_weights(ROOT / "config/weights.yaml")


def _report() -> tuple[RunReport, list]:  # type: ignore[type-arg]
    attempts = []
    info = {}
    for n in ["c4_grounded_qa", "c6_extraction"]:
        suite = load_suite(ROOT / "suites", n)
        info[suite.capability] = CapabilityInfo(
            name=suite.name, description=suite.description, tasks=len(suite.tasks)
        )
        attempts += run_suite([SPECS["mock-strong"], SPECS["mock-cheap"]], suite, 1)
    s = recommend(summarize(attempts, SPECS), WEIGHTS)
    report = RunReport(
        run_id="t",
        suites={c: "h" for c in info},
        reps=1,
        overall=overall(s, SPECS, WEIGHTS),
        capabilities=s,
        routing=routing_table(s, WEIGHTS),
        simulated=True,
        capability_info=info,
        weights=WEIGHTS,
    )
    return report, attempts


def test_overall_tracks_time_per_task() -> None:
    report, _ = _report()
    by_id = {o.model_id: o for o in report.overall}
    assert by_id["mock-cheap"].latency_per_task_s < by_id["mock-strong"].latency_per_task_s
    assert all(o.latency_per_task_s > 0 for o in report.overall)


def test_bottom_line_is_plain_sentences() -> None:
    report, _ = _report()
    lines = bottom_line(report)
    assert lines[0].startswith("Recommendation:")
    assert any(line.startswith("Safety:") for line in lines)
    assert any("3-repetition" in line for line in lines)
    assert all(line.endswith(".") for line in lines)


def test_html_has_both_charts_and_glossary_and_no_external_assets() -> None:
    report, attempts = _report()
    page = render_html(report, attempts)
    assert "Rox Fitness vs cost per task" in page and "Rox Fitness vs time per task" in page
    assert page.count("<svg") == 2
    assert "How to read this scorecard" in page and "Bottom line" in page
    assert "Record extraction" in page
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in page and "<script" not in page


def test_charts_plot_each_model_with_its_values() -> None:
    models = [
        ModelOverall(
            model_id=m,
            baseline=b,
            fitness=f,
            weight_coverage=1,
            cost_per_task_usd=c,
            latency_per_task_s=t,
            p95_latency_s=t * 2,
            capability_verdicts={},
            verdict=v,
        )
        for m, b, f, c, t, v in [
            ("big", True, 0.96, 0.002, 2.9, "BASELINE"),
            ("small", False, 0.94, 0.0001, 2.1, "ROUTE"),
        ]
    ]
    cost, time = cost_svg(models), time_svg(models)
    assert "big" in cost and "small" in cost and "$0.0001" in cost
    assert "2.9s" in time and "2.1s" in time and "polyline" in time


def test_without_baseline_compares_head_to_head() -> None:
    specs = {k: v.model_copy(update={"baseline": False}) for k, v in SPECS.items()}
    suite = load_suite(ROOT / "suites", "c6_extraction")
    attempts = run_suite([specs["mock-strong"], specs["mock-cheap"]], suite, 1)
    s = recommend(summarize(attempts, specs), WEIGHTS)
    assert {x.verdict for x in s} <= {"PASS", "HOLD"}
    ov = overall(s, specs, WEIGHTS)
    assert {o.verdict for o in ov} <= {"PASS", "PARTIAL", "HOLD"}
    report = RunReport(
        run_id="t",
        suites={"c6_extraction": "h"},
        reps=1,
        overall=ov,
        capabilities=s,
        routing=routing_table(s, WEIGHTS),
        weights=WEIGHTS,
    )
    lines = bottom_line(report)
    assert lines[0].startswith("Highest quality:")
    assert not any("baseline" in line.lower() for line in lines)
    page = render_html(report, attempts)
    assert "BASELINE" not in page and "Rox uses today" not in page
