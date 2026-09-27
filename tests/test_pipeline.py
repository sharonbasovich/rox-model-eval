from pathlib import Path

from rox_model_eval.aggregate import recommend, summarize
from rox_model_eval.cli import main
from rox_model_eval.config import ModelSpec, Weights, load_models, load_weights
from rox_model_eval.runner import run_suite
from rox_model_eval.suites import load_suite
from rox_model_eval.types import Usage

ROOT = Path(__file__).resolve().parent.parent


def test_cost_accounts_for_cached_tokens() -> None:
    spec = ModelSpec(
        id="m",
        adapter="mock",
        model="m",
        price_in_per_mtok=2.0,
        price_out_per_mtok=8.0,
        price_cached_in_per_mtok=0.5,
    )
    usage = Usage(prompt_tokens=1_000_000, cached_prompt_tokens=400_000, completion_tokens=500_000)
    assert spec.cost_usd(usage) == 0.6 * 2.0 + 0.4 * 0.5 + 0.5 * 8.0


def test_c6_suite_is_self_consistent() -> None:
    suite = load_suite(ROOT / "suites", "c6_extraction")
    assert len(suite.tasks) >= 10
    assert len({t.id for t in suite.tasks}) == len(suite.tasks)
    required = set(suite.json_schema["required"]) if suite.json_schema else set()
    for task in suite.tasks:
        assert set(task.expected) == required, task.id


def test_mock_run_produces_verdicts_and_is_deterministic() -> None:
    specs = load_models(ROOT / "config/models.yaml")
    suite = load_suite(ROOT / "suites", "c6_extraction")
    chosen = [specs["mock-strong"], specs["mock-cheap"]]
    first = run_suite(chosen, suite, reps=2)
    second = run_suite(chosen, suite, reps=2)
    assert [a.scores for a in first] == [a.scores for a in second]

    summaries = recommend(summarize(first, specs), load_weights(ROOT / "config/weights.yaml"))
    verdicts = {s.model_id: s.verdict for s in summaries}
    assert verdicts["mock-strong"] == "BASELINE"
    assert verdicts["mock-cheap"] in {"ADOPT", "ROUTE", "HOLD"}
    strong = next(s for s in summaries if s.model_id == "mock-strong")
    cheap = next(s for s in summaries if s.model_id == "mock-cheap")
    assert strong.mean_score > cheap.mean_score


def test_missing_api_key_is_reported_not_raised(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    spec = ModelSpec(id="real", adapter="openai", model="x", api_key_env="NO_SUCH_KEY")
    suite = load_suite(ROOT / "suites", "c6_extraction")
    suite.tasks = suite.tasks[:1]
    attempts = run_suite([spec], suite, reps=1)
    summaries = recommend(summarize(attempts, {"real": spec}), Weights())
    assert summaries[0].verdict == "HOLD"
    assert "missing API key" in attempts[0].scores.notes[0]


def test_cli_writes_scorecard(tmp_path: Path) -> None:
    code = main(
        [
            "run",
            "--models",
            "mock-strong,mock-cheap",
            "--suite",
            "c6_extraction",
            "--reps",
            "1",
            "--models-file",
            str(ROOT / "config/models.yaml"),
            "--weights-file",
            str(ROOT / "config/weights.yaml"),
            "--suites-dir",
            str(ROOT / "suites"),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    assert {p.name for p in run_dir.iterdir()} == {"attempts.jsonl", "summary.json", "scorecard.md"}
    assert "offline simulator" in (run_dir / "scorecard.md").read_text()
