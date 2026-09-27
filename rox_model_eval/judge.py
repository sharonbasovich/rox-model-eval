"""LLM-as-judge: rubric scoring, pairwise preference vs baseline, calibration.

Deterministic scorers run first and gate adoption; the judge only covers what
code cannot check (tone, insight, usefulness). Judge scores are blended into the
quality score, never used to override a hard gate. Calibrate the judge against
human labels (`rox-eval calibrate`) before trusting it for a decision.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .adapters import ModelAdapter
from .scorers.common import parse_json
from .types import Message, ModelRequest, Suite, Task

JUDGE_WEIGHT = 0.4

_SYSTEM = (
    "You are a strict, calibrated evaluator for outputs of an AI revenue platform used by "
    "enterprise sales teams. Judge only against the rubric and the provided inputs. Output "
    "that asserts facts absent from the inputs must score low on groundedness. Respond with "
    "JSON only."
)


def criteria(rubric: list[str]) -> list[str]:
    return [r.split(":", 1)[0].strip() for r in rubric]


def _inputs_block(task: Task, limit: int = 12_000) -> str:
    raw = json.dumps(task.inputs, indent=1, default=str)
    return raw if len(raw) <= limit else raw[:limit] + "\n...[truncated]"


def rubric_request(
    suite: Suite, task: Task, output_text: str, oracle: dict[str, Any] | None = None
) -> ModelRequest:
    rubric = suite.judge_rubric or []
    keys = criteria(rubric)
    prompt = (
        f"TASK TYPE: {suite.name}\n\nINSTRUCTIONS GIVEN TO THE MODEL:\n"
        f"{suite.fill(suite.prompt_template, task.inputs)[:4000]}\n\n"
        f"INPUTS:\n{_inputs_block(task)}\n\nOUTPUT TO GRADE:\n<<<\n{output_text}\n>>>\n\n"
        "Score each criterion from 1 (unacceptable) to 5 (excellent):\n"
        + "\n".join(f"- {r}" for r in rubric)
        + '\n\nReturn {"scores": {'
        + ", ".join(f'"{k}": <1-5>' for k in keys)
        + '}, "rationale": "<one sentence>"}'
    )
    return ModelRequest(
        messages=[Message(role="system", content=_SYSTEM), Message(role="user", content=prompt)],
        params={"rep": 0},
        offline_oracle=oracle,
    )


def parse_rubric(text: str, rubric: list[str]) -> float | None:
    """Mean criterion score mapped to 0..1, or None if the judge output is unusable."""
    parsed = parse_json(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("scores"), dict):
        return None
    values = []
    for key in criteria(rubric):
        v = parsed["scores"].get(key)
        if isinstance(v, int | float) and 1 <= v <= 5:
            values.append((float(v) - 1) / 4)
    return round(statistics.fmean(values), 4) if values else None


def judge_rubric(
    adapter: ModelAdapter,
    suite: Suite,
    task: Task,
    output_text: str,
    oracle: dict[str, Any] | None = None,
) -> tuple[float | None, float]:
    """Returns (judge score 0..1 or None, judge cost in USD)."""
    response = adapter.complete(rubric_request(suite, task, output_text, oracle))
    cost = adapter.spec.cost_usd(response.usage)
    if response.error:
        return None, cost
    return parse_rubric(response.text, suite.judge_rubric or []), cost


def pairwise_request(
    suite: Suite, task: Task, a: str, b: str, oracle: dict[str, Any] | None = None
) -> ModelRequest:
    prompt = (
        f"TASK TYPE: {suite.name}\n\nINPUTS:\n{_inputs_block(task, 8000)}\n\n"
        f"RESPONSE A:\n<<<\n{a}\n>>>\n\nRESPONSE B:\n<<<\n{b}\n>>>\n\n"
        "Which response would a revenue team rather ship? Consider: "
        + "; ".join(suite.judge_rubric or ["accuracy", "usefulness"])
        + '. Return {"winner": "A" | "B" | "tie", "rationale": "<one sentence>"}'
    )
    return ModelRequest(
        messages=[Message(role="system", content=_SYSTEM), Message(role="user", content=prompt)],
        params={"rep": 0},
        offline_oracle=oracle,
    )


def pairwise(
    adapter: ModelAdapter,
    suite: Suite,
    task: Task,
    candidate: str,
    baseline: str,
    oracle_pref: float | None = None,
) -> float | None:
    """Candidate preference vs baseline in [0, 1]; judged in both orders to cancel position bias.

    `oracle_pref` (offline simulator only): 1 candidate better, 0 worse, 0.5 tie.
    """
    results: list[float] = []
    for cand_first in (True, False):
        a, b = (candidate, baseline) if cand_first else (baseline, candidate)
        oracle = None
        if oracle_pref is not None:
            if oracle_pref == 0.5:
                oracle = {"pairwise_winner": "tie"}
            else:
                cand_wins = oracle_pref > 0.5
                oracle = {"pairwise_winner": "A" if cand_wins == cand_first else "B"}
        response = adapter.complete(pairwise_request(suite, task, a, b, oracle))
        parsed = parse_json(response.text) if not response.error else None
        if not isinstance(parsed, dict):
            continue
        winner = str(parsed.get("winner", "")).strip().upper()
        if winner == "TIE":
            results.append(0.5)
        elif winner in {"A", "B"}:
            results.append(1.0 if (winner == "A") == cand_first else 0.0)
    return statistics.fmean(results) if results else None


class CalibrationLabel(BaseModel):
    suite: str
    task_id: str
    output: str
    human: dict[str, int]


class CalibrationReport(BaseModel):
    n: int
    spearman: float | None
    mean_abs_error: float
    within_one_point: float
    rows: list[dict[str, Any]]


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    try:
        return round(statistics.correlation(rx, ry), 4)
    except statistics.StatisticsError:
        return None


def load_labels(path: str | Path) -> list[CalibrationLabel]:
    return [CalibrationLabel(**row) for row in yaml.safe_load(Path(path).read_text())["labels"]]


def calibrate(
    adapter: ModelAdapter, suites: dict[str, Suite], labels: list[CalibrationLabel]
) -> CalibrationReport:
    """Agreement between the judge and human 1-5 labels, on the judge's own 1-5 scale."""
    rows: list[dict[str, Any]] = []
    judge_vals: list[float] = []
    human_vals: list[float] = []
    for label in labels:
        suite = suites[label.suite]
        task = next(t for t in suite.tasks if t.id == label.task_id)
        oracle = {"judge_scores": label.human} if adapter.spec.adapter == "mock" else None
        score, _ = judge_rubric(adapter, suite, task, label.output, oracle)
        human = (statistics.fmean(label.human.values()) - 1) / 4
        rows.append(
            {
                "suite": label.suite,
                "task_id": label.task_id,
                "human": round(1 + 4 * human, 2),
                "judge": None if score is None else round(1 + 4 * score, 2),
            }
        )
        if score is not None:
            judge_vals.append(1 + 4 * score)
            human_vals.append(1 + 4 * human)
    diffs = [abs(j - h) for j, h in zip(judge_vals, human_vals, strict=True)]
    return CalibrationReport(
        n=len(judge_vals),
        spearman=spearman(judge_vals, human_vals),
        mean_abs_error=round(statistics.fmean(diffs), 4) if diffs else 0.0,
        within_one_point=round(sum(d <= 1 for d in diffs) / len(diffs), 4) if diffs else 0.0,
        rows=rows,
    )
