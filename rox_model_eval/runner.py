"""Runs candidate models across a suite with N repetitions."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .adapters import build_adapter
from .config import ModelSpec
from .scorers import score_extraction
from .types import Attempt, ModelRequest, ScoreBreakdown, Suite

ProgressFn = Callable[[Attempt], None]


def run_suite(
    specs: Iterable[ModelSpec],
    suite: Suite,
    reps: int = 3,
    on_attempt: ProgressFn | None = None,
) -> list[Attempt]:
    attempts: list[Attempt] = []
    for spec in specs:
        adapter = build_adapter(spec)
        for task in suite.tasks:
            for rep in range(reps):
                request = ModelRequest(
                    messages=suite.render(task),
                    json_schema=suite.json_schema,
                    params={"rep": rep},
                    offline_oracle=task.expected if spec.adapter == "mock" else None,
                )
                response = adapter.complete(request)
                if response.error:
                    scores = ScoreBreakdown(passed=False, score=0.0, notes=[response.error])
                else:
                    scores = score_extraction(response.text, task.expected, suite.json_schema)
                attempt = Attempt(
                    model_id=spec.id,
                    capability=suite.capability,
                    task_id=task.id,
                    rep=rep,
                    response=response,
                    scores=scores,
                    cost_usd=spec.cost_usd(response.usage),
                )
                attempts.append(attempt)
                if on_attempt:
                    on_attempt(attempt)
    return attempts
