"""Runs candidate models across suites: single-shot or multi-step tool loops, N reps."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .adapters import ModelAdapter, build_adapter
from .config import ModelSpec
from .crm_sim import CrmBackend
from .judge import JUDGE_WEIGHT, judge_rubric, pairwise
from .scorers import SCORERS
from .tools_sim import simulate_tool
from .types import (
    Attempt,
    Message,
    ModelRequest,
    ModelResponse,
    RunOutput,
    ScoreBreakdown,
    Suite,
    Task,
    Timings,
    ToolCall,
    Usage,
)

ProgressFn = Callable[[Attempt], None]


def _oracle(task: Task) -> dict[str, Any]:
    """Hints for the offline simulator only: the reference answer and what an attack wants."""
    exp = task.expected
    oracle: dict[str, Any] = {
        "reference": task.reference_output if task.reference_output is not None else exp
    }
    tempting = [*exp.get("canaries", []), *exp.get("protected", [])]
    if tempting:
        oracle["tempting_text"] = tempting[0]
    if exp.get("forbidden_tools"):
        oracle["tempting_call"] = {
            "name": exp["forbidden_tools"][0],
            "arguments": exp.get("tempting_arguments", {}),
        }
    return oracle


def execute(
    adapter: ModelAdapter, suite: Suite, task: Task, rep: int
) -> tuple[ModelResponse, RunOutput]:
    """One attempt. With tools, loops model -> tool results -> model until a final answer.

    Tasks with `inputs.followups` are multi-turn sessions: after each final answer the next
    user message is appended and the loop continues with the full history. Tasks with
    `inputs.crm` run against a stateful backend whose end state is returned for scoring.
    """
    messages = suite.render(task)
    oracle = _oracle(task) if adapter.spec.adapter == "mock" else None
    crm = CrmBackend(task.inputs["crm"]) if "crm" in task.inputs else None
    followups: list[str] = [str(f) for f in task.inputs.get("followups", [])]
    usage = Usage()
    ttft: float | None = None
    total = 0.0
    trajectory: list[ToolCall] = []
    turn_texts: list[str] = []
    last = ModelResponse()
    for turn in range(1 + len(followups)):
        if turn:
            messages.append(Message(role="assistant", content=last.text))
            messages.append(Message(role="user", content=followups[turn - 1]))
        for step in range(suite.max_steps if suite.tools else 1):
            last = adapter.complete(
                ModelRequest(
                    messages=messages,
                    tools=suite.tools,
                    json_schema=suite.json_schema,
                    params={"rep": rep},
                    offline_oracle=oracle,
                )
            )
            usage.add(last.usage)
            total += last.timings.total_s
            if ttft is None:
                ttft = last.timings.ttft_s
            if last.error or not last.tool_calls or not suite.tools:
                break
            calls = [
                c if c.id else c.model_copy(update={"id": f"call_{turn}_{step}_{i}"})
                for i, c in enumerate(last.tool_calls)
            ]
            trajectory += calls
            messages.append(Message(role="assistant", content=last.text, tool_calls=calls))
            messages += [
                Message(
                    role="tool",
                    content=crm.call(c) if crm else simulate_tool(task, c),
                    tool_call_id=c.id,
                )
                for c in calls
            ]
        turn_texts.append(last.text)
        if last.error:
            break
    combined = last.model_copy(
        update={"usage": usage, "timings": Timings(ttft_s=ttft, total_s=total)}
    )
    return combined, RunOutput(
        text=last.text,
        trajectory=trajectory,
        turn_texts=turn_texts,
        final_state=crm.state if crm else None,
    )


def score_attempt(
    suite: Suite, task: Task, response: ModelResponse, output: RunOutput
) -> ScoreBreakdown:
    if response.error:
        return ScoreBreakdown(passed=False, score=0.0, format_valid=False, notes=[response.error])
    try:
        scorer = SCORERS[suite.scorer]
    except KeyError as exc:
        raise ValueError(f"suite {suite.capability} uses unknown scorer '{suite.scorer}'") from exc
    return scorer(suite, task, output)


def run_suite(
    specs: Iterable[ModelSpec],
    suite: Suite,
    reps: int = 3,
    on_attempt: ProgressFn | None = None,
    judge: ModelSpec | None = None,
    workers: int = 1,
) -> list[Attempt]:
    """Run every (model, task, rep); results keep that order regardless of `workers`."""
    judge_adapter = build_adapter(judge) if judge and suite.judge_rubric else None
    jobs = [
        (spec, adapter, task, rep)
        for spec in specs
        for adapter in [build_adapter(spec)]
        for task in suite.tasks
        for rep in range(reps)
    ]

    def one(job: tuple[ModelSpec, ModelAdapter, Task, int]) -> Attempt:
        spec, adapter, task, rep = job
        attempt = _attempt(spec, adapter, suite, task, rep, judge_adapter)
        if on_attempt:
            on_attempt(attempt)
        return attempt

    if workers <= 1:
        return [one(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def _attempt(
    spec: ModelSpec,
    adapter: ModelAdapter,
    suite: Suite,
    task: Task,
    rep: int,
    judge_adapter: ModelAdapter | None,
) -> Attempt:
    response, output = execute(adapter, suite, task, rep)
    scores = score_attempt(suite, task, response, output)
    cost = spec.cost_usd(response.usage)
    if judge_adapter and not response.error:
        oracle = None
        if judge_adapter.spec.adapter == "mock":
            level = 1 + 4 * scores.score
            oracle = {"judge_scores": {k: level for k in _criteria(suite)}}
        j, _ = judge_rubric(judge_adapter, suite, task, output.text, oracle)
        if j is not None:
            blended = (1 - JUDGE_WEIGHT) * scores.score + JUDGE_WEIGHT * j
            scores = scores.model_copy(
                update={
                    "judge_score": j,
                    "score": round(blended, 4),
                    "passed": scores.passed and j >= 0.5,
                }
            )
    return Attempt(
        model_id=spec.id,
        capability=suite.capability,
        task_id=task.id,
        rep=rep,
        response=response,
        trajectory=output.trajectory,
        scores=scores,
        cost_usd=cost,
    )


def _criteria(suite: Suite) -> list[str]:
    return [r.split(":", 1)[0].strip() for r in suite.judge_rubric or []]


def pairwise_vs_baseline(
    judge: ModelSpec, suite: Suite, attempts: list[Attempt], baseline_id: str
) -> dict[str, float]:
    """Judge win-rate of each candidate against the baseline, on rep-0 outputs."""
    if not suite.judge_rubric:
        return {}
    adapter = build_adapter(judge)
    tasks = {t.id: t for t in suite.tasks}
    rep0 = {(a.model_id, a.task_id): a for a in attempts if a.rep == 0 and not a.response.error}
    models = sorted({m for m, _ in rep0} - {baseline_id})
    out: dict[str, float] = {}
    for model in models:
        prefs: list[float] = []
        for task_id, task in tasks.items():
            cand, base = rep0.get((model, task_id)), rep0.get((baseline_id, task_id))
            if cand is None or base is None:
                continue
            oracle_pref = None
            if adapter.spec.adapter == "mock":
                delta = cand.scores.score - base.scores.score
                oracle_pref = 0.5 if abs(delta) < 0.02 else float(delta > 0)
            p = pairwise(adapter, suite, task, cand.response.text, base.response.text, oracle_pref)
            if p is not None:
                prefs.append(p)
        if prefs:
            out[model] = round(sum(prefs) / len(prefs), 4)
    return out
