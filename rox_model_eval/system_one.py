"""Typed decisions from a local System One model (Laya, open weights, runs on CPU).

A System One model answers typed questions about a state instead of generating text:
`noul` -> P(true), `choice` -> one of the given options with a probability for each,
`score` -> expected level on an ordinal rubric. Laya (convaiinnovations/laya, Apache-2.0)
is an open-weight model of this kind; it is not TypeSafe's Jev.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

try:
    import laya
except ImportError:  # optional extra: pip install -e '.[laya]'
    laya = None

Question = dict[str, Any]


class Decisions(BaseModel):
    answers: dict[str, dict[str, Any]]
    input_tokens: int = 0
    seconds: float = 0.0
    model: str = ""

    def noul(self, qid: str) -> float:
        return float(self.answers[qid]["noul"])

    def choice(self, qid: str) -> tuple[str, float]:
        a = self.answers[qid]
        return str(a["choice"]), float(a["probabilities"][a["choice"]])


class Decider(Protocol):
    name: str

    def decide(self, state: str | dict[str, Any], questions: dict[str, Question]) -> Decisions: ...


class LayaDecider:
    """Laya loaded in-process. One instance is shared by every worker thread."""

    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        if laya is None:
            raise RuntimeError("laya is not installed: pip install -e '.[laya]'")
        self.name = model.rsplit("/", 1)[-1]
        self.agent = laya.load(model, device=device, revision=revision)
        self.lock = threading.Lock()

    def decide(self, state: str | dict[str, Any], questions: dict[str, Question]) -> Decisions:
        started = time.perf_counter()
        with self.lock:
            out = self.agent.predict(state, questions)
        return Decisions(
            answers=out["answers"],
            input_tokens=int(out.get("usage", {}).get("input_tokens", 0)),
            seconds=time.perf_counter() - started,
            model=self.name,
        )


class ScriptedDecider:
    """Deterministic decider for tests: answers come from a callback, not a model."""

    def __init__(self, answer: Any, name: str = "scripted") -> None:
        self.answer = answer
        self.name = name
        self.calls: list[tuple[str | dict[str, Any], dict[str, Question]]] = []

    def decide(self, state: str | dict[str, Any], questions: dict[str, Question]) -> Decisions:
        self.calls.append((state, questions))
        answers = {qid: self.answer(qid, q, state) for qid, q in questions.items()}
        return Decisions(answers=answers, input_tokens=10 * len(questions), model=self.name)


class DecisionLog(BaseModel):
    """What the decider decided during one attempt, kept for the scorecard notes."""

    entries: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    seconds: float = 0.0

    def add(self, d: Decisions, note: str) -> None:
        self.entries.append(note)
        self.input_tokens += d.input_tokens
        self.seconds += d.seconds
