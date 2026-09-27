"""Offline simulator adapter.

This adapter does NOT measure any real model. It starts from the task's
known-good reference output and injects failures at `flaw_rate`: dropped or
fabricated JSON fields, malformed JSON, invented statistics in prose, skipped or
corrupted tool calls, and -- on safety tasks -- obeying the planted injection.
Its only purpose is to exercise the harness end to end (scorers, agent loop,
judge, cost/latency accounting, aggregation, history, scorecard) with zero API
keys. Never report its numbers as model results.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from ..types import ModelRequest, ModelResponse, Timings, ToolCall, Usage
from .base import ModelAdapter

_FABRICATED_VALUES = ["Globex Holdings", "4,200", "2019", "Series Z", "unknown-corp.example"]
_FABRICATED_SENTENCE = "They grew revenue 340% last year and now serve 12,000 customers."


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockAdapter(ModelAdapter):
    @property
    def flaw_rate(self) -> float:
        return float(self.spec.params.get("flaw_rate", 0.1))

    @property
    def injection_rate(self) -> float:
        """How often the simulator falls for a planted injection or tempting write."""
        return float(self.spec.params.get("injection_rate", self.flaw_rate / 2))

    def complete(self, request: ModelRequest) -> ModelResponse:
        oracle = request.offline_oracle or {}
        prompt_text = "\n".join(m.content for m in request.messages)
        step = sum(1 for m in request.messages if m.role == "assistant" and m.tool_calls)
        rng = random.Random(
            _seed(self.spec.id, prompt_text[:4000], str(request.params.get("rep", 0)), str(step))
        )

        text = ""
        tool_calls: list[ToolCall] = []
        if "judge_scores" in oracle:
            text = self._judge(oracle["judge_scores"], rng)
        elif "pairwise_winner" in oracle:
            winner = str(oracle["pairwise_winner"])
            if rng.random() < self.flaw_rate:
                winner = rng.choice(["A", "B", "tie"])
            text = json.dumps({"winner": winner, "rationale": "offline simulated judge"})
        else:
            ref = oracle.get("reference")
            if isinstance(ref, dict) and "steps" in ref:
                tool_calls, text = self._agent_step(ref, oracle, step, rng)
            elif isinstance(ref, dict):
                text = self._corrupt_json(ref, rng)
            else:
                text = self._corrupt_text(str(ref or ""), oracle.get("tempting_text"), rng)

        latency = (0.35 if "cheap" in self.spec.id else 0.9) * (0.8 + rng.random() * 0.4)
        latency *= 1 + len(prompt_text) / 40_000
        out_len = len(text) + sum(len(json.dumps(tc.arguments)) for tc in tool_calls)
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=_approx_tokens(prompt_text),
                completion_tokens=max(1, out_len // 4),
            ),
            timings=Timings(ttft_s=latency * 0.4, total_s=latency),
            provider_version=f"offline-simulator/{self.spec.model}",
        )

    def _judge(self, scores: dict[str, float], rng: random.Random) -> str:
        noisy = {
            k: max(1, min(5, round(v + rng.uniform(-1, 1) * self.flaw_rate * 2)))
            for k, v in scores.items()
        }
        return json.dumps({"scores": noisy, "rationale": "offline simulated judge"})

    def _agent_step(
        self, ref: dict[str, Any], oracle: dict[str, Any], step: int, rng: random.Random
    ) -> tuple[list[ToolCall], str]:
        steps: list[dict[str, Any]] = ref["steps"]
        tempting_call = oracle.get("tempting_call")
        if step == 0 and tempting_call and rng.random() < self.injection_rate:
            return [
                ToolCall(
                    id="call_t",
                    name=tempting_call["name"],
                    arguments=dict(tempting_call.get("arguments", {})),
                )
            ], ""
        if step < len(steps) and not (step > 0 and rng.random() < self.flaw_rate / 2):
            planned = steps[step]
            args = dict(planned.get("arguments", {}))
            if args and rng.random() < self.flaw_rate:
                key = rng.choice(sorted(args))
                args[key] = "UNKNOWN"
            return [ToolCall(id=f"call_{step}", name=planned["tool"], arguments=args)], ""
        return [], self._corrupt_text(str(ref.get("final", "")), oracle.get("tempting_text"), rng)

    def _corrupt_json(self, ref: dict[str, Any], rng: random.Random) -> str:
        answer: dict[str, Any] = dict(ref)
        for key in list(answer):
            if rng.random() >= self.flaw_rate:
                continue
            value = answer[key]
            if isinstance(value, list) and len(value) > 1:
                shuffled = list(value)
                rng.shuffle(shuffled)
                answer[key] = shuffled
            elif isinstance(value, bool):
                answer[key] = not value
            elif rng.random() < 0.5:
                del answer[key]
            else:
                answer[key] = rng.choice(_FABRICATED_VALUES)
        text = json.dumps(answer, indent=2)
        if rng.random() < self.flaw_rate / 4:
            text = text[: max(1, len(text) - 3)]
        return text

    def _corrupt_text(self, text: str, tempting: str | None, rng: random.Random) -> str:
        lines = text.rstrip("\n").split("\n")
        if len(lines) > 2 and rng.random() < self.flaw_rate:
            lines.pop(rng.randrange(1, len(lines)))
        if rng.random() < self.flaw_rate:
            lines.insert(min(len(lines), 2), _FABRICATED_SENTENCE)
        if tempting and rng.random() < self.injection_rate:
            lines.append(tempting)
        return "\n".join(lines)
