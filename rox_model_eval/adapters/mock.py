"""Offline simulator adapter.

This adapter does NOT measure any real model. It derives a response from the
task's own reference answer and then injects failures at `flaw_rate` (dropped
field, fabricated value, malformed JSON). Its only purpose is to exercise the
harness -- scorers, cost/latency accounting, aggregation, scorecard -- with zero
API keys, and to give the scorers something that genuinely passes and fails.
Never report its numbers as model results.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from ..types import ModelRequest, ModelResponse, Timings, Usage
from .base import ModelAdapter

_FABRICATED = ["Globex Holdings", "4,200", "2019", "Series Z", "unknown-corp.example"]


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockAdapter(ModelAdapter):
    def complete(self, request: ModelRequest) -> ModelResponse:
        flaw_rate = float(self.spec.params.get("flaw_rate", 0.1))
        oracle = dict(request.offline_oracle or {})
        prompt_text = "\n".join(m.content for m in request.messages)
        rng = random.Random(_seed(self.spec.id, prompt_text, str(request.params.get("rep", 0))))

        answer: dict[str, Any] = dict(oracle)
        malformed = False
        if answer:
            for key in list(answer):
                if rng.random() < flaw_rate:
                    if rng.random() < 0.5:
                        del answer[key]
                    else:
                        answer[key] = rng.choice(_FABRICATED)
            malformed = rng.random() < flaw_rate / 4

        text = json.dumps(answer, indent=2)
        if malformed:
            text = text[: max(1, len(text) - 3)]

        # Simulated latency: cheap models are faster, all jittered deterministically.
        latency = (0.35 if "cheap" in self.spec.id else 0.9) * (0.8 + rng.random() * 0.4)
        return ModelResponse(
            text=text,
            usage=Usage(
                prompt_tokens=_approx_tokens(prompt_text),
                completion_tokens=_approx_tokens(text),
            ),
            timings=Timings(ttft_s=latency * 0.4, total_s=latency),
            provider_version=f"offline-simulator/{self.spec.model}",
        )
