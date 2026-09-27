"""Model/candidate configuration and cost accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .types import Usage


class ModelSpec(BaseModel):
    """A candidate model: how to call it and what it costs.

    Adding a new frontier model is a config entry, never a code change.
    """

    id: str
    adapter: str
    model: str
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    price_cached_in_per_mtok: float | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    baseline: bool = False

    def cost_usd(self, usage: Usage) -> float:
        cached = usage.cached_prompt_tokens
        fresh_in = max(usage.prompt_tokens - cached, 0)
        cached_rate = (
            self.price_cached_in_per_mtok
            if self.price_cached_in_per_mtok is not None
            else self.price_in_per_mtok
        )
        return (
            fresh_in * self.price_in_per_mtok
            + cached * cached_rate
            + usage.completion_tokens * self.price_out_per_mtok
        ) / 1_000_000


class Weights(BaseModel):
    """Capability weights for the composite score, plus gate thresholds."""

    capabilities: dict[str, float] = Field(default_factory=dict)
    quality_gate: float = 0.75
    fabrication_gate: float = 0.05
    format_gate: float = 0.9
    safety_gate: float = 0.0
    pass_rate_gate: float = 0.0
    route_margin: float = 0.05
    regression_score_drop: float = 0.05
    regression_fabrication_rise: float = 0.02
    regression_latency_ratio: float = 1.5


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(Path(path).read_text())
    specs = [ModelSpec(**m) for m in raw["models"]]
    return {s.id: s for s in specs}


def load_weights(path: str | Path) -> Weights:
    return Weights(**yaml.safe_load(Path(path).read_text()))
