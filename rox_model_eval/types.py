"""Core data types shared by adapters, scorers, runner and report."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_prompt_tokens += other.cached_prompt_tokens


class Timings(BaseModel):
    """Wall-clock timings for a model call (or the sum over an agent loop)."""

    ttft_s: float | None = None
    total_s: float = 0.0


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Chat message. Assistant turns may carry tool calls; `tool` turns carry results."""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ModelResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    timings: Timings = Field(default_factory=Timings)
    provider_version: str | None = None
    error: str | None = None
    cost_usd: float | None = None
    decisions: list[str] = Field(default_factory=list)


class ModelRequest(BaseModel):
    """A provider-agnostic model call.

    `offline_oracle` is read *only* by the offline simulator adapter; real
    provider adapters ignore it entirely.
    """

    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    json_schema: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    offline_oracle: dict[str, Any] | None = None


class Task(BaseModel):
    """One evaluation item.

    `inputs` fill the suite's `{{placeholders}}`; `expected` is what the suite's
    scorer checks against; `reference_output` is a known-good answer, used to
    document what good looks like and to drive the offline simulator.
    """

    id: str
    tags: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    reference_output: Any = None


class Suite(BaseModel):
    capability: str
    name: str
    description: str = ""
    scorer: str
    system: str
    prompt_template: str
    json_schema: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    max_steps: int = 6
    judge_rubric: list[str] | None = None
    tasks: list[Task]

    @staticmethod
    def fill(template: str, values: dict[str, Any]) -> str:
        def sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise KeyError(f"template placeholder '{{{{{key}}}}}' has no input")
            return str(values[key])

        return _PLACEHOLDER.sub(sub, template)

    def render(self, task: Task) -> list[Message]:
        return [
            Message(role="system", content=self.fill(self.system, task.inputs)),
            Message(role="user", content=self.fill(self.prompt_template, task.inputs)),
        ]

    @property
    def tool_names(self) -> set[str]:
        return {t["function"]["name"] for t in self.tools or []}


class RunOutput(BaseModel):
    """What a model produced for one task: final text plus any tool trajectory."""

    text: str
    trajectory: list[ToolCall] = Field(default_factory=list)
    turn_texts: list[str] = Field(default_factory=list)
    final_state: dict[str, Any] | None = None


class ScoreBreakdown(BaseModel):
    """Per-attempt scores.

    `score` is 0..1 quality. `fabrication` is the 0..1 share of output that the
    inputs do not support. `safety_violation` marks an attempt that obeyed
    injected instructions, leaked protected data or took a forbidden action.
    """

    passed: bool
    score: float
    format_valid: bool = True
    fabrication: float = 0.0
    safety_violation: bool = False
    judge_score: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class Attempt(BaseModel):
    model_id: str
    capability: str
    task_id: str
    rep: int
    response: ModelResponse
    trajectory: list[ToolCall] = Field(default_factory=list)
    scores: ScoreBreakdown
    cost_usd: float
