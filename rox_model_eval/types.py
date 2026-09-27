"""Core data types shared by adapters, scorers, runner and report."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0


class Timings(BaseModel):
    """Wall-clock timings for a single model call."""

    ttft_s: float | None = None
    total_s: float = 0.0


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    timings: Timings = Field(default_factory=Timings)
    provider_version: str | None = None
    error: str | None = None


class Message(BaseModel):
    role: str
    content: str


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
    """One evaluation item within a capability suite."""

    id: str
    source_text: str
    expected: dict[str, Any]
    tags: list[str] = Field(default_factory=list)


class Suite(BaseModel):
    capability: str
    name: str
    description: str = ""
    system: str
    prompt_template: str
    json_schema: dict[str, Any] | None = None
    tasks: list[Task]

    def render(self, task: Task) -> list[Message]:
        return [
            Message(role="system", content=self.system),
            Message(role="user", content=self.prompt_template.format(source_text=task.source_text)),
        ]


class ScoreBreakdown(BaseModel):
    """Per-attempt scores. `passed` gates cost-per-successful-task."""

    passed: bool
    score: float
    json_valid: bool = False
    schema_valid: bool = False
    field_accuracy: float = 0.0
    fabrication_rate: float = 0.0
    notes: list[str] = Field(default_factory=list)


class Attempt(BaseModel):
    model_id: str
    capability: str
    task_id: str
    rep: int
    response: ModelResponse
    scores: ScoreBreakdown
    cost_usd: float
