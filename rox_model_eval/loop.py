"""The per-task execution loop: model -> tool results -> model, across user turns."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .adapters.base import ModelAdapter
from .crm_sim import CrmBackend
from .tools_sim import simulate_tool
from .types import (
    Message,
    ModelRequest,
    ModelResponse,
    RunOutput,
    Suite,
    Task,
    Timings,
    ToolCall,
    Usage,
)

ToolGate = Callable[[ToolCall], str | None]


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
    adapter: ModelAdapter,
    suite: Suite,
    task: Task,
    rep: int,
    system_note: str | None = None,
    tool_gate: ToolGate | None = None,
) -> tuple[ModelResponse, RunOutput]:
    """One attempt. With tools, loops model -> tool results -> model until a final answer.

    Tasks with `inputs.followups` are multi-turn sessions: after each final answer the next
    user message is appended and the loop continues with the full history. Tasks with
    `inputs.crm` run against a stateful backend whose end state is returned for scoring.
    `system_note` is appended to the system prompt. `tool_gate` may veto a tool call by
    returning a reason; a vetoed call is not executed and not added to the trajectory.
    """
    messages = suite.render(task)
    if system_note:
        messages[0] = messages[0].model_copy(
            update={"content": f"{messages[0].content}\n\n{system_note}"}
        )
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
            messages.append(Message(role="assistant", content=last.text, tool_calls=calls))
            for c in calls:
                veto = tool_gate(c) if tool_gate else None
                if veto:
                    result = json.dumps({"error": f"blocked by policy: {veto}"})
                else:
                    trajectory.append(c)
                    result = crm.call(c) if crm else simulate_tool(task, c)
                messages.append(Message(role="tool", content=result, tool_call_id=c.id))
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
