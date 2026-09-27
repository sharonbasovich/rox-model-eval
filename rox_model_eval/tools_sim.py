"""Scripted tool backends for agent-loop suites.

A task's `inputs.tool_results` maps tool name -> either a fixed result, or a list
of `{match: {arg: value}, result: ...}` cases. Unmatched calls get an error
result, which is what a real API would return for a bad id or unknown record.
"""

from __future__ import annotations

import json
from typing import Any

from .types import Task, ToolCall


def _norm(v: Any) -> str:
    return str(v).strip().lower()


def simulate_tool(task: Task, call: ToolCall) -> str:
    table: dict[str, Any] = task.inputs.get("tool_results", {})
    if call.name not in table:
        return json.dumps({"error": f"tool '{call.name}' unavailable or returned nothing"})
    spec = table[call.name]
    if isinstance(spec, list) and all(isinstance(c, dict) and "match" in c for c in spec):
        for case in spec:
            want: dict[str, Any] = case["match"]
            if all(
                k in call.arguments and _norm(v) in _norm(call.arguments[k])
                for k, v in want.items()
            ):
                return json.dumps(case["result"])
        return json.dumps({"error": "no record matches those arguments"})
    return json.dumps(spec)
