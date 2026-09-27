import re
from pathlib import Path

import pytest

from rox_model_eval.adapters.anthropic import to_anthropic_messages
from rox_model_eval.adapters.base import ModelAdapter
from rox_model_eval.adapters.openai_compatible import to_openai_message
from rox_model_eval.config import ModelSpec
from rox_model_eval.runner import execute, run_suite
from rox_model_eval.suites import ALL_SUITES, build_transcript, load_suite
from rox_model_eval.tools_sim import simulate_tool
from rox_model_eval.types import Message, ModelRequest, ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ModelSpec(id="oracle", adapter="mock", model="oracle", params={"flaw_rate": 0.0})


@pytest.mark.parametrize("name", ALL_SUITES)
def test_suite_loads_renders_and_reference_passes(name: str) -> None:
    suite = load_suite(ROOT / "suites", name)
    assert suite.capability == name
    assert len({t.id for t in suite.tasks}) == len(suite.tasks) >= 4
    for task in suite.tasks:
        for m in suite.render(task):
            assert not re.search(r"\{\{\s*\w+\s*\}\}", m.content), task.id
        for tool in task.expected.get("forbidden_tools", []):
            assert tool in suite.tool_names, (task.id, tool)
        for call in task.expected.get("calls", []):
            assert call["tool"] in suite.tool_names, (task.id, call)
    attempts = run_suite([ORACLE], suite, reps=1)
    failed = [(a.task_id, a.scores.notes) for a in attempts if not a.scores.passed]
    assert not failed, failed


def test_transcript_generator_is_deterministic_and_plants_needles() -> None:
    spec = {
        "seed": 1,
        "target_words": 3000,
        "needles": [{"depth": 0.9, "text": "NEEDLE-A"}],
        "distractors": [{"depth": 0.1, "text": "DISTRACTOR-B"}],
    }
    t1, t2 = build_transcript(spec), build_transcript(spec)
    assert t1 == t2 and len(t1.split()) >= 3000
    assert t1.index("DISTRACTOR-B") < t1.index("NEEDLE-A")


def test_long_context_tasks_reach_target_length() -> None:
    suite = load_suite(ROOT / "suites", "c7_long_context")
    for task in suite.tasks:
        target = task.inputs["transcript_spec"]["target_words"]
        assert len(task.inputs["transcript"].split()) >= target


class _Scripted(ModelAdapter):
    """Calls one tool, then answers with whatever the tool returned."""

    def __init__(self) -> None:
        super().__init__(ModelSpec(id="s", adapter="scripted", model="s"))
        self.seen: list[list[Message]] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.seen.append(list(request.messages))
        if request.messages[-1].role == "tool":
            return ModelResponse(text=f"result: {request.messages[-1].content}")
        return ModelResponse(
            tool_calls=[ToolCall(name="search_accounts", arguments={"query": "Brightkeel"})]
        )


def test_agent_loop_feeds_tool_results_back() -> None:
    suite = load_suite(ROOT / "suites", "c5_tool_calling")
    task = suite.tasks[0]
    adapter = _Scripted()
    response, output = execute(adapter, suite, task, rep=0)
    assert [c.name for c in output.trajectory] == ["search_accounts"]
    assert "acc_101" in output.text
    tool_msg = adapter.seen[1][-1]
    assert tool_msg.role == "tool" and tool_msg.tool_call_id == output.trajectory[0].id


def test_simulated_tool_errors_on_unknown_record() -> None:
    suite = load_suite(ROOT / "suites", "c5_tool_calling")
    out = simulate_tool(
        suite.tasks[0], ToolCall(name="get_account", arguments={"account_id": "nope"})
    )
    assert "error" in out


def test_provider_message_serialization() -> None:
    call = ToolCall(id="c1", name="search", arguments={"q": "x"})
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(
            role="assistant", content="", tool_calls=[call, call.model_copy(update={"id": "c2"})]
        ),
        Message(role="tool", content="r1", tool_call_id="c1"),
        Message(role="tool", content="r2", tool_call_id="c2"),
    ]
    oa = to_openai_message(msgs[2])
    assert oa["content"] is None and oa["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    assert to_openai_message(msgs[3])["tool_call_id"] == "c1"
    an = to_anthropic_messages(msgs)
    assert [m["role"] for m in an] == ["user", "assistant", "user"]
    assert an[1]["content"][0]["type"] == "tool_use"
    assert [b["tool_use_id"] for b in an[2]["content"]] == ["c1", "c2"]
