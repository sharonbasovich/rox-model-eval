import json
from pathlib import Path
from typing import Any

from rox_model_eval.adapters.base import ModelAdapter
from rox_model_eval.adapters.hybrid import HybridAdapter
from rox_model_eval.config import ModelSpec
from rox_model_eval.scorers.grounded_qa import score_grounded_qa
from rox_model_eval.scorers.safety import score_safety
from rox_model_eval.suites import load_suite
from rox_model_eval.system_one import ScriptedDecider
from rox_model_eval.types import ModelRequest, ModelResponse, Timings, ToolCall, Usage

ROOT = Path(__file__).resolve().parent.parent
WRITER = ModelSpec(id="w", adapter="scripted", model="w", price_in_per_mtok=1.0)
HYBRID = ModelSpec(id="h", adapter="hybrid", model="h")


class _Writer(ModelAdapter):
    def __init__(self, reply: str = "", tool: ToolCall | None = None) -> None:
        super().__init__(WRITER)
        self.reply, self.tool = reply, tool
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        usage = Usage(prompt_tokens=1000)
        timings = Timings(total_s=1.0)
        if self.tool and request.messages[-1].role != "tool":
            return ModelResponse(tool_calls=[self.tool], usage=usage, timings=timings)
        return ModelResponse(text=self.reply, usage=usage, timings=timings)


def _noul(p: float) -> Any:
    return lambda qid, q, state: {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def test_confident_unanswerable_abstains_without_calling_the_writer() -> None:
    suite = load_suite(ROOT / "suites", "c4_grounded_qa")
    task = next(t for t in suite.tasks if not t.expected["answerable"])
    writer = _Writer()
    hybrid = HybridAdapter(HYBRID, writer, ScriptedDecider(_noul(0.02)))
    result = hybrid.run_task(suite, task, 0)
    assert result is not None
    response, output = result
    assert not writer.requests and response.cost_usd == 0.0
    assert json.loads(output.text)["answerable"] is False
    assert score_grounded_qa(suite, task, output).passed
    assert "writer skipped" in response.decisions[0]


def test_uncertain_answerability_defers_to_writer_and_sums_cost() -> None:
    suite = load_suite(ROOT / "suites", "c4_grounded_qa")
    reply = '{"answerable": true, "answer": "x", "citations": []}'
    writer = _Writer(reply)
    hybrid = HybridAdapter(HYBRID, writer, ScriptedDecider(_noul(0.6)))
    result = hybrid.run_task(suite, suite.tasks[0], 0)
    assert result is not None
    response, output = result
    assert len(writer.requests) == 1 and output.text == reply
    assert response.cost_usd == WRITER.cost_usd(Usage(prompt_tokens=1000))
    assert response.timings.total_s >= 1.0


def test_tool_gate_vetoes_injected_action_and_screen_warns_writer() -> None:
    suite = load_suite(ROOT / "suites", "c8_safety")
    task = next(
        t for t in suite.tasks if "export_contacts" in t.expected.get("forbidden_tools", [])
    )

    def answer(qid: str, q: Any, state: Any) -> dict[str, Any]:
        p = {"injection": 0.95, "allowed": 0.05}.get(qid, 0.0)
        return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}

    writer = _Writer(
        "Summary of the note.", ToolCall(name="export_contacts", arguments={"to": "x"})
    )
    hybrid = HybridAdapter(HYBRID, writer, ScriptedDecider(answer))
    result = hybrid.run_task(suite, task, 0)
    assert result is not None
    response, output = result
    assert output.trajectory == []
    assert "SECURITY SCREEN" in writer.requests[0].messages[0].content
    assert "blocked by policy" in writer.requests[1].messages[-1].content
    assert not score_safety(suite, task, output).safety_violation
    assert any("vetoed" in d for d in response.decisions)


def test_confident_ranking_uses_decider_order_and_writer_only_for_reason() -> None:
    suite = load_suite(ROOT / "suites", "c3_insights")
    task = suite.tasks[0]
    ideal = task.expected["ideal_order"]
    probs = {i: 0.95 - 0.3 * n for n, i in enumerate(ideal)}

    def answer(qid: str, q: Any, state: Any) -> dict[str, Any]:
        return {"type": "noul", "noul": max(probs[qid], 0.0), "confidence": 0.9}

    writer = _Writer('{"top_reason": "champion left"}')
    hybrid = HybridAdapter(HYBRID, writer, ScriptedDecider(answer))
    result = hybrid.run_task(suite, task, 0)
    assert result is not None
    _, output = result
    assert json.loads(output.text) == {"ranking": ideal, "top_reason": "champion left"}
    assert len(writer.requests) == 1 and writer.requests[0].json_schema is not None


def test_capabilities_outside_pipelines_go_to_the_writer_unchanged() -> None:
    suite = load_suite(ROOT / "suites", "c6_extraction")
    decider = ScriptedDecider(_noul(0.5))
    hybrid = HybridAdapter(HYBRID, _Writer("{}"), decider)
    assert hybrid.run_task(suite, suite.tasks[0], 0) is None and not decider.calls
