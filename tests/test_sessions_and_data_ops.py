from pathlib import Path

from rox_model_eval.adapters.base import ModelAdapter
from rox_model_eval.config import ModelSpec
from rox_model_eval.crm_sim import CrmBackend
from rox_model_eval.datagen import build_dataset
from rox_model_eval.runner import execute
from rox_model_eval.scorers.agent_session import score_agent_session
from rox_model_eval.scorers.data_ops import score_data_ops
from rox_model_eval.suites import load_suite
from rox_model_eval.types import ModelRequest, ModelResponse, RunOutput, Task, ToolCall

ROOT = Path(__file__).resolve().parent.parent


def test_crm_backend_reads_see_writes_and_rejects_cross_account_refs() -> None:
    crm = CrmBackend(
        {
            "accounts": [
                {"account_id": "a1", "name": "A", "domain": "a.com"},
                {"account_id": "a2", "name": "B", "domain": "b.com"},
            ],
            "deals": [{"deal_id": "d1", "account_id": "a2", "stage": "proposal"}],
        }
    )
    crm.call(ToolCall(name="update_deal", arguments={"deal_id": "d1", "stage": "closed_won"}))
    assert '"closed_won"' in crm.call(ToolCall(name="list_deals", arguments={"account_id": "a2"}))
    bad = crm.call(
        ToolCall(
            name="create_task",
            arguments={"account_id": "a1", "title": "x", "due_date": "2026-10-01", "deal_id": "d1"},
        )
    )
    assert "another account" in bad and not crm.state["tasks"]
    assert "error" in crm.call(ToolCall(name="update_deal", arguments={"deal_id": "d1", "x": 1}))


def _session_output(task: Task, calls: list[ToolCall], texts: list[str]) -> RunOutput:
    crm = CrmBackend(task.inputs["crm"])
    for c in calls:
        crm.call(c)
    return RunOutput(text=texts[-1], trajectory=calls, turn_texts=texts, final_state=crm.state)


def test_agent_session_flags_collateral_writes_and_missed_state() -> None:
    suite = load_suite(ROOT / "suites", "c9_agent_sessions")
    task = next(t for t in suite.tasks if t.id == "c9-004-renewal-math-session")
    right = [
        ToolCall(
            name="update_deal",
            arguments={"deal_id": "d_1", "amount": 112200, "stage": "negotiation"},
        ),
        ToolCall(
            name="update_deal", arguments={"deal_id": "d_2", "amount": 30000, "stage": "proposal"}
        ),
        ToolCall(
            name="create_task",
            arguments={
                "account_id": "acc_qm",
                "title": "Send revised order form",
                "due_date": "2026-09-29",
                "deal_id": "d_1",
            },
        ),
    ]
    texts = ["$96,000 proposal", "114,000", "112,200", "2026-09-29", "136,200", "ok", "142,200"]
    good = score_agent_session(suite, task, _session_output(task, right, texts))
    assert good.passed, good.notes

    stray = ToolCall(name="update_deal", arguments={"deal_id": "d_9", "stage": "closed_lost"})
    bad = score_agent_session(suite, task, _session_output(task, [*right, stray], texts))
    assert not bad.passed and any("unrequested write deals:d_9" in n for n in bad.notes)

    wrong_math = [right[0].model_copy(update={"arguments": {"deal_id": "d_1", "amount": 114000}})]
    worse = score_agent_session(suite, task, _session_output(task, wrong_math + right[1:], texts))
    assert not worse.passed and worse.metrics["end_state"] < 1


class _TwoTurn(ModelAdapter):
    def __init__(self) -> None:
        super().__init__(ModelSpec(id="t", adapter="scripted", model="t"))

    def complete(self, request: ModelRequest) -> ModelResponse:
        last = request.messages[-1]
        if last.role == "user" and "Update" in last.content:
            return ModelResponse(
                tool_calls=[ToolCall(name="update_deal", arguments={"deal_id": "d_1", "amount": 1})]
            )
        return ModelResponse(text=f"answer to: {last.content[:20]}")


def test_multi_turn_loop_appends_followups_and_returns_state() -> None:
    suite = load_suite(ROOT / "suites", "c9_agent_sessions")
    task = next(t for t in suite.tasks if t.id == "c9-004-renewal-math-session")
    _, out = execute(_TwoTurn(), suite, task, rep=0)
    assert len(out.turn_texts) == 1 + len(task.inputs["followups"])
    assert out.final_state is not None
    d1 = next(d for d in out.final_state["deals"] if d["deal_id"] == "d_1")
    assert d1["amount"] == 1
    assert next(d for d in task.inputs["crm"]["deals"] if d["deal_id"] == "d_1")["amount"] == 96000


def test_datagen_is_deterministic_and_answer_follows_rules() -> None:
    spec = {"kind": "dedupe_contacts", "seed": 3, "n": 120}
    assert build_dataset(spec) == build_dataset(spec)
    text, exp = build_dataset(spec)
    rows = {line.split(",")[0]: line for line in text.splitlines()[1:]}
    for group in exp["groups"]["duplicate_groups"]:
        assert len(group) >= 2 and all(g in rows for g in group)


def test_data_ops_scores_exactness_and_fabricated_ids() -> None:
    task = Task(
        id="x",
        expected={
            "sets": {"missing": ["a1", "a2"]},
            "groups": {"dups": [["r1", "r2", "r3"]]},
            "values": {"totals": {"Priya Nair": 1000, "Sam Ortiz": 0}},
            "valid_ids": ["a1", "a2", "a3", "r1", "r2", "r3"],
        },
    )
    suite = load_suite(ROOT / "suites", "c10_data_ops")
    perfect = (
        '{"missing": ["a2","a1"], "dups": [["r3","r1","r2"]], '
        '"totals": {"Priya Nair": "1,000", "Sam Ortiz": 0}}'
    )
    assert score_data_ops(suite, task, RunOutput(text=perfect)).passed
    partial = '{"missing": ["a1","zz9"], "dups": [["r1","r2"]], "totals": {"Priya Nair": 1000}}'
    s = score_data_ops(suite, task, RunOutput(text=partial))
    assert (
        not s.passed and s.fabrication > 0 and s.metrics["dups"] < 1 and s.metrics["totals"] == 0.5
    )
