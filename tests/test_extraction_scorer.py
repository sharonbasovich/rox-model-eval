import json

from rox_model_eval.scorers import score_record

SCHEMA = {
    "type": "object",
    "required": ["name", "employees"],
    "properties": {
        "name": {"type": ["string", "null"]},
        "employees": {"type": ["integer", "null"]},
    },
}


def test_perfect_match_passes_with_normalization() -> None:
    out = json.dumps({"name": "  ACME Corp ", "employees": 12.0})
    s = score_record(out, {"name": "acme corp", "employees": 12}, SCHEMA)
    assert s.passed and s.metrics["field_accuracy"] == 1.0 and s.fabrication == 0.0


def test_fabricated_value_where_source_has_none_is_flagged() -> None:
    out = json.dumps({"name": "Acme", "employees": 500})
    s = score_record(out, {"name": "Acme", "employees": None}, SCHEMA)
    assert not s.passed
    assert s.fabrication == 1.0
    assert any("fabricated employees" in n for n in s.notes)


def test_malformed_json_scores_zero() -> None:
    s = score_record('{"name": "Acme"', {"name": "Acme"}, SCHEMA)
    assert not s.passed and s.score == 0.0 and not s.format_valid and s.metrics["json_valid"] == 0.0


def test_code_fenced_json_is_accepted() -> None:
    out = '```json\n{"name": "Acme", "employees": 3}\n```'
    assert score_record(out, {"name": "Acme", "employees": 3}, SCHEMA).passed


def test_schema_violation_halves_score() -> None:
    out = json.dumps({"name": "Acme", "employees": "three"})
    s = score_record(out, {"name": "Acme", "employees": None}, SCHEMA)
    assert s.metrics["schema_valid"] == 0.0 and not s.passed
    assert s.score == 0.25
