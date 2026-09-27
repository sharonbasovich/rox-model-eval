"""SQLite run history and regression detection.

Every run is recorded per (model, capability) together with the suite content
hash. A regression is only flagged against a previous run of the same model on
an *unchanged* suite, so editing a suite never looks like a model getting worse.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel

from .aggregate import ModelSummary
from .config import Weights

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    model_id TEXT NOT NULL,
    capability TEXT NOT NULL,
    suite_hash TEXT NOT NULL,
    mean_score REAL NOT NULL,
    pass_rate REAL NOT NULL,
    fabrication_rate REAL NOT NULL,
    safety_violation_rate REAL NOT NULL,
    p95_latency_s REAL NOT NULL,
    cost_per_success_usd REAL,
    verdict TEXT NOT NULL,
    PRIMARY KEY (run_id, model_id, capability)
);
"""


class Regression(BaseModel):
    model_id: str
    capability: str
    metric: str
    previous: float
    current: float
    previous_run: str


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(_SCHEMA)
    return conn


def record(
    db: Path, run_id: str, ts: str, summaries: list[ModelSummary], hashes: dict[str, str]
) -> None:
    with _connect(db) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    run_id,
                    ts,
                    s.model_id,
                    s.capability,
                    hashes.get(s.capability, ""),
                    s.mean_score,
                    s.pass_rate,
                    s.fabrication_rate,
                    s.safety_violation_rate,
                    s.p95_latency_s,
                    s.cost_per_success_usd,
                    s.verdict,
                )
                for s in summaries
            ],
        )
    conn.close()


def detect(
    db: Path,
    run_id: str,
    summaries: list[ModelSummary],
    hashes: dict[str, str],
    weights: Weights,
) -> list[Regression]:
    if not db.exists():
        return []
    conn = _connect(db)
    found: list[Regression] = []
    for s in summaries:
        row = conn.execute(
            "SELECT run_id, mean_score, fabrication_rate, safety_violation_rate, p95_latency_s "
            "FROM results WHERE model_id=? AND capability=? AND suite_hash=? AND run_id<>? "
            "ORDER BY ts DESC LIMIT 1",
            (s.model_id, s.capability, hashes.get(s.capability, ""), run_id),
        ).fetchone()
        if row is None:
            continue
        prev_run, score, fab, safety, p95 = row

        checks = [
            (
                "mean_score",
                score,
                s.mean_score,
                s.mean_score < score - weights.regression_score_drop,
            ),
            (
                "fabrication_rate",
                fab,
                s.fabrication_rate,
                s.fabrication_rate > fab + weights.regression_fabrication_rise,
            ),
            (
                "safety_violation_rate",
                safety,
                s.safety_violation_rate,
                s.safety_violation_rate > safety,
            ),
            (
                "p95_latency_s",
                p95,
                s.p95_latency_s,
                p95 > 0 and s.p95_latency_s > p95 * weights.regression_latency_ratio,
            ),
        ]
        found += [
            Regression(
                model_id=s.model_id,
                capability=s.capability,
                metric=metric,
                previous=round(before, 4),
                current=round(after, 4),
                previous_run=prev_run,
            )
            for metric, before, after, hit in checks
            if hit
        ]
    conn.close()
    return found


def recent(db: Path, limit: int = 20) -> list[tuple[str, str, str, float, str]]:
    if not db.exists():
        return []
    conn = _connect(db)
    rows = conn.execute(
        "SELECT run_id, model_id, capability, mean_score, verdict FROM results "
        "ORDER BY ts DESC, model_id, capability LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [(str(a), str(b), str(c), float(d), str(e)) for a, b, c, d, e in rows]
