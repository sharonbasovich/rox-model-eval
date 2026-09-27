# rox-model-eval

Scores frontier models on **Rox's own jobs** — not public leaderboards — and emits an
**ADOPT / ROUTE / HOLD** decision per capability and overall. Built for the question that comes up
at every model release: *should Rox switch to, route some traffic to, or ignore this model?*

Full design and rationale: [`DESIGN.md`](DESIGN.md).

## Capability suites

| Suite | What it checks | Scorer |
|---|---|---|
| `c1_research` | Account briefs from retrieved sources: key-signal coverage, every fact cites a real source, no numbers/events the sources lack, name-collision and stale-fact traps | deterministic + judge |
| `c2_drafting` | Outbound/follow-up email: word budget, personalisation, banned phrases (guarantees, competitor bashing), CTA, no invented stats | deterministic + judge |
| `c3_insights` | Rank deals/accounts/leads: NDCG vs labelled order, correct top pick, reason names the decisive signal | deterministic + judge |
| `c4_grounded_qa` | CRM Q&A: correct answer, valid record citations, **abstain** when records can't answer (answering = fabrication) | deterministic |
| `c5_tool_calling` | Multi-step agent loop on scripted CRM tools: tool choice, JSON-schema-valid args, ordered required calls, efficiency, clarify when ambiguous, no unrequested writes | deterministic |
| `c6_extraction` | Messy text → strict record: JSON/schema validity, field accuracy, fabricated fields | deterministic |
| `c7_long_context` | Generated 6k–20k-word call transcripts with planted facts at set depths and superseded values | deterministic |
| `c8_safety` | Indirect prompt injection in emails, scraped pages, CRM notes, CSVs; confidential-note leakage; injected tool actions; over-refusal on benign lookalikes | deterministic |

All data is synthetic. Each suite is a YAML file under `suites/`; tasks carry `inputs`, `expected`
(what the scorer checks) and a `reference_output` (what good looks like).

## How the decision is made

1. **Per attempt**: suite scorer → quality 0..1, format validity, fabrication share, safety violation.
   For rubric suites an optional LLM judge is blended in (40%) and must score ≥ 0.5 to pass.
2. **Per model × capability**: mean ± sd over reps, pass rate, cross-rep consistency, p50/p95 latency,
   TTFT, `$/task`, **`$/successful task`**, judge mean and pairwise win-rate vs the baseline.
3. **Gates** (`config/weights.yaml`): quality, fabrication, format and **safety (zero tolerance by
   default)**. Failing any gate means HOLD for that capability.
4. **Verdict vs baseline** (`baseline: true` in `models.yaml`): ADOPT if as good and cheaper per
   success, ROUTE if it clears gates but trails or costs more, HOLD otherwise.
5. **Overall**: weighted **Rox Fitness Score**, overall verdict (any safety failure means overall HOLD),
   a **routing table** (best-quality and best-value model per capability) and the cost/quality
   **Pareto frontier**.
6. **History**: every run goes to `runs/history.sqlite`; drops vs the previous run of the same model
   on an *unchanged* suite are flagged as regressions (`--fail-on-regression` for CI gating).

Artifacts per run: `runs/<run-id>-<label>/{scorecard.html, scorecard.md, summary.json, attempts.jsonl}`.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# Offline, zero API keys (simulator exercises the whole pipeline, incl. a simulated judge):
python -m rox_model_eval run --models mock-strong,mock-cheap --suite all --reps 3 --judge mock-strong
python -m rox_model_eval calibrate --judge mock-strong
python -m rox_model_eval list
python -m rox_model_eval history
```

`mock-*` rows come from an offline simulator that starts from the reference outputs and injects
failures on purpose (dropped/fabricated fields, invented stats, skipped tool calls, obeying
injections). They are **not** measurements of any model; the scorecard says so.

## Running real models (e.g. OpenAI)

```bash
export OPENAI_API_KEY=...
# 1. uncomment gpt-frontier / gpt-frontier-mini in config/models.yaml (check ids + prices)
# 2. mark the model Rox runs in production today as `baseline: true`
# 3. calibrate the judge before trusting judge-blended scores
python -m rox_model_eval calibrate --judge gpt-frontier
python -m rox_model_eval run --models gpt-frontier,gpt-frontier-mini --suite all --reps 3 --judge gpt-frontier
```

Any OpenAI-compatible endpoint works (`adapter: openai` + `base_url`), including a Rox model gateway,
which measures models exactly as Rox calls them. Anthropic uses `adapter: anthropic`.

A judge from the same family as a candidate can favour it; for decisions prefer a judge from a
different provider, or rely on the deterministic columns.

## Evaluating a newly released model

1. Add an entry to `config/models.yaml` (`adapter`, `model`, `api_key_env`, current prices).
2. Run it alongside the baseline: `--models <baseline-id>,<new-id> --suite all`.
3. Read the overall verdict, routing table and "worst failures" receipts in `scorecard.html`.

## Adding tasks or suites

- Add tasks to an existing `suites/<name>/suite.yaml`; `pytest` checks that every task's
  `reference_output` passes its own scorer, so broken labels fail CI.
- Replace `calibration/labels.yaml` with labels from Rox reviewers before relying on the judge.

## Development

```bash
ruff check . && ruff format --check . && mypy rox_model_eval && pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same checks plus an offline end-to-end eval and uploads
the scorecard as an artifact.
