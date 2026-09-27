# rox-model-eval

Scores frontier models on **Rox's own jobs** — not public leaderboards — and emits an
**ADOPT / ROUTE / HOLD** decision per capability. Built for the question that comes up at every
model release: *should Rox switch to, route to, or ignore this model?*

Full design, roadmap and rationale: [`DESIGN.md`](DESIGN.md).

## What's here (Phase 0)

- **Provider-agnostic adapters**: `openai` (any OpenAI-compatible endpoint, including a Rox gateway),
  `anthropic`, and `mock` (offline simulator). A new model is a `config/models.yaml` entry with its
  pricing — no code change.
- **C6 structured-extraction suite**: 10 synthetic, messy inputs (signatures, CSV rows, forwarded
  threads, a prompt-injection-in-data case). Several tasks deliberately omit fields, so any non-null
  value there counts as **fabrication**.
- **Scoring**: JSON validity, JSON-schema validity, per-field accuracy, fabrication rate.
- **Economics**: provider-reported token usage (incl. cached), p50/p95 latency, streamed
  time-to-first-token, `$/task` and **`$/successful task`**.
- **Recommender**: quality + fabrication gates, then comparison against the `baseline: true` model
  (current production default).
- **Artifacts** per run: `runs/<ts>-<capability>/{attempts.jsonl, summary.json, scorecard.md}`.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# Offline, zero API keys (simulator exercises the pipeline):
python -m rox_model_eval run --models mock-strong,mock-cheap --suite c6_extraction --reps 3

# Real models: uncomment entries in config/models.yaml, export keys, then e.g.
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...
python -m rox_model_eval run --models gpt-frontier,claude-frontier --suite c6_extraction
```

`mock-*` rows come from an offline simulator that derives answers from the reference labels and
injects failures on purpose. They are **not** measurements of any model — they only test the harness.

## Evaluating a newly released model

1. Add an entry to `config/models.yaml` (`adapter`, `model`, `api_key_env`, current prices).
2. Run it alongside the baseline: `--models <baseline-id>,<new-id>`.
3. Read the verdict and the "worst failures" receipts in `scorecard.md`.

## Development

```bash
ruff check . && ruff format --check . && mypy rox_model_eval && pytest -q
```

## Roadmap

Next: C1 research, C4 grounded Q&A, C5 tool calling; then LLM judge with human calibration, C8
injection suite, HTML scorecard + Pareto, and regression tracking. See `DESIGN.md` §9.
