# Alpha Factory

Production-oriented alpha generation pipeline for WorldQuant Brain:
- continuous candidate generation,
- async simulation and submit orchestration,
- resilient submit retry + DLQ,
- post-submit review reconciliation,
- quality-diversity search and budget economy.

## Current System Status

This repository is beyond baseline scripting and now runs with:
- Async pipeline workers (`producer -> ranker -> simulator`) in `run_async_pipeline.py`.
- Two-tier candidate budget gate in `budget_allocator.py`.
- Continuous novelty + QD archive in `quality_diversity.py` with DB persistence.
- Submit state machine in `tracker.py` and retry/DLQ policies in `submit_governor.py`.
- Review-sync path (`submitted -> accepted/rejected`) using WQ status polling.
- Minute KPIs with separate submit transport vs true acceptance rates.

## Architecture

Core runtime:
- `alpha_factory_cli.py`: one-entry CLI for setup/run/test/kpi/reconciliation.
- `run_async_pipeline.py`: high-throughput async runtime.
- `run_daily.py` + `pipeline.py`: batch/continuous daily flow.

Search and allocation:
- `alpha_ranker.py`: pre-sim ranking heuristics.
- `quality_diversity.py`: descriptor-based archive + novelty scoring.
- `budget_allocator.py`: Tier-1 cheap gate + Tier-2 expected value.
- `generator.py`, `pattern_lab.py`, `alpha_policy.py`, `alpha_ast.py`.

Execution and governance:
- `wq_client.py`: auth/sim/submit/status API integration.
- `submit_governor.py`: queue dispatch, retry, review reconcile.
- `tracker.py`: SQLite schema, state transitions, KPI queries, replay DLQ.

## Submit Lifecycle

Closed lifecycle in DB:
`new -> gated -> queued -> submitted -> accepted|rejected`
and failure path:
`queued|failed -> dead_lettered -> replay -> queued`

Important semantic:
- Submit endpoint 2xx means `submitted`.
- `accepted` is only set by review reconciliation polling.

## KPIs

`python alpha_factory_cli.py kpi --minutes 60`

Primary metrics:
- `gate_pass_rate`: generated -> gated
- `submit_success_rate`: queued -> submitted (transport success)
- `submit_ok_rate`: queued -> (submitted + accepted + rejected_after_submit)
- `true_accept_rate`: accepted / (accepted + rejected_after_submit + submitted_pending_decided)
- `dlq_rate`: dead_lettered / queued

## Quickstart

1) Setup:
- `python alpha_factory_cli.py setup`

2) Fill `.env` from `.env.example`:
- required: `WQ_EMAIL`, `WQ_PASSWORD`
- optional: `OPENROUTER_API_KEY`

3) Run:
- Async mode: `python alpha_factory_cli.py async --limit 0 --score 50`
- Daily mode: `python alpha_factory_cli.py start`
- One-click auto profile:
  - Local: `python alpha_factory_cli.py auto --profile local`
  - VPS: `python alpha_factory_cli.py auto --profile vps`
  - GitHub burst profile: `python alpha_factory_cli.py auto --profile gha`

## CLI Commands

- `setup`: bootstrap virtualenv and dependencies
- `start`: run continuous daily pipeline
- `async`: run async streaming pipeline
- `auto`: one-click profile runner (`local|vps|gha`)
- `replay-dlq`: requeue dead-lettered submit jobs
- `sync-submit`: poll WQ review status for submitted alphas
- `kpi`: print minute-level pipeline + QD stats
- `test`: run unit tests
- `zip`: generate portable zip package

## Environment Knobs

Runtime:
- `ASYNC_USE_RAG`
- `ASYNC_RANKER_WORKERS`
- `ASYNC_SIMULATOR_WORKERS`
- `ASYNC_NOVELTY_MIN`
- `ASYNC_TIER1_MIN_QUALITY`
- `ASYNC_TIER2_MIN_EV`
- `ASYNC_REVIEW_JITTER_RATIO`
- `GHA_BURST_LIMIT`
- `GHA_PRE_RANK_SCORE`
- `GHA_SYNC_LIMIT`

WQ client:
- `WQ_MAX_CONCURRENT`
- `WQ_POLL_INTERVAL`
- `WQ_MAX_WAIT_TIME`
- `WQ_INTERACTIVE_AUTH`

## Testing

Run all tests:
- `python -m unittest discover -s tests -v`

Current suite covers:
- budget allocator,
- QD novelty/archive,
- async gate logic,
- submit governor retry/reconcile,
- tracker lifecycle/DLQ/KPI math,
- WQ submission status parser.

## Deployment Scope

Before pushing to private GitHub:
- Include source, docs, tests, `.env.example`, `requirements.txt`.
- Exclude secrets/runtime artifacts: `.env`, `.venv`, `results/`, `alpha_results.db*`, `*.log`, `*.zip`.

Detailed checklist:
- `DEPLOY_PRIVATE_AUDIT.md`
- `CODEX_HANDOFF_SCOPE.md`
- `USE_CASES.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `SECURITY.md`
- `ACTIONS_BUDGET.md`

## Automation Targets

GitHub Actions workflows:
- `.github/workflows/alpha-burst.yml` (every 5 minutes burst + sync + KPI artifact)
- `.github/workflows/nightly-health.yml` (nightly tests + KPI snapshot)
- `.github/workflows/ci.yml` (push/PR unit tests + CLI smoke)

VPS service files:
- `scripts/run_vps.sh`
- `deploy/alpha-factory.service`

