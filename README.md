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
- `generator.py` supports opt-in `GENERATOR_MODE=hypothesis_driven` for hypothesis-first generation.

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
  - Local (hybrid supervisor + singleton lock): `python alpha_factory_cli.py auto --profile local`
    - disable hybrid wrapper: `python alpha_factory_cli.py auto --profile local --no-hybrid`
  - VPS: `python alpha_factory_cli.py auto --profile vps`
  - GitHub burst profile: `python alpha_factory_cli.py auto --profile gha`
- Global command mode (run from any directory):
  - install once: `python alpha_factory_cli.py install-global --name alpha`
  - then run: `alpha --help`
  - one-click local run: `alpha --yolo`

## CLI Commands

- `setup`: bootstrap virtualenv and dependencies
- `start`: run continuous daily pipeline
- `async`: run async streaming pipeline
- `auto`: one-click profile runner (`local|vps|gha`)
- `replay-dlq`: requeue dead-lettered submit jobs
- `sync-submit`: poll WQ review status for submitted alphas
- `public-report`: export sanitized KPI JSON for external sharing
- `install-global`: install system-wide command wrapper (`alpha`)
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
- `GENERATOR_MODE` (`legacy` or `hypothesis_driven`)
- `GHA_BURST_LIMIT`
- `GHA_PRE_RANK_SCORE`
- `GHA_SYNC_LIMIT`
- `LOCAL_HYBRID_SYNC_INTERVAL`
- `LOCAL_HYBRID_KPI_INTERVAL`
- `LOCAL_HYBRID_SYNC_LIMIT`
- `LOCAL_HYBRID_RESTART_BACKOFF`
- `LOCAL_HYBRID_SCORE`
- `LOCAL_SINGLETON_LOCKFILE`

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
- `HANDOVER_DETAILED.md`
- `OPTIMIZATION_NEXT_TASKS.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `SECURITY.md`
- `ACTIONS_BUDGET.md`
- `TEAM_SHARING.md`

## Agent Onboarding (Read First)

For any coding agent/new contributor, read in this order before changing runtime logic:
1. `README.md` (this file)
2. `HANDOVER_DETAILED.md`
3. `CODEX_HANDOFF_SCOPE.md`
4. `USE_CASES.md`
5. `SECURITY.md`
6. `TEAM_SHARING.md`
7. `OPTIMIZATION_NEXT_TASKS.md`

Required constraints for agents:
- Keep repo private and never output secrets.
- Never publish alpha expressions, alpha IDs, or raw WQ payloads.
- Preserve submit semantics: `submitted` != `accepted`.
- Keep CLI contract backward-compatible unless explicitly approved.

## Automation Targets

GitHub Actions workflows:
- `.github/workflows/alpha-burst.yml` (weekday burst windows + sync + KPI/public artifact)
- `.github/workflows/nightly-health.yml` (nightly tests + KPI snapshot)
- `.github/workflows/ci.yml` (push/PR unit tests + CLI smoke)
- `.github/workflows/security-audit.yml` (secrets scan + SAST + dependency audit + CodeQL)

VPS service files:
- `scripts/run_vps.sh`
- `deploy/alpha-factory.service`

Windows autostart files:
- `scripts/windows_auto_runner.ps1`
- `scripts/install_windows_autostart.ps1`
- `scripts/uninstall_windows_autostart.ps1`

## Team Collaboration (Private Repo)

- Enforce protected `main`: require PR, required status checks (`CI`, `Security Audit`), and no force push.
- Use `.github/CODEOWNERS` to require maintainer review on runtime/governance/workflow files.
- Keep prod credentials only in GitHub Actions `Secrets` (never in repository files).

## Public Showcase Without Leaking IP

- Generate only sanitized metrics (no alpha expressions/signatures):
  - `python alpha_factory_cli.py public-report --minutes 60 --out results/public_report.json`
- Optional mirror to a public profile repo from `alpha-burst.yml` using:
  - `PUBLIC_STATUS_REPO` (e.g. `username/quant-status`)
  - `PUBLIC_STATUS_TOKEN` (fine-grained PAT with repo-content write on that public repo only)
- Safe to publish: throughput, acceptance, DLQ and QD aggregate numbers.
- Never publish: expressions, candidate payloads, credentials, raw API responses.

