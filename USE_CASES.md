# Use Cases

This file captures practical execution scenarios so deployment/docs do not miss behavioral context.

## Use Case 1: Continuous Discovery + Submission

Goal:
- Generate and evaluate continuously, submit quality alphas, and keep review states synchronized.

Commands:
- `python alpha_factory_cli.py async --limit 0 --score 50`
- `python alpha_factory_cli.py sync-submit --limit 30`
- `python alpha_factory_cli.py kpi --minutes 60`
- One-click equivalent: `python alpha_factory_cli.py auto --profile local`
  - Opt-in hypothesis mode: set `GENERATOR_MODE=hypothesis_driven`

Expected behavior:
- Rank/gate before simulation.
- Submit queue processed with retry policy.
- `submitted` alphas reconciled to `accepted/rejected` over time.

## Use Case 2: Recovery After API Instability

Goal:
- Recover safely after 429/5xx/network instability.

Commands:
- `python alpha_factory_cli.py replay-dlq --limit 50`
- `python alpha_factory_cli.py sync-submit --limit 30`

Expected behavior:
- Retry with capped exponential backoff.
- Polling backoff includes jitter to avoid synchronized spikes.
- Non-retryable semantic errors remain dead-lettered.

## Use Case 3: Portable Machine Migration

Goal:
- Move project to another Windows machine and run quickly.

Commands:
- `python alpha_factory_cli.py zip`
- unzip on target machine
- `python alpha_factory_cli.py setup`
- `python alpha_factory_cli.py start`

Expected behavior:
- venv and dependencies recreated.
- `.env` initialized from `.env.example`.
- Runtime artifacts remain local and uncommitted.

## Use Case 4: Private GitHub Deployment

Goal:
- Push a clean private repo without missing core context.

Checklist:
- Read `README.md`, `CODEX_HANDOFF_SCOPE.md`, `DEPLOY_PRIVATE_AUDIT.md`.
- Verify tests pass locally.
- Confirm no secrets or runtime artifacts are staged.

Expected behavior:
- PR includes full architecture and operational semantics.
- Submit/review/KPI definitions are consistent with implementation.

## Use Case 6: VPS 24/7 Engine + GH Sidecar

Goal:
- Keep engine alive 24/7 on VPS, while GH runs lightweight checkpoints.

Commands:
- VPS: `python alpha_factory_cli.py auto --profile vps`
- GH: workflow `.github/workflows/alpha-burst.yml` (5-minute schedule)

Expected behavior:
- VPS owns main simulate/submit loop.
- GH performs burst + sync-submit + KPI artifacts for visibility.

## Use Case 7: Local Hybrid Anti-Crash Runner

Goal:
- Keep local engine running continuously without duplicate instances.

Commands:
- `python alpha_factory_cli.py auto --profile local`
- Optional: `python alpha_factory_cli.py auto --profile local --no-hybrid`

Expected behavior:
- Singleton lock prevents duplicate local supervisors.
- Hybrid loop periodically runs `sync-submit` and KPI checks.
- Engine auto-restarts with backoff if subprocess exits.

## Use Case 5: KPI-Based Operational Monitoring

Goal:
- Distinguish transport-level submit health from true acceptance quality.

Command:
- `python alpha_factory_cli.py kpi --minutes 120`

Interpretation:
- `submit_success_rate`: queue transport health.
- `submit_ok_rate`: queue items that reached submit workflow outcomes.
- `true_accept_rate`: quality of reviewed submissions.
- `dlq_rate`: execution reliability signal.

