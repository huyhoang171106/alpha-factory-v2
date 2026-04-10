# Alpha Factory Detailed Handover

## 1) Current Production Scope

- Runtime entrypoint:
  - `python alpha_factory_cli.py auto --profile local|vps|gha --skip-install`
- Continuous engine:
  - `run_async_pipeline.py` (producer -> ranker -> simulator -> gate -> submit queue)
- Governance + submit lifecycle:
  - `tracker.py` + `submit_governor.py` + `wq_client.py`
- Search + quality:
  - `alpha_ranker.py`, `quality_diversity.py`, `budget_allocator.py`

## 2) Confirmed State Machine

- Submit states:
  - `new -> gated|rejected|queued`
  - `gated -> queued|rejected`
  - `queued -> submitted|failed|dead_lettered`
  - `failed -> queued|dead_lettered|rejected`
  - `submitted -> accepted|rejected`
  - `dead_lettered -> queued` (replay path)
- Final states:
  - `accepted`, `rejected`, `dead_lettered`

## 3) KPI Semantics (Important)

- `submitted` = transport/API submit success, not final WQ review decision.
- `accepted`/`rejected` = only after reconciliation polling from WQ.
- Primary minute KPIs are in `tracker.minute_kpis(...)`.
- Public-safe KPI report is available via:
  - `python alpha_factory_cli.py public-report --minutes 60 --out results/public_report.json`

## 4) What Is Already Hardened

- Retry policy by error class with DLQ fallback.
- Replay support from DLQ.
- Async pipeline with queue-based flow.
- CI smoke + unit tests.
- Security workflow:
  - secrets scan (`gitleaks`)
  - dependency audit (`pip-audit`)
  - static analysis (`bandit`)
  - CodeQL
- Dependabot config present.
- CODEOWNERS file present.

## 5) GitHub / Deploy Status

- Repo: private.
- Push and release flow already working.
- Branch protection for private repo is plan-dependent:
  - requires GitHub Pro/Student on personal account.
- Dependabot alerts and automated security fixes are enabled.

## 6) Team Collaboration Guardrails

- Keep `.env` local only.
- Store automation credentials in GitHub Secrets.
- Never publish raw expressions or alpha IDs to public channels.
- Use `TEAM_SHARING.md` as operational policy.

## 7) Local Runbook (Windows)

- Start manually:
  - `python alpha_factory_cli.py auto --profile local --skip-install`
- Sync review decisions:
  - `python alpha_factory_cli.py sync-submit --limit 40`
- Replay DLQ:
  - `python alpha_factory_cli.py replay-dlq --limit 50`
- Observe KPI:
  - `python alpha_factory_cli.py kpi --minutes 60`

Autostart service-like mode (Task Scheduler):
- Install + start:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_windows_autostart.ps1`
- Remove:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/uninstall_windows_autostart.ps1`
- Runtime logs:
  - `results/windows_auto_runner.log`
  - `results/windows_auto_runner.err.log`

## 8) VPS Migration Hand-off (Next Step)

- Keep CLI contract unchanged.
- Move long-running job to systemd (`deploy/alpha-factory.service`).
- Keep GitHub Actions for CI/security + lightweight burst only.
- Export sanitized public metrics from either VPS cron or GH artifact sync.

## 9) Known Constraints

- GH-hosted Actions is not unlimited.
- WQ endpoints can rate-limit; submit/review backoff must remain active.
- CPU bottlenecks can occur in ranker/collinearity if worker settings are too high for machine.

## 10) Minimal Acceptance Before “Stable Production”

- Unit tests pass.
- `auto --profile local` runs continuously without duplicate process storms.
- `sync-submit` updates `submitted -> accepted/rejected`.
- `public-report` output contains no sensitive keys/expressions.
