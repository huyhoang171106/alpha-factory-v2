<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-10T20:45:00 | Updated: 2026-04-10T20:45:00 -->

# .github/workflows

## Purpose
GitHub Actions CI/CD workflow definitions for the private alpha factory repo. Each workflow is triggered by specific events and enforces quality/security gates before merging.

## Key Files

| File | Description |
|------|-------------|
| `alpha-burst.yml` | Weekday burst pipeline (UTC 13,15,17,19,21) — runs async engine, syncs submit, publishes KPI artifact; auto-skips if WQ secrets missing; cancels overlap |
| `ci.yml` | Push/PR to main — unit tests + CLI smoke; cancels in-progress duplicates; ignores docs-only changes |
| `nightly-health.yml` | Weekday nightly health snapshot — short timeout + cached deps; runs tests + KPI report |
| `security-audit.yml` | Scheduled security: secrets scan (gitleaks), dependency audit (pip-audit), SAST (bandit), CodeQL |

## For AI Agents

### Minute Budget
See `ACTIONS_BUDGET.md` for monthly quota management. Free tier = 2,000 min/month; Student/Pro = 3,000 min/month.

### Required Status Checks
Branch protection requires `CI` and `Security Audit` to pass before merge to `main`.

### Secrets
Never hardcode credentials. Use `secrets.WQ_EMAIL`, `secrets.WQ_PASSWORD`, `secrets.OPENROUTER_API_KEY`, `secrets.PUBLIC_STATUS_TOKEN`.

<!-- MANUAL: -->