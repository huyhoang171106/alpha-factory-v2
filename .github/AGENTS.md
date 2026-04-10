<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-10T20:45:00 | Updated: 2026-04-10T20:45:00 -->

# .github

## Purpose
GitHub repository configuration: CI/CD workflows, CODEOWNERS, PR template, dependabot config.

## Key Files

| File | Description |
|------|-------------|
| `CODEOWNERS` | Require maintainer review on runtime/governance/workflow files |
| `PULL_REQUEST_TEMPLATE.md` | PR checklist before merging |
| `dependabot.yml` | Automatic dependency update schedule |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `workflows/` | GitHub Actions pipeline definitions (see `workflows/AGENTS.md`) |

## For AI Agents

### Workflow Review
Before adding/modifying workflows, check `ACTIONS_BUDGET.md` to understand monthly Actions minute budget constraints.

### Security
Never commit secrets in workflow files. Use `secrets.WQ_EMAIL`, `secrets.WQ_PASSWORD`, etc.

<!-- MANUAL: -->
