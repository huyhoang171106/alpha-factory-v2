# Security Policy

## Supported Deployment Model
- This project is intended for **private repositories**.
- Runtime credentials are local-only and must never be committed.

## Secrets Handling
- Keep `.env` local; do not push it.
- Rotate credentials immediately if `.env` is exposed.
- Required secrets:
  - `WQ_EMAIL`
  - `WQ_PASSWORD`
- Optional secret:
  - `OPENROUTER_API_KEY`
  - `PUBLIC_STATUS_TOKEN` (optional, for sanitized public metrics mirror only)

## Do Not Commit
- `.env`
- `alpha_results.db*`
- `results/`
- `.venv/`
- `*.log`, `*.zip`, temp artifacts

These are enforced by `.gitignore`, but still verify staged files before push.

## Repository Hardening Baseline
- Enable branch protection on `main`:
  - Require pull request
  - Require status checks (`CI`, `Security Audit`)
  - Block force push and branch deletion
- Require reviews for CODEOWNER files (`.github/CODEOWNERS`).
- Keep Actions permissions minimal (`contents: read` by default).
- Run scheduled security workflow:
  - secrets scan (`gitleaks`)
  - dependency audit (`pip-audit`)
  - static analysis (`bandit`, `CodeQL`)

## Team Sharing Rules
- Use least privilege:
  - maintainers: write/admin
  - contributors: triage/read unless explicitly needed
- Split secrets by environment (`dev`, `prod`) in GitHub Environments.
- Never share raw `.env` over chat tools. Use password manager or GitHub Secrets only.

## Public Profile Strategy (No IP Leak)
- Publish only sanitized report output (`public_report.json`) from `public-report`.
- Allowed public fields: aggregate KPI/QD counts and rates only.
- Forbidden public fields: alpha expressions, alpha IDs, URLs, raw error payloads, credentials.

## Incident Response (Credential Leak)
1. Revoke/rotate leaked credentials at source.
2. Remove leaked data from git history (if pushed).
3. Rotate all downstream tokens/sessions.
4. Re-run deploy checklist in `DEPLOY_PRIVATE_AUDIT.md`.

## Reporting
- For private-team issues, report directly to repository maintainers.
- Include: scope of exposure, suspected files, and commit hashes affected.
