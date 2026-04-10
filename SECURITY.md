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

## Do Not Commit
- `.env`
- `alpha_results.db*`
- `results/`
- `.venv/`
- `*.log`, `*.zip`, temp artifacts

These are enforced by `.gitignore`, but still verify staged files before push.

## Incident Response (Credential Leak)
1. Revoke/rotate leaked credentials at source.
2. Remove leaked data from git history (if pushed).
3. Rotate all downstream tokens/sessions.
4. Re-run deploy checklist in `DEPLOY_PRIVATE_AUDIT.md`.

## Reporting
- For private-team issues, report directly to repository maintainers.
- Include: scope of exposure, suspected files, and commit hashes affected.
