# GitHub Actions Budget Plan (Private Repo)

## Goal
Use private Actions quota efficiently under both:
- `Free`: 2,000 minutes/month
- `Student (GitHub Pro)`: 3,000 minutes/month
while preserving operational coverage.

## Current Strategy

- `ci.yml`
  - Runs on push/PR to `main/master`.
  - Ignores docs-only changes.
  - Cancels in-progress duplicate runs per ref.
- `alpha-burst.yml`
  - Scheduled only on weekdays at UTC `13,15,17,19,21` (5 runs/day).
  - Auto-skips when `WQ_EMAIL/WQ_PASSWORD` secrets are missing.
  - Cancels overlapping runs.
  - Manual dispatch supports runtime inputs (`burst_limit`, `pre_rank_score`, `sync_limit`).
- `nightly-health.yml`
  - Weekday nightly health snapshot.
  - Short timeout and cached dependencies.

## Why This Uses Minutes Better

- Removes high-waste `*/5` cron behavior.
- Prevents overlap storms via `concurrency`.
- Cuts redundant CI via `paths-ignore`.
- Uses pip cache to reduce setup time.

## Suggested Monthly Budget Envelope

### Mode A: Free (2,000 min/month)

- CI (code changes): variable, target `< 600 min`.
- Burst schedule (weekday): target `< 1,000 min`.
- Nightly health (weekday): target `< 200 min`.
- Reserve buffer: `~200 min` for manual runs/hotfixes.

If usage approaches quota:
1. Reduce burst windows (e.g., 3/day instead of 5/day).
2. Lower `GHA_BURST_LIMIT`.
3. Temporarily disable nightly.

### Mode B: Student / Pro (3,000 min/month)

- CI (code changes): target `< 800 min`.
- Burst schedule (weekday): target `< 1,600 min`.
- Nightly health (weekday): target `< 250 min`.
- Reserve buffer: `~350 min` for incident/manual recovery runs.

When moving from Free -> Student:
1. Keep current workflow settings for 1 week.
2. Increase burst intensity gradually:
   - First increase `GHA_BURST_LIMIT` (example `60 -> 80`).
   - Then optionally add one more burst window per weekday.
3. Track real minute burn in Billing before next increase.

## Student Plan / Upgrade Path

When upgrading account quota:
- Increase burst frequency gradually.
- Keep `concurrency` and cache settings unchanged.
- Re-evaluate timeout + limits after 1 week of metrics.

If Student expires and account falls back to Free:
- revert to Mode A budget immediately,
- reduce burst limit and/or schedule windows,
- keep CI and nightly guardrails unchanged.

## Self-Hosted Option

If moving heavy jobs to self-host:
- Keep CI smoke on GitHub-hosted runners.
- Shift burst workloads to self-hosted labels.
- Maintain the same CLI contract (`auto --profile gha` or dedicated self-host profile).
