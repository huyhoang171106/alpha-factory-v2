# Alpha Factory Private Deploy Audit
Date: 2026-04-10

## 1) Scope Audit (What to publish)
Recommended publish scope for private GitHub:

- Core runtime:
  - `run_async_pipeline.py`
  - `run_daily.py`
  - `pipeline.py`
  - `alpha_factory_cli.py`
- Strategy/search/budget:
  - `generator.py`
  - `alpha_ranker.py`
  - `quality_diversity.py`
  - `budget_allocator.py`
  - `alpha_policy.py`
  - `pattern_lab.py`
  - `alpha_dna.py`
  - `alpha_seeds.py`
  - `alpha_candidate.py`
  - `alpha_ast.py`
- Execution/governance:
  - `tracker.py`
  - `submit_governor.py`
  - `wq_client.py`
  - `validator.py`
  - `community_harvester.py`
  - `alpha_rag.py`
- Docs/ops:
  - `PORTABLE_SETUP.md`
  - `update.md`
  - `DEPLOY_PRIVATE_AUDIT.md`
  - `requirements.txt`
  - `.env.example`
  - `.gitignore`
  - `tests/`

## 2) Scope Audit (What must NOT be committed)
- Secrets:
  - `.env` (contains real credentials now)
  - any `credentials.json`
- Runtime data:
  - `alpha_results.db`, `alpha_results.db-wal`, `alpha_results.db-shm`
  - `results/` outputs
  - log files `*.log`
- Local build/runtime:
  - `.venv/`
  - `__pycache__/`, `*.pyc`
- Local artifacts:
  - temporary zips and scratch files

`.gitignore` has been tightened to include:
- `*.db-wal`, `*.db-shm`, `*.zip`, `*.tmp`, `*.bak`

## 3) Feature State (Current, real usage)
This reflects the current codebase behavior.

- Async streaming pipeline:
  - Producer -> Ranker -> Simulator workers
  - `SIMULATOR_WORKERS` is wired and active from env
- Search quality/diversity:
  - Continuous novelty score (signature + token distance + rarity)
  - QD archive with elite cells persisted in DB (`qd_archive`)
- Budget allocation:
  - Two-tier gate (`BudgetAllocator`):
    - Tier-1 quality+novelty gate
    - Tier-2 expected-value gate (Thompson-style arm sampling)
- Governance/state machine:
  - Closed submit transition rules in tracker
  - Submit jobs table + DLQ + replay
- Submit resilience:
  - Submit retry policy by error class
  - Dead-letter for non-retryable classes
- Post-submit review semantics:
  - Submit API success marks `submitted` only
  - `accepted/rejected` handled by review reconciliation path
- KPI:
  - Minute KPIs include gate, submit, DLQ
  - Separate `submit_ok_rate` and `true_accept_rate`

## 4) Tests / Readiness
- Current unit test suite: passing (`24/24`).
- Test coverage includes:
  - Async gate logic
  - Budget allocator
  - QD novelty/archive
  - Submit governor retry/reconcile flow
  - Tracker state transitions + DLQ + KPI math
  - WQ submission status parsing

## 5) Updated Deploy Plan (Private GitHub)
Use this sequence before first push.

1. Security preflight:
   - Rotate `WQ_EMAIL/WQ_PASSWORD` because local `.env` currently has real credentials.
   - Keep `.env` local only; never commit.
2. Clean staged scope:
   - Commit only source/docs/tests from Section 1.
   - Ensure no db/log/results files are staged.
3. Local verification:
   - `python -m unittest discover -s tests -v`
   - `python alpha_factory_cli.py --help`
4. Initialize/push private repo:
   - `git init`
   - `git add .`
   - `git commit -m "alpha-factory: async qd budget submit governance baseline"`
   - Create private GitHub repo
   - `git remote add origin <PRIVATE_REPO_URL>`
   - `git branch -M main`
   - `git push -u origin main`
5. Post-push checklist:
   - Add branch protection on `main`
   - Add required checks: unit tests
   - Add repo secrets in GitHub (if CI/deploy is added later)

## 6) Immediate Next Work (for your “project thật cháy” phase)
- Finalize review polling scheduler in runtime loop (if not already enabled in your branch).
- Add acceptance SLA dashboard panel (`submitted_pending`, `accepted`, `rejected_after_submit`).
- Add arm-level submit quota scheduler for minute budget allocation.
- Add CI workflow (`pytest/unittest`, lint) for private repo guardrails.
