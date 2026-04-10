# Codex Handoff Scope (GitHub Deploy)

This document is the anti-miss handoff contract for deployment work.

## 1) In-Scope For Deploy PR

- Runtime:
  - `alpha_factory_cli.py`
  - `run_async_pipeline.py`
  - `run_daily.py`
  - `pipeline.py`
- Search and quality:
  - `alpha_ranker.py`, `quality_diversity.py`, `budget_allocator.py`
  - `generator.py`, `alpha_policy.py`, `pattern_lab.py`, `alpha_ast.py`
- Governance and submission:
  - `tracker.py`, `submit_governor.py`, `wq_client.py`
- Docs and tests:
  - `README.md`, `PORTABLE_SETUP.md`, `DEPLOY_PRIVATE_AUDIT.md`, `USE_CASES.md`
  - `tests/`
  - `requirements.txt`, `.env.example`, `.gitignore`
  - `.github/workflows/alpha-burst.yml`
  - `.github/workflows/nightly-health.yml`
  - `scripts/run_vps.sh`
  - `deploy/alpha-factory.service`

## 2) Explicitly Out-Of-Scope (Must Not Commit)

- `.env`
- `alpha_results.db`, `alpha_results.db-wal`, `alpha_results.db-shm`
- `results/`
- `.venv/`
- `*.log`, `*.zip`, `*.pyc`, `__pycache__/`
- local scratch/temp files

## 3) Required Feature Assertions In PR Description

Codex should include all assertions below in PR summary/checklist:

1. Async pipeline has active ranker/simulator worker configuration.
2. Candidate filtering includes duplicate, collinearity, novelty, and budget gates.
3. Submit flow semantics:
   - 2xx submit => `submitted` only
   - review polling drives `accepted/rejected`
4. DLQ and replay are available and tested.
5. Review poll scheduler:
   - oldest `submitted_at` first
   - `next_review_at` gate respected
   - exponential backoff with jitter implemented
6. KPIs separate transport submit success and true acceptance.

## 4) Verification Commands (Must Run)

- `python -m unittest discover -s tests -v`
- `python alpha_factory_cli.py --help`
- `python alpha_factory_cli.py kpi --minutes 60`

## 5) Release Notes Template

Use this exact structure:

1. **What changed**
2. **Why it matters**
3. **Behavioral changes**
4. **Operational notes**
5. **Risk and rollback**

## 6) Context Integrity Checklist

- [ ] README command list matches CLI parser.
- [ ] KPI definitions in docs match `tracker.minute_kpis`.
- [ ] Submit semantics in docs match `submit_governor` and `tracker`.
- [ ] State-machine claims match `ALLOWED_SUBMIT_TRANSITIONS`.
- [ ] New env knobs documented where applicable.

