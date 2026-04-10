## What Changed
- 

## Why It Matters
- 

## Behavioral Changes
- 

## Operational Notes
- 

## Risk and Rollback
- Risk:
- Rollback:

## Deploy Scope Check (Must Pass)
- [ ] Only in-scope files from `CODEX_HANDOFF_SCOPE.md` are included.
- [ ] No secrets/runtime artifacts are included (`.env`, DB, logs, results, zip, venv).
- [ ] README command list matches `alpha_factory_cli.py --help`.
- [ ] KPI semantics in docs match `tracker.minute_kpis`.
- [ ] Submit semantics are correct: endpoint 2xx => `submitted`; review sync => `accepted/rejected`.

## Verification (Paste Outputs)
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python alpha_factory_cli.py --help`
- [ ] `python alpha_factory_cli.py kpi --minutes 60`
