<!-- Generated: 2026-04-10T20:45:00 | Updated: 2026-04-10T20:45:00 -->

# Alpha Factory

## Purpose
Production-oriented alpha generation pipeline for WorldQuant Brain: continuous candidate generation, async simulation and submit orchestration, resilient submit retry + DLQ, post-submit review reconciliation, and quality-diversity search with budget economy. This is a private quantitative trading research repository — never expose alpha expressions, credentials, or raw API payloads.

## Key Files

| File | Description |
|------|-------------|
| `alpha_factory_cli.py` | Single-entry CLI — one command to setup/run/test/kpi/reconciliation; handles bootstrap, venv, profiles (local/vps/gha), singleton lock, A/B safe compare |
| `run_async_pipeline.py` | High-throughput async runtime: Producer → Ranker → Simulator → Gate → Submit Queue |
| `run_daily.py` | Daily batch pipeline runner |
| `pipeline.py` | `AlphaFactory` orchestrator: Generate → Validate → Simulate → Submit → Evolve → Log |
| `generator.py` | Alpha expression generator v2 — 5 strategies (Theme-Driven, Template Mutation, Composite, Group-Aware, Seed-Based); supports `GENERATOR_MODE=hypothesis_driven` |
| `alpha_seeds.py` | Seed library: 101 Alphas (ArXiv), SMC, WQ community templates, Microstructure, Quality, Behavioural, Cross-Sectional, Regime, Fundamental factors |
| `alpha_ast.py` | AST utilities: canonicalize, parameter-agnostic signature, token/operator sets |
| `alpha_policy.py` | Quality policy and critic: gate thresholds, strategy clustering, risk flags, LLM budget ratio; **Tier-1 acceptance gate**: survivorship/lookahead bias detection, IC stability scoring, `pre_submission_gate()` combining all filters |
| `alpha_candidate.py` | `AlphaCandidate` dataclass — expression, metadata, score, lifecycle state |
| `alpha_ranker.py` | Pre-sim ranker: XGBoost model + heuristics scoring + **expression complexity penalty** (max depth 3, max ops 8, max 5 unique lookbacks) before WQ Brain simulation |
| `budget_allocator.py` | Two-tier budget economy: Tier-1 cheap gate + Tier-2 Thompson-sampling expected value |
| `quality_diversity.py` | Descriptor-based QD archive + novelty scoring (MAP-Elites style) |
| `alpha_dna.py` | DNA weights for generator strategy selection |
| `alpha_rag.py` | Optional RAG mutator for generator (imported conditionally) |
| `wq_client.py` | WorldQuant Brain API client: auth, simulate, submit, status polling |
| `submit_governor.py` | Submit queue governor: retry by error class, DLQ, reconciliation |
| `tracker.py` | SQLite schema + `AlphaTracker`: state machine, KPI queries, replay DLQ, CSV export |
| `validator.py` | Expression and batch validation |
| `evolve.py` | `AlphaEvolver` — mutate and recombine existing alphas |
| `pattern_lab.py` | Pattern library and mutation helpers |
| `community_harvester.py` | Harvest alpha patterns from WQ community |
| `dashboard.py` | Live dashboard renderer |
| `local_backtest.py` | Local backtesting harness |
| `generator_weights.json` | Static DNA weights for generator strategy selection |
| `pytest.ini` | pytest configuration |
| `requirements.txt` | Dependencies: requests, python-dotenv, joblib, numpy, pandas |
| `.env.example` | Template for required env vars (`WQ_EMAIL`, `WQ_PASSWORD`, optional `OPENROUTER_API_KEY`) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `.github/` | GitHub Actions workflows, CODEOWNERS, PR template, dependabot config (see `.github/AGENTS.md`) |
| `deploy/` | Production deployment files — systemd service unit for VPS (see `deploy/AGENTS.md`) |
| `schemas/` | JSON schemas — public KPI report schema (see `schemas/AGENTS.md`) |
| `scripts/` | Automation scripts — VPS runner, Windows autostart, public metrics export (see `scripts/AGENTS.md`) |
| `tests/` | Unit test suite covering pipeline, budget, QD, submit governor, tracker, WQ client (see `tests/AGENTS.md`) |

## For AI Agents

### Production Entry Point
- **Never expose** alpha expressions, alpha IDs, raw WQ API payloads, or credentials.
- Primary runtime: `python alpha_factory_cli.py auto --profile local|vps|gha`
- Submit lifecycle semantic: `submitted` (API 2xx) ≠ `accepted` (WQ review decision). Always preserve this distinction.
- Keep CLI command contracts backward-compatible.

### Before Modifying Runtime Logic
Read in this order:
1. `README.md`
2. `HANDOVER_DETAILED.md`
3. `CODEX_HANDOFF_SCOPE.md`
4. `USE_CASES.md`
5. `SECURITY.md`
6. `TEAM_SHARING.md`
7. `OPTIMIZATION_NEXT_TASKS.md`

### After Modifying Python Code
After any code change, rebuild the knowledge graph:
```
python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
```

### KPIs
`python alpha_factory_cli.py kpi --minutes 60`
Primary metrics: `gate_pass_rate`, `submit_success_rate`, `submit_ok_rate`, `true_accept_rate`, `dlq_rate`

### graphify
A graphify knowledge graph is maintained at `graphify-out/`. Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md`. If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.

## Key Environment Knobs

**Runtime:** `ASYNC_USE_RAG`, `ASYNC_RANKER_WORKERS`, `ASYNC_SIMULATOR_WORKERS`, `ASYNC_NOVELTY_MIN`, `ASYNC_TIER1_MIN_QUALITY`, `ASYNC_TIER2_MIN_EV`, `ASYNC_IC_STABILITY_MIN`, `ASYNC_COMPLEXITY_MIN`, `GENERATOR_MODE` (legacy/hypothesis_driven)

**Tier-1 Gate Thresholds (new):**
- `ASYNC_IC_STABILITY_MIN` (default 0.15) — minimum IC stability score to pass pre-submission gate
- `ASYNC_COMPLEXITY_MIN` (default 0.55) — minimum expression complexity score (1.0=simple/robust)

**Profiles:** `local`, `vps`, `gha` — each sets different concurrency and queue sizes.

**WQ client:** `WQ_MAX_CONCURRENT`, `WQ_POLL_INTERVAL`, `WQ_MAX_WAIT_TIME`, `WQ_INTERACTIVE_AUTH`

## Dependencies

### External
- requests — HTTP client for WQ Brain API
- python-dotenv — env var loading
- joblib — XGBoost model persistence
- numpy — numerical computing
- pandas — data manipulation

### Internal
- `alpha_candidate.py` → `generator.py`, `pipeline.py`
- `alpha_policy.py` → `generator.py`, `tracker.py`, `pipeline.py`, `alpha_ranker.py`
- `alpha_seeds.py` → `generator.py`
- `alpha_ast.py` → `tracker.py`, `quality_diversity.py`, `generator.py`
- `wq_client.py` → `submit_governor.py`, `pipeline.py`
- `submit_governor.py` → `alpha_factory_cli.py`
- `tracker.py` → `alpha_factory_cli.py`, `run_async_pipeline.py`, `pipeline.py`
- `budget_allocator.py` → `run_async_pipeline.py`
- `quality_diversity.py` → `run_async_pipeline.py`
- `evolve.py` → `pipeline.py`

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `alpha-burst.yml` — weekday burst windows + sync + KPI artifact
- `nightly-health.yml` — weekday nightly tests + KPI snapshot
- `ci.yml` — push/PR unit tests + CLI smoke
- `security-audit.yml` — secrets scan + SAST + dependency audit + CodeQL

<!-- MANUAL: -->
