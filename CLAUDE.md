# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## graphify

A graphify knowledge graph is maintained at `graphify-out/`. Use it as the primary navigation aid for architecture and codebase questions.

- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.
- After modifying Python files in this session, rebuild the graph:
  ```
  python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
  ```

## Commands

```bash
# First-time setup (bootstrap venv, install deps, create .env from .env.example)
python alpha_factory_cli.py setup

# Continuous daily pipeline
python alpha_factory_cli.py start

# High-throughput async streaming (0 = run forever)
python alpha_factory_cli.py async --limit 0 --score 50

# One-click profiles (local = hybrid supervisor + singleton lock)
python alpha_factory_cli.py auto --profile local   # hybrid supervisor
python alpha_factory_cli.py auto --profile local --no-hybrid
python alpha_factory_cli.py auto --profile vps
python alpha_factory_cli.py auto --profile gha     # GitHub Actions burst

# Run unit tests
python -m unittest discover -s tests -v

# Print minute-level KPIs
python alpha_factory_cli.py kpi --minutes 60

# Requeue dead-lettered submit jobs
python alpha_factory_cli.py replay-dlq --limit 50

# Poll WQ review status for submitted alphas
python alpha_factory_cli.py sync-submit --limit 30

# Export sanitized public KPI report
python alpha_factory_cli.py public-report --minutes 60 --out results/public_report.json

# Safe A/B comparison (cadence-sequenced, no parallel submit collision)
python alpha_factory_cli.py ab-safe --profile local --minutes-per-leg 10 --cycles 1 --d1-share-a 0.90 --d1-share-b 0.75

# Build portable zip
python alpha_factory_cli.py zip

# Global command install (then run `alpha` from any directory)
python alpha_factory_cli.py install-global --name alpha
alpha --help
alpha --yolo   # one-click local hybrid run
```

## Architecture

Alpha Factory is a production-oriented alpha generation pipeline for WorldQuant Brain with these core layers:

### Runtime entrypoints
- `alpha_factory_cli.py` — single CLI, all commands, profile env merging, singleton lock
- `run_async_pipeline.py` — async streaming: Generator → Ranker → Simulator → Gate → Submit Queue
- `run_daily.py` + `pipeline.py` — batch/continuous daily flow

### Candidate generation
- `generator.py` — 5 strategies: Theme-Driven, Template Mutation, Composite, Group-Aware, Seed-Based
- `alpha_seeds.py` — seed library (ArXiv 101 Alphas, SMC, WQ community, Microstructure, Quality, Regime, Fundamental)
- `alpha_ast.py` — AST utilities: canonicalize, parameter-agnostic signature, token/operator sets
- `alpha_policy.py` — quality gate thresholds, strategy clustering, risk flags
- `alpha_dna.py` — generator strategy weights
- `alpha_rag.py` — optional RAG mutator (imported conditionally)

### Simulation and ranking
- `alpha_ranker.py` — XGBoost pre-sim ranker + heuristics; saves WQ Brain quota
- `wq_client.py` — WorldQuant Brain API client: auth, simulate, submit, status polling
- `validator.py` — expression and batch validation

### Governance and lifecycle
- `submit_governor.py` — queue dispatch, retry by error class, DLQ, reconciliation
- `tracker.py` — SQLite schema, state machine, KPI queries, CSV export, DLQ replay
- State machine: `new → gated → queued → submitted → accepted|rejected`
  - Failure path: `queued|failed → dead_lettered → replay → queued`
  - **Critical semantic: `submitted` (API 2xx) ≠ `accepted` (WQ review decision). Always preserve this distinction.**

### Search and budget
- `budget_allocator.py` — Tier-1 cheap gate + Tier-2 Thompson-sampling expected value
- `quality_diversity.py` — descriptor-based MAP-Elites archive + novelty scoring
- `evolve.py` — mutate and recombine existing alphas
- `pattern_lab.py` — pattern library and mutation helpers
- `community_harvester.py` — harvest patterns from WQ community

## Mandatory Rules for All Agents

- **Never expose** alpha expressions, alpha IDs, raw WQ API payloads, or credentials.
- **Never publish** alpha expressions or raw WQ payloads outside this private repo.
- Keep CLI command contracts backward-compatible.
- After any Python code change, rebuild the graphify knowledge graph.

## Agent Onboarding (Read Before Changing Runtime Logic)

Read in this order before modifying runtime or governance logic:
1. `README.md`
2. `HANDOVER_DETAILED.md`
3. `CODEX_HANDOFF_SCOPE.md`
4. `USE_CASES.md`
5. `SECURITY.md`
6. `TEAM_SHARING.md`
7. `OPTIMIZATION_NEXT_TASKS.md`

## Project Documentation

For full codebase navigation and file descriptions, see `AGENTS.md` (root) and subdirectory `AGENTS.md` files.
