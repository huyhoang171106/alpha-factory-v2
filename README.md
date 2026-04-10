# Alpha Factory

Production-oriented alpha generation pipeline for WorldQuant Brain: continuous candidate generation, async simulation, resilient submit orchestration, post-submit review reconciliation, and quality-diversity search with budget economy.

> **Security:** This is a private quantitative research repository. Never expose alpha expressions, alpha IDs, raw WQ API payloads, or credentials.

---

## Architecture

```
Generator ──▶ Ranker ──▶ Simulator ──▶ Tier-1 Gate ──▶ Submit Queue ──▶ WQ Brain
   │           │            │               │                │
   ▼           ▼            ▼               ▼                ▼
 async       5-layer      WQ API        bias/IC/         SubmitGovernor
 producer    pre-filter    batch        quality          + DLQ + retry
 queue       (no cost)    simulation     gate (local)
```

### Pipeline stages

| Stage | What it does |
|-------|-------------|
| **Generator** | Produces alpha expressions via 5 strategies: Theme-Driven, Template Mutation, Composite, Group-Aware, Seed-Based. Optionally RAG-augmented via LLM when `OPENROUTER_API_KEY` is set. |
| **Ranker** | Fast 5-layer filter before simulation: duplicate signature → DB duplicate → collinearity → critic score → pre-ranker score (50+). Zero WQ API cost. |
| **Simulator** | Calls `WQClient.simulate_batch()` — WQ Brain runs backtest, returns Sharpe, Fitness, Turnover, Sub-Sharpe, pass/fail checks. |
| **Tier-1 Gate** | Local pre-submission check before submit queue — catches survivorship bias, lookahead bias, IC instability, and quality failures **without spending WQ quota**. |
| **Submit Queue** | `SubmitGovernor` dispatches, retries by error class, dead-letters failed jobs, and reconciles `submitted → accepted/rejected` via review polling. |

### Key files

| File | Purpose |
|------|---------|
| `alpha_factory_cli.py` | Single entry CLI — setup, run, test, KPIs, reconciliation |
| `run_async_pipeline.py` | Async streaming runtime (producer → ranker → simulator → gate → submit) |
| `generator.py` | Alpha expression generator with 5 strategies + hypothesis-driven mode |
| `alpha_ranker.py` | Pre-sim scoring: complexity penalty, structural patterns, XGBoost surrogate |
| `alpha_policy.py` | Quality gates, Tier-1 bias detection, IC stability, pre-submission gate |
| `wq_client.py` | WorldQuant Brain API client: auth, simulate, submit, status polling |
| `submit_governor.py` | Submit queue, retry by error class, DLQ, reconciliation |
| `tracker.py` | SQLite state machine, KPI queries, DLQ replay, CSV export |
| `budget_allocator.py` | Tier-1 cheap gate + Tier-2 Thompson-sampling expected value |
| `quality_diversity.py` | MAP-Elites-style QD archive with novelty scoring |

### Submit lifecycle (state machine)

```
new → gated → queued → submitted → accepted|rejected
                           ↓
              failed → dead_lettered → replay → queued
```

> **Critical semantic:** `submitted` (WQ API 2xx) ≠ `accepted` (WQ review decision). Always preserve this distinction. `accepted`/`rejected` are only set after review reconciliation polling.

---

## Setup

### 1. Bootstrap

```bash
python alpha_factory_cli.py setup
```

Creates `.venv`, installs all dependencies, seeds SQLite DB.

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` — required:
```env
WQ_EMAIL=your_wq_email@example.com
WQ_PASSWORD=your_wq_password
```

Optional — enables RAG-augmented mutation in the generator:
```env
OPENROUTER_API_KEY=sk-or-...
```

### 3. Verify

```bash
# All 64 tests pass
.venv\Scripts\python.exe -m pytest tests/ -q

# Empty KPI report (zeros expected on fresh DB)
python alpha_factory_cli.py kpi --minutes 60
```

---

## Running

### One-click profiles

```bash
# Local (Windows/mixed): hybrid supervisor + singleton lock
python alpha_factory_cli.py auto --profile local

# VPS (always-on server)
python alpha_factory_cli.py auto --profile vps

# GitHub Actions burst
python alpha_factory_cli.py auto --profile gha
```

### Fine-grained control

```bash
python alpha_factory_cli.py async --limit 0 --score 50
```

| Argument | Meaning |
|----------|---------|
| `--limit N` | Max alphas to simulate. `0` = run forever. |
| `--score N` | Pre-ranker threshold — expression must score ≥ N to enter simulation queue. Default: 50. |

### Daily / batch mode

```bash
python alpha_factory_cli.py start
```

---

## Key Environment Variables

### Tier-1 Gate (pre-submission filtering)

| Variable | Default | Effect |
|----------|---------|--------|
| `ASYNC_IC_STABILITY_MIN` | `0.15` | Minimum IC stability score. Alpha rejected if below — catches Sharpe/fitness gap, high turnover, weak sub-sharpe. |
| `ASYNC_COMPLEXITY_MIN` | `0.55` | Minimum expression simplicity score (1.0 = simple/robust). Penalizes depth > 3, > 8 operators, > 5 unique lookback constants. |

### Budget and simulation

| Variable | Default | Effect |
|----------|---------|--------|
| `ASYNC_RANKER_WORKERS` | `2` | Concurrent ranker filter workers |
| `ASYNC_SIMULATOR_WORKERS` | `1` | Concurrent simulator workers (WQ has global concurrency limit) |
| `ASYNC_MIN_CRITIC_SCORE` | `0.38` | Minimum critic score to enter ranker |
| `ASYNC_NOVELTY_MIN` | `0.28` | Minimum novelty score to pass Tier-1 budget gate |
| `GENERATOR_MODE` | `hypothesis_driven` | `legacy` or `hypothesis_driven` — changes generation strategy mix |

### Profiles apply different defaults — see `alpha_factory_cli.py` `PROFILE_DEFAULTS`.

---

## Observability

### Live KPIs

```bash
python alpha_factory_cli.py kpi --minutes 60
```

Primary metrics:

| Metric | Meaning |
|--------|---------|
| `gate_pass_rate` | % generated alphas pass Tier-1 gate |
| `submit_success_rate` | % queued → submitted (transport/API only) |
| `submit_ok_rate` | % queued → submitted + accepted + rejected |
| `true_accept_rate` | % accepted / all decided reviews |
| `dlq_rate` | % dead-lettered / queued |

### Monitor running pipeline

While `auto --profile local` is running, each loop prints:

```
Gen: 142 | Filt: 38 | Sim: 12 | Passed: 7 | Q'd: 6 | Sub: 4 | 🚫GATE: 3 | Err: 1
```

| Field | Meaning |
|-------|---------|
| `Gen` | Total alphas generated |
| `Filt` | Rejected by ranker (duplicate, collinear, low score) |
| `Sim` | Sent to WQ simulation |
| `Passed` | Passed Tier-1 gate + quality gate |
| `Q'd` | Entered submit queue |
| `Sub` | Submitted to WQ (API 2xx) |
| `🚫GATE` | Rejected by Tier-1 gate (no quota spent) |
| `Err` | Network error or timeout |

### Review sync (reconcile WQ decisions)

```bash
# Pull latest accepted/rejected states from WQ
python alpha_factory_cli.py sync-submit --limit 30
```

### Replay dead-lettered jobs

```bash
python alpha_factory_cli.py replay-dlq --limit 50
```

### Export public KPI report

```bash
python alpha_factory_cli.py public-report --minutes 60 --out results/public_report.json
```

---

## Maintenance

### Portable machine migration

```bash
python alpha_factory_cli.py zip          # bundle on source machine
# copy zip to new machine, unzip
python alpha_factory_cli.py setup       # bootstrap on new machine
python alpha_factory_cli.py start        # run
```

### Windows autostart (Task Scheduler)

```powershell
# Install service-like autostart
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_windows_autostart.ps1

# Remove
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/uninstall_windows_autostart.ps1
```

Logs: `results/windows_auto_runner.log`, `results/windows_auto_runner.err.log`

### Global command (run from any directory)

```bash
python alpha_factory_cli.py install-global --name alpha
alpha --help
alpha --yolo   # one-click local hybrid run
```

---

## Testing

```bash
# Full suite (64 tests)
.venv\Scripts\python.exe -m pytest tests/ -q

# Specific module
.venv\Scripts\python.exe -m pytest tests/test_alpha_policy.py -v

# With coverage
.venv\Scripts\python.exe -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## After Modifying Python Code

Rebuild the graphify knowledge graph:

```bash
python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
```

---

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

| Workflow | When | What |
|----------|------|------|
| `alpha-burst.yml` | Weekdays burst windows | Runs pipeline + syncs + publishes KPI artifact |
| `nightly-health.yml` | Weekday nights | Unit tests + KPI snapshot |
| `ci.yml` | Every push/PR | Unit tests + CLI smoke |
| `security-audit.yml` | Weekly | Secrets scan, SAST, dependency audit, CodeQL |