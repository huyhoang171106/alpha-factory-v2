<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-10T20:45:00 | Updated: 2026-04-10T20:45:00 -->

# tests

## Purpose
Unit test suite for the alpha factory pipeline. All tests use Python's `unittest` framework (configured via `pytest.ini`).

## Key Files

| File | Description |
|------|-------------|
| `test_budget_allocator.py` | Tests for two-tier budget economy: Tier-1 gate, Thompson sampling |
| `test_quality_diversity.py` | Tests for QD archive, novelty scoring, MAP-Elites behavior |
| `test_async_pipeline.py` | Tests for async gate logic, queue flow, producer/ranker/simulator |
| `test_submit_governor.py` | Tests for retry/reconcile logic and DLQ state transitions |
| `test_tracker_memory.py` | Tests for SQLite state machine and KPI math |
| `test_wq_client_submission.py` | Tests for WQ API submission status parser |
| `test_wq_client_simulate_retry.py` | Tests for WQ client retry logic |
| `test_alpha_policy.py` | Tests for quality policy, thresholds, strategy clustering |
| `test_generator_hypothesis_mode.py` | Tests for hypothesis-driven generation mode |

## For AI Agents

### Running Tests
```bash
python -m unittest discover -s tests -v
# or via CLI:
python alpha_factory_cli.py test
```

### Test Coverage Target
≥80% coverage. Tests must pass before any PR merge. CI enforces this via `.github/workflows/ci.yml`.

### Key Invariants
- Submit state machine: `new → gated → queued → submitted → accepted|rejected` with DLQ replay path.
- `submitted` ≠ `accepted` — must be preserved in all tests.
- Budget allocator Tier-1/Tier-2 gate logic must be deterministic.

<!-- MANUAL: -->
