# Optimization Backlog (Next Iterations)

## Priority A (Do First)

1. Submit truth loop hardening
- Add dedicated periodic reconciler loop in runtime (not only ad-hoc command).
- Separate dashboard metrics:
  - `submit_transport_ok_rate`
  - `true_accept_rate`
- Target:
  - reduce KPI ambiguity and avoid overestimating submit performance.

2. CPU efficiency in rank + collinearity
- Precompute and cache structural signatures per batch.
- Reduce repeated token/operator extraction on hot path.
- Add bounded LRU caches around expensive similarity checks.
- Target:
  - lower CPU time per candidate and increase throughput on local/VPS.

3. Strict single-instance runner
- Enforce singleton lock for local daemon to prevent duplicate cmd windows/processes.
- Add health heartbeat file and restart policy.
- Target:
  - stable long-run behavior on Windows.

## Priority B (High Impact)

1. Budget allocator upgrade
- Move from static thresholds to arm-level EV scheduler per minute.
- Allocate simulate/submit quota to highest expected-value arms first.
- Add exploration floor to avoid mode collapse.

2. Novelty model upgrade
- Shift novelty from near-binary thresholding to continuous distance score.
- Use descriptor distance + historical decay weighting.
- Track novelty drift over time.

3. Quality-diversity archive
- Introduce lightweight MAP-Elites bins with elite replacement policy.
- Add archive coverage KPI:
  - filled-cells ratio
  - per-cell improvement velocity.

## Priority C (Reliability / Ops)

1. Persistence robustness
- Add SQLite maintenance tasks:
  - periodic VACUUM/ANALYZE windows
  - WAL size guardrails.

2. Observability
- Structured JSON logs.
- Add run/session IDs across all commands.
- Add alert thresholds for:
  - DLQ spikes
  - submit error-class concentration
  - queue latency inflation.

3. Recovery automation
- Auto replay for retryable DLQ classes under safety budget.
- Backoff strategy tuning by historical error frequency.

## Priority D (VPS Migration)

1. Service profile
- Finalize `--profile vps` defaults by CPU core and memory class.
- Add environment preset file per host size.

2. Deployment automation
- Add one-command bootstrap for Ubuntu (`install + service + verify`).
- Add smoke check command post-deploy.

3. Secure secret flow
- Use environment file permissions and rotation playbook.
- Add short incident checklist for VPS credential compromise.

## Definition Of Done For Next Release

- Continuous runtime stable 7+ days.
- No duplicate-runner incidents.
- CPU utilization lower for same throughput baseline.
- KPI panel clearly separates transport vs true acceptance.
- Public showcase can update automatically without leaking algorithm/IP.
