"""
run_daily.py — CLI Entry Point for Alpha Factory
Usage:
    python run_daily.py                  # Full pipeline (generate + simulate + log)
    python run_daily.py --dry-run        # Generate + validate only, no API calls
    python run_daily.py --test           # Simulate 3 test expressions
    python run_daily.py --submit         # Auto-submit passing alphas
    python run_daily.py --dashboard      # Show current stats
    python run_daily.py --evolve-only    # Evolve from existing passing alphas
    python run_daily.py --continuous     # Run 24/7 with cooldown between rounds
"""

import argparse
import os
import sys
import time
import signal
from datetime import datetime
from pipeline import AlphaFactory


# Graceful shutdown
_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    print("\n\n⏹️  Graceful shutdown requested... finishing current round.")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)


def main():
    parser = argparse.ArgumentParser(
        description="🏭 WQ Alpha Factory — Auto-generate & submit alphas daily"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate + validate only, no API calls"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test: simulate 3 expressions"
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Auto-submit passing alphas (use with caution)"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Show current stats dashboard"
    )
    parser.add_argument(
        "--evolve-only", action="store_true",
        help="Only evolve from existing passing alphas"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run 24/7 with cooldown between rounds (Ctrl+C to stop)"
    )
    parser.add_argument(
        "--harvest", action="store_true",
        help="Harvest community winners → evolve → simulate → auto-submit"
    )
    parser.add_argument(
        "--auto-submit-tier", type=str, default="good",
        choices=["minimum", "good", "excellent", "elite"],
        help="Quality tier for auto-submit (default: good = Sharpe ≥ 1.5)"
    )
    parser.add_argument(
        "--harvest-sharpe", type=float, default=1.25,
        help="Minimum Sharpe to harvest from winners (default: 1.25)"
    )
    parser.add_argument(
        "--cooldown", type=int, default=60,
        help="Seconds between rounds in continuous mode (default: 60)"
    )
    parser.add_argument(
        "--target", type=int, default=10,
        help="Target number of passing alphas (default: 10)"
    )
    parser.add_argument(
        "--max-submit", type=int, default=30,
        help="Max submissions per round after diversity grouping (default: 30)"
    )
    parser.add_argument(
        "--candidates", type=int, default=50,
        help="Max candidates to generate per round (default: 50)"
    )
    parser.add_argument(
        "--pre-rank-score", type=float, default=50.0,
        help="Unified pre-rank minimum score for daily+continuous (default: 50.0)"
    )
    parser.add_argument(
        "--region", type=str, default="USA",
        choices=["USA", "EUR", "CHN", "ASI", "TUR", "TWN", "KOR", "GLB"],
        help="Market region (default: USA)"
    )
    parser.add_argument(
        "--universe", type=str, default="TOP3000",
        help="Universe (default: TOP3000)"
    )
    parser.add_argument(
        "--delay", type=int, default=1,
        help="Signal delay (default: 1)"
    )
    parser.add_argument(
        "--decay", type=int, default=6,
        help="Alpha decay (default: 6)"
    )
    parser.add_argument(
        "--neutralization", type=str, default="SUBINDUSTRY",
        choices=["SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET", "NONE"],
        help="Neutralization (default: SUBINDUSTRY)"
    )
    parser.add_argument(
        "--learn", action="store_true", default=True,
        help="Enable DNA learning loop (default: True)"
    )
    parser.add_argument(
        "--no-learn", action="store_false", dest="learn",
        help="Disable DNA learning loop"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Auto-delete bad simulations from WQ Brain"
    )
    parser.add_argument(
        "--level", type=int, default=5, choices=[3, 4, 5],
        help="Mining level: 3=combo, 4=regime, 5=economic-intuition (default: 5)"
    )

    args = parser.parse_args()

    # Initialize factory
    factory = AlphaFactory(
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        decay=args.decay,
        neutralization=args.neutralization,
        mining_level=args.level,
    )

    # Dashboard mode
    if args.dashboard:
        factory.show_dashboard()
        return

    # Evolve-only mode
    if args.evolve_only:
        passing = factory.tracker.get_passing_expressions()
        if not passing:
            print("❌ No passing alphas to evolve from. Run the full pipeline first.")
            return
        print(f"🧬 Evolving from {len(passing)} passing alphas...")
        evolved_strs = factory.evolver.evolve_batch(passing, n_per_alpha=5)
        print(f"  Generated {len(evolved_strs)} variants")
        from validator import validate_batch
        from alpha_candidate import AlphaCandidate
        evolved = [AlphaCandidate(expression=e, theme="evolved", mutation_type="evolve_only") for e in evolved_strs]
        evolved_valid = factory.validate_candidates(evolved)
        print(f"  Valid: {len(evolved_valid)}")
        if not args.dry_run and evolved_valid:
            results, cmap = factory.simulate_candidates(evolved_valid, max_batch=20)
            factory.log_results(results, candidates_map=cmap)
            passed = factory.filter_passed(results)
            if passed:
                factory.submit_alphas(passed, args.target, args.submit, candidates_map=cmap)
        return

    # ── Harvest mode — fetch winners → evolve → simulate → auto-submit ──
    if args.harvest:
        from community_harvester import CommunityHarvester
        harvester = CommunityHarvester()
        harvester.show_quality_targets()

        tier = getattr(args, "auto_submit_tier", "good")
        harvest_sharpe = getattr(args, "harvest_sharpe", 1.25)

        print(f"🌾 HARVEST MODE | min_sharpe={harvest_sharpe} | tier={tier}")
        candidates = harvester.harvest_and_evolve(
            min_sharpe=harvest_sharpe,
            top_n=args.candidates,
        )

        if not candidates:
            print("⚠️  No candidates from harvest (DB may be empty). Running standard generation.")
        else:
            print(f"  🧬 {len(candidates)} evolved candidates from community winners")
            from validator import validate_batch
            from alpha_candidate import AlphaCandidate
            harvest_cands = [AlphaCandidate(expression=e, theme="harvested", mutation_type="harvest_evolve") for e in candidates]
            valid = factory.validate_candidates(harvest_cands)
            print(f"  ✅ Valid: {len(valid)}")

            if not args.dry_run and valid:
                batch = valid[:15]
                results, cmap = factory.simulate_candidates(batch, max_batch=len(batch))
                factory.log_results(results, candidates_map=cmap)

                # Auto-submit using tier threshold
                if args.submit:
                    queued = harvester.auto_submit_passing(
                        results,
                        tier=tier,
                        dry_run=False,
                    )
                    print(f"  📦 Auto-queued: {queued} alphas (tier={tier})")
                else:
                    # Dry-run report
                    passed = [r for r in results if r.is_submittable]
                    print(f"  📊 {len(passed)}/{len(results)} would be submitted (add --submit to submit)")
                    for r in passed:
                        print(f"    Sharpe={r.sharpe:.3f} T={r.turnover:.1f}% | {r.expression[:60]}")
        return

    # Test mode
    if args.test:
        print("🧪 Test mode: simulating 3 sample expressions...")
        test_exprs = [
            "-rank(ts_delta(close, 5))",
            "rank(volume / ts_mean(volume, 20))",
            "rank(ts_corr(close, volume, 10))",
        ]
        results = factory.client.simulate_batch(test_exprs)
        factory.log_results(results)
        for r in results:
            status = "✅" if r.is_submittable else "❌"
            print(f"  {status} Sharpe={r.sharpe:.3f} | {r.expression}")
        return

    # Continuous mode — run 24/7
    if args.continuous:
        run_continuous(factory, args)
        return

    stats = factory.run_daily(
        target_alphas=args.target,
        max_candidates=args.candidates,
        max_submit_per_round=args.max_submit,
        min_pre_rank_score=args.pre_rank_score,
        auto_submit=args.submit,
        harvest=args.harvest,
        learn=args.learn,
        cleanup=args.cleanup,
        dry_run=args.dry_run,
    )


def run_continuous(factory: AlphaFactory, args):
    """
    Run the pipeline in continuous loop with rich live dashboard.

    Flow per cycle:
    1. Generate + Validate + Dedup + Rank (instant, with live counts)
    2. Simulate batch (with per-alpha live results)
    3. Filter + Submit + Evolve
    4. Cooldown → next cycle
    """
    from dashboard import (
        DashboardState, AlphaRow,
        render_cycle_header, render_phase_start, render_sim_row_live,
        render_sim_progress_inline, render_cycle_summary,
        render_cooldown, _c, _bar, _truncate,
    )
    from datetime import timedelta

    global _shutdown
    cycle = 0
    session_start = time.time()

    ds = DashboardState(session_start=session_start)
    try:
        from alpha_seeds import get_all_seeds
        ds.total_seeds = len(get_all_seeds())
    except Exception:
        ds.total_seeds = 0

    # Enable Windows ANSI support
    if sys.platform == "win32":
        os.system("")

    print(_c("═" * 70, "cyan"))
    print(_c("  🏭 ALPHA FACTORY — Live Dashboard Mode", "bold"))
    print(_c("═" * 70, "cyan"))
    print(f"  Region: {_c(factory.region, 'yellow')} │ Universe: {_c(factory.universe, 'yellow')} │ Seeds: {_c(str(ds.total_seeds), 'green')}")
    print(f"  Candidates/round: {args.candidates} │ Cooldown: {args.cooldown}s │ Level: {_c(str(args.level), 'magenta')} │ Auto-submit: {_c(str(args.submit), 'yellow')}")
    print(f"  Ctrl+C → graceful shutdown")
    print(_c("═" * 70, "cyan"))

    # Pre-generate first batch
    next_batch, next_stats = _prepare_batch(
        factory,
        args.candidates,
        return_stats=True,
        min_score=args.pre_rank_score,
    )

    while not _shutdown:
        cycle += 1
        cycle_start = time.time()
        ds.cycle = cycle
        ds.cycle_start = cycle_start

        # Reset cycle counters
        ds.generated = ds.validated = ds.unique = ds.ranked = 0
        ds.simulated = ds.passed = ds.submitted = 0
        ds.rows = []

        render_cycle_header(ds)

        # ── Phase 1: Use pre-generated batch ──
        current_batch = next_batch
        prep_stats = next_stats or {"generated": 0, "validated": 0, "unique": 0, "ranked": 0}

        if not current_batch:
            render_phase_start(ds, "GENERATE", "No pre-gen, generating fresh...")
            current_batch, prep_stats = _prepare_batch(
                factory,
                args.candidates,
                return_stats=True,
                min_score=args.pre_rank_score,
            )
            if not current_batch:
                print(f"  {_c('❌ No unique candidates. Waiting...', 'red')}")
                time.sleep(args.cooldown)
                continue

        ds.generated = prep_stats.get("generated", 0)
        ds.validated = prep_stats.get("validated", 0)
        ds.unique = prep_stats.get("unique", 0)
        ds.ranked = len(current_batch)
        render_phase_start(ds, "SIMULATE",
            f"{len(current_batch)} candidates ready")

        if args.dry_run:
            print(f"  🏃 DRY RUN: {len(current_batch)} expressions")
            for i, c in enumerate(current_batch[:5]):
                expr = c.expression if hasattr(c, 'expression') else str(c)
                print(f"    {i+1}. {_truncate(expr, 70)}")
            next_batch, next_stats = _prepare_batch(
                factory,
                args.candidates,
                return_stats=True,
                min_score=args.pre_rank_score,
            )
            time.sleep(5)
            continue

        # ── Phase 2: Simulate with live progress ──
        ds.sim_total = len(current_batch)
        ds.sim_done = 0

        # Print table header
        print(_c(f"\n  {'#':>3}  {'St':4}  {'Sharpe':>7}  {'Fit':>5}  {'Turn%':>6}  {'Chk':>5}  {'Time':>5}  Expression", "dim"))
        print(_c("  " + "─" * 90, "dim"))

        # Use simulate_candidates with true progress callback
        sim_start = time.time()
        def _progress_cb(result, done, total):
            ds.sim_done = done
            render_sim_progress_inline(done, total, ds.avg_sim_time)

        results, cmap = factory.simulate_candidates(
            current_batch,
            max_batch=len(current_batch),
            progress_callback=_progress_cb,
        )

        # Clear progress line
        sys.stdout.write("\r" + " " * 80 + "\r")

        # ── Display results as table ──
        total_sim_time = time.time() - sim_start
        per_alpha_time = total_sim_time / max(len(results), 1)
        # Rolling avg
        ds.avg_sim_time = 0.7 * ds.avg_sim_time + 0.3 * per_alpha_time

        for i, r in enumerate(results):
            row = AlphaRow(
                index=ds.session_simulated + i + 1,
                expression=r.expression,
                theme=getattr(r, 'theme', ''),
            )

            # Determine status from result
            if hasattr(r, 'all_passed') and r.all_passed:
                row.status = "✅"
            elif hasattr(r, 'is_submittable') and r.is_submittable:
                row.status = "✅"
            elif hasattr(r, 'error') and r.error:
                row.status = "❌"
                row.error = str(r.error)
            elif hasattr(r, 'sharpe') and r.sharpe and r.sharpe >= 1.0:
                row.status = "⚠️"
            else:
                row.status = "❌"

            row.sharpe = getattr(r, 'sharpe', None)
            row.fitness = getattr(r, 'fitness', None)
            row.turnover = getattr(r, 'turnover', None)

            # Checks info
            checks_passed = getattr(r, 'checks_passed', None)
            checks_total = getattr(r, 'checks_total', None)
            if checks_passed is not None and checks_total is not None:
                row.checks = f"{checks_passed}/{checks_total}"

            row.elapsed = per_alpha_time

            render_sim_row_live(row)
            ds.rows.append(row)

        # Update counters
        ds.simulated = len(results)
        ds.session_simulated += len(results)
        cycle_errors = [r for r in results if getattr(r, "error", "")]
        if cycle_errors:
            top_reason = cycle_errors[0].error.split("\n")[0][:90]
            print(
                f"  {_c('⚠ Fast-fail signals detected:', 'yellow')} "
                f"{len(cycle_errors)}/{len(results)} errors | {top_reason}"
            )
        batch_stats = getattr(factory.client, "last_batch_stats", {}) or {}
        if batch_stats:
            print(
                f"  📉 Batch VAR | ok={batch_stats.get('success',0)}/{batch_stats.get('total',0)} "
                f"submit_fail={batch_stats.get('submit_failed',0)} "
                f"timeout={batch_stats.get('timeouts',0)} "
                f"parse_fail={batch_stats.get('parse_failed',0)} "
                f"submittable={batch_stats.get('submittable',0)}"
            )
            total_obs = float(batch_stats.get("total", 0) or 0)
            if total_obs > 0:
                fail_ratio = float(batch_stats.get("submit_failed", 0) or 0) / total_obs
                if fail_ratio >= 0.75:
                    # Auto-rollback profile to keep system alive under API stress.
                    args.candidates = max(20, int(args.candidates * 0.7))
                    try:
                        factory.client.session.adapters["https://"]._pool_maxsize = 20
                    except Exception:
                        pass
                    print(
                        f"  {_c('🛟 Safe-profile auto rollback: reduced candidate load due to high submit_fail ratio', 'yellow')}"
                    )

        # ── Phase 3: Filter + Submit ──
        factory.log_results(results, candidates_map=cmap)
        passed = factory.filter_passed(results)
        ds.passed = len(passed)
        ds.session_passed += len(passed)

        if ds.passed > 0:
            render_phase_start(ds, "SUBMIT",
                f"{ds.passed} alphas passed! Best Sharpe: {max(r.sharpe for r in passed if r.sharpe):.3f}")

        if passed:
            # Track best
            best = max((r.sharpe for r in passed if r.sharpe), default=0)
            if best > ds.session_best_sharpe:
                ds.session_best_sharpe = best

            moved = factory.submit_alphas(
                passed,
                max_submit=args.max_submit,
                auto_submit=args.submit,
                candidates_map=cmap,
            )
            if moved:
                if args.submit:
                    ds.submitted = moved
                    ds.session_submitted += moved
                    print(f"  {_c(f'🚀 Submitted {moved} alphas!', 'green')}")
                else:
                    print(f"  {_c(f'📦 Queued {moved} alphas for governor', 'cyan')}")

            # MULTI-REGION REPLICATE
            all_passed = [r for r in results if getattr(r, 'all_passed', False)]
            if all_passed and not _shutdown:
                render_phase_start(ds, "REPLICATE",
                    f"Replicating {len(all_passed)} to EUR/ASI/CHN...")
                region_results = factory.replicate_to_regions(all_passed)
                if region_results:
                    factory.log_results(region_results, candidates_map=cmap)
                    region_passed = factory.filter_passed(region_results)
                    if region_passed:
                        region_sub = factory.submit_alphas(
                            region_passed, max_submit=args.max_submit, auto_submit=args.submit, candidates_map=cmap
                        )
                        if region_sub:
                            if args.submit:
                                ds.session_submitted += region_sub
                                print(f"  {_c(f'🌍🚀 Submitted {region_sub} multi-region alphas!', 'green')}")
                            else:
                                print(f"  {_c(f'🌍📦 Queued {region_sub} multi-region alphas', 'cyan')}")

        # ── Phase 4: Evolve winners ──
        if not _shutdown:
            evolved = factory.evolve_winners(results, candidates_map=cmap)
            if evolved:
                evolved_valid = factory.validate_candidates(evolved)
                evolved_unique = factory.deduplicate(evolved_valid)
                if evolved_unique and not args.dry_run:
                    render_phase_start(ds, "EVOLVE",
                        f"{len(evolved_unique)} evolved variants")

                    print(_c(f"  {'#':>3}  {'St':4}  {'Sharpe':>7}  {'Fit':>5}  {'Turn%':>6}  {'Chk':>5}  Expression", "dim"))
                    print(_c("  " + "─" * 75, "dim"))

                    evo_results, evo_cmap = factory.simulate_candidates(
                        evolved_unique, max_batch=len(evolved_unique)
                    )
                    factory.log_results(evo_results, candidates_map=evo_cmap)

                    for i, r in enumerate(evo_results):
                        row = AlphaRow(
                            index=ds.session_simulated + i + 1,
                            expression=r.expression,
                            theme="evolved",
                        )
                        if hasattr(r, 'is_submittable') and r.is_submittable:
                            row.status = "✅"
                        elif hasattr(r, 'sharpe') and r.sharpe and r.sharpe >= 1.0:
                            row.status = "⚠️"
                        else:
                            row.status = "❌"
                        row.sharpe = getattr(r, 'sharpe', None)
                        row.fitness = getattr(r, 'fitness', None)
                        row.turnover = getattr(r, 'turnover', None)
                        checks_p = getattr(r, 'checks_passed', None)
                        checks_t = getattr(r, 'checks_total', None)
                        if checks_p is not None and checks_t is not None:
                            row.checks = f"{checks_p}/{checks_t}"
                        render_sim_row_live(row)

                    ds.session_simulated += len(evo_results)
                    evo_passed = factory.filter_passed(evo_results)
                    ds.session_passed += len(evo_passed)
                    if evo_passed:
                        evo_sub = factory.submit_alphas(
                            evo_passed,
                            max_submit=args.max_submit,
                            auto_submit=args.submit,
                            candidates_map=evo_cmap,
                        )
                        if evo_sub:
                            if args.submit:
                                ds.session_submitted += evo_sub
                                print(f"  {_c(f'🧬🚀 Submitted {evo_sub} evolved alphas!', 'green')}")
                            else:
                                print(f"  {_c(f'🧬📦 Queued {evo_sub} evolved alphas', 'cyan')}")

        # Always run one governor flush per cycle for minute-cadence behavior.
        if args.submit and not _shutdown:
            flush = factory.governor.flush_once(limit=args.max_submit)
            if flush.get("submitted", 0) > 0:
                ds.session_submitted += flush["submitted"]
                flush_text = (
                    f"⏱️ Governor flush: {flush.get('submitted', 0)} submitted / "
                    f"{flush.get('selected', 0)} selected"
                )
                print(
                    f"  {_c(flush_text, 'green')}"
                )

        funnel = factory.tracker.submit_funnel_metrics(lookback_hours=1)
        print(
            f"  📎 Funnel(1h): gen={funnel['generated']} sim={funnel['simulated']} "
            f"queued={funnel['queued']} submitted={funnel['submitted']} accepted={funnel['accepted']}"
        )

        # ── Cycle summary ──
        render_cycle_summary(ds)

        # Prepare next batch sequentially to avoid SQLite cross-thread issues
        render_phase_start(ds, "GENERATE", "Preparing next batch...")
        next_batch, next_stats = _prepare_batch(
            factory,
            args.candidates,
            return_stats=True,
            min_score=args.pre_rank_score,
        )

        # Next batch info
        next_count = len(next_batch) if next_batch else 0
        print(f"  📋 Next batch ready: {_c(str(next_count), 'bold')} candidates")

        if _shutdown:
            break

        # ── Cooldown with live timer ──
        render_cooldown(args.cooldown, lambda: _shutdown)

    # ── Session end ──
    total_time = time.time() - session_start
    total_str = str(timedelta(seconds=int(total_time)))
    print()
    print(_c("═" * 70, "cyan"))
    print(_c(f"  ⏹️  SESSION COMPLETE — {cycle} cycles in {total_str}", "bold"))
    print(_c("═" * 70, "cyan"))
    print(f"  Total simulated: {_c(str(ds.session_simulated), 'bold')}")
    print(f"  Total passed:    {_c(str(ds.session_passed), 'green')}")
    print(f"  Total submitted: {_c(str(ds.session_submitted), 'cyan')}")
    print(f"  Best Sharpe:     {_c(f'{ds.session_best_sharpe:.3f}', 'yellow')}")
    rate = ds.session_simulated / (total_time / 3600) if total_time > 60 else 0
    print(f"  Throughput:      {_c(f'{rate:.0f} alphas/hr', 'magenta')}")
    print(_c("═" * 70, "cyan"))



# ============================================================
# Quota constants (IQC 2026 allowed allowance: ~50000 sims/day)
# ============================================================
DAILY_SIM_BUDGET = 50000    # target massive scale
BATCH_SIZE_CAP   = 150      # increased to maximize concurrent bandwidth
RANK_MIN_SCORE   = 50.0     # lowered to increase funnel size


def _prepare_batch(factory: AlphaFactory, n: int, return_stats: bool = False, min_score: float = RANK_MIN_SCORE):
    """
    Generate -> Validate -> Dedup -> Pre-Rank (aggressive filter)
    Returns list of AlphaCandidate objects.
    """
    n_gen = min(n * 3, 150)
    candidates = factory.generate_candidates(n_gen)
    valid = factory.validate_candidates(candidates)
    unique = factory.deduplicate(valid)

    ranked = factory.rank_candidates(
        unique,
        top_n=BATCH_SIZE_CAP,
        min_score=min_score,
    )
    if return_stats:
        stats = {
            "generated": len(candidates),
            "validated": len(valid),
            "unique": len(unique),
            "ranked": len(ranked),
        }
        return ranked, stats
    return ranked


if __name__ == "__main__":
    main()
