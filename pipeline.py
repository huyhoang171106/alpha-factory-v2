"""
pipeline.py — Alpha Factory Pipeline Orchestrator
Daily workflow: Generate → Validate → Simulate → Submit → Evolve → Log
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from wq_client import WQClient, SimResult
from generator import AlphaGenerator
from validator import validate_expression, validate_batch, normalize_expression_aliases
from tracker import AlphaTracker
from evolve import AlphaEvolver
from alpha_seeds import get_all_seeds
from alpha_ranker import filter_and_rank
from alpha_dna import AlphaDNA
from community_harvester import CommunityHarvester
from alpha_candidate import AlphaCandidate
from alpha_policy import passes_quality_gate, should_simulate_candidate
from submit_governor import SubmitGovernor
from pattern_lab import PatternLab

logger = logging.getLogger(__name__)


class AlphaFactory:
    """
    Daily alpha manufacturing pipeline.
    
    Workflow:
    1. Generate N candidate expressions (seeds + mutations + combos)
    2. Validate syntax (AST check)
    3. Deduplicate against historical DB
    4. Simulate on WQ Brain API
    5. Filter by Sharpe/Fitness/Turnover
    6. Auto-submit passing alphas
    7. Evolve top performers for next round
    8. Log everything to DB + CSV
    """

    def __init__(
        self,
        email: str = None,
        password: str = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 6,
        neutralization: str = "SUBINDUSTRY",
        mining_level: int = 5,
    ):
        self.region = region
        self.universe = universe
        self.delay = delay
        self.decay = decay
        self.neutralization = neutralization
        self.mining_level = max(1, min(5, mining_level))

        self.dna = AlphaDNA()
        self.generator = AlphaGenerator(
            dna_weights=self.dna.get_weights(),
            mining_level=self.mining_level,
        )
        self.evolver = AlphaEvolver()
        self.tracker = AlphaTracker()
        self.harvester = CommunityHarvester()
        self._client = None
        self._email = email
        self._password = password
        self._governor = None
        self._run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.pattern_lab = PatternLab()

    @property
    def client(self) -> WQClient:
        if self._client is None:
            self._client = WQClient(self._email, self._password)
            if getattr(self.harvester, "client", None) is None:
                self.harvester.client = self._client
        return self._client

    @property
    def governor(self) -> SubmitGovernor:
        if self._governor is None:
            self._governor = SubmitGovernor(self.tracker, self.client)
        return self._governor

    def generate_candidates(self, n: int = 50, harvest: bool = False) -> list[AlphaCandidate]:
        """Step 1: Generate candidate AlphaCandidates with lineage metadata"""
        logger.info(f"📝 Generating {n} candidate expressions...")

        candidates = []

        # Strategy A: Harvest from existing winners (if enabled)
        if harvest:
            harvested_exprs = self.harvester.harvest_and_evolve(top_n=int(n * 0.4))
            for expr in harvested_exprs:
                candidates.append(AlphaCandidate(
                    expression=expr,
                    theme="harvested",
                    mutation_type="community_evolve",
                ))
            logger.info(f"  Harvested: {len(harvested_exprs)} candidates from winners")

        # Strategy B: DNA-biased generation (already returns AlphaCandidate)
        needed = n - len(candidates)
        if needed > 0:
            for expr in self.pattern_lab.propose_expressions(n=max(0, min(needed // 4, 12))):
                candidates.append(
                    AlphaCandidate(
                        expression=expr,
                        theme="pattern_lab",
                        mutation_type="pattern_lab_proposal",
                    )
                )

        needed = n - len(candidates)
        if needed > 0:
            self.generator.weights = self.dna.get_weights()
            submit_fail_rate = 0.0
            if self._client is not None:
                stats = getattr(self.client, "last_batch_stats", {}) or {}
                total = float(stats.get("total", 0) or 0)
                if total > 0:
                    submit_fail_rate = float(stats.get("submit_failed", 0) or 0) / total
            self.generator.runtime_submit_fail_rate = submit_fail_rate
            use_rag = submit_fail_rate < 0.5
            batch = self.generator.generate_batch(needed, use_rag=use_rag)
            candidates.extend(batch)

        # Auto-Fitness Exo-Skeleton (Pasteurize + Decay Smoothing)
        for c in candidates:
            if not c.expression.startswith("ts_decay_linear"):
                # Wraps the core logic to mathematically suppress outliers and drastically lower turnover
                c.expression = f"ts_decay_linear(pasteurize({c.expression}), 8)"
            if self.mining_level >= 5 and self._is_turnover_risk(c.expression):
                if "hump(" not in c.expression and "trade_when(" not in c.expression:
                    c.expression = f"hump({c.expression}, hump=0.01)"

        logger.info(f"  Total Generated: {len(candidates)} raw candidates (Wrapped with Fitness Exo-Skeleton)")
        return candidates

    def _is_turnover_risk(self, expr: str) -> bool:
        """
        Heuristic turnover-risk detector:
        short-horizon deltas + frequent sign flips + raw volume shocks.
        """
        short_delta = ("ts_delta(" in expr and any(f",{d})" in expr or f", {d})" in expr for d in [1, 2, 3, 4, 5]))
        flip_heavy = "sign(" in expr or "ts_av_diff(" in expr
        volume_spike = "volume / adv20" in expr or "ts_delta(volume" in expr
        return short_delta and (flip_heavy or volume_spike)

    def validate_candidates(self, candidates: list[AlphaCandidate]) -> list[AlphaCandidate]:
        """Step 2: Validate syntax via safe CPU Multi-Threading"""
        import os
        from concurrent.futures import ThreadPoolExecutor
        
        valid = []
        max_threads = max(1, (os.cpu_count() or 4) - 2)
        
        def _check(c):
            expr = c.expression if isinstance(c, AlphaCandidate) else c
            expr = normalize_expression_aliases(expr)
            if isinstance(c, AlphaCandidate):
                c.expression = expr
            is_valid, _ = validate_expression(expr)
            return c if is_valid else None
            
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for result in executor.map(_check, candidates, chunksize=10):
                if result:
                    valid.append(result)
                    
        rejected = len(candidates) - len(valid)
        if rejected:
            logger.info(f"  Validated: {len(valid)} valid, {rejected} rejected (using {max_threads} CPU threads)")
        return valid

    def deduplicate(self, candidates: list[AlphaCandidate]) -> list[AlphaCandidate]:
        """Step 3: Remove duplicates (already tested in DB or too collinear)"""
        unique = []
        for c in candidates:
            expr = c.expression if isinstance(c, AlphaCandidate) else c
            if not self.tracker.is_duplicate(expr) and not self.tracker.is_collinear(expr):
                unique.append(c)
        duped = len(candidates) - len(unique)
        if duped:
            logger.info(f"  Deduped: {len(unique)} new, {duped} already tested or collinear")
        return unique

    def rank_candidates(
        self,
        candidates: list[AlphaCandidate],
        top_n: int = None,
        min_score: float = 35.0,
    ) -> list[AlphaCandidate]:
        """
        Step 3.5: Tier-2 pre-simulation ranking with
        family crowding + in-batch similarity penalties.
        """
        top_n = top_n or len(candidates)

        expr_to_candidate = {}
        expressions = []
        for c in candidates:
            expr = c.expression if isinstance(c, AlphaCandidate) else c
            if expr not in expr_to_candidate:
                expr_to_candidate[expr] = c
                expressions.append(expr)

        family_pass_counts = self.tracker.get_family_pass_counts()
        ranked_exprs = filter_and_rank(
            expressions,
            top_n=top_n,
            min_score=min_score,
            family_pass_counts=family_pass_counts,
        )
        ranked = [expr_to_candidate[e] for e in ranked_exprs if e in expr_to_candidate]
        ranked = [
            c for c in ranked
            if should_simulate_candidate(c.expression if isinstance(c, AlphaCandidate) else str(c))
        ]
        logger.info(
            f"🎯 Pre-rank: {len(ranked)}/{len(candidates)} pass "
            f"(min={min_score}, top_n={top_n}) with tier-2 penalties"
        )
        return ranked

    def simulate_candidates(
        self,
        candidates: list[AlphaCandidate],
        max_batch: int = 30,
        progress_callback=None,
    ) -> tuple[list[SimResult], dict]:
        """Step 4: Simulate on WQ Brain API. Returns (results, candidates_map)."""
        to_sim = candidates[:max_batch]
        logger.info(f"🔄 Simulating {len(to_sim)} expressions on WQ Brain...")

        # Build lookup map: expression -> AlphaCandidate
        candidates_map = {}
        expressions = []
        for c in to_sim:
            expr = c.expression if isinstance(c, AlphaCandidate) else c
            expressions.append(expr)
            candidates_map[expr] = c if isinstance(c, AlphaCandidate) else None

        results = self.client.simulate_batch(
            expressions,
            region=self.region,
            universe=self.universe,
            delay=self.delay,
            decay=self.decay,
            neutralization=self.neutralization,
            progress_callback=progress_callback,
        )
        return results, candidates_map

    def filter_passed(self, results: list[SimResult]) -> list[SimResult]:
        """Step 5: Filter for submittable alphas"""
        passed = [r for r in results if passes_quality_gate(r)]
        logger.info(f"📊 Filter: {len(passed)}/{len(results)} submittable")
        return passed

    def submit_alphas(
        self,
        passed: list[SimResult],
        max_submit: int = 30,
        auto_submit: bool = False,
        candidates_map: dict = None,
    ) -> int:
        """
        Step 6: Smart submission — best per family cluster.
        Groups by family, picks highest Sharpe per family to maximize diversity.
        """
        if not passed:
            return 0

        # Smart grouping: pick best per family
        cmap = candidates_map or {}
        family_best = {}
        for result in passed:
            candidate = cmap.get(result.expression)
            family = candidate.family if isinstance(candidate, AlphaCandidate) and candidate.family else result.expression[:30]
            if family not in family_best or result.sharpe > family_best[family].sharpe:
                family_best[family] = result

        # Submit top N from diverse families (Now just pushes to STAGING)
        to_submit = sorted(family_best.values(), key=lambda r: r.sharpe, reverse=True)[:max_submit]
        skipped = len(passed) - len(to_submit)
        if skipped > 0:
            logger.info(f"  🎯 Smart STAGING: {len(to_submit)} diverse / {len(passed)} total ({skipped} family duplicates skipped)")

        queued = self.governor.enqueue(to_submit, candidates_map=cmap)
        logger.info("  📦 Queued %s alpha(s) for WQ submit governor", queued)
        if auto_submit and queued > 0:
            flush = self.governor.flush_once(limit=max_submit)
            logger.info(
                "  🚀 Flush now: selected=%s submitted=%s failed=%s",
                flush.get("selected", 0),
                flush.get("submitted", 0),
                flush.get("failed", 0),
            )
            return int(flush.get("submitted", 0))
        return queued

    def replicate_to_regions(
        self,
        passed: list[SimResult],
        extra_regions: list[str] = None,
    ) -> list[SimResult]:
        """
        💰 SCORE MULTIPLIER: Take passing alpha from primary region
        and test it in other regions. Same expression can score in multiple regions!
        
        Strategy: USA alpha passes → test EUR, ASI, CHN → submit all that pass.
        """
        if extra_regions is None:
            extra_regions = self._select_replication_regions(top_k=3)
        
        # Only replicate alphas that passed all checks
        replicable = [r for r in passed if r.all_passed and r.expression]
        if not replicable:
            return []

        all_region_results = []
        for region in extra_regions:
            if region == self.region:  # Skip primary region
                continue
            
            expressions = [r.expression for r in replicable]
            logger.info(f"🌍 Replicating {len(expressions)} alphas to {region}...")
            
            results = self.client.simulate_batch(
                expressions,
                region=region,
                universe=self.universe,
                delay=self.delay,
                decay=self.decay,
                neutralization=self.neutralization,
            )
            
            region_passed = [r for r in results if r.all_passed]
            logger.info(f"  🌍 {region}: {len(region_passed)}/{len(results)} passed")
            all_region_results.extend(results)

        return all_region_results

    def _select_replication_regions(self, top_k: int = 3) -> list[str]:
        """
        Pick replication regions dynamically from recent pass-rate + sharpe stats.
        Falls back to a strong default basket if no history exists.
        """
        baseline = ["EUR", "ASI", "CHN", "TUR", "TWN", "KOR", "GLB"]
        candidates = [r for r in baseline if r != self.region]

        stats = self.tracker.get_region_performance(lookback_days=14)
        scored = [row[0] for row in stats if row[0] in candidates]
        ranked = scored + [r for r in candidates if r not in scored]
        selected = ranked[:top_k]
        logger.info(f"🌍 Region selector chose: {selected}")
        return selected

    def evolve_winners(
        self,
        results: list[SimResult],
        sharpe_threshold: float = 0.8,
        n_per_alpha: int = 3,
        candidates_map: dict = None,
    ) -> list[AlphaCandidate]:
        """Step 7: Create variants from high-Sharpe alphas with lineage"""
        good_exprs = [
            r.expression for r in results
            if r.sharpe > sharpe_threshold and not r.error
        ]

        if not good_exprs:
            return []

        variant_strs = self.evolver.evolve_batch(good_exprs, n_per_alpha=n_per_alpha)
        logger.info(f"🧬 Evolved {len(variant_strs)} variants from {len(good_exprs)} good alphas")

        # Wrap evolved strings as AlphaCandidate with parent lineage
        evolved_candidates = []
        for v in variant_strs:
            # Try to find parent candidate for lineage inheritance
            parent = (candidates_map or {}).get(v)
            if parent and isinstance(parent, AlphaCandidate):
                evolved_candidates.append(parent.derive(v, mutation_type="evolved"))
            else:
                evolved_candidates.append(AlphaCandidate(
                    expression=v,
                    theme="evolved",
                    mutation_type="evolved",
                ))
        return evolved_candidates

    def log_results(self, results: list[SimResult], candidates_map: dict = None, batch_id: str = ""):
        """Step 8: Save to DB + CSV (with lineage metadata if available)"""
        batch_ref = batch_id or f"batch-{uuid.uuid4().hex[:8]}"
        saved = self.tracker.save_batch(
            results,
            candidates_map=candidates_map,
            run_id=self._run_id,
            batch_id=batch_ref,
        )
        csv_path = self.tracker.export_csv()
        logger.info(f"💾 Saved {saved} results → {csv_path}")

    # =========================================================
    # Main Pipeline
    # =========================================================
    def run_daily(
        self,
        target_alphas: int = 10,
        max_candidates: int = 50,
        max_simulations: int = 30,
        max_submit_per_round: int = 30,
        min_pre_rank_score: float = 35.0,
        auto_submit: bool = False,
        evolve: bool = True,
        harvest: bool = True,
        learn: bool = True,
        cleanup: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """
        Run the complete daily pipeline.
        
        Args:
            target_alphas: Target number of passing alphas
            max_candidates: Max expressions to generate
            max_simulations: Max expressions to simulate per round
            auto_submit: Auto-submit passing alphas
            evolve: Run evolution on good alphas
            dry_run: Generate + validate only, no API calls
            
        Returns:
            Summary dict with stats
        """
        start = time.time()
        today = datetime.now().strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"  🏭 ALPHA FACTORY — Daily Run {today}")
        print(f"{'='*60}")
        print(f"  Region: {self.region} | Universe: {self.universe}")
        print(f"  Target: {target_alphas} alphas | Dry run: {dry_run}")
        print(f"{'='*60}\n")

        all_results = []
        round_num = 0
        total_passed = 0
        funnel = {
            "generated": 0,
            "validated": 0,
            "unique": 0,
            "ranked": 0,
            "simulated": 0,
            "submittable": 0,
            "all_checks_passed": 0,
            "submitted": 0,
            "gt_1_5": 0,
        }

        while total_passed < target_alphas and round_num < 3:
            round_num += 1
            print(f"\n--- Round {round_num} ---")

            # 1. Generate (with optional harvest)
            candidates = self.generate_candidates(max_candidates, harvest=harvest)
            funnel["generated"] += len(candidates)

            # 2. Validate
            valid = self.validate_candidates(candidates)
            funnel["validated"] += len(valid)

            # 3. Deduplicate
            unique = self.deduplicate(valid)
            funnel["unique"] += len(unique)

            if not unique:
                logger.warning("No new unique candidates to test!")
                break

            if dry_run:
                ranked_preview = self.rank_candidates(
                    unique,
                    top_n=max_simulations,
                    min_score=min_pre_rank_score,
                )
                funnel["ranked"] += len(ranked_preview)
                print(
                    f"  🏃 DRY RUN: {len(unique)} unique | "
                    f"{len(ranked_preview)} ranked (min_score={min_pre_rank_score})"
                )
                for c in ranked_preview[:10]:
                    expr = c.expression if isinstance(c, AlphaCandidate) else c
                    theme = c.theme if isinstance(c, AlphaCandidate) else "?"
                    print(f"    [{theme}] {expr}")
                continue

            # 3.5 Pre-rank (Tier-2 penalties)
            ranked = self.rank_candidates(
                unique,
                top_n=max_simulations,
                min_score=min_pre_rank_score,
            )
            funnel["ranked"] += len(ranked)
            if not ranked:
                logger.warning("No candidates survived pre-ranking!")
                break

            # 4. Simulate
            results, cmap = self.simulate_candidates(ranked, max_simulations)
            funnel["simulated"] += len(results)
            all_results.extend(results)

            # 5. Filter
            passed = self.filter_passed(results)
            funnel["submittable"] += len(passed)
            funnel["all_checks_passed"] += sum(1 for r in results if r.all_passed)
            funnel["gt_1_5"] += sum(1 for r in results if r.sharpe > 1.5 and not r.error)
            total_passed += len(passed)

            # 6. Submit (smart: best per family)
            if passed:
                moved = self.submit_alphas(
                    passed,
                    max_submit=max_submit_per_round,
                    auto_submit=auto_submit,
                    candidates_map=cmap,
                )
                funnel["submitted"] += moved if auto_submit else 0
                if moved:
                    verb = "Submitted" if auto_submit else "Queued"
                    icon = "🚀" if auto_submit else "📦"
                    print(f"  {icon} {verb} {moved} alphas!")
                    
            # 6.5. Cross-Region Arbitrage
            if passed and not dry_run:
                replicated = self.replicate_to_regions(passed)
                if replicated:
                    self.log_results(replicated, candidates_map=cmap)
                    reg_passed = self.filter_passed(replicated)
                    if reg_passed:
                        reg_sub = self.submit_alphas(
                            reg_passed,
                            max_submit=max_submit_per_round,
                            auto_submit=auto_submit,
                            candidates_map=cmap,
                        )
                        funnel["submitted"] += reg_sub if auto_submit else 0
                        if reg_sub:
                            verb = "Submitted" if auto_submit else "Queued"
                            print(f"  🌍🚀 {verb} {reg_sub} cross-region replicated alphas!")
                            total_passed += len(reg_passed)

            # 7. DNA Learning LOOP
            if learn:
                self.dna.learn_from_results(results)
                self.pattern_lab.learn_from_results(results)
                if int(self.pattern_lab.data.get("updates", 0)) % 5 == 0:
                    self.pattern_lab.emit_self_code_proposal()

            # 8. Cleanup (optional)
            if cleanup:
                self.dna.client = self.client
                self.dna.delete_bad_sims(results, dry_run=dry_run)

            # 9. Log main results with lineage
            self.log_results(results, candidates_map=cmap)

            # 10. Evolve — validate + simulate evolved variants inline
            if evolve and round_num < 3:
                evolved = self.evolve_winners(results, candidates_map=cmap)
                if evolved:
                    evolved_valid = self.validate_candidates(evolved)
                    evolved_unique = self.deduplicate(evolved_valid)
                    if evolved_unique and not dry_run:
                        logger.info(f"🧬 Simulating {len(evolved_unique)} evolved variants...")
                        evo_results, evo_cmap = self.simulate_candidates(evolved_unique, max_batch=max_simulations)
                        self.dna.learn_from_results(evo_results)
                        self.pattern_lab.learn_from_results(evo_results)
                        if int(self.pattern_lab.data.get("updates", 0)) % 5 == 0:
                            self.pattern_lab.emit_self_code_proposal()
                        all_results.extend(evo_results)
                        evo_passed = self.filter_passed(evo_results)
                        funnel["simulated"] += len(evo_results)
                        funnel["submittable"] += len(evo_passed)
                        funnel["all_checks_passed"] += sum(1 for r in evo_results if r.all_passed)
                        funnel["gt_1_5"] += sum(1 for r in evo_results if r.sharpe > 1.5 and not r.error)
                        total_passed += len(evo_passed)
                        if evo_passed:
                            evo_submitted = self.submit_alphas(
                                evo_passed,
                                max_submit=max_submit_per_round,
                                auto_submit=auto_submit,
                                candidates_map=evo_cmap,
                            )
                            funnel["submitted"] += evo_submitted if auto_submit else 0
                            if evo_submitted:
                                verb = "Submitted" if auto_submit else "Queued"
                                print(f"  🧬🚀 {verb} {evo_submitted} evolved alphas!")
                        self.log_results(evo_results, candidates_map=evo_cmap)

        # Final summary
        elapsed = time.time() - start
        stats = self.tracker.get_daily_stats(today)

        print(f"\n{'='*60}")
        print(f"  📊 DAILY SUMMARY")
        print(f"{'='*60}")
        print(f"  Total simulated : {stats['simulated']}")
        print(f"  Passed          : {stats['passed']}")
        print(f"  Submitted       : {stats['submitted']}")
        print(f"  Avg Sharpe      : {stats['avg_sharpe']:.3f}")
        print(f"  Max Sharpe      : {stats['max_sharpe']:.3f}")
        print(f"  Time            : {elapsed:.0f}s")
        print(f"{'='*60}")
        print(f"\n  🔬 FUNNEL (this run)")
        print(f"  generated       : {funnel['generated']}")
        print(f"  validated       : {funnel['validated']}")
        print(f"  unique          : {funnel['unique']}")
        print(f"  ranked          : {funnel['ranked']}")
        print(f"  simulated       : {funnel['simulated']}")
        print(f"  sharpe > 1.5    : {funnel['gt_1_5']}")
        print(f"  all checks pass : {funnel['all_checks_passed']}")
        print(f"  submittable     : {funnel['submittable']}")
        print(f"  submitted       : {funnel['submitted']}")

        # Show top alphas
        top = self.tracker.get_top_alphas(5)
        if top:
            print(f"\n🏆 Top 5 Alphas Today:")
            for i, (expr, sharpe, fitness, turnover, url, passed) in enumerate(top, 1):
                status = "✅" if passed else "⚠️"
                print(f"  {i}. {status} Sharpe={sharpe:.3f} Fit={fitness:.2f} "
                      f"TO={turnover:.1f}%")
                print(f"     {expr[:80]}...")
                if url:
                    print(f"     {url}")

        return stats

    def show_dashboard(self):
        """Show current stats dashboard with lineage analytics"""
        stats = self.tracker.get_daily_stats()
        top = self.tracker.get_top_alphas(10)
        submittable = self.tracker.get_submittable()

        print(f"\n{'='*60}")
        print(f"  📊 ALPHA FACTORY DASHBOARD")
        print(f"{'='*60}")
        print(f"  Today: {stats['total']} tested | {stats['passed']} passed | "
              f"{stats['submitted']} submitted")
        print(f"  Avg Sharpe: {stats['avg_sharpe']} | Max: {stats['max_sharpe']}")

        # --- Theme Analytics ---
        theme_stats = self.tracker.get_theme_stats()
        if theme_stats:
            print(f"\n  📈 Theme Performance:")
            print(f"  {'Theme':<20} {'Total':>6} {'Passed':>7} {'Rate':>7} {'AvgS':>7} {'MaxS':>7}")
            print(f"  {'─'*55}")
            for theme, total, passed, avg_s, max_s in theme_stats:
                rate = (passed / total * 100) if total > 0 else 0
                avg_s = avg_s or 0
                max_s = max_s or 0
                bar = "█" * int(rate / 5)
                print(f"  {theme:<20} {total:>6} {passed:>7} {rate:>6.1f}% {avg_s:>6.2f} {max_s:>6.2f}  {bar}")

        # --- Mutation Type Analytics ---
        mut_stats = self.tracker.get_mutation_stats()
        if mut_stats:
            print(f"\n  🧬 Mutation Effectiveness:")
            print(f"  {'Type':<22} {'Total':>6} {'Passed':>7} {'Rate':>7} {'AvgS':>7}")
            print(f"  {'─'*50}")
            for mut, total, passed, avg_s in mut_stats:
                rate = (passed / total * 100) if total > 0 else 0
                avg_s = avg_s or 0
                print(f"  {mut:<22} {total:>6} {passed:>7} {rate:>6.1f}% {avg_s:>6.2f}")

        # --- Family Crowding ---
        family_stats = self.tracker.get_family_stats()
        if family_stats:
            print(f"\n  👪 Top Families (by passes):")
            print(f"  {'Family':<14} {'Total':>6} {'Passed':>7} {'BestS':>7}")
            print(f"  {'─'*38}")
            for fam, total, passed, best_s in family_stats[:10]:
                print(f"  {fam:<14} {total:>6} {passed:>7} {best_s:>6.2f}")

        if submittable:
            print(f"\n  📋 Ready to submit ({len(submittable)}):")
            for expr, sharpe, fitness, to, aid, url in submittable[:5]:
                print(f"    Sharpe={sharpe:.3f} | {url}")

        if top:
            print(f"\n  🏆 All-time Top 10:")
            for i, (expr, sharpe, fitness, to, url, passed) in enumerate(top, 1):
                print(f"    {i}. Sharpe={sharpe:.3f} | {expr[:60]}...")
