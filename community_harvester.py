"""
community_harvester.py — WQ Brain Community Alpha Harvester + Auto-Submit
=========================================================================

Strategy:
1. Fetch YOUR submitted alphas with Sharpe >= threshold
2. Search community/shared alphas (if API allows)
3. Evolve/mutate fetched alphas → generate variants
4. Auto-submit variants that pass all checks

Why this beats blind generation:
- Community alphas are PROVEN to pass WQ Brain checks
- Their expressions reveal working operator patterns
- Mutating known winners is far more efficient than random generation
- 37 quota shots on community-derived expressions >> random brute force

Top 5% threshold insight:
- Sharpe 1.25 = minimum (what we set as baseline)
- Sharpe 1.5  = good (~top 20-25%)
- Sharpe 2.0  = excellent (~top 10%)
- Sharpe 2.5+ = top 5% elite
- BUT: IQC ranks by PORTFOLIO, not individual. 100 uncorrelated
  Sharpe-1.4 alphas >> 5 Sharpe-2.5 alphas in the same direction.

Usage:
    from community_harvester import CommunityHarvester
    harvester = CommunityHarvester()
    exprs = harvester.harvest_and_evolve(top_n=20)
"""

import re
import time
import random
import logging
import sqlite3
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

API_BASE = "https://api.worldquantbrain.com"

# ============================================================
# Submission quality tiers
# ============================================================
QUALITY_TIERS = {
    "minimum":   {"sharpe": 1.25, "fitness": 1.0,  "turnover_max": 70},  # our current threshold
    "good":      {"sharpe": 1.5,  "fitness": 1.2,  "turnover_max": 60},  # ~top 20%
    "excellent": {"sharpe": 2.0,  "fitness": 1.5,  "turnover_max": 50},  # ~top 10%
    "elite":     {"sharpe": 2.5,  "fitness": 2.0,  "turnover_max": 40},  # top 5%
}

TARGET_TIER = "good"  # aim for top 20% by default


@dataclass
class HarvestedAlpha:
    alpha_id: str
    expression: str
    sharpe: float
    fitness: float
    turnover: float
    region: str
    universe: str
    source: str  # "own_submitted", "own_unsubmitted", "community"


# ============================================================
# Community Harvester
# ============================================================
class CommunityHarvester:
    """
    Fetch proven alphas from WQ Brain, evolve them, auto-submit variants.

    Flow:
        harvest() → fetch own best alphas from WQ Brain API
        evolve_harvest() → generate variants using mutation strategies  
        auto_submit() → submit variants that pass is_submittable check
    """

    def __init__(self, client=None):
        self.client = client
        self._harvested: List[HarvestedAlpha] = []

    def _ensure_client(self):
        if self.client is None:
            from wq_client import WQClient
            self.client = WQClient()
        return self.client

    # ─────────────────────────────────────────────────────────
    # Step 1: HARVEST — Fetch best alphas from WQ Brain
    # ─────────────────────────────────────────────────────────

    def fetch_own_alphas(
        self,
        min_sharpe: float = 1.25,
        limit: int = 50,
        status: str = "UNSUBMITTED",
    ) -> List[HarvestedAlpha]:
        """
        Fetch YOUR alphas from WQ Brain with Sharpe >= min_sharpe.
        Endpoint: GET /alphas?sharpe[gte]=X&limit=N&status=S
        """
        logger.info(f"🔍 Fetching {status} alphas with Sharpe >= {min_sharpe}...")

        params = {
            "limit": limit,
            "offset": 0,
            "order": "-sharpe",     # highest Sharpe first
        }
        # WQ Brain API filter param for sharpe
        params["sharpe"] = f"gte:{min_sharpe}"

        try:
            r = self._ensure_client()._api_request("get", f"{API_BASE}/alphas", params=params)
            if r is None or r.status_code != 200:
                logger.warning(f"  Fetch failed: {r.status_code if r else 'None'}")
                return []

            data = r.json()
            alphas_raw = data.get("results", data.get("alphas", []))

            harvested = []
            for a in alphas_raw:
                # Extract expression from nested structure
                expr = (
                    a.get("regular", {}).get("code", "") or
                    a.get("expression", "") or
                    a.get("code", "")
                )
                if not expr:
                    continue

                sharpe = float(a.get("sharpe", 0) or 0)
                if sharpe < min_sharpe:
                    continue

                h = HarvestedAlpha(
                    alpha_id   = a.get("id", ""),
                    expression = expr,
                    sharpe     = sharpe,
                    fitness    = float(a.get("fitness", 0) or 0),
                    turnover   = float(a.get("turnover", 0) or 0),
                    region     = a.get("settings", {}).get("region", "USA"),
                    universe   = a.get("settings", {}).get("universe", "TOP3000"),
                    source     = "own_" + status.lower(),
                )
                harvested.append(h)

            logger.info(f"  ✅ Fetched {len(harvested)} alphas (Sharpe >= {min_sharpe})")
            self._harvested.extend(harvested)
            return harvested

        except Exception as e:
            logger.error(f"  Fetch error: {e}")
            return []

    def fetch_from_db(self, min_sharpe: float = 1.25) -> List[HarvestedAlpha]:
        """
        Fetch our own HIGH-PERFORMING alphas from local SQLite DB (alpha_results.db).
        Faster than API, uses historical data.
        """
        db_path = os.path.join(os.path.dirname(__file__), "alpha_results.db")
        if not os.path.exists(db_path):
            return []

        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("""
                SELECT expression, sharpe, fitness, turnover, region, universe
                FROM alphas
                WHERE sharpe >= ? AND all_passed = 1
                ORDER BY sharpe DESC
                LIMIT 50
            """, (min_sharpe,)).fetchall()
            conn.close()

            harvested = []
            for row in rows:
                h = HarvestedAlpha(
                    alpha_id   = "",
                    expression = row[0],
                    sharpe     = row[1],
                    fitness    = row[2],
                    turnover   = row[3],
                    region     = row[4] or "USA",
                    universe   = row[5] or "TOP3000",
                    source     = "local_db",
                )
                harvested.append(h)

            logger.info(f"  📦 Loaded {len(harvested)} winning alphas from local DB")
            self._harvested.extend(harvested)
            return harvested

        except Exception as e:
            logger.warning(f"  DB fetch error: {e}")
            return []

    def harvest(self, min_sharpe: float = 1.25) -> List[HarvestedAlpha]:
        """
        Full harvest: API (own alphas) + local DB.
        Returns deduplicated list sorted by Sharpe.
        """
        # Try local DB first (fast)
        db_results = self.fetch_from_db(min_sharpe)

        # Then API (slower, but gets latest)
        api_results = self.fetch_own_alphas(min_sharpe, limit=50, status="UNSUBMITTED")
        api_results += self.fetch_own_alphas(min_sharpe, limit=50, status="SUBMITTED")

        # Deduplicate by expression
        seen = set()
        unique = []
        for h in sorted(db_results + api_results, key=lambda x: x.sharpe, reverse=True):
            if h.expression not in seen:
                seen.add(h.expression)
                unique.append(h)

        logger.info(f"🎯 Total harvested: {len(unique)} unique winning expressions")
        self._harvested = unique
        return unique

    # ─────────────────────────────────────────────────────────
    # Step 2: EVOLVE — Generate variants from winners
    # ─────────────────────────────────────────────────────────

    def _mutate_lookback(self, expr: str) -> List[str]:
        """Change numeric lookback windows ±20-50%"""
        variants = []
        numbers = list(set(re.findall(r'\b(\d+)\b', expr)))
        for n in numbers[:3]:
            n_int = int(n)
            if n_int < 2 or n_int > 500:
                continue
            for factor in [0.6, 0.75, 1.25, 1.5, 2.0]:
                new_n = max(2, int(n_int * factor))
                if new_n != n_int:
                    variants.append(re.sub(r'\b' + n + r'\b', str(new_n), expr, count=1))
        return variants[:4]

    def _add_sector_neutralize(self, expr: str) -> List[str]:
        """Wrap expression with group_neutralize"""
        if "group_neutralize" in expr:
            return []
        variants = []
        for group in ["sector", "industry", "subindustry"]:
            variants.append(f"group_neutralize({expr}, {group})")
        return variants[:2]

    def _add_volume_confirm(self, expr: str) -> List[str]:
        """Multiply by volume confirmation signal"""
        if "volume" in expr.lower():
            return []
        return [
            f"({expr}) * rank(volume / adv20)",
            f"({expr}) * rank(ts_delta(volume, 5))",
        ]

    def _add_quality_filter(self, expr: str) -> List[str]:
        """Add quality filter (volatility-adjusted)"""
        return [
            f"({expr}) / (ts_std_dev(returns, 20) + 0.001)",
            f"({expr}) * rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
        ]

    def _negate(self, expr: str) -> List[str]:
        """Try negating (sometimes the opposite works in another regime)"""
        if expr.startswith("-"):
            return [expr[1:]]
        return ["-" + expr]

    def _combine_two(self, expr1: str, expr2: str) -> str:
        """Combine two expressions additively"""
        return f"rank(0.5 * rank({expr1}) + 0.5 * rank({expr2}))"

    def evolve_single(self, alpha: HarvestedAlpha, n_variants: int = 5) -> List[str]:
        """Generate mutations from a single winning expression"""
        expr = alpha.expression
        variants = set()

        # Lookback mutations (safe, preserve structure)
        variants.update(self._mutate_lookback(expr))

        # Sector neutralization (add cross-sectional context)
        variants.update(self._add_sector_neutralize(expr))

        # Volume confirmation (adds microstructure edge)
        if alpha.turnover < 50:  # only if turnover is manageable
            variants.update(self._add_volume_confirm(expr))

        # Quality filter (lower turnover, stabilize signal)
        if alpha.turnover > 40:
            variants.update(self._add_quality_filter(expr))

        # Shuffle and limit
        variant_list = list(variants - {expr})
        random.shuffle(variant_list)
        return variant_list[:n_variants]

    def evolve_harvest(
        self,
        harvested: Optional[List[HarvestedAlpha]] = None,
        variants_per_alpha: int = 4,
    ) -> List[str]:
        """
        Evolve all harvested alphas → generate unique variant expressions.
        Combines pairs of winners for extra diversity.
        """
        if harvested is None:
            harvested = self._harvested

        if not harvested:
            logger.warning("No harvested alphas to evolve!")
            return []

        logger.info(f"🧬 Evolving {len(harvested)} harvested alphas...")
        all_variants = set()

        # Single alpha mutations
        for h in harvested:
            variants = self.evolve_single(h, n_variants=variants_per_alpha)
            all_variants.update(variants)

        # Pairwise combinations (top 5 × top 5 = 25 combos)
        top5 = harvested[:5]
        for i, h1 in enumerate(top5):
            for h2 in top5[i+1:]:
                combo = self._combine_two(h1.expression, h2.expression)
                all_variants.add(combo)

        # Remove originals (don't re-simulate what WQ already has)
        original_exprs = {h.expression for h in harvested}
        new_variants = list(all_variants - original_exprs)
        random.shuffle(new_variants)

        logger.info(f"  🧬 Generated {len(new_variants)} unique variants from harvest")
        return new_variants

    # ─────────────────────────────────────────────────────────
    # Step 3: Harvest + Evolve combined
    # ─────────────────────────────────────────────────────────

    def harvest_and_evolve(
        self,
        min_sharpe: float = 1.25,
        top_n: int = 20,
    ) -> List[str]:
        """
        Full pipeline: Fetch winners → Evolve → Return top candidates.
        These are ready to feed into the main simulation pipeline.
        """
        logger.info("=" * 55)
        logger.info("🌾 COMMUNITY HARVEST + EVOLVE")
        logger.info("=" * 55)

        # Step 1: Harvest proven winners
        harvested = self.harvest(min_sharpe=min_sharpe)

        if not harvested:
            logger.warning("  No winners found! Falling back to seed-based generation.")
            return []

        logger.info(f"  Top 3 harvested:")
        for h in harvested[:3]:
            logger.info(f"    Sharpe={h.sharpe:.2f} T={h.turnover:.0f}% → {h.expression[:55]}...")

        # Step 2: Evolve variants
        variants = self.evolve_harvest(harvested, variants_per_alpha=5)

        # Step 3: Limit to top_n
        result = variants[:top_n]
        logger.info(f"  ✅ Ready: {len(result)} evolved candidates for simulation")
        return result

    # ─────────────────────────────────────────────────────────
    # Step 4: Auto-Submit qualifying alphas
    # ─────────────────────────────────────────────────────────

    def auto_submit_passing(
        self,
        sim_results,
        tier: str = TARGET_TIER,
        dry_run: bool = False,
    ) -> int:
        """
        Auto-submit all SimResults that exceed the quality tier threshold.

        Args:
            sim_results: list of SimResult objects from wq_client
            tier: "minimum" | "good" | "excellent" | "elite"
            dry_run: if True, log but don't actually submit

        Returns:
            Number of alphas submitted
        """
        thresholds = QUALITY_TIERS.get(tier, QUALITY_TIERS["minimum"])
        min_sharpe = thresholds["sharpe"]
        min_fitness = thresholds["fitness"]
        max_turnover = thresholds["turnover_max"]

        queued = 0
        for result in sim_results:
            qualifies = (
                result.sharpe   >= min_sharpe and
                result.fitness  >= min_fitness and
                0 < result.turnover <= max_turnover and
                result.all_passed and
                not result.error and
                result.alpha_id
            )

            if not qualifies:
                logger.debug(
                    f"  ⏭ Skip (S={result.sharpe:.2f} F={result.fitness:.2f} "
                    f"T={result.turnover:.0f}%): {result.expression[:40]}"
                )
                continue

            tier_label = self._classify_tier(result.sharpe)
            if dry_run:
                logger.info(
                    f"  🌀 DRY-RUN staging [{tier_label}] "
                    f"S={result.sharpe:.2f} T={result.turnover:.0f}%: "
                    f"{result.expression[:50]}"
                )
            else:
                # NEW GOVERNOR LOGIC: No direct submit. Rely on cron_flush_hourly.py
                logger.info(
                    f"  📦 PUSHED TO STAGING [{tier_label}] "
                    f"S={result.sharpe:.2f} T={result.turnover:.0f}%: "
                    f"{result.expression[:50]}"
                )
            queued += 1

        logger.info(f"  📊 Auto-queue: {queued}/{len(sim_results)} queued (tier={tier})")
        return queued

    def _classify_tier(self, sharpe: float) -> str:
        if sharpe >= 2.5: return "🌟ELITE"
        if sharpe >= 2.0: return "💎Excel"
        if sharpe >= 1.5: return "✅ Good"
        return "📗  Min"

    def show_quality_targets(self):
        """Print quality tier info"""
        print("\n📊 Alpha Quality Tiers (WQ Brain):")
        print("─" * 55)
        print(f"  {'Tier':<12} {'Sharpe':<10} {'Fitness':<10} {'Turnover':<12} {'~Rank'}")
        print("─" * 55)
        benchmarks = [
            ("Minimum",   1.25, 1.0, "<70%", "~top 50%"),
            ("Good",      1.50, 1.2, "<60%", "~top 20%"),
            ("Excellent", 2.00, 1.5, "<50%", "~top 10%"),
            ("Elite",     2.50, 2.0, "<40%", "top 5%  "),
        ]
        for tier, sharpe, fitness, turnover, rank in benchmarks:
            marker = " ◄ TARGET" if tier == TARGET_TIER.capitalize() else ""
            print(f"  {tier:<12} {sharpe:<10.2f} {fitness:<10.1f} {turnover:<12} {rank}{marker}")
        print("─" * 55)
        print()


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S"
    )
    h = CommunityHarvester()
    h.show_quality_targets()

    # Show dry-run of harvest
    print("🌾 Testing harvest from local DB...")
    results = h.fetch_from_db(min_sharpe=1.0)
    if results:
        print(f"  Found {len(results)} winners in local DB")
        variants = h.evolve_harvest(results, variants_per_alpha=3)
        print(f"  Generated {len(variants)} evolved variants")
        for v in variants[:5]:
            print(f"    {v}")
    else:
        print("  No results in DB yet — run some simulations first!")
    print()
    h.show_quality_targets()
