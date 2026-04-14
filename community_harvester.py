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

    # ─────────────────────────────────────────────────────────
    # Deep Mutation Strategies
    # ─────────────────────────────────────────────────────────

    # ── Helper: Regime detection ──────────────────────────────

    @staticmethod
    def _regime_from_volatility(alpha_expr: str) -> str:
        """
        Compute realized_vol = ts_std(returns, 20) / ts_mean(returns, 20)
        and return "low_vol" | "medium_vol" | "high_vol".

        Since we cannot execute expressions at harvest time, this method
        uses heuristic lookback detection in the alpha expression itself
        to guess the prevailing regime, and falls back to "medium_vol".
        """
        canonical = alpha_expr.lower()

        # Count how many long (>=60) vs short (<=20) lookback windows appear
        long_lookbacks = re.findall(r'ts_\w+\([^)]*,\s*(\d+)\)', canonical)
        short_lookbacks = re.findall(r'ts_\w+\([^)]*,\s*(\d+)\)', canonical)

        long_count = sum(1 for n in long_lookbacks if int(n) >= 60)
        short_count = sum(1 for n in short_lookbacks if int(n) <= 20)

        if long_count > short_count * 2:
            return "low_vol"
        elif short_count > long_count * 2:
            return "high_vol"
        return "medium_vol"

    # ─────────────────────────────────────────────────────────
    # Strategy 1: Operator Substitution Mutations
    # ─────────────────────────────────────────────────────────

    # Semantic equivalence groups (single-entry pairs for clean substitution)
    OPERATOR_EQUIVALENCE_GROUPS = [
        ("ts_corr(", "ts_cov("),
        ("ts_cov(", "ts_corr("),
        ("decay_linear(", "ts_mean("),
        ("ts_mean(", "decay_linear("),
        ("rank(", "ts_zscore("),
        ("ts_rank(", "ts_percentile("),
    ]

    def apply_operator_substitution(self, alpha_expr: str) -> list[str]:
        """
        Replace top-level operators with semantic alternates while preserving
        the original argument list and surrounding expression text.
        """
        original = alpha_expr.strip()
        mutants: list[str] = []

        for old_op, new_op in self.OPERATOR_EQUIVALENCE_GROUPS:
            i = 0
            while i < len(original):
                if original[i : i + len(old_op)] != old_op:
                    i += 1
                    continue
                if i > 0 and original[i - 1].isalnum():
                    i += 1
                    continue

                depth = 0
                for j in range(i):
                    if original[j] == "(":
                        depth += 1
                    elif original[j] == ")":
                        depth -= 1
                if depth != 0:
                    i += 1
                    continue

                rest = original[i + len(old_op) :]
                depth = 1
                arg_end = -1
                for k, ch in enumerate(rest):
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            arg_end = k
                            break
                if arg_end == -1:
                    i += len(old_op)
                    continue

                args = rest[:arg_end]
                suffix = rest[arg_end + 1 :]
                mutant = original[:i] + new_op + args + ")" + suffix
                if mutant != original and mutant.count("(") == mutant.count(")"):
                    mutants.append(mutant)
                i += len(old_op)

        return list(dict.fromkeys(mutants))

    def decompose_and_recombine(
        self,
        alpha_expr: str,
        other_winners: list[str],
    ) -> list[str]:
        """
        If top-level is a binary operator (+, -, *, /), extract left and right
        sub-expressions and recombine each with a different winner's signal.

        Returns mutated candidate expressions.
        """
        if len(other_winners) < 2:
            return []

        canonical = alpha_expr.strip()

        # Find the outermost top-level binary operator (+, -, *, /)
        # by scanning for the last valid binary op at nesting depth 0.
        binary_ops = ["+", "-", "*", "/"]
        best_pos = -1
        best_op: str | None = None
        depth = 0

        for i, ch in enumerate(canonical):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch in binary_ops and depth == 0 and i > 0:
                # Skip '-' that is part of a negative number (e.g. "x-5")
                if ch == "-" and canonical[i - 1].isalnum():
                    continue
                # Skip if the raw previous char is also a binary operator
                # (handles chains like "+-" or "++")
                prev_raw = canonical[i - 1]
                if prev_raw in binary_ops:
                    continue
                best_pos = i
                best_op = ch

        if best_pos == -1 or best_op is None:
            # DEBUG
            import sys as _sys
            _sys.stderr.write(f"[DEBUG decompose] no binary op found in: {canonical!r}\n")
            return []

        left = canonical[:best_pos].strip()
        right = canonical[best_pos + 1:].strip()

        # Guard: reject trivial operands
        if len(left) < 3 or len(right) < 3:
            return []

        mutants: list[str] = []
        used = set()
        for winner in other_winners[:6]:
            w = winner.strip()
            if not w or w in used:
                continue

            # Recombine left with winner's signal on the right
            m1 = f"({left}) {best_op} ({w})"
            # Recombine right with winner's signal on the left
            m2 = f"({w}) {best_op} ({right})"

            for m in [m1, m2]:
                if m != alpha_expr and m.count("(") == m.count(")"):
                    mutants.append(m)
                    used.add(w)

        return mutants[:6]

    # ─────────────────────────────────────────────────────────
    # Strategy 3: Regime-Conditional Mutation
    # ─────────────────────────────────────────────────────────

    def apply_regime_conditional_mutation(
        self,
        alpha_expr: str,
        regime: str,
    ) -> list[str]:
        """
        Apply regime-aware mutations:

        - high_vol:  replace long lookbacks (>=60) with shorter (<=20),
                     add ts_std weighting
        - low_vol:   replace short lookbacks (<=20) with longer (>=60),
                     favor mean-reversion signals
        - transition: add ts_delta wrapping and faster decay operators
        """
        canonical = alpha_expr.strip()
        mutants: list[str] = []

        if regime == "high_vol":
            # Replace long lookbacks (>=60) with shorter (20)
            mutants_long = re.sub(
                r'(ts_\w+)\([^,]+,\s*(\d+)\)',
                lambda m: (
                    m.group(1) + "(" + m.group(0).split(",")[0].split("(")[1] + ", 20)"
                    if int(m.group(2)) >= 60
                    else m.group(0)
                ),
                canonical,
            )
            if mutants_long != canonical:
                mutants.append(mutants_long)

            # Add ts_std weighting (volatility dampening)
            if "ts_std" not in canonical.lower():
                mutants.append(
                    f"rank({canonical}) * rank(ts_std({canonical}, 20))"
                )

        elif regime == "low_vol":
            # Replace short lookbacks (<=20) with longer (60)
            mutants_short = re.sub(
                r'(ts_\w+)\([^,]+,\s*(\d+)\)',
                lambda m: (
                    m.group(1) + "(" + m.group(0).split(",")[0].split("(")[1] + ", 60)"
                    if 2 <= int(m.group(2)) <= 20
                    else m.group(0)
                ),
                canonical,
            )
            if mutants_short != canonical:
                mutants.append(mutants_short)

            # Add mean-reversion wrapper: sign change on negative ts_delta
            if "sign(" not in canonical.lower():
                mutants.append(f"sign(-ts_delta({canonical}, 5))")

        elif regime == "transition":
            # Wrap with ts_delta and decay_linear
            if "ts_delta" not in canonical.lower():
                mutants.append(f"ts_delta({canonical}, 5)")

            if "decay_linear" not in canonical.lower():
                mutants.append(f"decay_linear({canonical}, 6)")

            if mutants:
                # Second-order delta (rate of change of rate)
                mutants.append(f"ts_delta(ts_delta({canonical}, 5), 5)")

        return mutants

    # ─────────────────────────────────────────────────────────
    # Strategy 4: Recent-Sharpe Community Fetch
    # ─────────────────────────────────────────────────────────

    def fetch_community_alphas(
        self,
        min_recent_sharpe: float = 1.0,
        limit: int = 50,
    ) -> List[HarvestedAlpha]:
        """
        Fetch community/shared alphas from WQ Brain sorted by recent Sharpe
        (not just top performance), filtered to top 20% by recent Sharpe.

        Endpoint: GET /community/alphas?sort=recent_sharpe&sharpe[gte]=X
        """
        logger.info(
            f"🌐 Fetching community alphas (recent_sharpe >= {min_recent_sharpe}, "
            f"top 20%, limit={limit})..."
        )

        # Top 20% threshold: in practice WQ community distributions vary;
        # recent_sharpe >= 1.0 is a reasonable proxy for top quartile
        params = {
            "limit": limit,
            "offset": 0,
            "sort": "recent_sharpe",          # ← recent Sharpe sort
            "sharpe": f"gte:{min_recent_sharpe}",
            "region": "USA",
        }

        try:
            r = self._ensure_client()._api_request(
                "get",
                f"{API_BASE}/community/alphas",
                params=params,
            )
            if r is None or r.status_code != 200:
                logger.warning(
                    f"  Community fetch failed: {r.status_code if r else 'None'}"
                )
                return []

            data = r.json()
            alphas_raw = data.get("results", data.get("alphas", []))

            harvested: list[HarvestedAlpha] = []
            for a in alphas_raw:
                expr = (
                    a.get("regular", {}).get("code", "") or
                    a.get("expression", "") or
                    a.get("code", "")
                )
                if not expr:
                    continue

                recent_sharpe = float(a.get("recentSharpe", a.get("recent_sharpe", 0)) or 0)
                if recent_sharpe < min_recent_sharpe:
                    continue

                h = HarvestedAlpha(
                    alpha_id   = a.get("id", ""),
                    expression = expr,
                    sharpe     = recent_sharpe,
                    fitness    = float(a.get("fitness", 0) or 0),
                    turnover   = float(a.get("turnover", 0) or 0),
                    region     = a.get("settings", {}).get("region", "USA"),
                    universe   = a.get("settings", {}).get("universe", "TOP3000"),
                    source     = "community",
                )
                harvested.append(h)

            logger.info(f"  🌐 Fetched {len(harvested)} community alphas (recent Sharpe)")
            return harvested

        except Exception as e:
            logger.error(f"  Community fetch error: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # Deep Mutation Orchestrator
    # ─────────────────────────────────────────────────────────

    def mutate_deep(
        self,
        alphas: list[str],
        regime: str | None = None,
    ) -> list[dict]:
        """
        Run all deep mutation strategies on a list of alpha expressions and
        return a list of mutation records.

        Args:
            alphas: list of alpha expression strings
            regime: optional regime hint ("high_vol" | "low_vol" | "transition")
                   if omitted, inferred per-alpha via _regime_from_volatility

        Returns:
            list of dicts with keys:
                expression   — mutated alpha string
                mutation_type — one of the strategy names below
                regime        — regime used ("inferred" if auto-detected)
        """
        if regime and regime not in ("high_vol", "low_vol", "transition", "medium_vol"):
            logger.warning(f"Unknown regime '{regime}', defaulting to medium_vol")
            regime = "medium_vol"

        results: list[dict] = []

        for alpha_expr in alphas:
            effective_regime = regime or self._regime_from_volatility(alpha_expr)

            # 1. Operator substitution
            for mut in self.apply_operator_substitution(alpha_expr):
                results.append({
                    "expression": mut,
                    "mutation_type": "operator_substitution",
                    "regime": effective_regime,
                })

            # 2. Decompose & recombine (requires other winners)
            # Use the alphas list itself as the winner pool (skip self)
            other_winners = [a for a in alphas if a != alpha_expr]
            for mut in self.decompose_and_recombine(alpha_expr, other_winners):
                results.append({
                    "expression": mut,
                    "mutation_type": "decompose_recombine",
                    "regime": effective_regime,
                })

            # 3. Regime-conditional mutation
            for mut in self.apply_regime_conditional_mutation(alpha_expr, effective_regime):
                results.append({
                    "expression": mut,
                    "mutation_type": f"regime_{effective_regime}",
                    "regime": effective_regime,
                })

        # Deduplicate by expression
        seen: set[str] = set()
        unique_results: list[dict] = []
        for r in results:
            if r["expression"] not in seen:
                seen.add(r["expression"])
                unique_results.append(r)

        logger.info(
            f"  🧬 Deep mutate: {len(alphas)} inputs → "
            f"{len(unique_results)} unique mutants "
            f"(regime={regime or 'inferred'})"
        )
        return unique_results


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
