"""
alpha_dna.py — Alpha DNA Learning Loop
======================================

Tư duy cốt lõi:
  "Thay vì generate ngẫu nhiên, học từ những gì WQ Brain đã reward
   và penalize → bias generator về phía winning DNA"

3 tính năng chính:
  1. EXTRACT DNA   — phân tích winners → extract operator patterns, 
                     lookback distributions, field usage
  2. LEARN & WEIGHT— cập nhật generator_weights.json (dùng bởi generator.py)
  3. DELETE BAD    — tự động xoá simulate xấu qua WQ Brain API
                     để giữ portfolio sạch

Vòng lặp học:
  simulate_batch() → analyze_results() → update_weights() → delete_bad() 
  → generate_biased_batch() → ... (lặp lại, cải thiện liên tục)

Usage:
    from alpha_dna import AlphaDNA
    dna = AlphaDNA()
    dna.learn_from_results(sim_results)  # học từ kết quả mới nhất
    dna.delete_bad_sims(sim_results)     # xoá sims xấu khỏi WQ Brain
    weights = dna.get_weights()          # dùng trong generator
"""

import re
import os
import json
import time
import math
import logging
import sqlite3
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

DNA_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "generator_weights.json")
DNA_DB_PATH      = os.path.join(os.path.dirname(__file__), "alpha_results.db")

# ── Tier thresholds ──────────────────────────────────────────
TIERS = {
    "elite":     {"sharpe": 2.5, "fitness": 2.0},
    "excellent": {"sharpe": 2.0, "fitness": 1.5},
    "good":      {"sharpe": 1.5, "fitness": 1.2},
    "minimum":   {"sharpe": 1.25, "fitness": 1.0},
    "bad":       {"sharpe": 0.0,  "fitness": 0.0},  # anything below minimum
}

# ── Default arm → strategy cluster weights (uniform start) ──
DEFAULT_ARM_WEIGHTS: Dict[str, float] = {
    "llm":          0.20,
    "evolved":      0.20,
    "harvested":    0.15,
    "rareop":       0.15,
    "seeded":       0.15,
    "deterministic": 0.15,
}
PRIOR = 0.1   # Bayesian prior for acceptance rate smoothing


# ============================================================
# Adaptive Arm Weights — wired to WQ acceptance rates
# ============================================================
def get_adaptive_weights(
    acceptance_rates: Dict[str, float],
) -> Dict[str, float]:
    """
    Bias generator arm weights toward arms with high WQ acceptance rates.

    Bayesian smoothing: effective_rate = (observed * resolved + prior * n)
                        / (resolved + n)
    Here `acceptance_rates` is assumed to already carry Bayesian smoothing
    (i.e. computed as (accepted + 1) / (resolved + 2)), so no further prior
    need be applied — the 0.4 floor in the formula already serves as the
    cold-start anchor.

    Formula: weight ∝ base_weight * (0.4 + 0.6 * smooth_accept_rate)

    Args:
        acceptance_rates: dict[arm_name] -> Bayesian-smoothed P(accept | submitted),
                          e.g. from BudgetAllocator.arm_snapshot() or
                          tracker.acceptance_rate_by_arm() with smoothing.

    Returns:
        dict[arm_name] -> float, all normalized to sum to 1.0.
        Arms not in acceptance_rates keep their DEFAULT_ARM_WEIGHTS share.
    """
    if not acceptance_rates:
        return dict(DEFAULT_ARM_WEIGHTS)

    # Compute raw (unscaled) weights
    raw: Dict[str, float] = {}
    for arm, base in DEFAULT_ARM_WEIGHTS.items():
        rate = acceptance_rates.get(arm, 0.5)   # 0.5 = cold-start prior
        raw[arm] = base * (0.4 + 0.6 * rate)

    total = sum(raw.values())
    if total <= 0.0:
        return dict(DEFAULT_ARM_WEIGHTS)
    return {arm: w / total for arm, w in raw.items()}

# ── Operators to track ───────────────────────────────────────
TRACKED_OPERATORS = [
    "ts_corr", "ts_delta", "ts_mean", "ts_std_dev", "ts_rank",
    "ts_sum", "ts_decay_linear", "ts_zscore", "ts_delay", "ts_min",
    "ts_max", "ts_skewness", "ts_covariance", "ts_arg_max",
    "group_neutralize", "group_rank", "group_mean", "group_zscore",
    "rank", "zscore", "signed_power", "sigmoid", "scale", "abs",
    "log", "sign", "power",
]

TRACKED_FIELDS = [
    "close", "open", "high", "low", "volume", "vwap", "returns",
    "adv20", "adv60", "cap", "sales", "eps", "book_value",
    "ebitda", "debt", "equity",
]


# ============================================================
# DNA Weights — what the generator should prioritize
# ============================================================
@dataclass
class DNAWeights:
    """
    Weights that bias the generator toward patterns that WQ Brain rewards.
    Updated after every simulation batch via learn_from_results().
    """
    # Operator win rates (0.0 - 1.0, higher = use more often)
    operator_weights: Dict[str, float] = field(default_factory=dict)

    # Field usage win rates
    field_weights: Dict[str, float] = field(default_factory=dict)

    # Preferred lookback window ranges [min, max]
    lookback_distribution: Dict[str, int] = field(default_factory=lambda: {
        "short_min": 3,  "short_max": 10,
        "mid_min":   10, "mid_max":   30,
        "long_min":  30, "long_max":  120,
        "pref_mode": "mid",  # which range to prefer
    })

    # Nesting depth preferences
    preferred_depth: int = 3  # 2-4 is sweet spot

    # Theme hit rates
    theme_weights: Dict[str, float] = field(default_factory=lambda: {
        "microstructure":  0.20,
        "quality":         0.20,
        "behavioural":     0.15,
        "cross_sectional": 0.20,
        "regime":          0.15,
        "fundamental":     0.10,
    })

    # Generation history
    total_simulated: int = 0
    total_winners: int = 0
    last_updated: str = ""

    @property
    def win_rate(self) -> float:
        if self.total_simulated == 0:
            return 0.0
        return self.total_winners / self.total_simulated

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DNAWeights":
        return cls(**d)

    @classmethod
    def default(cls) -> "DNAWeights":
        """Neutral starting weights"""
        w = cls()
        w.operator_weights = {op: 0.5 for op in TRACKED_OPERATORS}
        w.field_weights     = {f: 0.5 for f in TRACKED_FIELDS}
        return w


# ============================================================
# Alpha DNA
# ============================================================
class AlphaDNA:
    """
    Continuous learning system for alpha generation.

    After every batch:
      1. Classify results by tier
      2. Extract winning operators/fields/lookbacks
      3. Penalize losing operators
      4. Update generator_weights.json
      5. Optionally delete bad sims via API
    """

    def __init__(self, wq_client=None):
        self.client = wq_client  # optional WQClient for deletion
        self.weights = self._load_weights()

    # ─────────────────────────────────────────────────────────
    # Weights I/O
    # ─────────────────────────────────────────────────────────

    def _load_weights(self) -> DNAWeights:
        if os.path.exists(DNA_WEIGHTS_PATH):
            try:
                with open(DNA_WEIGHTS_PATH, "r") as f:
                    return DNAWeights.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")
        return DNAWeights.default()

    def save_weights(self):
        from datetime import datetime
        self.weights.last_updated = datetime.now().isoformat()
        with open(DNA_WEIGHTS_PATH, "w") as f:
            json.dump(self.weights.to_dict(), f, indent=2)
        logger.info(f"  💾 DNA weights saved → {DNA_WEIGHTS_PATH}")

    def get_weights(self) -> DNAWeights:
        return self.weights

    # ─────────────────────────────────────────────────────────
    # Expression Analysis
    # ─────────────────────────────────────────────────────────

    def _extract_operators(self, expr: str) -> List[str]:
        """Extract all function names used in an expression"""
        found = []
        for op in TRACKED_OPERATORS:
            if re.search(r'\b' + op + r'\s*\(', expr):
                found.append(op)
        return found

    def _extract_fields(self, expr: str) -> List[str]:
        """Extract all data fields used in an expression"""
        return [f for f in TRACKED_FIELDS if re.search(r'\b' + f + r'\b', expr)]

    def _extract_lookbacks(self, expr: str) -> List[int]:
        """Extract all numeric lookback parameters"""
        nums = re.findall(r'\b(\d+)\b', expr)
        return [int(n) for n in nums if 2 <= int(n) <= 500]

    def _classify_tier(self, sharpe: float, fitness: float) -> str:
        if sharpe >= 2.5 and fitness >= 2.0: return "elite"
        if sharpe >= 2.0 and fitness >= 1.5: return "excellent"
        if sharpe >= 1.5 and fitness >= 1.2: return "good"
        if sharpe >= 1.25 and fitness >= 1.0: return "minimum"
        return "bad"

    def _calc_depth(self, expr: str) -> int:
        """Calculate expression nesting depth"""
        max_d = d = 0
        for ch in expr:
            if ch == '(':   d += 1; max_d = max(max_d, d)
            elif ch == ')': d -= 1
        return max_d

    # ─────────────────────────────────────────────────────────
    # Core Learning
    # ─────────────────────────────────────────────────────────

    def analyze_results(self, sim_results) -> Dict[str, Any]:
        """
        Analyze a batch of SimResult objects.
        Returns DNA analysis report.
        """
        if not sim_results:
            return {}

        winners = [r for r in sim_results if r.sharpe >= 1.25 and not r.error]
        losers  = [r for r in sim_results if r.sharpe < 1.25 or r.error]

        report = {
            "total": len(sim_results),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(sim_results) if sim_results else 0,
            "tiers": Counter(),
            "winner_ops": Counter(),
            "loser_ops": Counter(),
            "winner_fields": Counter(),
            "loser_fields": Counter(),
            "winner_lookbacks": [],
            "winner_depth": [],
        }

        for r in sim_results:
            tier = self._classify_tier(r.sharpe, r.fitness)
            report["tiers"][tier] += 1

        for r in winners:
            expr = r.expression
            for op in self._extract_operators(expr):
                report["winner_ops"][op] += 1
            for f in self._extract_fields(expr):
                report["winner_fields"][f] += 1
            report["winner_lookbacks"].extend(self._extract_lookbacks(expr))
            report["winner_depth"].append(self._calc_depth(expr))

        for r in losers:
            expr = r.expression or ""
            for op in self._extract_operators(expr):
                report["loser_ops"][op] += 1
            for f in self._extract_fields(expr):
                report["loser_fields"][f] += 1

        # Best performers
        report["top3"] = sorted(
            [r for r in sim_results if not r.error],
            key=lambda x: x.sharpe, reverse=True
        )[:3]

        return report

    def update_weights(self, report: Dict[str, Any], learning_rate: float = 0.15):
        """
        Update DNA weights based on analysis report.
        Uses exponential moving average: new_w = (1-lr)*old_w + lr*signal
        """
        if not report:
            return

        total_w = report["winners"] + report["losers"] + 1e-6
        total   = report["total"]

        self.weights.total_simulated += total
        self.weights.total_winners   += report["winners"]

        # ── Operator weights ──────────────────────────────────
        for op in TRACKED_OPERATORS:
            w_count = report["winner_ops"].get(op, 0)
            l_count = report["loser_ops"].get(op, 0)

            # Signal: what fraction of its appearances are in winners?
            appearances = w_count + l_count
            if appearances > 0:
                win_fraction = w_count / appearances
            else:
                win_fraction = 0.5  # no data → neutral

            # EMA update
            old_w = self.weights.operator_weights.get(op, 0.5)
            new_w = (1 - learning_rate) * old_w + learning_rate * win_fraction

            # Soft bounds [0.05, 0.95] — never fully kill an operator
            self.weights.operator_weights[op] = max(0.05, min(0.95, new_w))

        # ── Field weights ─────────────────────────────────────
        for f in TRACKED_FIELDS:
            w_count = report["winner_fields"].get(f, 0)
            l_count = report["loser_fields"].get(f, 0)
            appearances = w_count + l_count
            win_fraction = (w_count / appearances) if appearances > 0 else 0.5

            old_w = self.weights.field_weights.get(f, 0.5)
            new_w = (1 - learning_rate) * old_w + learning_rate * win_fraction
            self.weights.field_weights[f] = max(0.05, min(0.95, new_w))

        # ── Preferred depth ───────────────────────────────────
        if report["winner_depth"]:
            mean_depth = sum(report["winner_depth"]) / len(report["winner_depth"])
            # EMA toward winning mean depth
            self.weights.preferred_depth = int(
                round(0.85 * self.weights.preferred_depth + 0.15 * mean_depth)
            )
            self.weights.preferred_depth = max(2, min(6, self.weights.preferred_depth))

        # ── Lookback distributions ────────────────────────────
        if report["winner_lookbacks"]:
            lbs = report["winner_lookbacks"]
            short = [l for l in lbs if l <= 10]
            mid   = [l for l in lbs if 10 < l <= 40]
            long_ = [l for l in lbs if l > 40]

            # Determine preferred range by count
            counts = {"short": len(short), "mid": len(mid), "long": len(long_)}
            self.weights.lookback_distribution["pref_mode"] = max(counts, key=counts.get)
            if lbs:
                avg_lb = sum(lbs) / len(lbs)
                # Update preferred range mid-point
                if avg_lb <= 10:
                    self.weights.lookback_distribution["short_max"] = max(5, int(avg_lb * 1.3))
                elif avg_lb <= 40:
                    self.weights.lookback_distribution["mid_max"] = max(15, int(avg_lb * 1.3))

        logger.info(
            f"  🧬 DNA updated | win_rate={report['win_rate']:.1%} "
            f"| total_sims={self.weights.total_simulated} "
            f"| total_winners={self.weights.total_winners}"
        )

    def learn_from_results(self, sim_results, learning_rate: float = 0.15):
        """
        Main entry: analyze results + update weights + save.
        Call this after every simulate_batch().
        """
        logger.info("🧬 DNA learning from batch results...")
        report = self.analyze_results(sim_results)

        if report:
            self._log_report(report)
            self.update_weights(report, learning_rate)
            self.save_weights()
            self._persist_winners_to_db(sim_results)

        return report

    # ─────────────────────────────────────────────────────────
    # Delete bad simulations from WQ Brain
    # ─────────────────────────────────────────────────────────

    def delete_bad_sims(
        self,
        sim_results,
        delete_threshold_sharpe: float = 0.5,
        dry_run: bool = False,
    ) -> int:
        """
        Auto-delete simulations with Sharpe < threshold from WQ Brain.
        This keeps your Unsubmitted portfolio clean.

        Args:
            sim_results: list of SimResult objects
            delete_threshold_sharpe: delete if Sharpe < this value
            dry_run: if True, log but don't delete

        Returns:
            Number of alphas deleted
        """
        if self.client is None:
            logger.warning("  No WQClient attached — cannot delete sims")
            return 0

        to_delete = [
            r for r in sim_results
            if (r.sharpe < delete_threshold_sharpe or r.error)
            and r.alpha_id
        ]

        if not to_delete:
            logger.info("  🧹 No bad sims to delete")
            return 0

        deleted = 0
        logger.info(f"  🗑️  Deleting {len(to_delete)} bad sims (Sharpe < {delete_threshold_sharpe})...")

        for r in to_delete:
            if dry_run:
                logger.info(f"    DRY DELETE: Sharpe={r.sharpe:.2f} | {r.expression[:50]}")
                deleted += 1
            else:
                ok = self._delete_alpha(r.alpha_id)
                if ok:
                    logger.info(f"    ✅ Deleted: Sharpe={r.sharpe:.2f} | {r.alpha_id}")
                    deleted += 1
                time.sleep(0.5)  # Rate limit

        logger.info(f"  🧹 Cleaned {deleted}/{len(to_delete)} bad sims")
        return deleted

    def _delete_alpha(self, alpha_id: str) -> bool:
        """DELETE /alphas/{alpha_id}"""
        try:
            from wq_client import API_BASE
            r = self.client._api_request(
                "delete",
                f"{API_BASE}/alphas/{alpha_id}"
            )
            return r is not None and r.status_code in (200, 204)
        except Exception as e:
            logger.warning(f"    Delete error: {e}")
            return False

    # ─────────────────────────────────────────────────────────
    # Biased Generation Tips — used by generator.py
    # ─────────────────────────────────────────────────────────

    def get_top_operators(self, n: int = 5) -> List[str]:
        """Return top N operators by DNA weight"""
        sorted_ops = sorted(
            self.weights.operator_weights.items(),
            key=lambda x: x[1], reverse=True
        )
        return [op for op, w in sorted_ops[:n]]

    def get_top_fields(self, n: int = 4) -> List[str]:
        """Return top N fields by DNA weight"""
        sorted_fields = sorted(
            self.weights.field_weights.items(),
            key=lambda x: x[1], reverse=True
        )
        return [f for f, w in sorted_fields[:n]]

    def get_preferred_lookback(self) -> int:
        """Sample a preferred lookback window based on DNA"""
        import random
        mode = self.weights.lookback_distribution.get("pref_mode", "mid")
        if mode == "short":
            lo = self.weights.lookback_distribution.get("short_min", 3)
            hi = self.weights.lookback_distribution.get("short_max", 10)
        elif mode == "long":
            lo = self.weights.lookback_distribution.get("long_min", 40)
            hi = self.weights.lookback_distribution.get("long_max", 120)
        else:  # mid
            lo = self.weights.lookback_distribution.get("mid_min", 10)
            hi = self.weights.lookback_distribution.get("mid_max", 30)
        return random.randint(lo, hi)

    def generate_dna_guided_expression(self) -> Optional[str]:
        """
        Generate a single expression biased by current DNA weights.
        Uses top operators and fields from winners.
        """
        import random
        top_ops    = self.get_top_operators(n=6)
        top_fields = self.get_top_fields(n=4)
        lb         = self.get_preferred_lookback()
        lb2        = self.get_preferred_lookback()
        lb3        = self.get_preferred_lookback()

        templates = [
            # Two-operator composite from top DNA
            f"rank(ts_corr({top_fields[0]}, {top_fields[1]}, {lb})) * rank(ts_mean({top_fields[0]}, {lb2}))",
            f"rank({top_ops[0]}({top_fields[0]}, {lb})) * rank({top_ops[1]}({top_fields[1]}, {lb2}))",
            f"-rank({top_ops[0]}({top_fields[0]}, {lb})) * rank({top_ops[1]}({top_fields[0]}, {lb2}))",
            # DNA-guided group ops
            f"group_neutralize(rank({top_ops[0]}({top_fields[0]}, {lb})), sector)",
            f"rank({top_ops[0]}({top_fields[0]}, {lb}) / ({top_ops[1]}({top_fields[0]}, {lb2}) + 0.001))",
            # Field value factor (if fundamentals are in top_fields)
            f"-rank({top_fields[0]} / ({top_fields[1]} + 1))",
        ]

        # weighted sample — pick template with best DNA match
        expr = random.choice(templates)
        return expr

    def generate_dna_batch(self, n: int = 10) -> List[str]:
        """Generate n DNA-guided expressions"""
        exprs = set()
        attempts = 0
        while len(exprs) < n and attempts < n * 3:
            e = self.generate_dna_guided_expression()
            if e:
                exprs.add(e)
            attempts += 1
        return list(exprs)

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def _persist_winners_to_db(self, sim_results):
        """Save winning alphas to local DB for future harvesting"""
        try:
            conn = sqlite3.connect(DNA_DB_PATH)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alphas (
                    id TEXT PRIMARY KEY,
                    expression TEXT,
                    sharpe REAL,
                    fitness REAL,
                    turnover REAL,
                    region TEXT,
                    universe TEXT,
                    all_passed INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for r in sim_results:
                if r.sharpe >= 1.0 and r.alpha_id:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO alphas 
                            (id, expression, sharpe, fitness, turnover, region, universe, all_passed)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            r.alpha_id, r.expression, r.sharpe, r.fitness,
                            r.turnover, "USA", "TOP3000",
                            1 if r.all_passed else 0,
                        ))
                    except Exception:
                        pass
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"DB persist error: {e}")

    # ─────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────

    def _log_report(self, report: Dict):
        tier = report.get("tiers", {})
        logger.info(
            f"  📊 Batch: {report['total']} sims | "
            f"W={report['winners']} ({report['win_rate']:.0%}) | "
            f"elite={tier.get('elite',0)} excellent={tier.get('excellent',0)} "
            f"good={tier.get('good',0)} min={tier.get('minimum',0)} bad={tier.get('bad',0)}"
        )
        if report.get("winner_ops"):
            top_ops = report["winner_ops"].most_common(3)
            logger.info(f"  🏆 Top winning ops: {top_ops}")
        if report.get("top3"):
            for r in report["top3"]:
                logger.info(f"    ★ S={r.sharpe:.2f} F={r.fitness:.2f} T={r.turnover:.0f}% | {r.expression[:55]}")

    def show_dna_status(self):
        """Print current DNA weights summary"""
        w = self.weights
        print("\n🧬 Alpha DNA Status")
        print("─" * 55)
        print(f"  Total simulated: {w.total_simulated}")
        print(f"  Total winners:   {w.total_winners}")
        print(f"  Win rate:        {w.win_rate:.1%}")
        print(f"  Preferred depth: {w.preferred_depth}")
        print(f"  Last updated:    {w.last_updated or 'Never'}")

        print("\n  🔝 Top operators:")
        top_ops = sorted(w.operator_weights.items(), key=lambda x: x[1], reverse=True)[:6]
        for op, score in top_ops:
            bar = "█" * int(score * 10)
            print(f"    {op:<22} {score:.2f} {bar}")

        print("\n  📊 Top fields:")
        top_fields = sorted(w.field_weights.items(), key=lambda x: x[1], reverse=True)[:6]
        for f, score in top_fields:
            bar = "█" * int(score * 10)
            print(f"    {f:<22} {score:.2f} {bar}")
        print()


# ============================================================
# Standalone run
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    dna = AlphaDNA()
    dna.show_dna_status()

    # Test DNA-guided generation
    print("🧬 DNA-guided expressions:")
    batch = dna.generate_dna_batch(n=5)
    for e in batch:
        print(f"  {e}")
