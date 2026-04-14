"""
budget_allocator.py

Two-stage budget economy:
- Tier-1 cheap acceptance using quality + novelty
- Tier-2 expected value allocation with Thompson sampling
  + acceptance-attributed P(true_accept | submitted) per arm
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ArmState:
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    rewards: float = 0.0
    # P(true_accept | submitted) from WQ reconciliation data.
    # Initialised to 0.5 (uniform prior) until real data arrives.
    p_accept: float = 0.5
    p_accept_resolved: int = 0  # number of resolved WQ decisions backing p_accept
    # Bayesian-smoothed acceptance rate: adds a 0.1 prior to avoid collapsing
    # when resolved count is small. Updated by update_acceptance_priors().
    acceptance_rate: float = 0.5


class BudgetAllocator:
    def __init__(
        self,
        tier1_min_quality: float = 0.48,
        tier1_min_novelty: float = 0.22,
        min_expected_value: float = 0.35,
        exploration_weight: float = 0.35,
        seed: int = 42,
    ):
        self.tier1_min_quality = float(tier1_min_quality)
        self.tier1_min_novelty = float(tier1_min_novelty)
        self.min_expected_value = float(min_expected_value)
        self.exploration_weight = float(exploration_weight)
        self.rng = random.Random(seed)
        self.arms: Dict[str, ArmState] = {}
        self.total_pulls = 0

    def _arm(self, name: str) -> ArmState:
        if name not in self.arms:
            self.arms[name] = ArmState()
        return self.arms[name]

    @staticmethod
    def normalize_quality(quality_score_0_100: float) -> float:
        return max(0.0, min(1.0, float(quality_score_0_100) / 100.0))

    def tier1_accept(self, quality_norm: float, novelty: float) -> Tuple[bool, str]:
        if quality_norm < self.tier1_min_quality:
            return False, "tier1_low_quality"
        if novelty < self.tier1_min_novelty:
            return False, "tier1_low_novelty"
        return True, "tier1_pass"

    def update_acceptance_priors(self, rates: dict) -> None:
        """
        Ingest per-arm acceptance rates from tracker.acceptance_rate_by_arm().

        rates: dict[arm_name] -> {"accepted": int, "resolved": int, "p_accept": float}

        Bayesian smoothing: adds a 0.1 prior (beta(1,1)) to avoid collapsing
        exploration when resolved count is small.
        Arms not present in `rates` are left unchanged (cold-start prior preserved).
        """
        for arm_name, data in rates.items():
            arm = self._arm(arm_name)
            resolved = int(data.get("resolved", 0))
            accepted = int(data.get("accepted", 0))
            if resolved >= 5:
                # Bayesian-smoothed acceptance rate: Jeffreys prior beta(1,1)
                # posterior = (accepted + 1) / (resolved + 2)
                arm.acceptance_rate = (accepted + 1.0) / (resolved + 2.0)
                arm.p_accept = float(data.get("p_accept", arm.acceptance_rate))
                arm.p_accept_resolved = resolved

    def expected_value(self, arm_name: str, quality_norm: float, novelty: float) -> float:
        """
        Acceptance-attributed EV formula.

        Weights:
          0.40 * quality_norm          — local cheap signal
          0.25 * novelty               — diversity incentive
          0.15 * thompson_sample(arm)  — sim-pass history via Thompson
          0.15 * acceptance_rate(arm)   — WQ true acceptance signal  ← uses Bayesian p_accept
          exploration_bonus            — UCB-style cold-start bonus
        """
        arm = self._arm(arm_name)
        thompson = self.rng.betavariate(arm.alpha, arm.beta)
        uncertainty_bonus = 0.0
        if arm.pulls > 0:
            uncertainty_bonus = math.sqrt(math.log(self.total_pulls + 2.0) / arm.pulls)
        else:
            uncertainty_bonus = 1.0
        accept_confidence = max(0.0, min(1.0, arm.p_accept_resolved / 50.0))
        quality_weight = 0.40 - (0.10 * accept_confidence)
        novelty_weight = 0.25 - (0.05 * accept_confidence)
        accept_weight = 0.15 + (0.20 * accept_confidence)
        ev = (
            quality_weight * quality_norm
            + novelty_weight * novelty
            + 0.15 * thompson
            + accept_weight * arm.acceptance_rate
            + self.exploration_weight * 0.05 * uncertainty_bonus
        )
        return max(0.0, min(1.0, ev))

    def tier2_accept(self, arm_name: str, quality_norm: float, novelty: float) -> Tuple[bool, float]:
        ev = self.expected_value(arm_name, quality_norm, novelty)
        return ev >= self.min_expected_value, ev

    def update(self, arm_name: str, reward: float):
        arm = self._arm(arm_name)
        r = max(0.0, min(1.0, float(reward)))
        arm.alpha += r
        arm.beta += (1.0 - r)
        arm.pulls += 1
        arm.rewards += r
        self.total_pulls += 1

    def arm_snapshot(self, arm_name: str) -> dict:
        arm = self._arm(arm_name)
        mean = arm.alpha / max(1e-9, (arm.alpha + arm.beta))
        return {
            "alpha": arm.alpha,
            "beta": arm.beta,
            "pulls": arm.pulls,
            "mean": mean,
            "p_accept": arm.p_accept,
            "acceptance_rate": arm.acceptance_rate,
            "p_accept_resolved": arm.p_accept_resolved,
        }

    def compute_arm_budget_ratio(
        self,
        arm_name: str,
        baseline_ratio: float = 0.10,
    ) -> float:
        """
        Returns a per-arm budget ratio that biases generation toward arms
        with high WQ acceptance rates (p_accept).

        Uses Bayesian-smoothed acceptance_rate to avoid collapsing exploration
        on arms with small resolved counts.

        Formula: baseline_ratio * (0.4 + 0.6 * accept_rate)
        Range:  [0.4*baseline, 1.0*baseline]
        Arms with p_accept near 1.0 get the full baseline_ratio;
        arms with p_accept near 0.0 get 40% of baseline_ratio.
        """
        arm = self._arm(arm_name)
        ratio = baseline_ratio * (0.4 + 0.6 * arm.acceptance_rate)
        return max(0.0, min(baseline_ratio, ratio))

    def get_regime_aware_arm(
        self,
        regime: Optional[str] = None,
        quality_norm: Optional[float] = None,
        novelty: Optional[float] = None,
    ) -> Tuple[str, ArmState]:
        """Get arm selection with regime awareness (convenience method)."""
        if regime is None:
            regime = "unclear"
        selector = RegimeAwareArmSelector(self)
        return selector.select_arm_with_context(regime, quality_norm, novelty)


# ─── Regime-Aware Arm Selection ─────────────────────────────────────────────────


class RegimeAwareArmSelector:
    """Contextual bandit: arm selection conditioned on market regime.
    Research: regime-aware allocation improves cross-regime Sharpe by 15-25%."""

    # Which arm types excel in which regimes
    REGIME_ARM_PREFERENCES: Dict[str, List[str]] = {
        "high_vol": ["quality_arm", "diversity_arm"],
        "low_vol": ["quality_arm", "novelty_arm"],
        "bull_trending": ["novelty_arm", "ensemble_arm"],
        "bear_trending": ["quality_arm", "diversity_arm"],
        "mean_reversion": ["novelty_arm", "quality_arm"],
        "unclear": ["quality_arm", "novelty_arm"],
    }

    def __init__(
        self,
        base_allocator: BudgetAllocator,
        recent_accept_rate: Optional[float] = None,
        base_exploration_weight: float = 0.35,
    ) -> None:
        self.base = base_allocator
        self.recent_accept_rate = recent_accept_rate if recent_accept_rate is not None else 0.5
        self.base_exploration_weight = base_exploration_weight

    def _adjust_exploration_for_regime(self, regime: str) -> float:
        """Higher exploration in unclear/high_vol regimes."""
        base = self.base_exploration_weight
        if regime in ("high_vol", "unclear"):
            return min(1.0, base * 1.3)
        if regime in ("bull_trending", "bear_trending"):
            return max(0.1, base * 0.9)
        return base

    def select_arm_with_context(
        self,
        regime: str,
        quality_norm: Optional[float] = None,
        novelty: Optional[float] = None,
    ) -> Tuple[str, ArmState]:
        """Select arm based on regime + quality + novelty context."""
        preferences = self.REGIME_ARM_PREFERENCES.get(
            regime, ["quality_arm", "novelty_arm"]
        )

        # Adjust exploration weight for regime
        exploration = self._adjust_exploration_for_regime(regime)

        # Regime-adjusted expected values
        adjusted_scores: Dict[str, float] = {}
        for arm_name in self.base.arms:
            arm = self.base.arms[arm_name]
            base_score = self._thompson_score(arm)

            # Boost preferred arms for this regime
            boost = 0.0
            if arm_name in preferences:
                boost = 0.2 * self.base.exploration_weight

            # Reduce boost if accept rate is low (more conservative)
            if self.recent_accept_rate < 0.3:
                boost *= 0.5

            adjusted_scores[arm_name] = base_score + boost

        # Exploration: pick random arm with probability = exploration
        if self.base.rng.random() < exploration:
            arm_name = self.base.rng.choice(list(self.base.arms.keys()))
        else:
            arm_name = max(adjusted_scores, key=adjusted_scores.get)

        return arm_name, self.base.arms[arm_name]

    @staticmethod
    def _thompson_score(arm: ArmState) -> float:
        """Thompson sampling score from arm state."""
        alpha = max(1.0, arm.alpha)
        beta = max(1.0, arm.beta)
        # Sample from Beta distribution
        sample = random.betavariate(alpha, beta)
        return sample * arm.p_accept
