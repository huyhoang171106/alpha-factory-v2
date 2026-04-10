"""
budget_allocator.py

Two-stage budget economy:
- Tier-1 cheap acceptance using quality + novelty
- Tier-2 expected value allocation with Thompson sampling
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ArmState:
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    rewards: float = 0.0


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

    def expected_value(self, arm_name: str, quality_norm: float, novelty: float) -> float:
        arm = self._arm(arm_name)
        sample = self.rng.betavariate(arm.alpha, arm.beta)
        uncertainty_bonus = 0.0
        if arm.pulls > 0:
            uncertainty_bonus = math.sqrt(math.log(self.total_pulls + 2.0) / arm.pulls)
        else:
            uncertainty_bonus = 1.0
        ev = (
            0.45 * quality_norm
            + 0.35 * novelty
            + 0.20 * sample
            + self.exploration_weight * 0.10 * uncertainty_bonus
        )
        return max(0.0, min(1.0, ev))

    def tier2_accept(self, arm_name: str, quality_norm: float, novelty: float) -> Tuple[bool, float]:
        ev = self.expected_value(arm_name, quality_norm, novelty)
        return ev >= self.min_expected_value, ev

    def update(self, arm_name: str, reward: float):
        arm = self._arm(arm_name)
        r = 1.0 if reward > 0 else 0.0
        arm.alpha += r
        arm.beta += (1.0 - r)
        arm.pulls += 1
        arm.rewards += r
        self.total_pulls += 1

    def arm_snapshot(self, arm_name: str) -> dict:
        arm = self._arm(arm_name)
        mean = arm.alpha / max(1e-9, (arm.alpha + arm.beta))
        return {"alpha": arm.alpha, "beta": arm.beta, "pulls": arm.pulls, "mean": mean}
