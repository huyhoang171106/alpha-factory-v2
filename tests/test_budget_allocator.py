import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from budget_allocator import BudgetAllocator


class BudgetAllocatorTests(unittest.TestCase):
    def test_tier1_rejects_low_quality(self):
        allocator = BudgetAllocator()
        accepted, reason = allocator.tier1_accept(quality_norm=0.2, novelty=0.8)
        self.assertFalse(accepted)
        self.assertEqual(reason, "tier1_low_quality")

    def test_tier2_expected_value_and_update(self):
        allocator = BudgetAllocator(seed=7)
        accepted, ev = allocator.tier2_accept("momentum:mutate", quality_norm=0.7, novelty=0.6)
        self.assertIsInstance(accepted, bool)
        self.assertGreaterEqual(ev, 0.0)
        self.assertLessEqual(ev, 1.0)
        before = allocator.arm_snapshot("momentum:mutate")
        allocator.update("momentum:mutate", reward=1.0)
        after = allocator.arm_snapshot("momentum:mutate")
        self.assertEqual(after["pulls"], before["pulls"] + 1)
        self.assertGreater(after["mean"], before["mean"])

    def test_fractional_reward_updates_arm_by_strength(self):
        weak = BudgetAllocator(seed=7)
        strong = BudgetAllocator(seed=7)

        weak.update("same_arm", reward=0.25)
        strong.update("same_arm", reward=0.75)

        self.assertGreater(
            strong.arm_snapshot("same_arm")["mean"],
            weak.arm_snapshot("same_arm")["mean"],
        )

    def test_acceptance_prior_shifts_ev(self):
        """Arms with high p_accept should produce consistently higher EV than low p_accept arms."""
        allocator = BudgetAllocator(seed=42, exploration_weight=0.0)
        allocator.update_acceptance_priors({
            "winning_arm": {"accepted": 9, "resolved": 10, "p_accept": 0.90},
            "losing_arm":  {"accepted": 1, "resolved": 10, "p_accept": 0.10},
        })
        # Average over multiple samples to smooth Thompson noise
        winning_evs = [allocator.expected_value("winning_arm", 0.6, 0.5) for _ in range(50)]
        losing_evs  = [allocator.expected_value("losing_arm",  0.6, 0.5) for _ in range(50)]
        self.assertGreater(sum(winning_evs) / len(winning_evs), sum(losing_evs) / len(losing_evs))

    def test_resolved_acceptance_history_increases_acceptance_weight(self):
        """More resolved WQ reviews should make p_accept matter more in EV."""
        low_history = BudgetAllocator(seed=42, exploration_weight=0.0)
        high_history = BudgetAllocator(seed=42, exploration_weight=0.0)

        low_history.update_acceptance_priors({
            "hot": {"accepted": 5, "resolved": 5, "p_accept": 0.90},
            "cold": {"accepted": 1, "resolved": 5, "p_accept": 0.10},
        })
        high_history.update_acceptance_priors({
            "hot": {"accepted": 45, "resolved": 50, "p_accept": 0.90},
            "cold": {"accepted": 5, "resolved": 50, "p_accept": 0.10},
        })

        low_spread = sum(
            low_history.expected_value("hot", 0.6, 0.5)
            - low_history.expected_value("cold", 0.6, 0.5)
            for _ in range(100)
        ) / 100
        high_spread = sum(
            high_history.expected_value("hot", 0.6, 0.5)
            - high_history.expected_value("cold", 0.6, 0.5)
            for _ in range(100)
        ) / 100

        self.assertGreater(high_spread, low_spread)


if __name__ == "__main__":
    unittest.main()
