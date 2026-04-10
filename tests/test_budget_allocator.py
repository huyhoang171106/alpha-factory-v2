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


if __name__ == "__main__":
    unittest.main()
