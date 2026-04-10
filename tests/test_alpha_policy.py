import unittest

from alpha_policy import (
    HIGH_THROUGHPUT_THRESHOLDS,
    classify_quality_tier,
    compute_llm_budget_ratio,
    critic_score,
    passes_quality_gate,
    should_simulate_candidate,
)


class DummyResult:
    def __init__(self, sharpe=0.0, fitness=0.0, turnover=0.0, all_passed=False, error=""):
        self.sharpe = sharpe
        self.fitness = fitness
        self.turnover = turnover
        self.all_passed = all_passed
        self.error = error


class AlphaPolicyTests(unittest.TestCase):
    def test_quality_tier_classification(self):
        self.assertEqual(classify_quality_tier(2.6, 2.1), "elite")
        self.assertEqual(classify_quality_tier(2.1, 1.7), "excellent")
        self.assertEqual(classify_quality_tier(1.6, 1.3), "good")
        self.assertEqual(classify_quality_tier(1.3, 1.0), "minimum")
        self.assertEqual(classify_quality_tier(0.8, 0.8), "reject")

    def test_quality_gate_requires_all_checks_in_high_throughput_profile(self):
        ok = DummyResult(sharpe=1.3, fitness=1.0, turnover=20, all_passed=True, error="")
        bad_checks = DummyResult(sharpe=1.6, fitness=1.4, turnover=20, all_passed=False, error="")
        self.assertTrue(passes_quality_gate(ok, HIGH_THROUGHPUT_THRESHOLDS))
        self.assertFalse(passes_quality_gate(bad_checks, HIGH_THROUGHPUT_THRESHOLDS))

    def test_quality_gate_rejects_errors_and_turnover_out_of_range(self):
        err = DummyResult(sharpe=2.0, fitness=1.5, turnover=20, all_passed=True, error="x")
        low_turnover = DummyResult(sharpe=2.0, fitness=1.5, turnover=0.5, all_passed=True, error="")
        self.assertFalse(passes_quality_gate(err))
        self.assertFalse(passes_quality_gate(low_turnover))

    def test_critic_score_and_simulation_decision(self):
        rich = "group_neutralize(ts_decay_linear(ts_zscore(ts_corr(close, volume, 20), 10), 6), industry)"
        weak = "rank(close)"
        self.assertGreater(critic_score(rich), critic_score(weak))
        self.assertTrue(should_simulate_candidate(rich))
        self.assertFalse(should_simulate_candidate(""))

    def test_llm_budget_controller(self):
        self.assertEqual(compute_llm_budget_ratio(0.2, 0.1, 0.1, False), 0.0)
        self.assertLessEqual(compute_llm_budget_ratio(0.2, 0.7, 0.1, True), 0.05)
        self.assertLessEqual(compute_llm_budget_ratio(0.2, 0.1, 0.4, True), 0.10)
        self.assertEqual(compute_llm_budget_ratio(0.2, 0.1, 0.1, True), 0.2)


if __name__ == "__main__":
    unittest.main()
