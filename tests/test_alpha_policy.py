import unittest
import os

from alpha_policy import (
    HIGH_THROUGHPUT_THRESHOLDS,
    classify_quality_tier,
    compute_llm_budget_ratio,
    critic_score,
    passes_quality_gate,
    passes_quality_gate_v2,
    robust_quality_score,
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
    def setUp(self):
        self._env_backup = {
            "ASYNC_REQUIRE_ALL_CHECKS": os.getenv("ASYNC_REQUIRE_ALL_CHECKS"),
            "ASYNC_MIN_CHECKS_RATIO": os.getenv("ASYNC_MIN_CHECKS_RATIO"),
        }
        os.environ.pop("ASYNC_REQUIRE_ALL_CHECKS", None)
        os.environ.pop("ASYNC_MIN_CHECKS_RATIO", None)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

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

    def test_quality_gate_can_use_partial_checks_ratio(self):
        os.environ["ASYNC_REQUIRE_ALL_CHECKS"] = "0"
        os.environ["ASYNC_MIN_CHECKS_RATIO"] = "0.5"
        partial_ok = DummyResult(sharpe=1.4, fitness=1.1, turnover=20, all_passed=False, error="")
        partial_ok.total_checks = 8
        partial_ok.passed_checks = 5
        partial_bad = DummyResult(sharpe=1.4, fitness=1.1, turnover=20, all_passed=False, error="")
        partial_bad.total_checks = 8
        partial_bad.passed_checks = 3
        self.assertTrue(passes_quality_gate(partial_ok, HIGH_THROUGHPUT_THRESHOLDS))
        self.assertFalse(passes_quality_gate(partial_bad, HIGH_THROUGHPUT_THRESHOLDS))

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

    def test_critic_score_rewards_brain_advanced_patterns(self):
        basic = "group_neutralize(ts_decay_linear(rank(returns),6), industry)"
        advanced = (
            "group_neutralize("
            "ts_decay_linear(rank(regression_neut(returns, ts_mean(returns, 20))) * "
            "rank(pasteurize(volume / adv20)), 6), "
            "densify(industry * 1000 + bucket(rank(cap), range=\"0.2,1,0.2\"))"
            ")"
        )
        self.assertGreater(critic_score(advanced), critic_score(basic))

    def test_quality_gate_v2_blocks_weak_subuniverse_profile(self):
        good = DummyResult(sharpe=1.9, fitness=1.4, turnover=22, all_passed=True, error="")
        good.sub_sharpe = 0.4
        weak_sub = DummyResult(sharpe=2.2, fitness=1.6, turnover=22, all_passed=True, error="")
        weak_sub.sub_sharpe = -0.3

        self.assertTrue(passes_quality_gate(good, HIGH_THROUGHPUT_THRESHOLDS))
        self.assertTrue(passes_quality_gate_v2(good, HIGH_THROUGHPUT_THRESHOLDS))
        self.assertFalse(passes_quality_gate_v2(weak_sub, HIGH_THROUGHPUT_THRESHOLDS))
        self.assertLess(robust_quality_score(weak_sub), robust_quality_score(good))


if __name__ == "__main__":
    unittest.main()
