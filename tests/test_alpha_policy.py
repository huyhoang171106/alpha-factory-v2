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
    detect_survivorship_bias,
    detect_lookahead_bias,
    detect_survivorship_and_lookahead,
    estimate_ic_stability,
    passes_ic_stability,
    pre_submission_gate,
    pre_submission_gate_from_result,
    sub_sharpe_ensemble_gate,
    SubSharpeEnsemble,
)

from alpha_ranker import (
    count_unique_lookbacks,
    expression_complexity_penalty,
    complexity_score,
    passes_complexity_check,
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
            "ASYNC_ROBUST_SCORE_MIN": os.getenv("ASYNC_ROBUST_SCORE_MIN"),
        }
        os.environ.pop("ASYNC_REQUIRE_ALL_CHECKS", None)
        os.environ.pop("ASYNC_MIN_CHECKS_RATIO", None)
        os.environ["ASYNC_ROBUST_SCORE_MIN"] = "1.0"

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


# ============================================================
# Tier-1 Acceptance Gate Tests
# ============================================================

class BiasDetectionTests(unittest.TestCase):

    def test_survivorship_bias_raw_close_flagged(self):
        """Raw close without rank/protection should be flagged."""
        self.assertTrue(detect_survivorship_bias("close - ts_mean(close, 20)"))

    def test_survivorship_bias_safe_with_rank(self):
        """Rank() neutralises survivorship bias risk."""
        self.assertFalse(detect_survivorship_bias("rank(close - ts_mean(close, 20))"))

    def test_survivorship_bias_safe_with_group_neutralize(self):
        self.assertFalse(detect_survivorship_bias(
            "group_neutralize(close, sector)"))

    def test_survivorship_bias_safe_with_winsorize(self):
        self.assertFalse(detect_survivorship_bias(
            "winsorize(returns, lower=0.01, upper=0.99)"))

    def test_lookahead_bias_negative_delay(self):
        """ts_delay with negative delta is future leak."""
        self.assertTrue(detect_lookahead_bias("ts_delay(close, -5)"))

    def test_lookahead_bias_negative_ts_delta(self):
        self.assertTrue(detect_lookahead_bias("ts_delta(returns, -3)"))

    def test_lookahead_bias_ts_delta_zero(self):
        """ts_delta with 0 delay ≈ current value, borderline but flagged."""
        self.assertTrue(detect_lookahead_bias("ts_delta(volume, 0)"))

    def test_lookahead_bias_normal_positive_delay_ok(self):
        self.assertFalse(detect_lookahead_bias("ts_delay(close, 5)"))

    def test_combined_bias_check_pass(self):
        safe = "rank(ts_mean(returns, 20))"
        result = detect_survivorship_and_lookahead(safe)
        self.assertTrue(result["passed"])
        self.assertEqual(result["bias_flags"], "")

    def test_combined_bias_check_fail_survivorship(self):
        risky = "ts_delta(close, 5) / volume"
        result = detect_survivorship_and_lookahead(risky)
        self.assertFalse(result["passed"])
        self.assertTrue(result["survivorship_bias"])
        self.assertIn("survivorship_bias", result["bias_flags"])


class ICStabilityTests(unittest.TestCase):

    def test_ic_stability_consistent_sharpe_fitness(self):
        """Consistent Sharpe/fitness → high stability."""
        score = estimate_ic_stability(sharpe=1.5, fitness=1.4, turnover=20, sub_sharpe=-1.0)
        self.assertGreater(score, 0.40)

    def test_ic_stability_degraded_by_high_fitness_sharpe_gap(self):
        """Fitness >> Sharpe suggests IS curve-fit."""
        high_gap = estimate_ic_stability(sharpe=1.0, fitness=2.5, turnover=20, sub_sharpe=-1.0)
        consistent = estimate_ic_stability(sharpe=1.5, fitness=1.4, turnover=20, sub_sharpe=-1.0)
        self.assertLess(high_gap, consistent)

    def test_ic_stability_penalised_by_high_turnover(self):
        high_to = estimate_ic_stability(sharpe=1.5, fitness=1.4, turnover=70, sub_sharpe=-1.0)
        low_to = estimate_ic_stability(sharpe=1.5, fitness=1.4, turnover=20, sub_sharpe=-1.0)
        self.assertLess(high_to, low_to)

    def test_ic_stability_penalised_by_negative_sub_sharpe(self):
        neg_sub = estimate_ic_stability(sharpe=1.5, fitness=1.4, turnover=20, sub_sharpe=-0.5)
        neutral_sub = estimate_ic_stability(sharpe=1.5, fitness=1.4, turnover=20, sub_sharpe=-1.0)
        self.assertLess(neg_sub, neutral_sub)

    def test_passes_ic_stability_gate(self):
        self.assertTrue(passes_ic_stability(1.5, 1.4, 20, -1.0))
        # Poor IC stability should fail with default floor=0.15
        self.assertFalse(passes_ic_stability(0.8, 0.5, 70, -0.3))

    def test_passes_ic_stability_custom_floor(self):
        # Even a decent score should fail with an unrealistically high floor
        self.assertFalse(passes_ic_stability(2.0, 1.8, 20, -1.0, floor=0.99))


class PreSubmissionGateTests(unittest.TestCase):

    def _gate(self, expr, sharpe=1.5, fitness=1.4, turnover=20, sub_sharpe=-1.0, error=""):
        res = type("Res", (), {
            "expression": expr,
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "sub_sharpe": sub_sharpe,
            "error": error,
            "all_passed": (error == ""),
            "passed_checks": 8 if error == "" else 0,
            "total_checks": 8 if error == "" else 0,
        })
        return pre_submission_gate_from_result(res)

    def test_passes_valid_expression(self):
        good = "rank(ts_corr(ts_zscore(close), ts_zscore(volume), 20))"
        result = self._gate(good)
        self.assertTrue(result["passed"])
        self.assertEqual(result["stage"], "passed")

    def test_rejects_runtime_error(self):
        result = self._gate("rank(close)", error="Division by zero")
        self.assertFalse(result["passed"])
        self.assertEqual(result["stage"], "runtime_error")

    def test_rejects_bias_detection(self):
        risky = "ts_delta(close, 5) / volume"
        result = self._gate(risky)
        self.assertFalse(result["passed"])
        self.assertEqual(result["stage"], "bias_detection")

    def test_rejects_ic_instability(self):
        # High turnover + poor sub_sharpe → IC unstable
        result = self._gate("rank(ts_mean(returns, 20))", sharpe=1.2, fitness=1.1, turnover=70, sub_sharpe=-0.2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["stage"], "ic_stability")

    def test_wrapper_from_result(self):
        class FakeResult:
            expression = "rank(ts_mean(returns, 20))"
            sharpe = 1.6
            fitness = 1.5
            turnover = 25
            sub_sharpe = -1.0
            error = ""
            all_passed = True
            passed_checks = 8
            total_checks = 8
        result = pre_submission_gate_from_result(FakeResult())
        self.assertTrue(result["passed"])


class ComplexityScoringTests(unittest.TestCase):

    def test_simple_expression_has_low_penalty(self):
        simple = "rank(close)"
        penalty = expression_complexity_penalty(simple)
        self.assertEqual(penalty, 0.0)

    def test_over_nested_expression_penalised(self):
        # 8 levels deep
        nested = "a(b(c(d(e(f(g(h(x))))))))"
        penalty = expression_complexity_penalty(nested)
        self.assertGreater(penalty, 0)

    def test_lookback_proliferation_penalised(self):
        """Many distinct lookback constants → suspicious."""
        many_lbs = "rank(ts_mean(close,3) + ts_mean(close,5) + ts_mean(close,7) + ts_mean(close,10) + ts_mean(close,14) + ts_mean(close,20))"
        penalty = expression_complexity_penalty(many_lbs)
        self.assertGreater(penalty, 0)

    def test_complexity_score_returns_full_dict(self):
        expr = "rank(ts_delta(close, 5))"
        result = complexity_score(expr)
        self.assertIn("score", result)
        self.assertIn("depth", result)
        self.assertIn("n_ops", result)
        self.assertIn("unique_lookbacks", result)
        self.assertIn("penalty", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_count_unique_lookbacks(self):
        self.assertEqual(count_unique_lookbacks("f(3) + f(5) + f(3) + f(5)"), 2)
        self.assertEqual(count_unique_lookbacks("f(x)"), 0)

    def test_passes_complexity_check_simple(self):
        simple = "rank(close)"
        self.assertTrue(passes_complexity_check(simple))

    def test_passes_complexity_check_complex(self):
        complex_expr = "a(b(c(d(e(f(g(h(i(x))))))))))"
        self.assertFalse(passes_complexity_check(complex_expr))

    def test_rank_score_penalty_applied(self):
        """Expression that is both over-nested and has many lookbacks should be penalised."""
        expr = "a(b(c(d(e(f(g(h(close)))))))) + g(a(1)) + g(a(2)) + g(a(3)) + g(a(4)) + g(a(5)) + g(a(6))"
        score_with_penalty = complexity_score(expr)["score"]
        simple_score = complexity_score("rank(close)")["score"]
        self.assertLess(score_with_penalty, simple_score)


# ============================================================
# Sub-Sharpe Ensemble Tests
# ============================================================

class DummyWQClient:
    """Mock WQ client that returns controlled sub_sharpe values."""

    def __init__(self, results_per_split: list[float | None]):
        self._results = results_per_split
        self._idx = 0

    def simulate(self, expression, region="USA", universe="TOP3000",
                 delay=1, decay=6, neutralization="SUBINDUSTRY", truncation=0.08):
        class FakeResult:
            def __init__(self, sub):
                self.sub_sharpe = sub
                self.error = "" if sub is not None else "mock_error"
        res = self._results[self._idx % len(self._results)]
        self._idx += 1
        return FakeResult(res)


class SubSharpeEnsembleTests(unittest.TestCase):

    def test_skips_already_passing(self):
        """Skip ensemble when sharpe > 1.8 and sub_sharpe >= 0."""
        client = DummyWQClient([0.5])
        ensemble = SubSharpeEnsemble(client, n_splits=3)
        result = ensemble.evaluate("rank(ts_mean(returns, 20))",
                                   primary_sharpe=2.1,
                                   primary_sub_sharpe=0.3)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["skipped_reason"], "already_passing")
        self.assertEqual(result["mean_sub_sharpe"], 0.3)

    def test_skips_low_sharpe(self):
        """Skip ensemble when primary Sharpe < 1.0 — won't pass gate anyway."""
        client = DummyWQClient([0.1])
        ensemble = SubSharpeEnsemble(client, n_splits=3)
        result = ensemble.evaluate("rank(ts_mean(returns, 20))",
                                   primary_sharpe=0.8,
                                   primary_sub_sharpe=-0.5)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["skipped_reason"], "primary_sharpe_too_low")

    def test_runs_ensemble_borderline(self):
        """Run ensemble for borderline case (sharpe between 1.0 and 1.8)."""
        client = DummyWQClient([0.1, -0.2, 0.05, -0.1, 0.15])
        ensemble = SubSharpeEnsemble(client, n_splits=5)
        result = ensemble.evaluate("rank(ts_corr(close, volume, 20))",
                                   primary_sharpe=1.5,
                                   primary_sub_sharpe=-0.3)
        self.assertFalse(result["skipped"])
        self.assertEqual(result["n_splits"], 5)
        self.assertIn("mean_sub_sharpe", result)
        self.assertIn("std_sub_sharpe", result)
        # mean of [0.1, -0.2, 0.05, -0.1, 0.15] = 0.0
        self.assertAlmostEqual(result["mean_sub_sharpe"], 0.0, places=3)

    def test_insufficient_splits_falls_back(self):
        """Fewer than 2 successful splits → falls back to primary."""
        client = DummyWQClient([None, None, 0.2])  # 2 failures
        ensemble = SubSharpeEnsemble(client, n_splits=3)
        result = ensemble.evaluate("rank(ts_mean(returns, 20))",
                                   primary_sharpe=1.4,
                                   primary_sub_sharpe=-0.5)
        self.assertFalse(result["skipped"])
        self.assertIn("insufficient_splits", result["skipped_reason"])
        self.assertEqual(result["mean_sub_sharpe"], -0.5)  # fell back

    def test_result_dict_structure(self):
        """Every result dict has required keys."""
        client = DummyWQClient([0.1, 0.2, 0.1])
        ensemble = SubSharpeEnsemble(client, n_splits=3)
        result = ensemble.evaluate("rank(ts_delta(close, 5))",
                                   primary_sharpe=1.6,
                                   primary_sub_sharpe=0.0)
        for key in ("mean_sub_sharpe", "std_sub_sharpe", "n_splits", "skipped", "skipped_reason"):
            self.assertIn(key, result)


class SubSharpeEnsembleGateTests(unittest.TestCase):

    def test_skipped_uses_primary_pass(self):
        """skipped=True + primary sub_sharpe >= 0 → PASS."""
        res = {"skipped": True, "skipped_reason": "already_passing",
               "mean_sub_sharpe": 0.3, "std_sub_sharpe": 0.0, "n_splits": 0}
        self.assertTrue(sub_sharpe_ensemble_gate("x", 0.3, res))

    def test_skipped_uses_primary_fail(self):
        """skipped=True + primary sub_sharpe in (-0.99, 0) → FAIL."""
        res = {"skipped": True, "skipped_reason": "primary_sharpe_too_low",
               "mean_sub_sharpe": -0.5, "std_sub_sharpe": 0.0, "n_splits": 0}
        self.assertFalse(sub_sharpe_ensemble_gate("x", -0.5, res))

    def test_ensemble_pass_mean_above_threshold(self):
        """mean >= -0.5 AND std < 0.8 → PASS (override primary)."""
        res = {"skipped": False, "skipped_reason": "",
               "mean_sub_sharpe": -0.3, "std_sub_sharpe": 0.5, "n_splits": 5}
        # Primary is -0.3 which is in the reject band, but ensemble overrides
        self.assertTrue(sub_sharpe_ensemble_gate("x", -0.3, res))

    def test_ensemble_fail_high_std(self):
        """mean >= -0.5 but std >= 0.8 → fall back to primary."""
        res = {"skipped": False, "skipped_reason": "",
               "mean_sub_sharpe": -0.2, "std_sub_sharpe": 1.1, "n_splits": 5}
        # Primary -0.2 is in reject band; std too high → use primary → FAIL
        self.assertFalse(sub_sharpe_ensemble_gate("x", -0.2, res))

    def test_ensemble_pass_primary_already_passing(self):
        """mean >= -0.5, std < 0.8, but primary already positive → PASS."""
        res = {"skipped": False, "skipped_reason": "",
               "mean_sub_sharpe": 0.1, "std_sub_sharpe": 0.3, "n_splits": 5}
        self.assertTrue(sub_sharpe_ensemble_gate("x", 0.1, res))

    def test_ensemble_fail_mean_too_low(self):
        """mean < -0.5 → fall back to primary."""
        res = {"skipped": False, "skipped_reason": "",
               "mean_sub_sharpe": -0.8, "std_sub_sharpe": 0.4, "n_splits": 5}
        # Primary falls back, is -0.5 in reject band?  -0.5 > -0.99 and < 0 → FAIL
        self.assertFalse(sub_sharpe_ensemble_gate("x", -0.5, res))


class EnsembleIntegrationTests(unittest.TestCase):
    """Test passes_quality_gate_v2 with ensemble override logic."""

    def setUp(self):
        self._env_backup = {
            "ASYNC_ROBUST_SCORE_MIN": os.getenv("ASYNC_ROBUST_SCORE_MIN"),
        }
        os.environ["ASYNC_ROBUST_SCORE_MIN"] = "1.0"

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _result(self, s, sub):
        class R:
            pass
        R.sharpe = s
        R.sub_sharpe = sub
        R.fitness = 1.5
        R.turnover = 20.0
        R.error = ""
        R.all_passed = True
        R.passed_checks = 8
        R.total_checks = 8
        R.self_corr = 0.0
        R.expression = "rank(ts_mean(returns, 20))"
        return R(), None

    def test_ensemble_overrides_negative_sub_sharpe(self):
        """Alpha with sub_sharpe=-0.3 should pass via ensemble override."""
        ensemble_res = {
            "skipped": False, "skipped_reason": "",
            "mean_sub_sharpe": 0.05, "std_sub_sharpe": 0.3, "n_splits": 5,
        }
        result, _ = self._result(1.9, -0.3)
        self.assertTrue(passes_quality_gate_v2(result, HIGH_THROUGHPUT_THRESHOLDS,
                                               ensemble_result=ensemble_res))

    def test_primary_used_when_ensemble_fails(self):
        """Alpha with sub_sharpe=-0.3 should FAIL when ensemble also fails."""
        ensemble_res = {
            "skipped": False, "skipped_reason": "",
            "mean_sub_sharpe": -0.7, "std_sub_sharpe": 0.9, "n_splits": 5,
        }
        result, _ = self._result(1.9, -0.3)
        self.assertFalse(passes_quality_gate_v2(result, HIGH_THROUGHPUT_THRESHOLDS,
                                                 ensemble_result=ensemble_res))

    def test_backward_compatible_no_ensemble(self):
        """Without ensemble_result, existing logic unchanged."""
        result, _ = self._result(1.9, 0.5)
        self.assertTrue(passes_quality_gate_v2(result, HIGH_THROUGHPUT_THRESHOLDS))

        result2, _ = self._result(2.0, -0.5)
        self.assertFalse(passes_quality_gate_v2(result2, HIGH_THROUGHPUT_THRESHOLDS))
