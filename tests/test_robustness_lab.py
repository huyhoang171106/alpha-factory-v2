import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from robustness_lab import (
    BiasDetector,
    ICStabilityScorer,
    WalkForwardValidator,
    CrossUniverseValidator,
    WalkForwardResult,
)


class TestBiasDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = BiasDetector()

    def test_survivorship_bias_raw_close(self) -> None:
        # Raw close without protection should be flagged
        flags = self.detector.detect_survivorship_bias("close / ts_mean(close, 20)")
        self.assertIn("raw_close_no_protection", flags)

    def test_survivorship_bias_protected(self) -> None:
        # Ranked close should NOT be flagged
        flags = self.detector.detect_survivorship_bias(
            "rank(close) / ts_mean(close, 20)"
        )
        self.assertNotIn("raw_close_no_protection", flags)

    def test_survivorship_bias_group_neutralize(self) -> None:
        # group_neutralize protects as well
        flags = self.detector.detect_survivorship_bias(
            "group_neutralize(close, sector)"
        )
        self.assertNotIn("raw_close_no_protection", flags)

    def test_lookahead_bias_negative_delta(self) -> None:
        flags = self.detector.detect_lookahead_bias("ts_delta(close, -5)")
        self.assertIn("ts_delta_negative_delay", flags)

    def test_lookahead_bias_normal(self) -> None:
        flags = self.detector.detect_lookahead_bias("ts_delta(close, 5)")
        self.assertNotIn("ts_delta_negative_delay", flags)

    def test_lookahead_bias_zero_window(self) -> None:
        flags = self.detector.detect_lookahead_bias("ts_mean(close, 0)")
        self.assertIn("ts_mean_zero_window", flags)

    def test_data_quality_division_by_zero(self) -> None:
        flags = self.detector.detect_data_quality_issues("close / 0")
        self.assertIn("division_by_zero", flags)

    def test_data_quality_log_of_zero(self) -> None:
        flags = self.detector.detect_data_quality_issues("log(0)")
        self.assertIn("log_of_zero", flags)

    def test_full_bias_check_structure(self) -> None:
        result = self.detector.full_bias_check("ts_delta(close, 5)")
        self.assertIn("survivorship", result)
        self.assertIn("lookahead", result)
        self.assertIn("data_quality", result)

    def test_full_bias_check_no_flags(self) -> None:
        result = self.detector.full_bias_check("rank(close) + ts_mean(volume, 20)")
        self.assertEqual(result["survivorship"], [])
        self.assertEqual(result["lookahead"], [])

    def test_passes_bias_check_clean(self) -> None:
        passed, reason = self.detector.passes_bias_check("rank(close)")
        self.assertTrue(passed)
        self.assertIsNone(reason)

    def test_passes_bias_check_dirty(self) -> None:
        passed, reason = self.detector.passes_bias_check("close / 0 + ts_delta(close, -1)")
        self.assertFalse(passed)
        self.assertIsNotNone(reason)


class TestICStabilityScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = ICStabilityScorer()

    def test_rolling_ic(self) -> None:
        ic_series = [0.05] * 300
        rolling = self.scorer.compute_rolling_ic(ic_series)
        self.assertIn("ic_21d", rolling)
        self.assertIn("ic_63d", rolling)
        self.assertIn("ic_252d", rolling)

    def test_rolling_ic_short_series(self) -> None:
        ic_series = [0.05, 0.04, 0.06]
        rolling = self.scorer.compute_rolling_ic(ic_series)
        self.assertAlmostEqual(rolling["ic_21d"], 0.05, places=4)

    def test_rolling_ic_custom_windows(self) -> None:
        ic_series = [0.05] * 100
        rolling = self.scorer.compute_rolling_ic(ic_series, windows=[5, 10, 50])
        self.assertIn("ic_5d", rolling)
        self.assertIn("ic_10d", rolling)
        self.assertIn("ic_50d", rolling)

    def test_ic_autocorrelation_stable(self) -> None:
        import random

        random.seed(42)
        ic_series = [random.uniform(-0.1, 0.1) for _ in range(50)]
        autocorr = self.scorer.ic_autocorrelation(ic_series)
        self.assertLess(abs(autocorr), 1.0)

    def test_ic_autocorrelation_persistent(self) -> None:
        # Monotonic IC = high autocorrelation
        ic_series = [0.05 + i * 0.001 for i in range(50)]
        autocorr = self.scorer.ic_autocorrelation(ic_series)
        self.assertGreater(autocorr, 0.5)

    def test_ic_autocorrelation_short_series(self) -> None:
        ic_series = [0.05, 0.04]
        autocorr = self.scorer.ic_autocorrelation(ic_series)
        self.assertEqual(autocorr, 0.0)

    def test_ic_decay_rate_negative(self) -> None:
        ic_series = [0.08 - i * 0.001 for i in range(50)]
        decay = self.scorer.ic_decay_rate(ic_series)
        self.assertLess(decay, 0.0)

    def test_ic_decay_rate_positive(self) -> None:
        ic_series = [0.04 + i * 0.001 for i in range(50)]
        decay = self.scorer.ic_decay_rate(ic_series)
        self.assertGreater(decay, 0.0)

    def test_ic_decay_rate_short_series(self) -> None:
        # 3-point series: upward trend → positive slope (non-zero)
        ic_series = [0.05, 0.04, 0.06]
        decay = self.scorer.ic_decay_rate(ic_series)
        self.assertGreater(decay, 0.0)  # Upward trend = positive slope

    def test_ic_stability_score_structure(self) -> None:
        ic_series = [0.05, 0.04, 0.06, 0.05, 0.04] * 10
        result = self.scorer.ic_stability_score(ic_series)
        self.assertIn("ic_stability_passed", result)
        self.assertIn("ic_autocorrelation", result)
        self.assertIn("ic_decay_rate", result)
        self.assertIn("gate_details", result)

    def test_ic_stability_score_passes(self) -> None:
        # Stable, positive IC series
        import random

        random.seed(0)
        ic_series = [0.05 + random.uniform(-0.02, 0.02) for _ in range(100)]
        result = self.scorer.ic_stability_score(ic_series)
        self.assertGreater(result["mean_ic"], 0.01)


class TestWalkForwardValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = WalkForwardValidator(n_splits=5)

    def test_walk_forward_analysis_returns_result(self) -> None:
        result = self.validator.walk_forward_analysis("rank(close)")
        self.assertIsInstance(result, WalkForwardResult)
        self.assertEqual(result.n_splits, 5)
        self.assertIsInstance(result.is_robust, bool)
        self.assertGreater(result.sharpe_train, 0)
        self.assertGreater(result.sharpe_test, 0)

    def test_walk_forward_analysis_splits_count(self) -> None:
        result = self.validator.walk_forward_analysis("rank(close)")
        self.assertEqual(len(result.sharpes_per_split), 5)

    def test_sharpe_drop_ratio_bounds(self) -> None:
        result = self.validator.walk_forward_analysis("rank(close)")
        self.assertGreaterEqual(result.sharpe_drop_ratio, 0.0)
        self.assertLessEqual(result.sharpe_drop_ratio, 1.0)

    def test_walk_forward_analysis_from_ic(self) -> None:
        ic_series = [0.05 + i * 0.001 for i in range(100)]
        result = self.validator.walk_forward_analysis("rank(close)", ic_series=ic_series)
        self.assertIsInstance(result, WalkForwardResult)
        self.assertEqual(result.n_splits, 5)

    def test_walk_forward_analysis_custom_splits(self) -> None:
        validator = WalkForwardValidator(n_splits=3)
        result = validator.walk_forward_analysis("rank(close)")
        self.assertEqual(result.n_splits, 3)
        self.assertEqual(len(result.sharpes_per_split), 3)

    def test_walk_forward_analysis_is_robust_flag(self) -> None:
        result = self.validator.walk_forward_analysis("rank(close)")
        self.assertIsInstance(result.is_robust, bool)


class TestCrossUniverseValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = CrossUniverseValidator()

    def test_get_universes_short_lookback(self) -> None:
        universes = self.validator.get_universes_to_test("ts_mean(close, 5)")
        self.assertIn("US", universes)
        self.assertIn("APAC", universes)
        self.assertIn("EM", universes)

    def test_get_universes_long_lookback(self) -> None:
        universes = self.validator.get_universes_to_test("ts_mean(close, 252)")
        self.assertIn("US", universes)
        self.assertLessEqual(len(universes), 4)

    def test_get_universes_no_lookback(self) -> None:
        # Expression with no numeric lookbacks defaults to all universes
        universes = self.validator.get_universes_to_test("rank(close)")
        self.assertIn("US", universes)

    def test_validate_cross_universe_framework(self) -> None:
        result = self.validator.validate_cross_universe("rank(close)")
        self.assertEqual(result["status"], "framework_only")
        self.assertIn("universes_to_test", result)
        self.assertIn("min_positive_rate", result)
        self.assertIn("note", result)

    def test_validate_cross_universe_returns_universes(self) -> None:
        result = self.validator.validate_cross_universe("ts_mean(close, 20)")
        self.assertIsInstance(result["universes_to_test"], list)


if __name__ == "__main__":
    unittest.main()
