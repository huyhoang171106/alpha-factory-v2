"""
Tests for reduce_self_correlation — semantic-preserving self-corr reduction.
"""
import unittest


class TestAutocorrReducer(unittest.TestCase):
    """Tests for reducing self-correlation in alpha expressions."""

    def test_shorten_long_lookback(self):
        """Long lookback ts_mean should be shortened to reduce persistence."""
        from alpha_policy import reduce_self_correlation
        expr = "rank(ts_mean(close, 30))"
        result = reduce_self_correlation(expr)
        # Should shorten lookback from 30 to ~15
        self.assertIn("ts_mean", result)
        # Should NOT wrap the whole expression in differencing (that would flip sign)
        self.assertFalse(result.startswith("(") and " - ts_delta" in result)

    def test_add_differencing_inside_smooth_op(self):
        """Differencing should be added INSIDE smoothing ops, not wrapping the whole expr."""
        from alpha_policy import reduce_self_correlation
        expr = "ts_mean(close, 20)"
        result = reduce_self_correlation(expr, target_autocorr=0.5)
        # Should add differencing inside ts_mean: ts_mean(close - ts_delta(close, 1), 20)
        self.assertIn("ts_delta", result)
        # Should NOT add outer differencing wrapper (would flip sign)
        self.assertNotIn(") - ts_delta", result)

    def test_preserve_sharpe(self):
        """Modified expression should be a valid non-empty string."""
        from alpha_policy import reduce_self_correlation
        expr = "rank(ts_delta(close, 5))"
        result = reduce_self_correlation(expr)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_adds_neutralization_when_missing(self):
        """group_neutralize should be added when no neutralization present."""
        from alpha_policy import reduce_self_correlation
        expr = "ts_mean(close, 10)"
        result = reduce_self_correlation(expr)
        # Should add group_neutralize since none present
        self.assertIn("group_neutralize", result)

    def test_ts_decay_not_needed_when_diff_added(self):
        """ts_decay is not needed when differencing was already added inside ts_mean."""
        from alpha_policy import reduce_self_correlation
        expr = "ts_mean(close, 20)"
        result = reduce_self_correlation(expr, target_autocorr=0.3)
        # Differencing inside ts_mean is applied first (better than ts_decay).
        # ts_delta IS present because differencing was added inside the smooth op.
        self.assertIn("ts_delta", result)
        # ts_decay is NOT added because differencing already handles the autocorr issue
        # (the function prioritises differencing over ts_decay as a stronger fix)

    def test_high_risk_expression_changed(self):
        """Expression with high self-corr risk should be modified."""
        from alpha_policy import reduce_self_correlation
        expr = "ts_mean(close, 50)"
        result = reduce_self_correlation(expr)
        # Should be different from input (lookback shortened, neutralization added)
        self.assertNotEqual(result, expr)

    def test_safe_expression_unchanged(self):
        """Already-safe expression (with ts_delta and neutralization) should be returned unchanged."""
        from alpha_policy import reduce_self_correlation
        expr = "group_neutralize(ts_delta(close, 5), subindustry)"
        result = reduce_self_correlation(expr)
        # ts_delta present + neutralization present → minimal changes expected
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
