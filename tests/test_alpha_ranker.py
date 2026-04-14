"""
tests/test_alpha_ranker.py
Unit tests for alpha_ranker.py and lineage_decay_tracker.py
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_ranker import (
    regime_sensitivity_score,
    ic_decay_probability,
    cross_regime_score,
    expression_simplicity_score,
    turnover_prediction,
    score_with_meta_model,
    count_nesting_depth,
    count_operators,
    count_unique_lookbacks,
)
from lineage_decay_tracker import (
    LineageDecayTracker,
    LineageStats,
    record_lineage_sim,
    is_lineage_alive,
    get_tracker,
)


# ============================================================
# Meta-Model Scoring Tests
# ============================================================

class TestRegimeSensitivityScore(unittest.TestCase):
    def test_high_regime_adaptive_expression(self):
        # ts_std_dev on returns + trade_when → regime-adaptive
        expr = "trade_when(vol_regime > 0.5, rank(ts_delta(close, 5))) * ts_std_dev(returns, 20)"
        score = regime_sensitivity_score(expr)
        self.assertGreaterEqual(score, 0.4, f"Expected regime-adaptive expression to score >= 0.4, got {score}")

    def test_static_momentum_low_score(self):
        # Pure momentum — no adaptation
        expr = "-rank(ts_delta(close, 5))"
        score = regime_sensitivity_score(expr)
        self.assertLess(score, 0.4, f"Expected static momentum to score low, got {score}")

    def test_if_else_branching(self):
        expr = "if_else(ts_std_dev(returns, 10) > 0.02, rank(ts_delta(close, 5)), rank(returns))"
        score = regime_sensitivity_score(expr)
        self.assertGreater(score, 0.3)

    def test_empty_returns_zero(self):
        self.assertEqual(regime_sensitivity_score(""), 0.0)
        self.assertEqual(regime_sensitivity_score(None), 0.0)

    def test_score_bounded_0_to_1(self):
        exprs = [
            "rank(ts_delta(close, 5))",
            "ts_skewness(returns, 20) + ts_kurtosis(returns, 20)",
            "trade_when(vol > 0.5, sign(returns))",
            "if_else(regime > 0, alpha_a, alpha_b)",
        ]
        for e in exprs:
            s = regime_sensitivity_score(e)
            self.assertGreaterEqual(s, 0.0, f"{e} scored below 0: {s}")
            self.assertLessEqual(s, 1.0, f"{e} scored above 1: {s}")


class TestICDecayProbability(unittest.TestCase):
    def test_real_data_declining_ic(self):
        history_ic = [0.55, 0.52, 0.50, 0.48, 0.45, 0.30, 0.22, 0.18]
        prob = ic_decay_probability("rank(close)", history_ic=history_ic)
        self.assertGreater(prob, 0.3, "Declining IC history should yield elevated decay probability")

    def test_real_data_stable_ic(self):
        history_ic = [0.45, 0.48, 0.44, 0.47, 0.46, 0.43, 0.45]
        prob = ic_decay_probability("rank(close)", history_ic=history_ic)
        self.assertLess(prob, 0.5, "Stable IC history should have low decay probability")

    def test_heuristic_long_lookback_high_risk(self):
        expr = "rank(ts_mean(returns, 500))"
        prob = ic_decay_probability(expr)
        self.assertGreater(prob, 0.4, "Long lookback should increase decay risk")

    def test_heuristic_complex_expression_risk(self):
        expr = "ts_mean(ts_corr(ts_delta(volume, 3), ts_delta(close, 3), 30), 60) + ts_std_dev(returns, 252)"
        prob = ic_decay_probability(expr)
        self.assertGreater(prob, 0.1, "Complex expression should have elevated decay risk")

    def test_simple_adaptive_low_risk(self):
        expr = "ts_zscore(ts_delta(close, 20))"
        prob = ic_decay_probability(expr)
        self.assertLess(prob, 0.4, "Simple adaptive expression should have low decay risk")

    def test_bounded_0_to_1(self):
        exprs = ["rank(close)", "ts_mean(returns, 500)", "ts_corr(volume, returns, 120)"]
        for e in exprs:
            p = ic_decay_probability(e)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestCrossRegimeScore(unittest.TestCase):
    def test_multi_field_group_neutralize(self):
        expr = "group_neutralize(rank(ts_corr(returns, volume, 20)), sector)"
        score = cross_regime_score(expr)
        self.assertGreater(score, 0.3, "Multi-field cross-sectional expression should score well")

    def test_vol_awareness(self):
        expr = "rank(ts_std_dev(returns, 20)) * rank(returns)"
        score = cross_regime_score(expr)
        self.assertGreater(score, 0.2)

    def test_simple_rank_low_score(self):
        expr = "rank(close)"
        score = cross_regime_score(expr)
        self.assertLess(score, 0.3, "Simple rank(close) should have low cross-regime score")

    def test_bounded_0_to_1(self):
        exprs = ["rank(close)", "group_neutralize(rank(returns), sector)", "ts_std_dev(returns, 20) * rank(returns)"]
        for e in exprs:
            s = cross_regime_score(e)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class TestExpressionSimplicityScore(unittest.TestCase):
    def test_simple_expression_high_score(self):
        expr = "rank(ts_delta(close, 5))"
        score = expression_simplicity_score(expr)
        self.assertGreater(score, 0.7, f"Simple expression should score high, got {score}")

    def test_complex_over_nested(self):
        # depth 5 + n_ops 8 + 5 unique lookbacks → significant penalty
        expr = "ts_mean(ts_corr(ts_delta(volume, 3), ts_mean(close, 20), 30), ts_mean(abs(returns), 10)) / ts_std_dev(returns, 60)"
        score = expression_simplicity_score(expr)
        # depth=5 (>4) → 0.20 penalty; n_ops=7 (>6) → 0.10 penalty; unique_lookbacks=5 (>3) → 0.15 penalty
        # score = 1.0 - 0.20 - 0.10 - 0.15 = 0.55
        self.assertGreater(score, 0.4, f"Complex expression should score > 0.4, got {score}")
        self.assertLess(score, 0.8, f"Complex expression should score < 0.8, got {score}")

    def test_many_lookbacks(self):
        # 5 unique lookbacks → penalty
        expr = "ts_mean(ts_corr(ts_delta(close, 5), ts_delta(volume, 10), 20), 60) / ts_std_dev(returns, 252)"
        score = expression_simplicity_score(expr)
        # unique_lookbacks=5 (>3) → 0.15 penalty; depth=3; n_ops=4
        # score = 1.0 - 0.15 = 0.85
        self.assertGreater(score, 0.5, f"Expression with many lookbacks should still score > 0.5, got {score}")

    def test_bounded_0_to_1(self):
        exprs = ["rank(close)", "ts_mean(ts_corr(volume, returns, 20), 60)"]
        for e in exprs:
            s = expression_simplicity_score(e)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class TestTurnoverPrediction(unittest.TestCase):
    def test_short_delta_high_turnover(self):
        expr = "ts_delta(close, 1) * ts_delta(volume, 1)"
        pred = turnover_prediction(expr)
        self.assertGreater(pred, 1.0, "Short ts_delta should predict high turnover")

    def test_decay_reduces_turnover(self):
        expr_with_decay = "ts_decay_linear(rank(ts_delta(close, 5)), 20)"
        pred = turnover_prediction(expr_with_decay)
        self.assertLess(pred, 1.5, "ts_decay_linear should reduce turnover prediction")

    def test_rank_wrapping_reduces_turnover(self):
        expr = "rank(ts_delta(close, 5))"
        pred = turnover_prediction(expr)
        self.assertLess(pred, 1.3, "rank() wrapping should reduce turnover")

    def test_no_smoothing_high_turnover(self):
        expr = "close / ts_delay(close, 1) - 1"
        pred = turnover_prediction(expr)
        self.assertGreater(pred, 1.0, "No smoothing with short delay should predict high turnover")

    def test_non_negative(self):
        exprs = ["rank(close)", "ts_mean(returns, 60)", "ts_decay_linear(rank(ts_delta(close, 20)), 60)"]
        for e in exprs:
            p = turnover_prediction(e)
            self.assertGreaterEqual(p, 0.0, f"Turnover prediction should be non-negative: {e} -> {p}")


class TestMetaModelScoring(unittest.TestCase):
    def test_all_fields_present(self):
        expr = "group_neutralize(rank(ts_corr(returns, volume, 20)), sector)"
        result = score_with_meta_model(expr)
        expected_keys = {
            "expected_sharpe", "decay_probability", "cross_regime_score",
            "ic_stability_score", "simplicity_score", "turnover_prediction", "composite_score",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_empty_candidate_returns_zeros(self):
        result = score_with_meta_model("")
        self.assertEqual(result["composite_score"], 0.0)
        self.assertEqual(result["decay_probability"], 1.0)

    def test_composite_is_bounded_0_to_1(self):
        exprs = [
            "rank(close)",
            "group_neutralize(rank(ts_delta(close, 5)), sector)",
            "ts_decay_linear(rank(ts_corr(returns, volume, 20)), 60)",
            "trade_when(vol > 0.5, rank(ts_delta(close, 5))) * ts_std_dev(returns, 20)",
        ]
        for e in exprs:
            result = score_with_meta_model(e)
            self.assertGreaterEqual(result["composite_score"], 0.0)
            self.assertLessEqual(result["composite_score"], 1.0)

    def test_strong_alpha_high_composite(self):
        # Good expression: multi-field, cross-regime, simple, not overfitted
        expr = "group_neutralize(rank(ts_corr(returns, volume, 20)), sector)"
        result = score_with_meta_model(expr)
        self.assertGreater(result["cross_regime_score"], 0.2, "Cross-regime scoring should be positive")
        self.assertLess(result["decay_probability"], 0.6, "Should have moderate decay risk")


# ============================================================
# LineageDecayTracker Tests
# ============================================================

class TestLineageStats(unittest.TestCase):
    def test_default_values(self):
        ls = LineageStats(family_id="test_family")
        self.assertEqual(ls.family_id, "test_family")
        self.assertEqual(ls.sharpe_history, [])
        self.assertEqual(ls.p_accept, 0.5)
        self.assertEqual(ls.sim_count, 0)
        self.assertTrue(ls.is_alive)

    def test_to_dict(self):
        ls = LineageStats(family_id="x", sharpe_history=[1.5, 1.2], p_accept=0.7, sim_count=2)
        d = ls.to_dict()
        self.assertEqual(d["family_id"], "x")
        self.assertEqual(d["sharpe_history"], [1.5, 1.2])
        self.assertEqual(d["p_accept"], 0.7)


class TestLineageDecayTracker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tracker = LineageDecayTracker(
            decay_slope_threshold=-0.1,
            p_accept_threshold=0.3,
            data_dir=self.tmpdir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_simulation_new_lineage(self):
        self.tracker.record_simulation("family_1", sharpe=1.5, p_accept=0.6)
        ls = self.tracker.lineages.get("family_1")
        self.assertIsNotNone(ls)
        self.assertEqual(len(ls.sharpe_history), 1)
        self.assertEqual(ls.sharpe_history[0], 1.5)
        self.assertEqual(ls.sim_count, 1)

    def test_record_simulation_updates_p_accept(self):
        self.tracker.record_simulation("family_1", sharpe=1.0, p_accept=0.5)
        self.tracker.record_simulation("family_1", sharpe=0.8, p_accept=0.2)
        ls = self.tracker.lineages["family_1"]
        self.assertEqual(ls.p_accept, 0.2)
        self.assertEqual(len(ls.sharpe_history), 2)

    def test_compute_decay_slope_stable(self):
        # Stable lineage — flat slope near 0
        history = [1.1, 1.2, 1.0, 1.15, 1.1]
        for sh in history:
            self.tracker.record_simulation("stable_family", sharpe=sh, p_accept=0.5)
        slope = self.tracker.compute_decay_slope("stable_family", window=5)
        self.assertGreater(slope, -0.1, "Stable lineage should have non-negative slope")

    def test_compute_decay_slope_declining(self):
        # Declining lineage — negative slope
        history = [1.5, 1.4, 1.2, 0.9, 0.5, 0.2, -0.1]
        for sh in history:
            self.tracker.record_simulation("declining_family", sharpe=sh, p_accept=0.5)
        slope = self.tracker.compute_decay_slope("declining_family", window=7)
        self.assertLess(slope, -0.05, "Declining Sharpe history should have negative slope")

    def test_should_kill_lineage_triggered(self):
        # Decaying slope + low p_accept → should kill
        history = [1.5, 1.3, 1.0, 0.6, 0.3, 0.0, -0.2]
        for sh in history:
            self.tracker.record_simulation("kill_me", sharpe=sh, p_accept=0.2)
        self.assertTrue(self.tracker.should_kill_lineage("kill_me"))
        self.assertFalse(self.tracker.lineages["kill_me"].is_alive)

    def test_should_kill_lineage_not_triggered_healthy(self):
        # Good Sharpe, good p_accept → alive
        history = [1.5, 1.4, 1.6, 1.3, 1.5, 1.4, 1.5]
        for sh in history:
            self.tracker.record_simulation("healthy_family", sharpe=sh, p_accept=0.7)
        self.assertFalse(self.tracker.should_kill_lineage("healthy_family"))
        self.assertTrue(self.tracker.lineages["healthy_family"].is_alive)

    def test_should_kill_lineage_early_kill(self):
        # Early kill: few sims, deeply negative Sharpe, low p_accept
        history = [-1.5, -1.2, -1.3, -1.4, -1.1]
        for sh in history:
            self.tracker.record_simulation("doomed_family", sharpe=sh, p_accept=0.2)
        self.assertTrue(self.tracker.should_kill_lineage("doomed_family"))

    def test_should_kill_unknown_lineage_returns_false(self):
        self.assertFalse(self.tracker.should_kill_lineage("unknown_family"))

    def test_get_alive_lineages(self):
        history_good = [1.5, 1.4, 1.6]
        history_bad = [1.5, 0.8, 0.3, 0.0, -0.1]
        for sh in history_good:
            self.tracker.record_simulation("alive_family", sharpe=sh, p_accept=0.7)
        for sh in history_bad:
            self.tracker.record_simulation("dead_family", sharpe=sh, p_accept=0.2)
        alive = self.tracker.get_alive_lineages()
        self.assertIn("alive_family", alive)
        self.assertNotIn("dead_family", alive)

    def test_get_kill_report(self):
        history_bad = [1.5, 0.8, 0.3, 0.0, -0.1]
        for sh in history_bad:
            self.tracker.record_simulation("kill_report_family", sharpe=sh, p_accept=0.2)
        # Also add an alive family so total_count = 2
        self.tracker.record_simulation("alive_in_report", sharpe=1.5, p_accept=0.7)
        self.tracker.should_kill_lineage("kill_report_family")
        report = self.tracker.get_kill_report()
        self.assertEqual(report["killed_count"], 1)
        self.assertEqual(report["total_count"], 2)  # 1 killed + 1 alive
        self.assertGreater(len(report["killed_lineages"]), 0)
        killed_entry = report["killed_lineages"][0]
        self.assertEqual(killed_entry["family_id"], "kill_report_family")
        self.assertLess(captured_slope := killed_entry["decay_slope"], 0.0)

    def test_revoke_lineage(self):
        history_bad = [1.5, 0.8, 0.3, 0.0, -0.1]
        for sh in history_bad:
            self.tracker.record_simulation("revive_family", sharpe=sh, p_accept=0.2)
        self.assertTrue(self.tracker.should_kill_lineage("revive_family"))
        self.assertTrue(self.tracker.revive_lineage("revive_family"))
        self.assertTrue(self.tracker.lineages["revive_family"].is_alive)
        self.assertIn("revive_family", self.tracker.get_alive_lineages())

    def test_persistence(self):
        history = [1.2, 1.4, 1.1]
        for sh in history:
            self.tracker.record_simulation("persist_family", sharpe=sh, p_accept=0.5)
        # Create a new tracker instance — should load persisted state
        tracker2 = LineageDecayTracker(data_dir=self.tmpdir)
        ls = tracker2.lineages.get("persist_family")
        self.assertIsNotNone(ls)
        self.assertEqual(len(ls.sharpe_history), 3)
        self.assertEqual(ls.sim_count, 3)

    def test_prune_old_history(self):
        many = list(range(100))
        for sh in many:
            self.tracker.record_simulation("prune_family", sharpe=float(sh % 10) / 10.0, p_accept=0.5)
        self.assertGreater(len(self.tracker.lineages["prune_family"].sharpe_history), 50)
        self.tracker.prune_old_history(max_history=20)
        self.assertEqual(len(self.tracker.lineages["prune_family"].sharpe_history), 20)


class TestStandaloneHelpers(unittest.TestCase):
    """Test the module-level convenience functions."""

    def setUp(self):
        # Reset the global tracker between tests to avoid cross-contamination
        import lineage_decay_tracker as ldt
        ldt._default_tracker = None

    def tearDown(self):
        import lineage_decay_tracker as ldt
        ldt._default_tracker = None

    def test_record_lineage_sim_singleton(self):
        record_lineage_sim("singleton_test", sharpe=1.5, p_accept=0.6)
        tracker = get_tracker()
        self.assertIn("singleton_test", tracker.lineages)

    def test_is_lineage_alive_unknown(self):
        # Unknown lineage should be considered alive
        self.assertTrue(is_lineage_alive("never_seen_before"))

    def test_is_lineage_alive_after_kill(self):
        record_lineage_sim("alive_then_dead", sharpe=1.0, p_accept=0.5)
        record_lineage_sim("alive_then_dead", sharpe=0.8, p_accept=0.5)
        record_lineage_sim("alive_then_dead", sharpe=0.6, p_accept=0.5)
        record_lineage_sim("alive_then_dead", sharpe=0.3, p_accept=0.2)
        record_lineage_sim("alive_then_dead", sharpe=0.1, p_accept=0.2)
        # Force kill
        tracker = get_tracker()
        tracker.should_kill_lineage("alive_then_dead")
        self.assertFalse(is_lineage_alive("alive_then_dead"))


if __name__ == "__main__":
    unittest.main(verbosity=2)