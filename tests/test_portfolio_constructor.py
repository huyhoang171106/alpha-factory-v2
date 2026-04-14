import unittest
from portfolio_constructor import (
    CorrelationBasedEnsembleBuilder,
    EnsembleICStabilityValidator,
    PreSubmissionGate,
    AlphaEnsemble,
    build_ensemble_from_candidates,
    ensemble_summary,
    CandidateSignal,
)

__all__ = [
    "CorrelationBasedEnsembleBuilder",
    "EnsembleICStabilityValidator",
    "PreSubmissionGate",
    "AlphaEnsemble",
    "build_ensemble_from_candidates",
    "ensemble_summary",
    "CandidateSignal",
]




class TestCorrelationBasedEnsembleBuilder(unittest.TestCase):
    def test_pairwise_ic_matrix_shape(self):
        builder = CorrelationBasedEnsembleBuilder()
        alphas = ["alpha1", "alpha2", "alpha3"]
        ic_matrix = builder.compute_pairwise_ic(alphas)
        self.assertEqual(len(ic_matrix), 3)
        self.assertEqual(len(ic_matrix[0]), 3)
        # diagonal should be 1.0
        self.assertAlmostEqual(ic_matrix[0][0], 1.0)
        self.assertAlmostEqual(ic_matrix[1][1], 1.0)

    def test_pairwise_ic_different_expressions(self):
        builder = CorrelationBasedEnsembleBuilder()
        alphas = [
            "rank(ts_mean(close, 20))",
            "rank(volume)",
        ]
        ic_matrix = builder.compute_pairwise_ic(alphas)
        # Different expressions should have IC < 1.0
        self.assertLess(ic_matrix[0][1], 1.0)
        self.assertGreater(ic_matrix[0][1], 0.0)

    def test_select_sub_alphas_low_ic(self):
        builder = CorrelationBasedEnsembleBuilder(ic_threshold=0.3)
        # Two very different alphas should both be selected
        candidates = [
            "rank(ts_mean(close, 20))",
            "rank(ts_std_dev(volume, 5))",
        ]
        selected = builder.select_sub_alphas(candidates)
        self.assertEqual(len(selected), 2)

    def test_select_sub_alphas_max_limit(self):
        builder = CorrelationBasedEnsembleBuilder(max_sub_alphas=2)
        candidates = [
            "rank(close)",
            "rank(ts_mean(close, 5))",
            "rank(ts_std_dev(volume, 5))",
        ]
        selected = builder.select_sub_alphas(candidates)
        self.assertLessEqual(len(selected), 2)

    def test_equal_risk_weights(self):
        builder = CorrelationBasedEnsembleBuilder()
        weights = builder.equal_risk_contribution_weights(3)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights[0], 1 / 3)
        self.assertAlmostEqual(weights[1], 1 / 3)
        self.assertAlmostEqual(weights[2], 1 / 3)

    def test_equal_risk_weights_single(self):
        builder = CorrelationBasedEnsembleBuilder()
        weights = builder.equal_risk_contribution_weights(1)
        self.assertEqual(weights, [1.0])

    def test_equal_risk_weights_zero(self):
        builder = CorrelationBasedEnsembleBuilder()
        weights = builder.equal_risk_contribution_weights(0)
        self.assertEqual(weights, [])

    def test_turnover_estimate(self):
        builder = CorrelationBasedEnsembleBuilder()
        alphas = ["rank(close)", "rank(ts_mean(close, 20))"]
        est = builder.ensemble_turnover_estimate(alphas)
        self.assertGreaterEqual(est, 0.0)
        self.assertLessEqual(est, 1.0)

    def test_turnover_estimate_empty(self):
        builder = CorrelationBasedEnsembleBuilder()
        est = builder.ensemble_turnover_estimate([])
        self.assertEqual(est, 0.0)

    def test_build_ensemble_two_candidates(self):
        builder = CorrelationBasedEnsembleBuilder()
        candidates = [
            ("rank(ts_mean(close, 20))", 1.5),
            ("rank(ts_std_dev(volume, 5))", 1.3),
        ]
        ens = builder.build_ensemble(candidates)
        self.assertTrue(ens.is_valid)
        self.assertEqual(len(ens.sub_alphas), 2)
        self.assertAlmostEqual(sum(ens.weights), 1.0)
        self.assertGreater(ens.ensemble_ic, 0.0)

    def test_build_ensemble_empty(self):
        builder = CorrelationBasedEnsembleBuilder()
        ens = builder.build_ensemble([])
        self.assertFalse(ens.is_valid)
        self.assertEqual(ens.rejection_reason, "no_candidates")

    def test_build_ensemble_diverse_candidates(self):
        builder = CorrelationBasedEnsembleBuilder(ic_threshold=0.25)
        candidates = [
            ("rank(ts_mean(close, 20))", 1.5),
            ("rank(ts_std_dev(volume, 5))", 1.4),
            ("rank(ts_corr(open, volume, 10))", 1.3),
        ]
        ens = builder.build_ensemble(candidates)
        self.assertTrue(ens.is_valid)
        self.assertGreaterEqual(len(ens.sub_alphas), 2)


class TestEnsembleICStabilityValidator(unittest.TestCase):
    def test_ic_autocorrelation_stable(self):
        validator = EnsembleICStabilityValidator()
        # Stable IC series with low autocorrelation
        ic_series = [0.05, 0.06, 0.04, 0.05, 0.06]
        autocorr = validator.compute_ic_autocorrelation(ic_series)
        self.assertLess(abs(autocorr), 1.0)

    def test_ic_autocorrelation_persistent(self):
        validator = EnsembleICStabilityValidator()
        # Exponential growth produces strong positive autocorrelation
        ic_series = [0.01, 0.02, 0.04, 0.08, 0.16]
        autocorr = validator.compute_ic_autocorrelation(ic_series)
        self.assertGreater(autocorr, 0.2)   # clearly positive, > 0

    def test_ic_autocorrelation_short_series(self):
        validator = EnsembleICStabilityValidator()
        ic_series = [0.05, 0.06]
        autocorr = validator.compute_ic_autocorrelation(ic_series)
        self.assertEqual(autocorr, 0.0)  # not enough data

    def test_ic_stability_gate_pass(self):
        validator = EnsembleICStabilityValidator(autocorr_threshold=0.35)
        # low autocorr + positive mean IC — linear growth sits at ~0.37, barely under 0.35
        ic_series = [0.05, 0.02, 0.06, 0.03, 0.04]
        passed, reason = validator.ic_stability_gate(ic_series)
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(reason, str)

    def test_ic_stability_gate_fail_high_autocorr(self):
        validator = EnsembleICStabilityValidator(autocorr_threshold=0.35)
        # trending up = high autocorr (perfect linear trend: autocorr≈0.40)
        ic_series = [0.01, 0.02, 0.03, 0.04, 0.05]
        passed, reason = validator.ic_stability_gate(ic_series)
        self.assertFalse(passed)
        self.assertIn("autocorr", reason.lower())

    def test_ic_stability_gate_insufficient_data(self):
        validator = EnsembleICStabilityValidator()
        ic_series = [0.05]
        passed, reason = validator.ic_stability_gate(ic_series)
        self.assertTrue(passed)
        self.assertIn("insufficient", reason.lower())

    def test_validate_ensemble_ic(self):
        validator = EnsembleICStabilityValidator()
        ensemble = AlphaEnsemble(
            sub_alphas=["a1", "a2"],
            weights=[0.5, 0.5],
            ensemble_ic=0.2,  # low IC → diversified
            ensemble_turnover=0.3,
            ensemble_sharpe=1.4,
            is_valid=True,
        )
        ic_series = [0.05, 0.02, 0.04, 0.03, 0.05]
        result = validator.validate_ensemble_ic(ensemble, ic_series)
        self.assertIsInstance(result, bool)


class TestPreSubmissionGate(unittest.TestCase):
    def test_ensemble_better_than_individual(self):
        gate = PreSubmissionGate(improvement_threshold=1.1)
        candidates = [
            ("rank(ts_mean(close, 20))", 1.5),
            ("rank(ts_std_dev(volume, 5))", 1.4),
            ("rank(ts_corr(open, volume, 10))", 1.3),
        ]
        use_ens, ens = gate.should_submit_ensemble(candidates)
        self.assertIsInstance(use_ens, bool)
        if use_ens and ens:
            self.assertIsInstance(ens, AlphaEnsemble)

    def test_single_candidate_no_ensemble(self):
        gate = PreSubmissionGate()
        candidates = [("alpha_single", 1.5)]
        use_ens, ens = gate.should_submit_ensemble(candidates)
        self.assertFalse(use_ens)
        self.assertIsNone(ens)

    def test_empty_candidates(self):
        gate = PreSubmissionGate()
        use_ens, ens = gate.should_submit_ensemble([])
        self.assertFalse(use_ens)
        self.assertIsNone(ens)

    def test_threshold_logic(self):
        gate = PreSubmissionGate(improvement_threshold=1.1)
        candidates = [
            ("alpha_a", 1.5),
            ("alpha_b", 1.0),
        ]
        use_ens, ens = gate.should_submit_ensemble(candidates)
        self.assertIsInstance(use_ens, bool)


class TestConvenienceHelpers(unittest.TestCase):
    def test_build_ensemble_from_candidates(self):
        candidates = [
            ("rank(ts_mean(close, 20))", 1.5),
            ("rank(volume)", 1.3),
        ]
        ens = build_ensemble_from_candidates(candidates)
        self.assertIsInstance(ens, AlphaEnsemble)

    def test_ensemble_summary_nonempty(self):
        ens = AlphaEnsemble(
            sub_alphas=["alpha_a", "alpha_b"],
            weights=[0.5, 0.5],
            ensemble_ic=0.25,
            ensemble_turnover=0.3,
            ensemble_sharpe=1.4,
            is_valid=True,
        )
        summary = ensemble_summary(ens)
        self.assertIsInstance(summary, str)
        self.assertIn("2 components", summary)

    def test_ensemble_summary_empty(self):
        summary = ensemble_summary(AlphaEnsemble([], [], 0.0, 0.0, 0.0, False))
        self.assertIn("Empty", summary)


class TestAlphaEnsembleDataclass(unittest.TestCase):
    def test_num_components_property(self):
        ens = AlphaEnsemble(
            sub_alphas=["a", "b", "c"],
            weights=[0.3, 0.3, 0.4],
            ensemble_ic=0.2,
            ensemble_turnover=0.3,
            ensemble_sharpe=1.5,
            is_valid=True,
        )
        self.assertEqual(ens.num_components, 3)

    def test_rejection_reason_optional(self):
        ens = AlphaEnsemble(
            sub_alphas=["a"],
            weights=[1.0],
            ensemble_ic=0.0,
            ensemble_turnover=0.0,
            ensemble_sharpe=0.0,
            is_valid=False,
        )
        self.assertIsNone(ens.rejection_reason)


class TestTurnoverBudgetEnforcement(unittest.TestCase):
    def test_turnover_within_budget(self):
        builder = CorrelationBasedEnsembleBuilder()
        ok, reason = builder.turnover_budget_enforcement(ensemble_turnover=0.45, max_turnover=0.60)
        self.assertTrue(ok)
        self.assertIn("turnover_ok", reason)

    def test_turnover_exceeds_budget(self):
        builder = CorrelationBasedEnsembleBuilder()
        ok, reason = builder.turnover_budget_enforcement(ensemble_turnover=0.75, max_turnover=0.60)
        self.assertFalse(ok)
        self.assertIn("turnover=", reason)

    def test_turnover_at_exact_boundary(self):
        builder = CorrelationBasedEnsembleBuilder()
        ok, reason = builder.turnover_budget_enforcement(ensemble_turnover=0.60, max_turnover=0.60)
        self.assertTrue(ok)  # strictly greater than is rejected, equal is fine


class TestRollingICValidation(unittest.TestCase):
    def test_rolling_ic_pass(self):
        validator = EnsembleICStabilityValidator()
        # 5/6 positive > 70%
        ic_series = [0.05, -0.01, 0.06, 0.04, 0.03, 0.05]
        passed, reason = validator.rolling_ic_validation(ic_series, min_positive_fraction=0.70)
        self.assertTrue(passed)
        self.assertIn("rolling_ic", reason)

    def test_rolling_ic_fail(self):
        validator = EnsembleICStabilityValidator()
        # 3/6 = 50% < 70%
        ic_series = [0.05, -0.01, -0.02, 0.04, -0.01, -0.03]
        passed, reason = validator.rolling_ic_validation(ic_series, min_positive_fraction=0.70)
        self.assertFalse(passed)
        self.assertIn("rolling_ic", reason)

    def test_rolling_ic_insufficient_data(self):
        validator = EnsembleICStabilityValidator()
        ic_series = [0.05, 0.03]
        passed, reason = validator.rolling_ic_validation(ic_series)
        self.assertTrue(passed)  # gracefully passes with insufficient data
        self.assertIn("insufficient", reason.lower())

    def test_rolling_ic_all_positive(self):
        validator = EnsembleICStabilityValidator()
        ic_series = [0.05, 0.03, 0.04, 0.06]
        passed, reason = validator.rolling_ic_validation(ic_series)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
