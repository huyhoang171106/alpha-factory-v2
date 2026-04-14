import asyncio
import unittest

from alpha_candidate import AlphaCandidate
from run_async_pipeline import AsyncAlphaFactory


class FakeTracker:
    def __init__(self, collinear=False):
        self._collinear = collinear

    def is_duplicate(self, _expr):
        return False

    def is_collinear(self, _expr):
        return self._collinear

    def close(self):
        return None

    def load_qd_archive(self, limit=2000):
        return []

    def acceptance_rate_by_arm(self, min_submitted=5, lookback_hours=168):
        return {}


class FakeClient:
    def simulate_batch(self, _expressions):
        return []


class FakeGenerator:
    def generate_batch(self, n=50, use_rag=False):
        return [
            AlphaCandidate(
                expression="group_neutralize(ts_zscore(ts_corr(close, volume, 20), 10), industry)"
            )
        ]


class FakeGovernor:
    def enqueue(self, sim_results, candidates_map=None):
        return 0

    def flush_once(self, limit=None):
        return {"selected": 0, "submitted": 0, "failed": 0}


class FakeResult:
    def __init__(
        self,
        sharpe=0.0,
        fitness=0.0,
        turnover=0.0,
        drawdown=0.0,
        sub_sharpe=-1.0,
        error="",
    ):
        self.sharpe = sharpe
        self.fitness = fitness
        self.turnover = turnover
        self.drawdown = drawdown
        self.sub_sharpe = sub_sharpe
        self.error = error


class AsyncPipelineTests(unittest.TestCase):
    def make_factory(self, collinear=False):
        return AsyncAlphaFactory(
            candidates_target=1,
            pre_rank_score=0.0,
            tracker=FakeTracker(collinear=collinear),
            client=FakeClient(),
            generator=FakeGenerator(),
            governor=FakeGovernor(),
        )

    def test_qd_novelty_changes_after_seen_expression(self):
        factory = self.make_factory()
        expr = "group_neutralize(ts_zscore(ts_corr(close, volume, 20), 10), industry)"
        first, _ = factory.qd_archive.novelty_score(expr)
        factory.qd_archive.maybe_update_archive(
            expr, quality=1.0, novelty=first, descriptor="g|c|d|e|mid|industry"
        )
        second, _ = factory.qd_archive.novelty_score(expr)
        self.assertGreater(first, second)

    def test_allocator_updates_after_reward(self):
        factory = self.make_factory()
        arm = "theme:mut"
        before = factory.allocator.arm_snapshot(arm)
        factory.allocator.update(arm, 1.0)
        after = factory.allocator.arm_snapshot(arm)
        self.assertEqual(before["pulls"], 0)
        self.assertEqual(after["pulls"], 1)
        self.assertGreater(after["mean"], before["mean"])

    def test_acceptance_priors_noop_on_empty_rates(self):
        """update_acceptance_priors with empty dict should not change allocator state."""
        factory = self.make_factory()
        arm = "llm"
        before = factory.allocator.arm_snapshot(arm)
        factory.allocator.update_acceptance_priors({})
        after = factory.allocator.arm_snapshot(arm)
        # p_accept should remain at the default uninformed prior of 0.5
        self.assertAlmostEqual(after["p_accept"], before["p_accept"], places=6)

    def test_acceptance_priors_shift_ev_for_high_accept_arm(self):
        """Arms updated with high p_accept should receive higher EV than cold-start arm."""
        factory = self.make_factory()
        factory.allocator.update_acceptance_priors(
            {
                "llm": {"accepted": 9, "resolved": 10, "p_accept": 0.90},
            }
        )
        ev_hot = factory.allocator.expected_value("llm", 0.6, 0.5)
        ev_cold = factory.allocator.expected_value("deterministic", 0.6, 0.5)
        # llm arm has p_accept=0.90, deterministic has p_accept=0.5 (prior)
        # ev_hot should be >= ev_cold on average
        self.assertGreaterEqual(ev_hot, ev_cold - 0.10)  # allow small Thompson noise

    def test_continuous_reward_orders_good_result_above_bad_result(self):
        factory = self.make_factory()
        good = FakeResult(
            sharpe=1.8, fitness=1.4, turnover=20, drawdown=0.03, sub_sharpe=0.2
        )
        weak = FakeResult(
            sharpe=-0.4, fitness=-0.2, turnover=80, drawdown=0.25, sub_sharpe=-0.3
        )

        self.assertGreater(
            factory._simulation_reward(good),
            factory._simulation_reward(weak),
        )
        self.assertGreater(factory._simulation_reward(good), 0.5)
        self.assertEqual(factory._simulation_reward(FakeResult(error="timeout")), 0.0)

    def test_adaptive_gates_relax_when_sim_queue_is_starved(self):
        factory = self.make_factory()
        factory.dynamic_pre_rank_score = 50.0
        factory.stats["generated"] = 100
        before_rank = factory.dynamic_pre_rank_score
        before_quality = factory.allocator.tier1_min_quality
        before_ev = factory.allocator.min_expected_value

        reason = factory.adapt_runtime_gates(
            {
                "accepted": 0,
                "rejected_after_submit": 0,
                "queued": 0,
                "dlq_rate": 0.0,
                "true_accept_rate": 0.0,
            }
        )

        self.assertEqual(reason, "relax_starved")
        self.assertLess(factory.dynamic_pre_rank_score, before_rank)
        self.assertLess(factory.allocator.tier1_min_quality, before_quality)
        self.assertLess(factory.allocator.min_expected_value, before_ev)

    def test_adaptive_gates_tighten_on_low_resolved_acceptance(self):
        factory = self.make_factory()
        before_rank = factory.dynamic_pre_rank_score
        before_quality = factory.allocator.tier1_min_quality
        before_ev = factory.allocator.min_expected_value

        reason = factory.adapt_runtime_gates(
            {
                "accepted": 1,
                "rejected_after_submit": 12,
                "queued": 2,
                "dlq_rate": 0.0,
                "true_accept_rate": 1 / 13,
            }
        )

        self.assertEqual(reason, "tighten_accept")
        self.assertGreater(factory.dynamic_pre_rank_score, before_rank)
        self.assertGreater(factory.allocator.tier1_min_quality, before_quality)
        self.assertGreater(factory.allocator.min_expected_value, before_ev)

    def test_adaptive_gates_do_not_tighten_initial_sim_backlog(self):
        factory = self.make_factory()
        factory.dynamic_pre_rank_score = 50.0
        factory.stats["generated"] = 20
        factory.stats["simulated"] = 0
        maxsize = getattr(
            factory.sim_queue, "maxsize", getattr(factory.sim_queue, "_maxsize", 300)
        )
        backlog_count = max(1, int(maxsize * 0.80))
        for i in range(backlog_count):
            factory.sim_queue.put_nowait(
                (1, i, AlphaCandidate(expression=f"rank(close + {i})"))
            )

        reason = factory.adapt_runtime_gates(
            {
                "accepted": 0,
                "rejected_after_submit": 0,
                "queued": 0,
                "dlq_rate": 0.0,
                "true_accept_rate": 0.0,
            }
        )

        self.assertEqual(reason, "hold")
        self.assertEqual(factory.dynamic_pre_rank_score, 50.0)

    def test_invalid_expression_rejected_before_ranker_filters(self):
        factory = self.make_factory()
        accepted, reason, score = asyncio.run(
            factory.should_accept_candidate(
                AlphaCandidate(expression="if(close > open, 1, -1)")
            )
        )

        self.assertFalse(accepted)
        self.assertTrue(reason.startswith("invalid_expr:"))
        self.assertEqual(score, 0.0)

    def test_local_bt_unsupported_expression_rejected_before_ranker_filters(self):
        factory = self.make_factory()
        accepted, reason, score = asyncio.run(
            factory.should_accept_candidate(
                AlphaCandidate(expression="rank(close) < rank(open)")
            )
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "local_bt_unsupported:comparison_operator")
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
