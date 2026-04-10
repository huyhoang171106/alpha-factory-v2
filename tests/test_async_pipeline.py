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


class FakeClient:
    def simulate_batch(self, _expressions):
        return []


class FakeGenerator:
    def generate_batch(self, n=50, use_rag=False):
        return [AlphaCandidate(expression="group_neutralize(ts_zscore(ts_corr(close, volume, 20), 10), industry)")]


class FakeGovernor:
    def enqueue(self, sim_results, candidates_map=None):
        return 0

    def flush_once(self, limit=None):
        return {"selected": 0, "submitted": 0, "failed": 0}


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
        factory.qd_archive.maybe_update_archive(expr, quality=1.0, novelty=first, descriptor="g|c|d|e|mid|industry")
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

    def test_should_accept_candidate_rejects_collinear(self):
        factory = self.make_factory(collinear=True)
        cand = AlphaCandidate(expression="group_neutralize(ts_mean(returns,20),industry)")
        accepted, reason = factory.should_accept_candidate(cand)
        self.assertFalse(accepted)
        self.assertEqual(reason, "collinear")

    def test_should_accept_candidate_accepts_valid_expression(self):
        factory = self.make_factory(collinear=False)
        cand = AlphaCandidate(expression="group_neutralize(ts_zscore(ts_corr(close, volume, 20), 10), industry)")
        accepted, reason = factory.should_accept_candidate(cand)
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")


if __name__ == "__main__":
    unittest.main()
