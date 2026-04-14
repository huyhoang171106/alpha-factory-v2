import os
import tempfile
import unittest

from tracker import AlphaTracker


class DummyResult:
    def __init__(
        self,
        expression,
        sharpe=1.6,
        fitness=1.2,
        turnover=22.0,
        all_passed=True,
        error="",
        alpha_id="ALPHA1",
        alpha_url="http://x",
    ):
        self.expression = expression
        self.sharpe = sharpe
        self.fitness = fitness
        self.turnover = turnover
        self.returns = 0.1
        self.drawdown = 0.05
        self.passed_checks = 5
        self.total_checks = 5
        self.all_passed = all_passed
        self.alpha_id = alpha_id
        self.alpha_url = alpha_url
        self.error = error
        self.sub_sharpe = 1.0
        self.region = "USA"
        self.universe = "TOP3000"
        self.delay = 1
        self.decay = 6
        self.neutralization = "SUBINDUSTRY"


class DummyCandidate:
    def __init__(self):
        self.theme = "test_theme"
        self.family = "test_family"
        self.mutation_type = "test_mutation"
        self.hypothesis = "h"
        self.nearest_sibling = ""


class TrackerMemoryTests(unittest.TestCase):
    def test_persistent_signature_memory_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alpha_results.db")
            tracker1 = AlphaTracker(db_path=db_path)
            expr = "rank(volume/(adv20+1))"
            tracker1.save_result(DummyResult(expr, alpha_id="ID-A"), candidate=DummyCandidate())
            tracker1.close()

            tracker2 = AlphaTracker(db_path=db_path)
            self.assertTrue(tracker2.is_duplicate(expr))
            # Same structure with different number should be detected by param signature.
            self.assertTrue(tracker2.is_collinear("rank(volume/(adv20+60))"))
            tracker2.close()

    def test_submit_state_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alpha_results.db")
            tracker = AlphaTracker(db_path=db_path)
            tracker.save_result(DummyResult("rank(close)", alpha_id="ID-B"), candidate=DummyCandidate())

            tracker.mark_queued("ID-B")
            rows = tracker.get_submit_queue(limit=10)
            self.assertEqual(len(rows), 1)

            tracker.mark_submit_failed("ID-B", "network", error_class="submit_network_exception", next_retry_seconds=0)
            rows2 = tracker.get_submit_queue(limit=10)
            self.assertEqual(len(rows2), 1)  # failed jobs remain retryable

            tracker.mark_submitted("ID-B")
            rows3 = tracker.get_submit_queue(limit=10)
            self.assertEqual(len(rows3), 0)
            tracker.close()

    def test_dead_letter_and_replay_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alpha_results.db")
            tracker = AlphaTracker(db_path=db_path)
            tracker.save_result(DummyResult("group_neutralize(rank(close),industry)", alpha_id="ID-C"), candidate=DummyCandidate())
            tracker.mark_queued("ID-C")
            tracker.mark_dead_lettered("ID-C", reason="semantic_4xx", error_class="submit_semantic_4xx_422")
            # dead-lettered should not appear in queue
            queued_before = tracker.get_submit_queue(limit=10)
            self.assertEqual(len(queued_before), 0)
            replayed = tracker.replay_dlq(limit=10)
            self.assertEqual(replayed, 1)
            queued_after = tracker.get_submit_queue(limit=10)
            self.assertEqual(len(queued_after), 1)
            tracker.close()

    def test_qd_archive_upsert_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alpha_results.db")
            tracker = AlphaTracker(db_path=db_path)
            tracker.upsert_qd_archive(
                descriptor="g|c|nd|ne|mid|industry",
                expression="group_neutralize(ts_corr(close, volume, 20), industry)",
                quality_score=1.3,
                novelty_score=0.7,
            )
            tracker.upsert_qd_archive(
                descriptor="g|c|nd|ne|mid|industry",
                expression="group_neutralize(ts_corr(close, volume, 30), industry)",
                quality_score=1.1,  # lower, should not replace elite expression
                novelty_score=0.5,
            )
            rows = tracker.load_qd_archive(limit=10)
            self.assertEqual(len(rows), 1)
            self.assertIn("volume, 20", rows[0][1])
            stats = tracker.qd_archive_stats()
            self.assertEqual(stats["cells"], 1)
            self.assertGreaterEqual(stats["updates"], 2)
            tracker.close()

    def test_finalize_submit_review_and_kpis(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alpha_results.db")
            tracker = AlphaTracker(db_path=db_path)
            try:
                tracker.save_result(DummyResult("rank(close)", alpha_id="ID-D"), candidate=DummyCandidate())
                tracker.mark_queued("ID-D")
                tracker.mark_submitted("ID-D")
                ok = tracker.finalize_submit_review("ID-D", decision="accepted", reason="approved by review")
                self.assertTrue(ok)
                kpi = tracker.minute_kpis(lookback_minutes=60)
                self.assertGreaterEqual(kpi["accepted"], 1)
                self.assertGreaterEqual(kpi["true_accept_rate"], 1.0)
            finally:
                tracker.close()

    def test_submitted_review_scheduler_orders_oldest_and_respects_backoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alpha_results.db")
            tracker = AlphaTracker(db_path=db_path)
            try:
                tracker.save_result(DummyResult("rank(open)", alpha_id="ID-E"), candidate=DummyCandidate())
                tracker.mark_queued("ID-E")
                tracker.mark_submitted("ID-E")

                tracker.save_result(DummyResult("rank(vwap)", alpha_id="ID-F"), candidate=DummyCandidate())
                tracker.mark_queued("ID-F")
                tracker.mark_submitted("ID-F")
                tracker.mark_review_pending("ID-F", reason="pending", backoff_seconds=3600)

                pending = tracker.get_submitted_pending_review(limit=10)
                self.assertGreaterEqual(len(pending), 1)
                ids = [row[0] for row in pending]
                self.assertIn("ID-E", ids)
                self.assertNotIn("ID-F", ids)
            finally:
                tracker.close()


    def test_acceptance_rate_by_arm(self):
        """acceptance_rate_by_arm must return per-cluster p_accept from resolved WQ decisions."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alpha_results.db")
            tracker = AlphaTracker(db_path=db_path)
            try:
                # Seed 6 alphas with known final states (3 per cluster)
                for i in range(3):
                    # cluster: llm — all accepted
                    cand = DummyCandidate()
                    cand.theme = "rag"
                    cand.mutation_type = "llm"
                    r = DummyResult(f"rank(close-{i})", alpha_id=f"LLM-{i}")
                    tracker.save_result(r, candidate=cand)
                    tracker.mark_queued(f"LLM-{i}")
                    tracker.mark_submitted(f"LLM-{i}")
                    tracker.finalize_submit_review(f"LLM-{i}", "accepted")

                for i in range(3):
                    # cluster: deterministic — all rejected
                    r = DummyResult(f"rank(volume-{i})", alpha_id=f"DET-{i}")
                    tracker.save_result(r, candidate=DummyCandidate())
                    tracker.mark_queued(f"DET-{i}")
                    tracker.mark_submitted(f"DET-{i}")
                    tracker.finalize_submit_review(f"DET-{i}", "rejected")

                # min_submitted=1 to allow small samples
                rates = tracker.acceptance_rate_by_arm(min_submitted=1)
                llm_rate = rates.get("llm", {}).get("p_accept", -1)
                det_rate = rates.get("deterministic", {}).get("p_accept", -1)
                self.assertAlmostEqual(llm_rate, 1.0, places=3)
                self.assertAlmostEqual(det_rate, 0.0, places=3)
            finally:
                tracker.close()


if __name__ == "__main__":
    unittest.main()
