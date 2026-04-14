import unittest

from submit_governor import SubmitGovernor


class DummyResult:
    def __init__(self, expression, alpha_id, sharpe, fitness=1.2, turnover=20.0, all_passed=True, error=""):
        self.expression = expression
        self.alpha_id = alpha_id
        self.sharpe = sharpe
        self.fitness = fitness
        self.turnover = turnover
        self.all_passed = all_passed
        self.error = error


class DummyCandidate:
    def __init__(self, family):
        self.family = family


class FakeTracker:
    def __init__(self):
        self.queued = []
        self.submitted = []
        self.failed = []
        self.dead_lettered = []
        self.rows = []
        self.pending_review = []
        self.review_finalized = []
        self.review_pending_marks = []

    def mark_queued(self, alpha_id):
        self.queued.append(alpha_id)

    def transition_submit_state(self, alpha_id, next_state, reason=""):
        return True

    def get_submit_queue(self, limit=30):
        return self.rows[:limit]

    def mark_submitted(self, alpha_id):
        self.submitted.append(alpha_id)

    def mark_submit_failed(self, alpha_id, reason, error_class="", next_retry_seconds=60):
        self.failed.append((alpha_id, reason, error_class, next_retry_seconds))

    def mark_dead_lettered(self, alpha_id, reason, error_class=""):
        self.dead_lettered.append((alpha_id, reason, error_class))

    def get_submitted_pending_review(self, limit=50):
        return self.pending_review[:limit]

    def finalize_submit_review(self, alpha_id, decision, reason="", self_corr=None):
        self.review_finalized.append((alpha_id, decision, reason, self_corr))
        return True

    def mark_review_pending(self, alpha_id, reason="", backoff_seconds=60):
        self.review_pending_marks.append((alpha_id, reason, backoff_seconds))


class FakeClient:
    def __init__(self, success_ids=None, decisions=None):
        self.success_ids = set(success_ids or [])
        self.decisions = decisions or {}
        self.calls = []

    def submit_alpha(self, alpha_id):
        self.calls.append(alpha_id)
        return alpha_id in self.success_ids

    def get_submission_decision(self, alpha_id):
        return self.decisions.get(alpha_id, ("submitted", "", "", 0.0))


class SubmitGovernorTests(unittest.TestCase):
    def test_enqueue_skips_family_duplicates(self):
        tracker = FakeTracker()
        client = FakeClient()
        governor = SubmitGovernor(tracker, client, max_submits_per_minute=4)

        r1 = DummyResult("expr_a", "A1", sharpe=2.0)
        r2 = DummyResult("expr_b", "A2", sharpe=1.9)
        cmap = {"expr_a": DummyCandidate("fam_x"), "expr_b": DummyCandidate("fam_x")}

        queued = governor.enqueue([r1, r2], candidates_map=cmap)
        self.assertEqual(queued, 1)
        self.assertEqual(tracker.queued, ["A1"])

    def test_flush_submits_and_marks_failed(self):
        tracker = FakeTracker()
        tracker.rows = [
            (1, "expr1", "ID1", 2.0, "fam1", "theme1", 0, "", 0, ""),
            (2, "expr2", "ID2", 1.9, "fam2", "theme2", 0, "", 0, ""),
        ]
        client = FakeClient(success_ids={"ID1"})
        governor = SubmitGovernor(tracker, client, max_submits_per_minute=4)

        out = governor.flush_once(limit=2)
        self.assertEqual(out["selected"], 2)
        self.assertEqual(out["submitted"], 1)
        self.assertEqual(out["failed"], 1)
        self.assertEqual(tracker.submitted, ["ID1"])
        self.assertEqual(tracker.failed[0][0], "ID2")

    def test_flush_log_redacts_expression(self):
        tracker = FakeTracker()
        raw_expr = "group_neutralize(rank(ts_delta(close, 12)), subindustry)"
        tracker.rows = [(1, raw_expr, "ID1", 2.0, "fam1", "theme1", 0, "", 0, "")]
        client = FakeClient(success_ids={"ID1"})
        governor = SubmitGovernor(tracker, client, max_submits_per_minute=4)

        with self.assertLogs("submit_governor", level="INFO") as logs:
            governor.flush_once(limit=1)

        rendered = "\n".join(logs.output)
        self.assertIn("expr_id=", rendered)
        self.assertNotIn(raw_expr, rendered)

    def test_retry_policy_for_semantic_error_goes_dlq(self):
        retryable, delay, dlq = SubmitGovernor._retry_policy("submit_semantic_4xx_422", attempts=1)
        self.assertFalse(retryable)
        self.assertTrue(dlq)

    def test_reconcile_submitted_updates_review_states(self):
        tracker = FakeTracker()
        tracker.pending_review = [("ID1", 0, "", ""), ("ID2", 1, "", ""), ("ID3", 2, "", "")]
        client = FakeClient(
            decisions={
                "ID1": ("accepted", "", "", 0.55),
                "ID2": ("rejected", "", "fitness too low", 0.82),
                "ID3": ("submitted", "", "", 0.0),
            }
        )
        governor = SubmitGovernor(tracker, client, max_submits_per_minute=4)
        out = governor.reconcile_submitted(limit=10)
        self.assertEqual(out["checked"], 3)
        self.assertEqual(out["accepted"], 1)
        self.assertEqual(out["rejected"], 1)
        self.assertEqual(out["pending"], 1)
        self.assertEqual(len(tracker.review_finalized), 2)
        self.assertEqual(len(tracker.review_pending_marks), 1)
        self.assertEqual(tracker.review_pending_marks[0][0], "ID3")

    def test_reconcile_submitted_applies_error_backoff(self):
        tracker = FakeTracker()
        tracker.pending_review = [("ID9", 2, "", "")]
        client = FakeClient(decisions={"ID9": ("unknown", "submit_status_rate_limited_429", "", 0.0)})
        governor = SubmitGovernor(tracker, client, max_submits_per_minute=4)
        out = governor.reconcile_submitted(limit=10)
        self.assertEqual(out["errors"], 1)
        self.assertEqual(len(tracker.review_pending_marks), 1)
        alpha_id, reason, delay = tracker.review_pending_marks[0]
        self.assertEqual(alpha_id, "ID9")
        self.assertIn("rate", reason.lower())
        self.assertGreaterEqual(delay, 192)
        self.assertLessEqual(delay, 288)

    def test_review_backoff_has_jittered_range(self):
        delays = [SubmitGovernor._review_backoff_seconds(attempts=0, error_class="pending") for _ in range(20)]
        # base=45s with +/-20% jitter => [36, 54]
        self.assertTrue(all(36 <= d <= 54 for d in delays))


if __name__ == "__main__":
    unittest.main()
