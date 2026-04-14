import os
import unittest

from alpha_rag import RAGMutator


class RAGPromptingTests(unittest.TestCase):
    def setUp(self):
        self._raw_context = os.getenv("LLM_INCLUDE_RAW_ALPHA_CONTEXT")
        os.environ.pop("LLM_INCLUDE_RAW_ALPHA_CONTEXT", None)

    def tearDown(self):
        if self._raw_context is None:
            os.environ.pop("LLM_INCLUDE_RAW_ALPHA_CONTEXT", None)
        else:
            os.environ["LLM_INCLUDE_RAW_ALPHA_CONTEXT"] = self._raw_context

    def make_rag(self):
        return RAGMutator(seed_file="missing-for-tests.json")

    def test_build_messages_redacts_raw_alpha_context_by_default(self):
        rag = self.make_rag()
        expr = "group_neutralize(rank(ts_delta(close, 5)), subindustry)"

        messages = rag._build_messages([expr], recent_fails=[expr], batch_size=2)
        body = "\n".join(m["content"] for m in messages)

        self.assertNotIn(expr, body)
        self.assertIn(RAGMutator._safe_expr_id(expr), body)
        self.assertIn("valid JSON", body)
        self.assertIn("Generate 2 diverse alpha candidates", body)

    def test_json_response_parses_valid_candidates_and_rejects_invalid(self):
        rag = self.make_rag()
        raw = """
        ```json
        {
          "candidates": [
            {
              "expression": "group_neutralize(rank(ts_delta(close, 5)), subindustry)",
              "hypothesis": "event_reversal",
              "expected_turnover": "medium",
              "risk_flags": []
            },
            {
              "expression": "foo(close)",
              "hypothesis": "bad"
            }
          ]
        }
        ```
        """

        candidates, rejected = rag._candidates_from_response(raw, limit=4)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].theme, "rag_llm")
        self.assertEqual(candidates[0].mutation_type, "llm_rag_json")
        self.assertEqual(candidates[0].hypothesis, "event_reversal")
        self.assertTrue(any("Unknown operator" in reason for reason in rejected))

    def test_legacy_single_expression_response_still_works(self):
        rag = self.make_rag()
        raw = "group_neutralize(rank(ts_delta(close, 5)), industry)"

        candidates, rejected = rag._candidates_from_response(raw, limit=1)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(rejected, [])

    def test_generated_candidate_logs_do_not_include_raw_expression(self):
        rag = self.make_rag()
        raw = "group_neutralize(rank(ts_delta(close, 5)), industry)"
        candidates, _ = rag._candidates_from_response(raw, limit=1)

        with self.assertLogs("alpha_rag", level="INFO") as logs:
            rag._log_generated_candidates("TestProvider", candidates)

        rendered = "\n".join(logs.output)
        self.assertIn(RAGMutator._safe_expr_id(raw), rendered)
        self.assertNotIn(raw, rendered)

    def test_repair_feedback_redacts_raw_expression_by_default(self):
        rag = self.make_rag()
        raw = "rank(ts_delta(close, 5))"
        candidates, _ = rag._candidates_from_response(raw, limit=1)

        feedback = rag._build_repair_feedback(
            candidates[0],
            "CRITIC: Missing mandatory 'group_neutralize' wrapper.",
        )
        messages = rag._build_messages([], repair_feedback=[feedback])
        body = "\n".join(m["content"] for m in messages)

        self.assertIn("REFINEMENT NEEDED", body)
        self.assertIn(RAGMutator._safe_expr_id(raw), body)
        self.assertNotIn(raw, body)
        self.assertNotIn("Chain of Thought", body)


if __name__ == "__main__":
    unittest.main()
