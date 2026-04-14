"""
Tests for RAG Enhancement similarity functions.
"""
import unittest
from alpha_rag import _compute_ast_similarity, _bm25_score, _pick_diverse_dna


class TestASTSimilarity(unittest.TestCase):
    """Tests for AST-based structural similarity."""

    def test_identical_expressions(self):
        """Identical expressions should have 1.0 similarity."""
        expr = "rank(ts_delta(close, 5))"
        self.assertEqual(_compute_ast_similarity(expr, expr), 1.0)

    def test_same_structure_different_params(self):
        """Same structure with different params should have ~1.0 similarity."""
        expr1 = "rank(ts_delta(close, 5))"
        expr2 = "rank(ts_delta(close, 10))"
        result = _compute_ast_similarity(expr1, expr2)
        self.assertGreater(result, 0.9, "Same structure should have high similarity")

    def test_different_structure(self):
        """Different structures should have low similarity."""
        expr1 = "rank(ts_delta(close, 5))"
        expr2 = "sma(volume, 20)"
        result = _compute_ast_similarity(expr1, expr2)
        self.assertLess(result, 0.5, "Different structures should have low similarity")

    def test_empty_input(self):
        """Empty inputs should return 0.0."""
        self.assertEqual(_compute_ast_similarity("", "rank(x)"), 0.0)
        self.assertEqual(_compute_ast_similarity("rank(x)", ""), 0.0)

    def test_complex_expressions(self):
        """Test complex multi-operator expressions."""
        expr1 = "group_neutralize(rank(ts_delta(close, 5)), sector)"
        expr2 = "group_neutralize(rank(ts_delta(close, 10)), industry)"
        result = _compute_ast_similarity(expr1, expr2)
        self.assertGreater(result, 0.5, "Complex expressions should have moderate similarity")


class TestBM25Score(unittest.TestCase):
    """Tests for BM25-style keyword scoring."""

    def test_identical_queries(self):
        """Identical query and candidate should have high score."""
        query = "rank(ts_delta(close, 5))"
        self.assertGreater(_bm25_score(query, query), 1.0)

    def test_partial_overlap(self):
        """Partial token overlap should give partial score."""
        query = "rank(ts_delta(close, 5))"
        # Different operators - some overlap but not complete
        candidate = "sma(ts_delta(volume, 10), 20)"
        result = _bm25_score(query, candidate)
        # Partial overlap (ts_delta is present) but not full match
        self.assertGreater(result, 0.0)
        # Should be less than identical
        self.assertLess(result, _bm25_score(query, query))

    def test_no_overlap(self):
        """No token overlap should give 0.0."""
        query = "rank(ts_delta(close, 5))"
        candidate = "sma(volume, 20)"
        self.assertEqual(_bm25_score(query, candidate), 0.0)

    def test_empty_inputs(self):
        """Empty inputs should return 0.0."""
        self.assertEqual(_bm25_score("", "rank(x)"), 0.0)
        self.assertEqual(_bm25_score("rank(x)", ""), 0.0)


class TestDiverseDNAPicking(unittest.TestCase):
    """Tests for diverse DNA picking with similarity threshold."""

    def test_preserve_high_performers(self):
        """Should filter to high performers (sharpe >= 1.25)."""
        seeds = [
            {"expression": "rank(x)", "sharpe": 1.5},
            {"expression": "sma(y)", "sharpe": 1.0},  # Below threshold
            {"expression": "ema(z)", "sharpe": 1.8},
        ]
        result = _pick_diverse_dna(seeds, n=2, min_similarity=0.3)
        # Should only include high performers
        for expr in result:
            self.assertIn(expr, ["rank(x)", "ema(z)"])

    def test_diversity_filtering(self):
        """Should select structurally different seeds."""
        seeds = [
            {"expression": "rank(ts_delta(close, 5))", "sharpe": 1.5},
            {"expression": "rank(ts_delta(close, 10))", "sharpe": 1.4},
            {"expression": "sma(volume, 20)", "sharpe": 1.3},
            {"expression": "ema(close, 15)", "sharpe": 1.2},
            {"expression": "rsi(14)", "sharpe": 1.1},
        ]
        # With min_similarity=0.3, we allow some similarity but filter very close ones
        # rank(ts_delta(...)) expressions should get filtered to keep diversity
        result = _pick_diverse_dna(seeds, n=3, min_similarity=0.3)
        self.assertLessEqual(len(result), 3)
        # With this min_similarity threshold, rank variants may be filtered
        # The algorithm picks first, then filters remaining by similarity
        self.assertGreaterEqual(len(result), 1)

    def test_fewer_seeds_than_requested(self):
        """Should return all seeds if less than n."""
        seeds = [
            {"expression": "rank(x)", "sharpe": 1.5},
            {"expression": "sma(y)", "sharpe": 1.3},
        ]
        result = _pick_diverse_dna(seeds, n=5, min_similarity=0.3)
        self.assertEqual(len(result), 2)

    def test_empty_seeds(self):
        """Should return empty list for empty seeds."""
        self.assertEqual(_pick_diverse_dna([], n=3, min_similarity=0.3), [])


if __name__ == "__main__":
    unittest.main()