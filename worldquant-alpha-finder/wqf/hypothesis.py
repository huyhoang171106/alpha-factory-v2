"""wqf/hypothesis.py — Hypothesis-driven alpha generation.

Philosophy: Alpha = Hypothesis about market behavior + evidence of validation.
Not just a formula. Each candidate is tagged with its market hypothesis.
"""
import random
import re
from typing import List, Tuple, Callable

# ── Hypothesis Registry ─────────────────────────────────────────
# Each entry: (id, name, weight, templates_fn, description)
HYPOTHESES: List[Tuple[str, str, float, Callable, str]] = [
    (
        "microstructure",
        "Liquidity & Microstructure",
        0.18,
        lambda: [
            "rank(-ts_mean(abs(returns) / (volume + 1), 20))",
            "rank(volume / adv20) * rank(-abs(returns))",
            "rank(ts_corr(vwap, volume, 10)) * rank(-ts_mean(abs(returns), 5))",
            "-rank(ts_delta(close, 1) / (volume + 1))",
            "rank(ts_delta(volume, 1) / (ts_mean(volume, 5) + 1))",
            "rank(vwap / close - 1) * rank(-ts_std_dev(returns, 5))",
        ],
        "Volume-price dynamics at intraday frequency. ADV normalization critical."
    ),
    (
        "quality",
        "Quality Factor (Mean Reversion)",
        0.18,
        lambda: [
            "rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
            "-rank(close / ts_mean(close, 60) - 1) * rank(volume / adv20)",
            "rank(-ts_delta(close, 5) / (ts_std_dev(close, 10) + 0.001))",
            "rank(-returns / (ts_std_dev(returns, 20) + 0.001)) * rank(volume / adv20)",
            "-group_neutralize(rank(ts_delta(close, 5) / (ts_std_dev(close, 5) + 0.001)), sector)",
            "rank(ts_mean(returns, 20)) * rank(-ts_std_dev(returns, 20))",
        ],
        "Quality = strong risk-adjusted returns. Long losers, short winners (reversion)."
    ),
    (
        "behavioral",
        "Behavioral / Sentiment",
        0.14,
        lambda: [
            "-rank(ts_delta(close, 5)) * rank(ts_std_dev(returns, 20))",
            "rank(ts_skewness(returns, 20)) * rank(-ts_delta(volume, 1))",
            "rank(ts_mean(volume, 5) / ts_mean(volume, 20)) * rank(returns)",
            "-rank(ts_delta(sign(returns), 3))",
            "rank(ts_corr(ts_mean(close, 5), ts_mean(volume, 5), 10))",
        ],
        "Investor behavior creates predictable patterns. Uses skewness, volume anomaly."
    ),
    (
        "cross_sectional",
        "Cross-Sectional Relative Value",
        0.18,
        lambda: [
            "group_neutralize(rank(ts_mean(returns, 20)), sector)",
            "-group_neutralize(rank(ts_delta(close, 5)), sector) * rank(ts_mean(volume, 5))",
            "group_zscore(rank(ts_mean(returns, 10)), sector) - rank(volume / adv20)",
            "-group_mean(rank(ts_delta(close, 5)), industry) + rank(ts_mean(returns, 5))",
            "group_neutralize(rank(-ts_delta(close, 10) / (ts_std_dev(close, 10) + 0.001)), sector)",
            "group_rank(rank(ts_delta(close, 5)), industry) - rank(ts_mean(returns, 5))",
        ],
        "Long cheap sectors, short expensive ones. group_* operators add neutralization."
    ),
    (
        "regime",
        "Regime / Volatility",
        0.14,
        lambda: [
            "-rank(ts_std_dev(returns, 20) / ts_mean(ts_std_dev(returns, 20), 60))",
            "rank(ts_delta(ts_std_dev(returns, 10), 1) / (ts_std_dev(returns, 10) + 0.001))",
            "-rank(ts_mean(abs(returns) / (volume + 1), 5) / (ts_mean(abs(returns) / (volume + 1), 20) + 0.001))",
            "rank(ts_std_dev(volume, 10) / (ts_mean(volume, 20) + 1))",
        ],
        "Volatility is predictable. High-vol regimes mean-revert. Volume vol also predictive."
    ),
    (
        "stat_arb",
        "Statistical Arbitrage",
        0.10,
        lambda: [
            "rank(ts_corr(close, volume, 20)) - rank(ts_corr(ts_mean(close, 5), ts_mean(volume, 5), 20))",
            "rank(ts_corr(returns, ts_delta(close, 1), 10)) * rank(-ts_std_dev(returns, 20))",
            "-rank(ts_corr(ts_delta(close, 1), volume, 15)) * rank(ts_std_dev(returns, 10))",
        ],
        "Mispricing between related securities. Short-term mean reversion in correlated pairs."
    ),
    (
        "fundamental",
        "Fundamental / Value",
        0.08,
        lambda: [
            "rank(ebitda / debt) * rank(-ts_delta(returns, 20))",
            "rank(1 / (book_value / cap - ts_mean(book_value / cap, 60)))",
        ],
        "Cheap by fundamentals outperforms. Use ratios, sector-relative comparisons."
    ),
]


class HypothesisEngine:
    """
    Generates alpha candidates from structured hypotheses.

    Usage:
        engine = HypothesisEngine()
        candidates = engine.generate_batch(50)
        # Returns list of (expression, hypothesis_id, description)
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _pick_hypothesis(self) -> Tuple[str, str, float, Callable, str]:
        """Weighted random hypothesis selection."""
        names = [h[0] for h in HYPOTHESES]
        weights = [h[2] for h in HYPOTHESES]
        picked = self.rng.choices(names, weights=weights, k=1)[0]
        return next(h for h in HYPOTHESES if h[0] == picked)

    def _pick_template(self, templates_fn: Callable) -> str:
        return self.rng.choice(templates_fn())

    def _maybe_mutate(self, expr: str) -> str:
        """
        Light mutation: sign flip + safe lookback variation.
        Only mutates top-level function calls (never touches nested parens).
        """
        import random as _r

        def _replace_lookback(e: str, op: str) -> str:
            """Replace lookback in the outermost op(...) call only."""
            # Find the outermost occurrence of op(
            start = e.find(f"{op}(")
            if start == -1:
                return e
            # Walk forward from start, track paren depth
            depth = 0
            arg_start = start + len(op) + 1
            i = arg_start
            comma_count = 0
            while i < len(e):
                ch = e[i]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    if depth == 0:
                        # End of this call's closing paren
                        break
                    depth -= 1
                elif ch == ',' and depth == 0:
                    comma_count += 1
                    if comma_count == 1:
                        # We're at the first comma — lookback is after this
                        lb_start = i + 1
                        # Skip whitespace
                        while lb_start < len(e) and e[lb_start] in ' \t':
                            lb_start += 1
                        # Extract the current number
                        j = lb_start
                        while j < len(e) and e[j].isdigit():
                            j += 1
                        if j > lb_start:
                            new_lb = str(_r.randint(5, 60))
                            return e[:lb_start] + new_lb + e[j:]
                        return e
                i += 1
            return e

        mutations = [
            lambda e: ("-" + e.strip()) if not e.strip().startswith("-") else e.strip()[1:],
            lambda e: _replace_lookback(e, "ts_mean"),
            lambda e: _replace_lookback(e, "ts_delta"),
            lambda e: _replace_lookback(e, "ts_std_dev"),
            lambda e: _replace_lookback(e, "ts_rank"),
        ]

        mutated = expr
        # Apply 1-2 mutations
        n_muts = _r.randint(1, 2)
        for _ in range(n_muts):
            fn = _r.choice(mutations)
            candidate = fn(mutated)
            # Sanity check: balanced parens, non-empty, changed
            if candidate and candidate != mutated and candidate.count("(") == candidate.count(")"):
                mutated = candidate
        return mutated

    def generate(self, n: int = 1) -> List[Tuple[str, str, str]]:
        """
        Generate n candidates. Returns: (expression, hypothesis_name, description)
        """
        results = []
        seen = set()
        attempts = 0
        while len(results) < n and attempts < n * 5:
            attempts += 1
            hypo_id, hypo_name, weight, templates_fn, desc = self._pick_hypothesis()
            expr = self._pick_template(templates_fn)
            # ~30% chance of mutation
            if self.rng.random() < 0.3:
                expr = self._maybe_mutate(expr)
            if expr in seen:
                continue
            seen.add(expr)
            results.append((expr, hypo_name, desc))
        return results

    def generate_batch(self, n: int = 20) -> List[str]:
        """Generate n candidates, return just expressions."""
        return [expr for expr, _, _ in self.generate(n)]

    def list_hypotheses(self) -> List[Tuple[str, str, float, str]]:
        """Return summary of all hypotheses."""
        return [(h[0], h[1], h[2], h[4]) for h in HYPOTHESES]
