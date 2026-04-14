"""wqf/ranker.py — Pre-simulation alpha scorer with hypothesis inference."""
import re
from typing import List, Tuple

HIGH_VALUE = {
    "group_neutralize": 20, "group_rank": 18, "group_mean": 15,
    "group_zscore": 15,
    "ts_corr": 10, "ts_covariance": 9, "ts_regression": 12,
    "ts_skewness": 10, "ts_entropy": 15,
    "signed_power": 10, "sign": 7, "scale": 6,
    "trade_when": 18, "ts_quantile": 14,
    "adv20": 12, "adv60": 8, "vwap": 10,
    "ts_decay_linear": 8, "ts_std_dev": 6, "ts_zscore": 6,
    "ts_sum": 5, "ts_delay": 4,
    "rank": 3,
}

LOW_VALUE_PATTERNS = [
    (r"^-?rank\(\w+\)$", -20),
    (r"^-?rank\(ts_\w+\(\w+,\s*\d+\)\)$", -25),
    (r"rank\(rank\(", -15),
    (r"ts_delta\([a-z]+,\s*[12]\)", -8),
    (r"^rank\(close/open\)", -5),
    (r"^rank\(open/close\)", -5),
]


def _depth(expr: str) -> int:
    d = mx = 0
    for ch in expr:
        if ch == "(": d += 1; mx = max(mx, d)
        elif ch == ")": d -= 1
    return mx


def _n_ops(expr: str) -> int:
    return len(re.findall(r'[a-zA-Z_]\w*\s*\(', expr))


def _unique_lookbacks(expr: str) -> int:
    nums = re.findall(r'\b\d+\b', expr)
    return len(set(nums))


def infer_hypothesis(expr: str) -> str:
    """Infer what market hypothesis this alpha encodes."""
    e = expr.lower()
    if "group_neutral" in e or "group_rank" in e:
        return "cross-sectional relative value"
    if "ts_corr" in e and "volume" in e:
        return "volume-price correlation"
    if "ts_skewness" in e or "ts_kurtosis" in e:
        return "tail risk / distribution asymmetry"
    if "ts_mean(abs(" in e:
        return "volatility arbitrage"
    if "ts_std_dev" in e and "returns" in e:
        return "volatility risk premium"
    if "adv20" in e or "vwap" in e:
        return "liquidity microstructure"
    if "signed_power" in e or "sign(" in e:
        return "conditional / regime-aware signal"
    if "ts_delta" in e and "volume" in e:
        return "order flow imbalance"
    if "ts_decay_linear" in e:
        return "time-weighted mean reversion"
    if "ts_delta" in e and ("close" in e or "open" in e):
        return "short-term price reversal"
    if "ts_mean" in e and "returns" in e:
        return "momentum / trend following"
    return "generic price factor"


def score_expression(expr: str) -> Tuple[float, str, str]:
    """Score 0-100. Returns (score, reasons, hypothesis)."""
    if not expr or len(expr) < 10:
        return 0.0, "too_short", "none"

    score = 50.0
    reasons = []
    depth = _depth(expr)
    n_ops = _n_ops(expr)
    unique_lb = _unique_lookbacks(expr)

    # ── Complexity checks ───────────────────────────────────────
    if depth < 2:
        score -= 15
        reasons.append("too_simple")
    elif depth <= 4:
        score += 10
        reasons.append("good_depth")
    elif depth <= 6:
        score += 5
    else:
        score -= 10
        reasons.append("over_nested")

    if n_ops < 2:
        score -= 10
    elif 2 <= n_ops <= 5:
        score += 8
        reasons.append("good_complexity")
    elif n_ops > 8:
        score -= 5
        reasons.append("too_many_ops")

    if len(expr) < 30:
        score -= 10
    elif 40 <= len(expr) <= 200:
        score += 5
    elif len(expr) > 400:
        score -= 8

    if unique_lb > 5:
        score -= 8
        reasons.append("lookback_proliferation")

    # ── High-value bonuses ─────────────────────────────────────
    for marker, bonus in HIGH_VALUE.items():
        if marker in expr:
            score += bonus
            reasons.append(f"+{bonus}:{marker}")

    # Composite signal × filter
    if re.search(r'\)\s*\*\s*\(', expr):
        score += 15
        reasons.append("+15:composite")
    if re.search(r'\)\s*\*\s*\w*rank\(', expr):
        score += 12
        reasons.append("+12:signal_filter")
    if "group_neutralize" in expr and "rank(" in expr:
        score += 10
        reasons.append("+10:neutralized_rank")

    # Multi-field (price + volume)
    price_fields = {"close", "open", "high", "low", "vwap"}
    vol_fields = {"volume", "adv20", "adv60"}
    tokens = set(re.findall(r'\w+', expr.lower()))
    has_price = bool(price_fields & tokens)
    has_vol = bool(vol_fields & tokens)
    if has_price and has_vol:
        score += 8
        reasons.append("+8:multifield")

    # ── Penalty anti-patterns ───────────────────────────────────
    for pat, pen in LOW_VALUE_PATTERNS:
        if re.search(pat, expr, re.I):
            score += pen
            reasons.append(f"antipattern")

    # Trivially simple seed
    if re.match(r'^-?rank\(ts_\w+\(\w+,\s*\d+\)\)$', expr.strip()):
        score -= 25
        reasons.append("-25:basic_seed")

    score = max(0.0, min(100.0, score))
    hypo = infer_hypothesis(expr)
    return round(score, 1), "; ".join(reasons[:6]), hypo


def rank_candidates(exprs: List[str], min_score: float = 40.0) -> List[tuple]:
    """
    Score and sort expressions.
    Returns: list of (expression, score, hypothesis, reasons)
    """
    scored = [(e, *score_expression(e)) for e in exprs]
    filtered = [(e, s, h, r) for e, s, r, h in scored if s >= min_score]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered


# ── CLI quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    test_exprs = [
        "rank(ts_delta(close, 1))",
        "rank(group_neutralize(rank(ts_mean(returns, 20)), sector)",
        "rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
        "-group_neutralize(rank(ts_delta(close, 5)), sector)",
        "rank(ts_corr(vwap, volume, 10)) * rank(-ts_mean(abs(returns), 5))",
        "rank(ts_std_dev(returns, 20) / ts_mean(ts_std_dev(returns, 20), 60))",
        "group_rank(rank(ts_delta(close, 5)), industry) - rank(ts_mean(returns, 5))",
        "-rank(ts_mean(abs(returns) / (volume + 1), 20))",
        "rank(ts_skewness(returns, 20)) * rank(-ts_delta(volume, 1))",
        "rank(ts_corr(close, volume, 20)) - rank(ts_corr(ts_mean(close, 5), ts_mean(volume, 5), 20))",
    ]
    ranked = rank_candidates(test_exprs, min_score=0)
    for expr, score, hypo, reasons in ranked:
        bar = "█" * int(score / 5)
        print(f"[{score:5.1f}] {bar:<20} {hypo}")
        print(f"         {expr[:75]}")
        print(f"         ({reasons})")
        print()
