"""
alpha_ranker.py — Pre-Simulation Alpha Ranker
Score alpha candidates BEFORE sending to WQ Brain.
Saves quota by prioritizing high-quality expressions.

Scoring logic based on:
- Expression complexity (not too simple, not too nested)
- Presence of proven factors (microstructure, quality, regime markers)
- Anti-patterns (pure noise indicators)
- Structural diversity signals
"""

import re
from typing import List, Tuple
import os
import joblib
import warnings

# Suppress sklearn warnings if model loaded
warnings.filterwarnings('ignore', category=UserWarning)

_XGB_MODEL = None
_MODEL_LOADED = False

def get_xgb_model():
    global _XGB_MODEL, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL_LOADED = True
        model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'xgboost_ranker.pkl')
        if os.path.exists(model_path):
            try:
                _XGB_MODEL = joblib.load(model_path)
            except Exception:
                pass
    return _XGB_MODEL

def extract_features(expr: str) -> dict:
    return {
        'length': len(expr),
        'num_ops': expr.count('('),
        'has_ts': 1 if 'ts_' in expr else 0,
        'has_rank': 1 if 'rank' in expr else 0,
        'has_group': 1 if 'group_' in expr else 0,
        'has_volume': 1 if 'volume' in expr else 0,
        'has_adv': 1 if 'adv' in expr.lower() else 0,
        'has_returns': 1 if 'returns' in expr else 0,
        'num_numbers': sum(c.isdigit() for c in expr)
    }



# ============================================================
# Bonus signals (known to work on WQ Brain)
# ============================================================
HIGH_VALUE_MARKERS = {
    # Cross-sectional / group ops: add relative value → high edge
    "group_neutralize": 20,
    "group_rank": 18,
    "group_mean": 15,
    "group_zscore": 15,

    # Microstructure signals
    "adv20": 12,
    "adv60": 8,
    "vwap": 10,

    # Time-series operators that capture non-linear patterns
    "ts_corr": 10,
    "ts_covariance": 9,
    "ts_regression": 12,
    "ts_skewness": 10,
    "ts_decay_linear": 8,
    "ts_product": 8,

    # Quality factors: sharpe-adjusted, std normalized
    "ts_std_dev": 6,

    # Volume signals
    "volume": 5,

    # Returns (clean signal field)
    "returns": 6,

    # Signed/conditional operators
    "signed_power": 10,
    "sign": 7,
    "scale": 6,

    # Rare operators (low competition = high edge)
    "trade_when": 18,
    "winsorize": 15,
    "ts_quantile": 14,
    "bucket": 12,
    "hump": 12,
    "ts_scale": 10,
    "ts_av_diff": 10,
    "vector_neut": 18,  # Boosted: extremely powerful for neutralization
    "jump_decay": 10,
    "normalize": 8,
    "days_from_last_change": 12,
    "ts_entropy": 15,   # New: measures signal complexity/information depth
    "ts_step": 10,      # New: regime shift detection
}

# ============================================================
# Penalty signals (known weak patterns)
# ============================================================
LOW_VALUE_PENALTIES = {
    # Trivial single-field rank — too simple
    r"^rank\(\w+\)$": -20,
    r"^-rank\(\w+\)$": -20,

    # Direct price ratio without any transform
    r"rank\(close/open\)": -5,
    r"rank\(open/close\)": -5,

    # Nested ranks without meaningful transformation (rank(rank(...)))
    r"rank\(rank\(": -15,

    # Over-nested expressions (>6 levels deep)
    # Handled separately by depth check

    # Very short lookbacks without combo (noise-like)
    r"ts_delta\([a-z]+, [12]\)": -8,

    # Pure trend-following without volume confirmation (commodity alpha, WQ penalizes)
    r"^rank\(ts_mean\(returns, \d+\)\)$": -10,
    r"^-rank\(ts_mean\(returns, \d+\)\)$": -10,
}


def count_nesting_depth(expr: str) -> int:
    """Count maximum nesting depth of parentheses"""
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    return max_depth


def count_operators(expr: str) -> int:
    """Count number of operators used"""
    ops = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', expr)
    return len(ops)


def has_multi_field(expr: str) -> bool:
    """Check if expression combines multiple data types"""
    price_fields = {"open", "high", "low", "close", "vwap"}
    volume_fields = {"volume", "adv20", "adv60"}
    return_fields = {"returns"}

    categories = 0
    if any(f in expr for f in price_fields):
        categories += 1
    if any(f in expr for f in volume_fields):
        categories += 1
    if any(f in expr for f in return_fields):
        categories += 1
    return categories >= 2


def has_cross_sectional(expr: str) -> bool:
    """Check if expression uses cross-sectional operators"""
    return any(op in expr for op in ["group_neutralize", "group_rank", "group_mean", "group_zscore"])


def has_time_comparison(expr: str) -> bool:
    """Check if expression compares across time (ratio or delta)"""
    return any(op in expr for op in ["ts_delay", "ts_delta", "ts_corr", "ts_regression"])


def score_expression(expr: str) -> Tuple[float, str]:
    """
    Score an alpha expression from 0-100.
    Higher = better chance of passing WQ Brain checks.
    
    Returns: (score, reason)
    """
    if not expr or len(expr) < 10:
        return 0.0, "Too short"

    score = 50.0  # Base score
    reasons = []

    # === COMPLEXITY SWEET SPOT ===
    depth = count_nesting_depth(expr)
    n_ops = count_operators(expr)
    expr_len = len(expr)

    # Complexity: not too simple (score 0-20 → low), not too nested (>6 → likely invalid)
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

    # ML Surrogate Model Pass (Targeted Penalty)
    xgb_model = get_xgb_model()
    ml_penalty = 0
    if xgb_model:
        import pandas as pd
        feats = extract_features(expr)
        X = pd.DataFrame([feats])
        try:
            prob = xgb_model.predict_proba(X)[0][1] # Probability of getting Sharpe > 1.0
            if prob < 0.3:
                ml_penalty = -50
                reasons.append(f"ML_REJECT:prob={prob:.2f}")
            elif prob > 0.7:
                score += 20
                reasons.append(f"ML_BOOST:prob={prob:.2f}")
        except Exception:
            pass
    score += ml_penalty

    # Number of operators: 2-5 is ideal
    if n_ops < 2:
        score -= 10
    elif 2 <= n_ops <= 5:
        score += 8
        reasons.append("good_complexity")
    elif n_ops > 8:
        score -= 5

    # Expression length sweet spot: 40-200 chars
    if expr_len < 30:
        score -= 10
    elif 40 <= expr_len <= 200:
        score += 5
    elif expr_len > 400:
        score -= 8

    # === HIGH VALUE BONUSES ===
    for marker, bonus in HIGH_VALUE_MARKERS.items():
        if marker in expr:
            score += bonus
            reasons.append(f"+{bonus}:{marker}")

    # Extra bonus for multi-factor combos
    if has_multi_field(expr):
        score += 15
        reasons.append("+multi_field")

    if has_cross_sectional(expr):
        score += 12
        reasons.append("+cross_sectional")

    if has_time_comparison(expr):
        score += 10
        reasons.append("+time_comparison")

    # Bonus for compositions: A * B pattern (signal × filter)
    if re.search(r'\)\s*\*\s*\w*rank\(', expr):
        score += 15
        reasons.append("+composite_signal")

    # Bonus for sign/signed_power (conditional logic)
    if "sign(" in expr:
        score += 7

    # === PENALTY ANTI-PATTERNS ===
    for pattern, penalty in LOW_VALUE_PENALTIES.items():
        if re.search(pattern, expr, re.IGNORECASE):
            score += penalty  # penalty is negative
            reasons.append(f"{penalty}:antipattern")

    # Penalty: trivially simple (only one operator)
    if n_ops == 1 and depth == 1:
        score -= 20
        reasons.append("-trivial")

    # Penalty: looks like brute-force mutation (just changing numbers)
    # e.g.: rank(ts_delta(close, 7)) — basic seed without composition
    simple_seed_pattern = r'^-?rank\(ts_\w+\(\w+, \d+\)\)$'
    if re.match(simple_seed_pattern, expr.strip()):
        score -= 25
        reasons.append("-basic_seed")

    # Clamp to 0-100
    score = max(0.0, min(100.0, score))
    reason_str = ", ".join(reasons[:5])  # top 5 reasons

    return round(score, 1), reason_str


def rank_candidates(
    expressions: List[str],
    top_n: int = None,
    min_score: float = 30.0,
) -> List[Tuple[str, float, str]]:
    """
    Score and rank alpha candidates before simulation.
    
    Args:
        expressions: List of alpha expressions
        top_n: If set, return only top N candidates
        min_score: Minimum score threshold (filter out weak ones)
    
    Returns:
        List of (expression, score, reason) sorted by score descending
    """
    scored = []
    for expr in expressions:
        score, reason = score_expression(expr)
        if score >= min_score:
            scored.append((expr, score, reason))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    if top_n:
        scored = scored[:top_n]

    return scored


def apply_family_crowding_penalty(
    scored: List[Tuple[str, float, str]],
    family_pass_counts: dict = None,
    max_per_family: int = 3,
    penalty_per_extra: float = 15.0,
) -> List[Tuple[str, float, str]]:
    """
    Penalize candidates from families that already have passes in DB.
    This forces diversity: if Family_X already has 3 passes, new Family_X
    candidates get penalized to make room for novel ideas.
    """
    if not family_pass_counts:
        return scored

    import hashlib
    adjusted = []
    for expr, score, reason in scored:
        family = hashlib.md5(expr.strip().lower()[:60].encode()).hexdigest()[:12]
        existing = family_pass_counts.get(family, 0)
        if existing >= max_per_family:
            crowding_penalty = penalty_per_extra * (existing - max_per_family + 1)
            score = max(0, score - crowding_penalty)
            reason += f", -{crowding_penalty:.0f}:family_crowding({existing})"
        adjusted.append((expr, score, reason))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


def compute_similarity_penalty(
    scored: List[Tuple[str, float, str]],
    penalty_threshold: float = 0.80,
    penalty_amount: float = 20.0,
) -> List[Tuple[str, float, str]]:
    """
    Penalize expressions that are too similar to higher-ranked ones.
    Uses token overlap ratio as a fast similarity proxy.
    """
    accepted = []
    for expr, score, reason in scored:
        tokens = set(re.findall(r'[a-zA-Z_]+', expr.lower()))
        is_similar = False
        for prev_expr, _, _ in accepted:
            prev_tokens = set(re.findall(r'[a-zA-Z_]+', prev_expr.lower()))
            if not tokens or not prev_tokens:
                continue
            overlap = len(tokens & prev_tokens) / max(len(tokens | prev_tokens), 1)
            if overlap >= penalty_threshold:
                score = max(0, score - penalty_amount)
                reason += f", -{penalty_amount:.0f}:similar({overlap:.0%})"
                is_similar = True
                break
        accepted.append((expr, score, reason))

    accepted.sort(key=lambda x: x[1], reverse=True)
    return accepted


def filter_and_rank(
    expressions: List[str],
    top_n: int = 30,
    min_score: float = 35.0,
    family_pass_counts: dict = None,
) -> List[str]:
    """
    Filter and rank expressions with family crowding + similarity penalties.
    Primary interface for pipeline integration.
    """
    ranked = rank_candidates(expressions, top_n=None, min_score=min_score)

    # Tier 2: family crowding penalty
    ranked = apply_family_crowding_penalty(ranked, family_pass_counts)

    # Tier 2: similarity penalty (dedup within batch)
    ranked = compute_similarity_penalty(ranked)

    # Final cut
    ranked = ranked[:top_n] if top_n else ranked
    filtered = [expr for expr, score, reason in ranked]

    n_filtered = len(expressions) - len(ranked)
    if n_filtered > 0:
        import logging
        logger = logging.getLogger(__name__)
        if ranked:
            top_score = ranked[0][1]
            avg_score = sum(s for _, s, _ in ranked) / len(ranked)
            logger.info(
                f"🎯 Pre-rank: {len(ranked)}/{len(expressions)} pass "
                f"(filtered {n_filtered}) | top={top_score:.0f} avg={avg_score:.0f}"
            )

    return filtered


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    test_exprs = [
        # Simple / expected low score
        "rank(close)",
        "rank(ts_delta(close, 5))",
        "rank(ts_mean(returns, 20))",

        # Medium complexity
        "-rank(ts_corr(open, volume, 10))",
        "rank(volume / ts_mean(volume, 20)) * rank(returns)",
        "rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",

        # High value — group ops + multi-field
        "group_neutralize(rank(ts_mean(returns, 20)), sector)",
        "rank(ts_delta(close, 5)) - group_mean(rank(ts_delta(close, 5)), 1, industry)",
        "group_neutralize(rank(volume / adv20), sector)",

        # Composite signal × filter
        "rank(ts_std_dev(returns, 5)) * (-rank(ts_delta(close, 5)))",
        "rank(ts_sum(returns * volume, 10) / ts_sum(abs(returns) * volume, 10) + 0.001)",
        "(-rank(ts_std_dev(returns, 5))) * rank(ts_mean(returns, 20))",

        # Microstructure
        "-rank(ts_mean(abs(returns) / (volume + 1), 20))",
        "rank(ts_corr(returns, ts_delta(volume, 1), 10)) * rank(ts_mean(returns, 20))",
    ]

    print("=== Alpha Pre-Simulation Ranker ===\n")
    ranked = rank_candidates(test_exprs, min_score=0)
    for expr, score, reason in ranked:
        bar = "█" * int(score / 5)
        print(f"  [{score:5.1f}] {bar:<20} {expr[:70]}")
        print(f"          Reason: {reason}")
        print()

    print(f"\nTop 5 after filter (min_score=40):")
    top5 = filter_and_rank(test_exprs, top_n=5, min_score=40)
    for i, e in enumerate(top5, 1):
        print(f"  {i}. {e}")
