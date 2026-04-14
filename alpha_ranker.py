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
from functools import lru_cache
from typing import List, Tuple
import os
import joblib
import warnings

from alpha_policy import estimate_ic_stability, estimate_self_corr_risk

# Suppress sklearn warnings if model loaded
warnings.filterwarnings("ignore", category=UserWarning)

_XGB_MODEL = None
_MODEL_LOADED = False


def get_xgb_model():
    global _XGB_MODEL, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL_LOADED = True
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "models", "xgboost_ranker.pkl"
        )
        if os.path.exists(model_path):
            try:
                _XGB_MODEL = joblib.load(model_path)
            except Exception:
                pass
    return _XGB_MODEL


def extract_features(expr: str) -> dict:
    return {
        "length": len(expr),
        "num_ops": expr.count("("),
        "has_ts": 1 if "ts_" in expr else 0,
        "has_rank": 1 if "rank" in expr else 0,
        "has_group": 1 if "group_" in expr else 0,
        "has_volume": 1 if "volume" in expr else 0,
        "has_adv": 1 if "adv" in expr.lower() else 0,
        "has_returns": 1 if "returns" in expr else 0,
        "num_numbers": sum(c.isdigit() for c in expr),
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
    "ts_decay_linear": 15,  # Boosted for Fitness
    "ts_mean": 12,  # Added for Fitness smoothing
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
    "ts_av_diff": 12,  # Boosted for reversion signals
    "vector_neut": 18,  # Boosted: extremely powerful for neutralization
    "jump_decay": 10,
    "normalize": 8,
    "days_from_last_change": 12,
    "ts_entropy": 15,  # New: measures signal complexity/information depth
    "ts_step": 10,  # New: regime shift detection
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


@lru_cache(maxsize=20000)
def count_nesting_depth(expr: str) -> int:
    """Count maximum nesting depth of parentheses"""
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


@lru_cache(maxsize=20000)
def count_operators(expr: str) -> int:
    """Count number of operators used"""
    ops = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\s*\(", expr)
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
    return any(
        op in expr
        for op in ["group_neutralize", "group_rank", "group_mean", "group_zscore"]
    )


def has_time_comparison(expr: str) -> bool:
    """Check if expression compares across time (ratio or delta)"""
    return any(
        op in expr for op in ["ts_delay", "ts_delta", "ts_corr", "ts_regression"]
    )


# ============================================================
# Tier-1: Expression Complexity Scoring
# ============================================================

# Thresholds derived from WQ acceptance research:
# - >5 unique lookback windows → suspicious overfitting
# - >3 nesting levels deep → fragile
# - >8 operators → untrustworthy
MAX_UNIQUE_LOOKBACKS = 6
MAX_NESTING_DEPTH = 6
MAX_OPERATORS = 16


@lru_cache(maxsize=20000)
def count_unique_lookbacks(expr: str) -> int:
    """Count distinct numeric lookback constants in expression."""
    numbers = re.findall(r"\b\d+\b", expr)
    return len(set(numbers))


@lru_cache(maxsize=20000)
def expression_complexity_penalty(expr: str) -> float:
    """
    Return a penalty (positive float) to subtract from score.

    Research finding: simpler expressions with better OOS stability
    have higher acceptance rates on WQ Brain.
    """
    depth = count_nesting_depth(expr)
    n_ops = count_operators(expr)
    unique_lookbacks = count_unique_lookbacks(expr)

    penalty = 0.0

    # Nesting depth penalty
    if depth > 6:
        penalty += 15.0
    elif depth > MAX_NESTING_DEPTH:
        penalty += 5.0 * (depth - MAX_NESTING_DEPTH)

    # Operator count penalty
    if n_ops > 8:
        penalty += 10.0
    elif n_ops > MAX_OPERATORS:
        penalty += 5.0 * (n_ops - MAX_OPERATORS)

    # Lookback proliferation penalty
    if unique_lookbacks >= MAX_UNIQUE_LOOKBACKS:
        penalty += 8.0 * max(0, unique_lookbacks - MAX_UNIQUE_LOOKBACKS + 1)

    return penalty


def complexity_score(expression: str) -> dict:
    """
    Score expression simplicity / robustness.
    Returns dict with: score (0-1), depth, n_ops, unique_lookbacks,
    penalty, and flags.
    """
    depth = count_nesting_depth(expression)
    n_ops = count_operators(expression)
    unique_lookbacks = count_unique_lookbacks(expression)
    penalty = expression_complexity_penalty(expression)

    # Score: 1.0 = simple/robust, 0.0 = dangerously complex
    base_score = 1.0 - min(penalty / 30.0, 1.0)

    flags: list[str] = []
    if depth > 6:
        flags.append("over_nested")
    if n_ops > 8:
        flags.append("too_many_ops")
    if unique_lookbacks > MAX_UNIQUE_LOOKBACKS:
        flags.append("lookback_proliferation")

    return {
        "score": max(0.0, min(1.0, base_score)),
        "depth": depth,
        "n_ops": n_ops,
        "unique_lookbacks": unique_lookbacks,
        "penalty": penalty,
        "flags": flags,
    }


EXPRESSION_COMPLEXITY_FLOOR = float(os.getenv("ASYNC_COMPLEXITY_MIN", "0.55"))


def passes_complexity_check(
    expression: str, floor: float = EXPRESSION_COMPLEXITY_FLOOR
) -> bool:
    """Gate: reject overly complex expressions."""
    return complexity_score(expression)["score"] >= floor


@lru_cache(maxsize=20000)
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
            prob = xgb_model.predict_proba(X)[0][
                1
            ]  # Probability of getting Sharpe > 1.0
            if prob < 0.15:
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
    if re.search(r"\)\s*\*\s*\w*rank\(", expr):
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
    simple_seed_pattern = r"^-?rank\(ts_\w+\(\w+, \d+\)\)$"
    if re.match(simple_seed_pattern, expr.strip()):
        score -= 25
        reasons.append("-basic_seed")

    # === TIER-1 COMPLEXITY PENALTY ===
    complexity_penalty = expression_complexity_penalty(expr)
    if complexity_penalty > 0:
        score -= complexity_penalty
        reasons.append(f"-{complexity_penalty:.0f}:complexity")

    # === SELF-CORRELATION RISK PENALTY ===
    # Pre-simulation proxy using structural heuristics / learned weights.
    # Risk score in [0, 1]: penalise high-risk expressions before wasting WQ quota.
    self_corr_risk = estimate_self_corr_risk(expr)
    if self_corr_risk > 0.60:
        penalty = min(30.0, self_corr_risk * 50.0)
        score -= penalty
        reasons.append(f"-{penalty:.0f}:self_corr_risk({self_corr_risk:.2f})")
    elif self_corr_risk < 0.20:
        score += 5
        reasons.append("+5:low_self_corr_risk")

    # Clamp to 0-100
    score = max(0.0, min(100.0, score))
    reason_str = ", ".join(reasons[:6])  # top 6 reasons

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
        tokens = set(re.findall(r"[a-zA-Z_]+", expr.lower()))
        is_similar = False
        for prev_expr, _, _ in accepted:
            prev_tokens = set(re.findall(r"[a-zA-Z_]+", prev_expr.lower()))
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
# Meta-Ranker ML Model — expression-level scoring signals
# ============================================================


def regime_sensitivity_score(expression: str) -> float:
    """
    Score based on how well alpha adapts to market regimes.
    Returns 0-1 where 1 = regime-adaptive.

    Look for:
    - ts_skewness, ts_kurtosis (distribution awareness)
    - vol regime operators (ts_std_dev on returns)
    - trade_when (conditional logic)
    - if_else (regime branching)

    Penalize:
    - Pure momentum without any adaptation
    - Static lookback without conditional logic
    """
    expr = (expression or "").lower()
    if not expr:
        return 0.0

    score = 0.0

    # Adaptive regime operators — positive signals
    if any(op in expr for op in ["ts_skewness", "ts_kurtosis"]):
        score += 0.30
    if "ts_std_dev" in expr and "returns" in expr:
        score += 0.25  # vol-awareness
    if "trade_when" in expr:
        score += 0.25
    if "if_else" in expr:
        score += 0.20
    if "signed_power" in expr or "sign(" in expr:
        score += 0.10  # sign-awareness = mild regime sensitivity
    if "ts_zscore" in expr or "zscore(" in expr:
        score += 0.15  # normalization = mild regime robustness

    # Static momentum with no adaptation — penalize
    static_momentum = bool(
        re.search(r"^-?rank\(ts_mean\(returns,\s*\d+\)\)$", expr)
    ) or bool(re.search(r"^-?rank\(ts_delta\(close,\s*\d+\)\)$", expr))
    if static_momentum and score < 0.3:
        score = max(0.0, score - 0.25)

    # Penalize very short expressions that are pure momentum
    if len(expr) < 40 and "ts_std_dev" not in expr and "trade_when" not in expr:
        score = max(0.0, score - 0.15)

    return max(0.0, min(1.0, score))


def ic_decay_probability(
    expression: str,
    history_ic: list[float] | None = None,
) -> float:
    """
    Estimate probability that IC will decay (become worse over time).
    If history_ic provided: use real rolling IC data.
    Otherwise: use expression structure heuristics.

    Returns 0-1 where 1 = high decay risk (avoid).

    Penalize:
    - Long lookbacks (overfitted to history)
    - Mean-reversion without adaptation
    - High complexity
    """
    # Use real data if available
    if history_ic is not None and len(history_ic) >= 3:
        ic_series = list(history_ic)
        n = len(ic_series)
        # Rolling decay: compare recent half vs older half
        mid = n // 2
        older = ic_series[:mid]
        recent = ic_series[mid:]
        if older and recent:
            older_mean = sum(older) / len(older)
            recent_mean = sum(recent) / len(recent)
            if older_mean > 0 and recent_mean < older_mean * 0.5:
                return min(1.0, 0.7 + 0.1 * (1 - recent_mean / max(older_mean, 0.001)))
            elif recent_mean < older_mean:
                decay_rate = (older_mean - recent_mean) / max(older_mean, 0.001)
                return max(0.0, min(1.0, decay_rate * 0.8))
        return 0.2  # stable IC — low decay risk

    # Heuristic scoring from expression structure
    expr = (expression or "").lower()
    if not expr:
        return 0.5

    risk = 0.0

    # Long lookbacks → overfitting risk
    numbers = re.findall(r"\b\d+\b", expr)
    max_lookback = max((int(n) for n in numbers if int(n) > 0), default=0)
    if max_lookback > 252:
        risk += 0.30
    elif max_lookback > 120:
        risk += 0.20
    elif max_lookback > 60:
        risk += 0.10

    # Mean-reversion without adaptation signals
    if "ts_mean" in expr and "ts_std_dev" not in expr and "zscore" not in expr:
        risk += 0.15
    if "ts_decay_linear" in expr and "trade_when" not in expr and "if_else" not in expr:
        risk += 0.10

    # High complexity → overfitting
    depth = count_nesting_depth(expression)
    n_ops = count_operators(expression)
    if depth > 5:
        risk += 0.20
    elif depth > 3:
        risk += 0.10
    if n_ops > 7:
        risk += 0.15
    elif n_ops > 5:
        risk += 0.08

    # Long expression = more ways to overfit
    if len(expr) > 300:
        risk += 0.15
    elif len(expr) > 200:
        risk += 0.08

    return max(0.0, min(1.0, risk))


def cross_regime_score(expression: str) -> float:
    """
    Estimate how well alpha performs across different market regimes.
    Score based on expression structural features:
    - Diversification (multi-field)
    - Normalization (rank, scale, zscore)
    - Hedging (signed_power, sign)
    - Volatility awareness (ts_std_dev on returns)
    Returns 0-1 where 1 = strong cross-regime alpha.
    """
    expr = (expression or "").lower()
    if not expr:
        return 0.0

    score = 0.0

    # Diversification signals
    if has_multi_field(expr):
        score += 0.20
    if has_cross_sectional(expr):
        score += 0.20  # group ops span regimes well

    # Normalization = cross-regime robustness
    if "rank(" in expr:
        score += 0.12
    if "ts_zscore" in expr or "zscore(" in expr:
        score += 0.10
    if "scale(" in expr or "signed_power" in expr:
        score += 0.08

    # Volatility awareness — key for cross-regime
    if "ts_std_dev" in expr and "returns" in expr:
        score += 0.20
    if "ts_mean(abs(" in expr or "ts_mean(abs(returns" in expr:
        score += 0.15  # abs-returns = vol proxy

    # Regime conditioning
    if "trade_when" in expr or "if_else" in expr:
        score += 0.15

    # Quality filter — masks regime effects
    if "winsorize" in expr or "densify" in expr:
        score += 0.10

    return max(0.0, min(1.0, score))


def expression_simplicity_score(expression: str) -> float:
    """
    Penalize overly complex expressions.
    Returns 0-1 where 1 = simple/preferred.

    Penalties:
    - >5 unique lookback values → penalty
    - >3 nesting depth → penalty
    - >8 operators → penalty
    """
    if not expression:
        return 0.0

    unique_lookbacks = count_unique_lookbacks(expression)
    depth = count_nesting_depth(expression)
    n_ops = count_operators(expression)

    score = 1.0

    # Lookback proliferation penalty
    if unique_lookbacks > 5:
        score -= 0.30
    elif unique_lookbacks > 3:
        score -= 0.15

    # Nesting depth penalty
    if depth > 6:
        score -= 0.35
    elif depth > 4:
        score -= 0.20
    elif depth > 3:
        score -= 0.10

    # Operator count penalty
    if n_ops > 10:
        score -= 0.30
    elif n_ops > 8:
        score -= 0.20
    elif n_ops > 6:
        score -= 0.10

    return max(0.0, min(1.0, score))


def turnover_prediction(expression: str) -> float:
    """
    Predict annual turnover % from expression structure.
    Returns estimated annual turnover (0.0 to 2.0+ = 200%).

    High-frequency signals:
    - Short delays (ts_delta with small values like 1-3)
    - ts_delta on small windows
    - Large weights on short-term volume
    - Large weights on raw close without smoothing
    - No rank() wrapping (raw signals change daily)
    """
    expr = (expression or "").lower()
    if not expr:
        return 0.5

    # Base turnover estimate
    base = 1.0  # 100% annual turnover baseline

    # Short ts_delta lookbacks → high turnover
    short_delta_matches = re.findall(r"ts_delta\([^)]*,\s*(\d+)\s*\)", expr)
    for raw in short_delta_matches:
        val = int(raw)
        if val <= 1:
            base += 0.30
        elif val <= 3:
            base += 0.15
        elif val <= 5:
            base += 0.05

    # Short ts_mean lookbacks → smoothing reduces turnover
    short_mean_matches = re.findall(r"ts_mean\([^)]*,\s*(\d+)\s*\)", expr)
    for raw in short_mean_matches:
        val = int(raw)
        if val <= 5:
            base -= 0.10  # still some smoothing
        elif val >= 40:
            base -= 0.15  # strong smoothing

    # ts_decay_linear — reduces turnover
    if "ts_decay_linear" in expr:
        decay_matches = re.findall(r"ts_decay_linear\([^)]*,\s*(\d+)\s*\)", expr)
        for raw in decay_matches:
            val = int(raw)
            if val >= 10:
                base -= 0.20
            elif val >= 5:
                base -= 0.10

    # rank() wrapping — dramatically reduces turnover
    rank_count = expr.count("rank(")
    if rank_count >= 2:
        base -= 0.25
    elif rank_count == 1:
        base -= 0.10

    # ts_delay — can increase or decrease depending on direction
    delay_matches = re.findall(r"ts_delay\([^)]*,\s*(-?\d+)\s*\)", expr)
    for raw in delay_matches:
        val = int(raw)
        if val < 0:
            base += 0.20  # negative delay = future peek → high turnover signal
        elif val >= 20:
            base -= 0.10

    # sign/signed_power — momentum signal, medium turnover
    if "signed_power" in expr or expr.count("sign(") > 0:
        base += 0.05

    # No smoothing at all → high turnover
    if "ts_mean" not in expr and "ts_decay" not in expr and "rank(" not in expr:
        base += 0.15

    return max(0.0, base)


def score_with_meta_model(candidate: str) -> dict:
    """
    Composite meta-model scoring for an alpha candidate.

    Returns:
        {
            'expected_sharpe': float,         # 0-1 proxy from structural signals
            'decay_probability': float,       # 0-1 IC decay risk
            'cross_regime_score': float,      # 0-1 cross-regime robustness
            'ic_stability_score': float,      # 0-1 IC stability
            'simplicity_score': float,        # 0-1 structural simplicity
            'turnover_prediction': float,      # estimated annual turnover
            'composite_score': float,         # weighted combination
        }
    """
    if not candidate:
        return {
            "expected_sharpe": 0.0,
            "decay_probability": 1.0,
            "cross_regime_score": 0.0,
            "ic_stability_score": 0.0,
            "simplicity_score": 0.0,
            "turnover_prediction": 0.0,
            "composite_score": 0.0,
        }

    expr = candidate
    score, _ = score_expression(expr)

    # Normalize to 0-1
    expected_sharpe = max(0.0, min(1.0, score / 100.0))

    decay_prob = ic_decay_probability(expr)
    cross_regime = cross_regime_score(expr)
    ic_stability = estimate_ic_stability(
        sharpe=expected_sharpe * 3.0,  # approximate
        fitness=expected_sharpe * 2.0,
        turnover=turnover_prediction(expr) * 50.0,  # approximate to real units
        sub_sharpe=-1.0,
    )
    simplicity = expression_simplicity_score(expr)
    turnover = turnover_prediction(expr)

    # Composite score — weighted blend emphasising non-decay and stability
    # Higher = better
    composite = (
        0.25 * expected_sharpe
        + 0.20 * (1.0 - decay_prob)  # penalise decay risk
        + 0.20 * cross_regime
        + 0.15 * ic_stability
        + 0.10 * simplicity
        + 0.10 * (1.0 - min(turnover / 2.0, 1.0))  # penalise high turnover
    )

    return {
        "expected_sharpe": round(expected_sharpe, 4),
        "decay_probability": round(decay_prob, 4),
        "cross_regime_score": round(cross_regime, 4),
        "ic_stability_score": round(ic_stability, 4),
        "simplicity_score": round(simplicity, 4),
        "turnover_prediction": round(turnover, 4),
        "composite_score": round(composite, 4),
    }


# ============================================================
# Pre-Simulation Gate Estimator  (Fix gate=0% failure rate)
# ============================================================

# Gate thresholds — must sync with alpha_policy.py HIGH_THROUGHPUT_THRESHOLDS
_GATE_MIN_SHARPE = 1.25
_GATE_MIN_FITNESS = 1.00
_GATE_MIN_ROBUST_SCORE = 1.35
_GATE_MIN_TURNOVER = 1.0
_GATE_MAX_TURNOVER = 70.0


def estimate_gate_probability(expr: str) -> dict:
    """
    Pre-simulation gate probability estimator.
    Predicts P(pass_gate) before wasting WQ Brain quota.

    Returns:
        {
            "passed": bool,          # True if should proceed to WQ simulation
            "probability": float,    # 0-1 gate pass probability
            "expected_sharpe": float, # estimated Sharpe proxy
            "expected_fitness": float,
            "expected_turnover": float,
            "gate_score": float,      # composite gate proxy score
            "reason": str,            # rejection or acceptance reason
        }

    Key insight: rank_score/100 → expected_sharpe is too pessimistic because
    rank_score measures STRUCTURE not PERFORMANCE. We blend multiple signals.
    """
    if not expr or len(expr) < 10:
        return {
            "passed": False,
            "probability": 0.0,
            "expected_sharpe": 0.0,
            "expected_fitness": 0.0,
            "expected_turnover": 0.0,
            "gate_score": 0.0,
            "reason": "too_short",
        }

    # --- Signal 1: ranker score (structural quality) ---
    rank_score, _ = score_expression(expr)
    rank_sharpe = rank_score / 100.0 * 2.5  # scale to expected Sharpe range

    # --- Signal 2: meta-model expected Sharpe ---
    meta = score_with_meta_model(expr)
    meta_sharpe = meta["expected_sharpe"] * 2.5

    # --- Signal 3: XGBoost model (if available) ---
    xgb = get_xgb_model()
    xgb_sharpe = 0.0
    if xgb:
        import pandas as pd

        feats = extract_features(expr)
        X = pd.DataFrame([feats])
        try:
            prob = xgb.predict_proba(X)[0][1]  # P(Sharpe > 1.0)
            xgb_sharpe = prob * 2.0  # convert to expected Sharpe
        except Exception:
            pass

    # --- Signal 4: self-corr risk (low = better) ---
    self_corr_risk = estimate_self_corr_risk(expr)
    scorr_bonus = (1.0 - self_corr_risk) * 0.3  # up to +0.3 for low risk

    # --- Weighted blend of signals ---
    n_signals = 1.0
    if xgb:
        n_signals += 1.0
        blended_sharpe = rank_sharpe * 0.30 + meta_sharpe * 0.30 + xgb_sharpe * 0.40
    else:
        blended_sharpe = rank_sharpe * 0.40 + meta_sharpe * 0.60

    blended_sharpe += scorr_bonus
    blended_sharpe = max(0.0, blended_sharpe)

    # --- Expected fitness (correlates with Sharpe but lower) ---
    expected_fitness = blended_sharpe * 0.75

    # --- Expected turnover (from meta model or heuristic) ---
    turnover_pred = meta.get("turnover_prediction", 0.0) * 50.0  # scale to %
    if turnover_pred == 0.0:
        # Fallback heuristic: simple expressions → lower turnover
        n_ops = count_operators(expr)
        turnover_pred = min(30.0 + n_ops * 4.0, 65.0)

    # --- Composite gate score (proxy for robust_quality_score) ---
    # Approximates: 0.52*sharpe + 0.33*fitness + 0.25*checks - penalties
    gate_score = (0.52 * blended_sharpe) + (0.33 * expected_fitness)
    if turnover_pred > 55.0:
        gate_score -= min(0.5, (turnover_pred - 55.0) / 60.0)
    if turnover_pred < 1.5:
        gate_score -= 0.2
    if self_corr_risk > 0.60:
        gate_score -= 0.25
    gate_score = max(0.0, gate_score)

    # --- Hard filter thresholds ---
    hard_fail = False
    reasons = []

    if blended_sharpe < 0.6:
        hard_fail = True
        reasons.append(f"sharpe_too_low({blended_sharpe:.2f}<0.6)")
    elif blended_sharpe < 1.0:
        reasons.append(f"sharpe_marginal({blended_sharpe:.2f})")

    if expected_fitness < 0.6:
        hard_fail = True
        reasons.append(f"fitness_too_low({expected_fitness:.2f}<0.6)")

    if gate_score < 0.8:
        hard_fail = True
        reasons.append(f"gate_score_low({gate_score:.2f}<0.8)")

    if not (1.0 < turnover_pred < 70.0):
        hard_fail = True
        reasons.append(f"turnover_out_of_range({turnover_pred:.1f}%)")

    # Known quality markers → bonus (if present, relax hard_fail)
    quality_markers = ["group_neutralize", "group_rank", "ts_corr", "ts_regression"]
    marker_bonus = sum(0.1 for m in quality_markers if m in expr)
    if hard_fail and marker_bonus >= 0.3:
        hard_fail = False
        reasons.append("quality_markers_override")

    probability = min(blended_sharpe / _GATE_MIN_SHARPE, 1.0)
    probability = max(0.0, min(1.0, probability))

    passed = not hard_fail and gate_score >= 0.8
    reason = "; ".join(reasons) if reasons else "gate_estimate_pass"

    return {
        "passed": passed,
        "probability": round(probability, 3),
        "expected_sharpe": round(blended_sharpe, 3),
        "expected_fitness": round(expected_fitness, 3),
        "expected_turnover": round(turnover_pred, 1),
        "gate_score": round(gate_score, 3),
        "reason": reason,
    }


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
