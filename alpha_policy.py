"""
alpha_policy.py - Shared quality policy and lightweight critic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class QualityThresholds:
    sharpe: float
    fitness: float
    turnover_min: float
    turnover_max: float
    require_all_checks: bool = True


# High-throughput profile per user choice:
# allow lower sharpe than strict mode but still require robust checks.
HIGH_THROUGHPUT_THRESHOLDS = QualityThresholds(
    sharpe=1.25,
    fitness=1.0,
    turnover_min=1.0,
    turnover_max=70.0,
    require_all_checks=True,
)


def classify_quality_tier(sharpe: float, fitness: float) -> str:
    if sharpe >= 2.5 and fitness >= 2.0:
        return "elite"
    if sharpe >= 2.0 and fitness >= 1.5:
        return "excellent"
    if sharpe >= 1.5 and fitness >= 1.2:
        return "good"
    if sharpe >= 1.25 and fitness >= 1.0:
        return "minimum"
    return "reject"


def infer_strategy_cluster(theme: str, mutation_type: str) -> str:
    t = (theme or "").lower()
    m = (mutation_type or "").lower()
    if "rag" in t or "llm" in t or "llm" in m:
        return "llm"
    if "evolve" in t or "evolve" in m or "crossover" in m:
        return "evolved"
    if "harvest" in t or "community" in t:
        return "harvested"
    if "level5" in t or "rareop" in m:
        return "rareop"
    if "seed" in m:
        return "seeded"
    return "deterministic"


def build_risk_flags(expression: str, turnover: float, error: str) -> str:
    flags: list[str] = []
    expr = (expression or "").lower()
    if error:
        flags.append("runtime_error")
    if turnover >= 60:
        flags.append("high_turnover")
    if "trade_when(" in expr:
        flags.append("event_conditioned")
    if "hump(" in expr:
        flags.append("hump_wrapped")
    if "group_neutralize(" not in expr:
        flags.append("missing_group_neutralize")
    return ",".join(flags)


def passes_quality_gate(result, thresholds: QualityThresholds = HIGH_THROUGHPUT_THRESHOLDS) -> bool:
    if getattr(result, "error", ""):
        return False
    env_require_all = os.getenv("ASYNC_REQUIRE_ALL_CHECKS")
    require_all_checks = thresholds.require_all_checks
    if env_require_all not in (None, ""):
        require_all_checks = env_require_all.strip().lower() in ("1", "true", "yes")

    all_passed = bool(getattr(result, "all_passed", False))
    total_checks = int(getattr(result, "total_checks", 0) or 0)
    passed_checks = int(getattr(result, "passed_checks", 0) or 0)
    min_checks_ratio = float(os.getenv("ASYNC_MIN_CHECKS_RATIO", "1.0" if require_all_checks else "0.60"))
    min_checks_ratio = max(0.0, min(1.0, min_checks_ratio))

    if require_all_checks:
        if not all_passed:
            return False
    elif total_checks > 0:
        if (passed_checks / max(1, total_checks)) < min_checks_ratio:
            return False

    sharpe = float(getattr(result, "sharpe", 0.0) or 0.0)
    fitness = float(getattr(result, "fitness", 0.0) or 0.0)
    turnover = float(getattr(result, "turnover", 0.0) or 0.0)
    min_sharpe = float(os.getenv("ASYNC_MIN_SHARPE", str(thresholds.sharpe)))
    min_fitness = float(os.getenv("ASYNC_MIN_FITNESS", str(thresholds.fitness)))
    min_turnover = float(os.getenv("ASYNC_TURNOVER_MIN", str(thresholds.turnover_min)))
    max_turnover = float(os.getenv("ASYNC_TURNOVER_MAX", str(thresholds.turnover_max)))
    return (
        sharpe >= min_sharpe
        and fitness >= min_fitness
        and min_turnover < turnover < max_turnover
    )


def critic_score(expression: str) -> float:
    """
    Fast deterministic critic score in [0, 1].
    Rewards structural richness and basic diversification patterns.
    """
    expr = (expression or "").lower()
    if not expr:
        return 0.0
    score = 0.0
    score += 0.18 if "group_neutralize(" in expr else 0.0
    score += 0.16 if "ts_zscore(" in expr or "zscore(" in expr else 0.0
    score += 0.14 if "ts_decay_linear(" in expr else 0.0
    score += 0.14 if "ts_corr(" in expr or "ts_covariance(" in expr else 0.0
    score += 0.14 if "rank(" not in expr[:16] else 0.0
    score += 0.12 if "trade_when(" in expr or "hump(" in expr else 0.0
    score += 0.12 if len(expr) > 60 else 0.0
    # Bonus for advanced BRAIN-style robustness patterns.
    score += 0.08 if "regression_neut(" in expr else 0.0
    score += 0.06 if "pasteurize(" in expr else 0.0
    score += 0.06 if "densify(" in expr or "bucket(" in expr else 0.0
    return min(score, 1.0)


def should_simulate_candidate(expression: str, min_critic_score: float = 0.28) -> bool:
    return critic_score(expression) >= min_critic_score


def robust_quality_score(result) -> float:
    """
    Composite quality score to reduce single-metric overfitting.
    Higher is better.
    """
    sharpe = float(getattr(result, "sharpe", 0.0) or 0.0)
    fitness = float(getattr(result, "fitness", 0.0) or 0.0)
    turnover = float(getattr(result, "turnover", 0.0) or 0.0)
    total_checks = int(getattr(result, "total_checks", 0) or 0)
    passed_checks = int(getattr(result, "passed_checks", 0) or 0)
    checks_ok = float(passed_checks) / max(1, total_checks) if total_checks > 0 else 0.0
    raw_sub = getattr(result, "sub_sharpe", None)
    sub_sharpe = float(raw_sub) if raw_sub is not None else None

    turnover_penalty = 0.0
    if turnover > 55.0:
        turnover_penalty += min(0.5, (turnover - 55.0) / 60.0)
    if turnover < 1.5:
        turnover_penalty += 0.2

    sub_penalty = 0.0
    if sub_sharpe is not None:
        if sub_sharpe > -0.99 and sub_sharpe < 0.0:
            sub_penalty = 0.20   # mild penalty for negative sub
        # Positive sub-sharpe is a strong acceptance signal — reward it
        elif sub_sharpe >= 0.0:
            sub_penalty = -min(0.20, sub_sharpe * 0.15)

    # Weighted linear blend with practical penalties.
    raw = (0.52 * sharpe) + (0.33 * fitness) + (0.25 * checks_ok)
    return raw - turnover_penalty - sub_penalty


def passes_quality_gate_v2(result, thresholds: QualityThresholds = HIGH_THROUGHPUT_THRESHOLDS) -> bool:
    """
    Stricter, overfit-aware gate:
    - preserve hard constraints
    - require composite quality above floor
    - penalize weak sub-universe robustness
    """
    if not passes_quality_gate(result, thresholds):
        return False
    if getattr(result, "error", ""):
        return False
    raw_sub = getattr(result, "sub_sharpe", None)
    if raw_sub is not None:
        sub_sharpe = float(raw_sub)
        # Known negative sub-universe sharpe is a strong rejection signal.
        if sub_sharpe > -0.99 and sub_sharpe < 0.0:
            return False
    score = robust_quality_score(result)
    # Configurable floor to trade off strictness vs throughput.
    floor = float(os.getenv("ASYNC_ROBUST_SCORE_MIN", "1.35"))
    return score >= floor


def estimate_competition_priority(result) -> float:
    """
    Priority proxy for IQC-style ranking:
    IS ~= D1 + D0/3, so D1 gets 3x weight vs D0.
    """
    score = robust_quality_score(result)
    delay = int(getattr(result, "delay", 1) or 1)
    return score if delay == 1 else (score / 3.0)


def compute_llm_budget_ratio(
    baseline_ratio: float,
    llm_error_rate: float,
    submit_fail_rate: float,
    has_api_key: bool,
) -> float:
    if not has_api_key:
        return 0.0
    penalty = max(llm_error_rate, submit_fail_rate)
    if penalty >= 0.60:
        return min(baseline_ratio, 0.05)
    if penalty >= 0.35:
        return min(baseline_ratio, 0.10)
    return baseline_ratio


def novelty_ratio(expression: str, reference_tokens: Iterable[str]) -> float:
    expr_tokens = {tok for tok in expression.lower().replace("(", " ").replace(")", " ").split() if tok}
    ref = set(reference_tokens)
    if not expr_tokens:
        return 0.0
    if not ref:
        return 1.0
    overlap = len(expr_tokens & ref)
    return 1.0 - (overlap / len(expr_tokens))


# ============================================================
# Tier-1: Bias Detection  (WQ automatic disqualifiers)
# ============================================================

def detect_survivorship_bias(expression: str) -> bool:
    """
    Detect likely survivorship bias — uses fields that implicitly assume
    currently-traded securities only (no delisted/microcap handling).

    WQ simulates on full universe including delisted/microcap securities.
    Expressions that divide by volume without guard or use raw close ratios
    without rank() are vulnerable.
    """
    expr = expression.lower()
    dangerous_patterns = [
        r'\bclose\b(?!\s*/)',          # raw close without denominator
        r'\bvolume\b(?!\s*(/|/))',     # raw volume without ratio
        r'\bcap\b(?!/)',               # raw market cap
        r'\breturns\b(?!.*(?:rank|winsorize|group_neutralize))',  # raw returns unguarded
    ]
    has_danger = any(re.search(p, expr) for p in dangerous_patterns)
    has_rank = 'rank(' in expr
    has_winsorize = 'winsorize(' in expr
    has_group_neutralize = 'group_neutralize(' in expr
    has_if_else = 'if_else(' in expr  # conditional logic suggests null handling
    # Low-risk if at least one defensive pattern present
    safe = has_rank or has_winsorize or has_group_neutralize or has_if_else
    return has_danger and not safe


def detect_lookahead_bias(expression: str) -> bool:
    """
    Detect likely look-ahead bias — future information leaking into signals.

    Common patterns:
    - ts_delay with negative delay (future leak)
    - References to 'universe' or 'index' without proper lag
    - ts_delta/ts_mean on very short windows that might peek
    - Negative lookback constants
    """
    expr = expression.lower()
    # 1. Future-looking delay: ts_delay(x, -N)
    if re.search(r'ts_delay\s*\([^)]*,\s*-\d+\s*\)', expr):
        return True
    # 2. ts_delta/ts_mean with explicit negative delta
    if re.search(r'ts_delta\s*\([^)]*,\s*-\d+\s*\)', expr):
        return True
    # 3. Division by a lookback parameter that is 0 or negative
    #    e.g. (x / (ts_delta(close, 0) + 1)) — ts_delta(close, 0) ≈ 0 leak
    if re.search(r'ts_delta\s*\([^)]*,\s*0\s*\)', expr):
        return True
    return False


def detect_survivorship_and_lookahead(expression: str) -> dict:
    """
    Combined bias check. Returns dict with 'passed', 'survivorship_bias',
    'lookahead_bias', and 'bias_flags' (comma-separated).
    """
    flags: list[str] = []
    if detect_survivorship_bias(expression):
        flags.append("survivorship_bias")
    if detect_lookahead_bias(expression):
        flags.append("lookahead_bias")
    return {
        "passed": len(flags) == 0,
        "survivorship_bias": "survivorship_bias" in flags,
        "lookahead_bias": "lookahead_bias" in flags,
        "bias_flags": ",".join(flags),
    }


# ============================================================
# Tier-1: IC Stability Scoring
# ============================================================

def estimate_ic_stability(sharpe: float, fitness: float, turnover: float, sub_sharpe: float) -> float:
    """
    Estimate IC (Information Coefficient) stability from available metrics.

    Signals:
    - Sharpe/fitness gap: large gap suggests IS overfitting
    - sub_sharpe (sub-universe robustness): negative → fragile
    - Turnover > 55%: overtrading → IC not stable
    - sub_sharpe < 0 AND |sub_sharpe| large: known failure mode
    """
    score = 0.0

    # Fitness-to-Sharpe consistency
    if sharpe > 0 and fitness > 0:
        ratio = min(fitness / max(sharpe, 0.01), 2.0)
        if 0.7 <= ratio <= 1.3:
            score += 0.30   # consistent — no IS/OOS gap
        elif ratio > 1.5:
            score -= 0.25   # fitness >> sharpe — likely IS curve-fit
        else:
            # Partial credit for being close to the sweet spot
            # e.g. ratio=0.57 → 0.13 away from 0.7 → partial 0.30 * (1 - 0.13/0.8)
            dist = abs(ratio - 0.7) if ratio < 0.7 else abs(ratio - 1.3)
            score += max(0, 0.20 * (1 - dist / 0.6))

    # Sub-universe robustness (most important positive signal from WQ)
    if sub_sharpe is not None:
        if sub_sharpe < -0.99:
            score += 0.20   # sub-universe neutral — standard WQ behaviour
        elif sub_sharpe < 0:
            score -= 0.20   # mild negative sub — penalise but not lethal
        elif sub_sharpe >= 0:
            # Positive sub-sharpe is a STRONG acceptance signal
            score += 0.20 + min(0.15, sub_sharpe * 0.10)
            # Fitness below 1.0 but sub_sharpe positive = interesting edge case
            if fitness < 1.0 and fitness > 0:
                score += 0.10  # bonus: sub-universe holds even when IS fitness is weak

    # Turnover penalty (lenient: allow up to 60%)
    if turnover > 60:
        score -= 0.15
    elif turnover > 55:
        score -= 0.05
    elif turnover < 2:
        score -= 0.10   # too illiquid to trust

    return max(0.0, min(1.0, score))


IC_STABILITY_FLOOR = float(os.getenv("ASYNC_IC_STABILITY_MIN", "0.15"))


def passes_ic_stability(
    sharpe: float,
    fitness: float,
    turnover: float,
    sub_sharpe: float,
    floor: float = IC_STABILITY_FLOOR,
) -> bool:
    """
    Gate: reject alphas with estimated IC instability.

    Research: IC persistence across rolling periods is a key WQ acceptance criterion.
    We approximate this cheaply from simulation results.
    """
    stability = estimate_ic_stability(sharpe, fitness, turnover, sub_sharpe)
    return stability >= floor


# ============================================================
# Tier-1: Pre-Submission Gate  (combines all Tier-1 filters)
# ============================================================

def pre_submission_gate(
    expression: str,
    sharpe: float,
    fitness: float,
    turnover: float,
    sub_sharpe: float,
    error: str,
) -> dict:
    """
    All Tier-1 acceptance filters before an alpha is queued for WQ submit.

    Returns a dict with:
      - passed (bool): True only if all gates pass
      - reason (str): human-readable failure reason
      - stage (str): which filter failed

    Run this before SubmitGovernor.enqueue().
    """
    # Stage 0: runtime error
    if error:
        return {"passed": False, "reason": error, "stage": "runtime_error"}

    # Stage 1: bias detection (automatic disqualifiers)
    bias = detect_survivorship_and_lookahead(expression)
    if not bias["passed"]:
        return {
            "passed": False,
            "reason": bias["bias_flags"],
            "stage": "bias_detection",
        }

    # Stage 2: IC stability
    if not passes_ic_stability(sharpe, fitness, turnover, sub_sharpe):
        stability = estimate_ic_stability(sharpe, fitness, turnover, sub_sharpe)
        return {
            "passed": False,
            "reason": f"ic_unstable:stability={stability:.2f}",
            "stage": "ic_stability",
        }

    # Stage 3: base quality gate (Sharpe, fitness, turnover)
    if not passes_quality_gate_v2(
        _SimResultProxy(sharpe, fitness, turnover, sub_sharpe, error)
    ):
        return {"passed": False, "reason": "quality_gate_failed", "stage": "quality_gate"}

    return {"passed": True, "reason": "all_tiers_passed", "stage": "passed"}


def pre_submission_gate_from_result(result) -> dict:
    """
    Convenience wrapper — accepts any object with sharpe, fitness, turnover,
    sub_sharpe, and error attributes (e.g. SimpleNamespace, dict, dataclass).
    """
    return pre_submission_gate(
        expression=getattr(result, "expression", ""),
        sharpe=float(getattr(result, "sharpe", 0.0) or 0.0),
        fitness=float(getattr(result, "fitness", 0.0) or 0.0),
        turnover=float(getattr(result, "turnover", 0.0) or 0.0),
        sub_sharpe=float(getattr(result, "sub_sharpe", -1.0) or -1.0),
        error=str(getattr(result, "error", "") or ""),
    )


class _SimResultProxy:
    """Thin wrapper so pre_submission_gate can reuse passes_quality_gate_v2."""
    __slots__ = ("sharpe", "fitness", "turnover", "sub_sharpe", "error", "all_passed", "passed_checks", "total_checks")

    def __init__(self, sharpe, fitness, turnover, sub_sharpe, error):
        self.sharpe = sharpe
        self.fitness = fitness
        self.turnover = turnover
        self.sub_sharpe = sub_sharpe
        self.error = error
        self.all_passed = bool(error == "")
        self.passed_checks = 0
        self.total_checks = 0
