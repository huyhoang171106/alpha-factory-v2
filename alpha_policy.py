"""
alpha_policy.py - Shared quality policy and lightweight critic.
"""

from __future__ import annotations

import os
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
    min_checks_ratio = float(os.getenv("ASYNC_MIN_CHECKS_RATIO", "1.0" if require_all_checks else "0.5"))
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
    checks_ok = 1.0 if getattr(result, "all_passed", False) else 0.0
    raw_sub = getattr(result, "sub_sharpe", None)
    sub_sharpe = float(raw_sub) if raw_sub is not None else None

    turnover_penalty = 0.0
    if turnover > 55.0:
        turnover_penalty += min(0.5, (turnover - 55.0) / 60.0)
    if turnover < 1.5:
        turnover_penalty += 0.2

    sub_penalty = 0.0
    if sub_sharpe is not None and sub_sharpe > -0.99:
        if 0.0 <= sub_sharpe < 0.25:
            sub_penalty = 0.25
        elif sub_sharpe < 0.0:
            sub_penalty = 0.35

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
