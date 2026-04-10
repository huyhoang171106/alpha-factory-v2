"""
alpha_policy.py - Shared quality policy and lightweight critic.
"""

from __future__ import annotations

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
    if thresholds.require_all_checks and not getattr(result, "all_passed", False):
        return False
    sharpe = float(getattr(result, "sharpe", 0.0) or 0.0)
    fitness = float(getattr(result, "fitness", 0.0) or 0.0)
    turnover = float(getattr(result, "turnover", 0.0) or 0.0)
    return (
        sharpe >= thresholds.sharpe
        and fitness >= thresholds.fitness
        and thresholds.turnover_min < turnover < thresholds.turnover_max
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
    return min(score, 1.0)


def should_simulate_candidate(expression: str, min_critic_score: float = 0.28) -> bool:
    return critic_score(expression) >= min_critic_score


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
