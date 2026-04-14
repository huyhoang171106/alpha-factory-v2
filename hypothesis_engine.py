"""
hypothesis_engine.py

Hypothesis Engine v2 — Dynamic Hypothesis-Driven Alpha Generation
Research basis: LLM → Hypothesis ONLY (not expression generation)
Ref: WQ research, Alpha Generation Strategy Optimization research

Hypothesis = economic intuition about market behavior
Expression = mathematical implementation of hypothesis
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ─── Market Regime Detection ─────────────────────────────────────────────────


@dataclass
class MarketRegime:
    """Represents a detected market regime with confidence."""

    type: str  # bull_trending, bear_trending, high_vol, low_vol, mean_reversion, unclear
    confidence: float  # 0-1
    indicators: Dict[str, float]


class MarketRegimeDetector:
    """Rule-based market regime detection.
    v1: rule-based. Future: HMM or LSTM-based.
    """

    def detect_from_market_data(
        self,
        vix_proxy: Optional[float] = None,
        market_return: Optional[float] = None,
        market_vol: Optional[float] = None,
        avg_turnover: Optional[float] = None,
    ) -> MarketRegime:
        """Detect current market regime from available indicators.

        Args:
            vix_proxy: VIX level or proxy (e.g., realized vol). Normal = ~15-20
            market_return: Recent market return (e.g., 20-day)
            market_vol: Realized volatility
            avg_turnover: Average market turnover
        Returns:
            MarketRegime with type and confidence
        """
        indicators: Dict[str, float] = {}
        if vix_proxy is not None:
            indicators["vix_proxy"] = vix_proxy
        if market_return is not None:
            indicators["market_return"] = market_return
        if market_vol is not None:
            indicators["market_vol"] = market_vol
        if avg_turnover is not None:
            indicators["avg_turnover"] = avg_turnover

        # Approximate VIX from vol if not provided
        vix = vix_proxy if vix_proxy is not None else (market_vol * 16.0 if market_vol else None)
        ret = market_return if market_return is not None else 0.0
        vol = market_vol if market_vol is not None else (vix / 16.0 if vix else 0.15)

        regime_type: str
        confidence: float

        if vix is not None:
            if vix > 25:
                regime_type = "high_vol"
                confidence = min(1.0, (vix - 25) / 15.0 + 0.5)
            elif vix < 15:
                if ret > 0.02:
                    regime_type = "bull_trending"
                    confidence = min(1.0, (0.05 - vix) / 10.0 + 0.5)
                elif ret < -0.02:
                    regime_type = "bear_trending"
                    confidence = 0.6
                else:
                    regime_type = "low_vol"
                    confidence = 0.6
            else:  # 15 <= vix <= 25
                if abs(ret) > 0.01:
                    regime_type = "mean_reversion"
                    confidence = 0.55
                else:
                    regime_type = "low_vol"
                    confidence = 0.5
        elif abs(ret) > 0.03:
            regime_type = "bull_trending" if ret > 0 else "bear_trending"
            confidence = 0.6
        else:
            regime_type = "mean_reversion"
            confidence = 0.5

        return MarketRegime(type=regime_type, confidence=confidence, indicators=indicators)

    def detect_from_expression_context(self, expression: str) -> List[str]:
        """Infer likely suitable regimes from expression features."""
        regimes: List[str] = []
        expr = expression.lower()
        if "ts_skewness" in expr or "ts_kurtosis" in expr:
            regimes.extend(["high_vol", "mean_reversion"])
        if "volume" in expr or "adv" in expr:
            regimes.extend(["high_vol", "low_vol"])
        if "ts_corr" in expr or "ts_regression" in expr:
            regimes.extend(["bull_trending", "bear_trending", "mean_reversion"])
        if "group_rank" in expr or "group_neutralize" in expr:
            regimes.extend(["bull_trending", "bear_trending", "high_vol", "low_vol"])
        if not regimes:
            regimes = ["low_vol", "mean_reversion"]
        return regimes


# ─── Hypothesis Registry ──────────────────────────────────────────────────────


@dataclass
class HypothesisRecord:
    """Tracks performance of a hypothesis across generated alphas."""

    name: str
    regime: str
    direction: str  # long, short, neutral
    expected_outcome: str
    success_count: int = 0
    total_count: int = 0
    avg_sharpe: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_count)


# Template library keyed by hypothesis type
HYPOTHESIS_TEMPLATES: Dict[str, Dict] = {
    "mean_reversion": {
        "description": "Price deviations from recent range revert to mean",
        "regimes": ["high_vol", "mean_reversion", "low_vol"],
        "direction": "neutral",
        "base_expressions": [
            "(close - ts_mean(close, {d})) / (ts_max(high, {d}) - ts_min(low, {d}) + 0.001)",
            "ts_rank(returns, {d})",
            "rank(close / ts_mean(close, {d}))",
        ],
        "filters": [
            "ts_std_dev(returns, {d}) / ts_mean(ts_std_dev(returns, {d}), {d2})",
            "volume / adv20",
            "rank(volume / adv20)",
        ],
        "lookback_range": (5, 60),
    },
    "momentum": {
        "description": "Trending prices continue their direction",
        "regimes": ["bull_trending", "bear_trending", "high_vol"],
        "direction": "long_short",
        "base_expressions": [
            "ts_mean(returns, {d})",
            "ts_delta(close, {d}) / ts_std_dev(close, {d})",
            "rank(ts_mean(returns, {d}))",
        ],
        "filters": [
            "ts_rank(abs(returns), {d})",
            "ts_corr(returns, volume, {d})",
            "rank(volume / adv20)",
        ],
        "lookback_range": (5, 40),
    },
    "liquidity_impact": {
        "description": "Abnormal volume predicts price impact",
        "regimes": ["high_vol", "low_vol", "mean_reversion"],
        "direction": "neutral",
        "base_expressions": [
            "abs(returns) / (volume * close + 1)",
            "(volume - ts_mean(volume, {d})) / ts_std_dev(volume, {d})",
            "vwap / close - 1",
        ],
        "filters": [
            "ts_std_dev(returns, {d})",
            "rank(volume / adv20)",
        ],
        "lookback_range": (1, 20),
    },
    "volatility_regime": {
        "description": "Volatility regimes predict themselves (vol clustering)",
        "regimes": ["high_vol", "low_vol"],
        "direction": "neutral",
        "base_expressions": [
            "ts_std_dev(returns, {d}) / ts_mean(ts_std_dev(returns, {d}), {d2})",
            "rank(ts_std_dev(returns, {d}))",
        ],
        "filters": [
            "ts_mean(ts_std_dev(returns, {d}), {d2})",
        ],
        "lookback_range": (5, 40),
    },
    "microstructure": {
        "description": "Bid-ask spread and order flow predicts short-term returns",
        "regimes": ["high_vol", "low_vol"],
        "direction": "neutral",
        "base_expressions": [
            "ts_corr(returns, volume, {d})",
            "ts_covariance(returns, volume, {d}) / (ts_std_dev(returns, {d}) * ts_std_dev(volume, {d}) + 0.001)",
        ],
        "filters": [
            "rank(volume / adv20)",
            "ts_std_dev(returns, {d})",
        ],
        "lookback_range": (1, 20),
    },
}


class HypothesisRegistry:
    """Tracks which hypotheses produce successful alphas."""

    def __init__(self, data_dir: str = "data/hypothesis_registry"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.data_dir / "hypothesis_registry.json"
        self.records: Dict[str, HypothesisRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for name, rec in data.items():
                        self.records[name] = HypothesisRecord(**rec)
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self) -> None:
        data = {
            name: {
                "name": r.name,
                "regime": r.regime,
                "direction": r.direction,
                "expected_outcome": r.expected_outcome,
                "success_count": r.success_count,
                "total_count": r.total_count,
                "avg_sharpe": r.avg_sharpe,
            }
            for name, r in self.records.items()
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_outcome(self, hypothesis_name: str, sharpe: float, accepted: bool) -> None:
        """Record the outcome of an alpha generated from a hypothesis."""
        if hypothesis_name not in self.records:
            self.records[hypothesis_name] = HypothesisRecord(
                name=hypothesis_name,
                regime="unknown",
                direction="unknown",
                expected_outcome="unknown",
            )
        rec = self.records[hypothesis_name]
        rec.total_count += 1
        if accepted:
            rec.success_count += 1
        # Running average of Sharpe
        rec.avg_sharpe = (
            rec.avg_sharpe * (rec.total_count - 1) + sharpe
        ) / rec.total_count
        self._save()

    def get_proven_hypotheses(
        self, min_sharpe: float = 1.0, min_success_rate: float = 0.3
    ) -> List[HypothesisRecord]:
        """Return hypotheses that consistently produce good alphas."""
        return [
            r
            for r in self.records.values()
            if r.avg_sharpe >= min_sharpe and r.success_rate >= min_success_rate
        ]

    def get_regime_hypotheses(self, regime: str) -> List[HypothesisRecord]:
        """Return best hypotheses for a specific regime."""
        regime_recs = [r for r in self.records.values() if r.regime == regime]
        return sorted(regime_recs, key=lambda r: r.avg_sharpe, reverse=True)


# ─── Hypothesis-Driven Generator ──────────────────────────────────────────────


class HypothesisDrivenGenerator:
    """Generates alpha expressions from known economic hypotheses."""

    def __init__(self, registry: Optional[HypothesisRegistry] = None):
        self.registry = registry or HypothesisRegistry()
        self.templates = HYPOTHESIS_TEMPLATES

    def get_hypothesis_for_regime(self, regime: str) -> str:
        """Select best hypothesis for current regime."""
        proven = self.registry.get_regime_hypotheses(regime)
        if proven:
            return proven[0].name
        # Fallback: pick template for regime
        for name, tmpl in self.templates.items():
            if regime in tmpl["regimes"]:
                return name
        return "mean_reversion"

    def generate_from_hypothesis(
        self,
        hypothesis_type: str,
        regime: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        """Generate alpha expression from hypothesis template."""
        if hypothesis_type not in self.templates:
            hypothesis_type = "mean_reversion"
        tmpl = self.templates[hypothesis_type]

        rng = random.Random(seed)
        lb_min, lb_max = tmpl["lookback_range"]
        d = rng.randint(lb_min, lb_max)
        d2 = d * 2

        # Select base expression
        base_exprs = tmpl["base_expressions"]
        base = rng.choice(base_exprs).format(d=d, d2=d2)

        # Select filter
        filters = tmpl["filters"]
        filter_expr = rng.choice(filters).format(d=d, d2=d2)

        direction_sign = "-" if tmpl["direction"] == "neutral" else ""

        # Combine: base * filter, then rank
        combined = f"({direction_sign}{base} * {filter_expr})"
        return f"rank({combined})"

    def generate_for_current_regime(
        self,
        regime: MarketRegime,
        hypothesis_type: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate alpha suited for the current market regime."""
        hyp_type = hypothesis_type or self.get_hypothesis_for_regime(regime.type)
        s = seed if seed is not None else random.randint(0, 9999)
        return self.generate_from_hypothesis(hyp_type, regime.type, s)
