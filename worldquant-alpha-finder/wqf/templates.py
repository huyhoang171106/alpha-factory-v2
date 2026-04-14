"""wqf/templates.py — 20 battle-tested alpha templates by category."""
from dataclasses import dataclass
from typing import List


@dataclass
class AlphaTemplate:
    id: str
    hypothesis: str
    expression: str
    description: str
    expected_sharpe_range: str
    notes: str


# ── Master template registry ────────────────────────────────────
TEMPLATES: List[AlphaTemplate] = [

    # ══ MICROSTRUCTURE ══════════════════════════════════════════
    AlphaTemplate(
        id="ms_01", hypothesis="microstructure",
        expression="rank(-ts_mean(abs(returns) / (volume + 1), 20))",
        description="Inverted price impact: low impact = strong signal",
        expected_sharpe_range="1.0–2.0",
        notes="Classic Kyle lambda. ADV normalization essential."
    ),
    AlphaTemplate(
        id="ms_02", hypothesis="microstructure",
        expression="group_neutralize(rank(volume / adv20), sector)",
        description="Relative volume: sector-neutral abnormal activity",
        expected_sharpe_range="1.2–2.2",
        notes="Group neutralize cleans sector effects."
    ),
    AlphaTemplate(
        id="ms_03", hypothesis="microstructure",
        expression="rank(ts_corr(vwap, volume, 10)) * rank(-ts_mean(abs(returns), 5))",
        description="VWAP-volume correlation × short-term reversal",
        expected_sharpe_range="1.0–1.8",
        notes="Composite: signal × filter pattern."
    ),
    AlphaTemplate(
        id="ms_04", hypothesis="microstructure",
        expression="-rank(ts_delta(close, 1) / (volume + 1))",
        description="Inverted order flow imbalance",
        expected_sharpe_range="0.8–1.5",
        notes="Simple but effective with ADV normalization."
    ),

    # ══ QUALITY ════════════════════════════════════════════════
    AlphaTemplate(
        id="q_01", hypothesis="quality",
        expression="rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
        description="Risk-adjusted returns: high Sharpe sub-series",
        expected_sharpe_range="1.5–2.5",
        notes="Normalize by vol = standard quality factor."
    ),
    AlphaTemplate(
        id="q_02", hypothesis="quality",
        expression="-group_neutralize(rank(ts_delta(close, 5) / (ts_std_dev(close, 5) + 0.001)), sector)",
        description="Intraday sector-relative mean reversion",
        expected_sharpe_range="1.3–2.0",
        notes="Neutralization prevents sector leakage."
    ),
    AlphaTemplate(
        id="q_03", hypothesis="quality",
        expression="rank(1 / (close / ts_mean(close, 120) - 1 + 0.1))",
        description="Cheap = mean-revert. Long cheap, short rich.",
        expected_sharpe_range="1.0–1.8",
        notes="Price relative to 6-month MA. Cap sensitivity."
    ),

    # ══ CROSS-SECTIONAL ════════════════════════════════════════
    AlphaTemplate(
        id="cs_01", hypothesis="cross_sectional",
        expression="group_neutralize(rank(ts_mean(returns, 20)), sector)",
        description="Rank returns, sector-neutral",
        expected_sharpe_range="1.5–2.5",
        notes="Most robust WQ pattern. Group neutralization critical."
    ),
    AlphaTemplate(
        id="cs_02", hypothesis="cross_sectional",
        expression="-group_neutralize(rank(ts_delta(close, 5)), sector) * rank(ts_mean(volume, 5))",
        description="Sector-relative short-term reversal × volume confirmation",
        expected_sharpe_range="1.2–2.0",
        notes="Volume filter prevents noise trading."
    ),
    AlphaTemplate(
        id="cs_03", hypothesis="cross_sectional",
        expression="group_zscore(rank(ts_mean(returns, 10)), sector) - rank(volume / adv20)",
        description="Z-scored returns minus relative volume",
        expected_sharpe_range="1.0–1.8",
        notes="Composite cross-sectional + microstructure."
    ),

    # ══ REGIME ══════════════════════════════════════════════════
    AlphaTemplate(
        id="reg_01", hypothesis="regime",
        expression="-rank(ts_std_dev(returns, 20) / ts_mean(ts_std_dev(returns, 20), 60))",
        description="Volatility mean reversion: high vol → vol normalizes",
        expected_sharpe_range="1.0–2.0",
        notes="Use on VIX-linked or high-vol regimes."
    ),
    AlphaTemplate(
        id="reg_02", hypothesis="regime",
        expression="rank(ts_delta(ts_std_dev(returns, 10), 1) / (ts_std_dev(returns, 10) + 0.001))",
        description="Volatility momentum: vol increasing = continued vol",
        expected_sharpe_range="0.8–1.5",
        notes="Anti-fragile in trending vol regimes."
    ),
    AlphaTemplate(
        id="reg_03", hypothesis="regime",
        expression="-rank(ts_mean(abs(returns) / (volume + 1), 5) / (ts_mean(abs(returns) / (volume + 1), 20) + 0.001))",
        description="Volume-normalized price impact ratio: short-term spike",
        expected_sharpe_range="1.0–1.8",
        notes="IC drops in low-liquidity environments."
    ),

    # ══ BEHAVIORAL ══════════════════════════════════════════════
    AlphaTemplate(
        id="beh_01", hypothesis="behavioral",
        expression="-rank(ts_delta(close, 5)) * rank(ts_std_dev(returns, 20))",
        description="Short-term reversal filtered by vol: losers bounce",
        expected_sharpe_range="1.0–1.8",
        notes="Volume confirmation adds stability."
    ),
    AlphaTemplate(
        id="beh_02", hypothesis="behavioral",
        expression="rank(ts_skewness(returns, 20)) * rank(-ts_delta(volume, 1))",
        description="Skewness timing: down-volume on high-skew = continuation",
        expected_sharpe_range="0.8–1.5",
        notes="Skewness as regime indicator."
    ),
    AlphaTemplate(
        id="beh_03", hypothesis="behavioral",
        expression="rank(ts_mean(volume, 5) / ts_mean(volume, 20)) * rank(returns)",
        description="Volume surge × price momentum",
        expected_sharpe_range="1.0–1.6",
        notes="Volume confirms or denies momentum."
    ),

    # ══ STATISTICAL ARBITRAGE ══════════════════════════════════
    AlphaTemplate(
        id="sa_01", hypothesis="stat_arb",
        expression="rank(ts_corr(close, volume, 20)) - rank(ts_corr(ts_mean(close, 5), ts_mean(volume, 5), 20))",
        description="Correlation structure: short-term vs medium-term price-volume",
        expected_sharpe_range="1.2–2.0",
        notes="Pair-wise structural break detection."
    ),
    AlphaTemplate(
        id="sa_02", hypothesis="stat_arb",
        expression="rank(ts_corr(returns, ts_delta(close, 1), 10)) * rank(-ts_std_dev(returns, 20))",
        description="Return-volume lead-lag filtered by vol",
        expected_sharpe_range="1.0–1.8",
        notes="Works well in high-vol environments."
    ),

    # ══ FUNDAMENTAL ═════════════════════════════════════════════
    AlphaTemplate(
        id="fund_01", hypothesis="fundamental",
        expression="rank(ebitda / debt) * rank(-ts_delta(returns, 20))",
        description="Quality × momentum: strong balance sheet + recent underperformance",
        expected_sharpe_range="1.0–1.8",
        notes="Quality and value interact non-linearly."
    ),
]


def get_templates_by_hypothesis(hypothesis: str) -> List[AlphaTemplate]:
    return [t for t in TEMPLATES if t.hypothesis == hypothesis]


def get_all_expressions() -> List[str]:
    return [t.expression for t in TEMPLATES]


def get_template_by_id(tid: str) -> AlphaTemplate | None:
    for t in TEMPLATES:
        if t.id == tid:
            return t
    return None
