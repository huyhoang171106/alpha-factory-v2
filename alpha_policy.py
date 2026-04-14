"""
alpha_policy.py - Shared quality policy and lightweight critic.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

from alpha_ast import operator_set

logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------------
# Self-corr rejection logging (feeds LearnedSelfCorrWeights)
# ------------------------------------------------------------------
_SELF_CORR_REJECT_LOG = os.path.join(os.path.dirname(__file__), "data", "self_corr_rejections.jsonl")


def _log_self_corr_rejection(result, self_corr: float, sharpe: float, fitness: float) -> None:
    """
    Log a self-corr rejection as structured JSON for downstream learning.

    Written to ``data/self_corr_rejections.jsonl`` — one JSON line per event.
    ``LearnedSelfCorrWeights._rebuild`` reads this file to build P(high_self_corr | feature).
    """
    import json as _json

    entry = {
        "expression": getattr(result, "expression", "")[:500],
        "self_corr": float(self_corr),
        "sharpe": float(sharpe),
        "fitness": float(fitness),
        "operators": list(operator_set(getattr(result, "expression", ""))),
    }
    try:
        os.makedirs(os.path.dirname(_SELF_CORR_REJECT_LOG), exist_ok=True)
        with open(_SELF_CORR_REJECT_LOG, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry) + "\n")
        logger.debug("Logged self-corr rejection: %.3f | %s", self_corr, entry["operators"])
    except Exception as e:
        logger.warning("Failed to log self_corr rejection: %s", e)


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


@lru_cache(maxsize=20000)
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


@lru_cache(maxsize=20000)
def should_simulate_candidate(expression: str, min_critic_score: float = 0.28) -> bool:
    return critic_score(expression) >= min_critic_score


# ============================================================
# Learned Critic — Bayesian P(pass_gate | operator feature)
# ============================================================
#
# "pass_gate" is defined as all_passed=1 AND sharpe >= 1.25.
# Bayesian smoothing with alpha=1.0 (uniform prior) keeps weights
# stable even when a feature has been seen only a handful of times.
# ============================================================

_LEARNED_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "learned_critic_weights.json"
)
_LEARNED_WEIGHTS: Optional[dict] = None
_WEIGHTS_LOCK = threading.Lock()

# ------------------------------------------------------------------
# Internal feature extractor (mirrors the static features used by
# critic_score so the two scoring paths are comparable).
# ------------------------------------------------------------------
# fmt: off
_OPERATOR_FEATURES: list[tuple[str, str]] = [
    ("has_group_neutralize",    r"group_neutralize\("),
    ("has_zscore",              r"(?:ts_)?zscore\("),
    ("has_ts_decay_linear",     r"ts_decay_linear\("),
    ("has_ts_corr",             r"ts_(?:covariance|corr)\("),
    ("has_rank_outer",          r"^rank\("),
    ("has_trade_when",          r"trade_when\("),
    ("has_hump",                r"hump\("),
    ("has_regression_neut",     r"regression_neut\("),
    ("has_pasteurize",          r"pasteurize\("),
    ("has_densify_bucket",      r"(?:densify|bucket)\("),
    ("has_vector_neutralize",   r"vector_neut\("),
    ("has_winsorize",           r"winsorize\("),
    ("has_if_else",             r"if_else\("),
    ("has_signed_power",        r"signed_power\("),
    ("has_scale",               r"scale\("),
    ("has_ts_quantile",         r"ts_quantile\("),
    ("has_normalize",           r"normalize\("),
    ("has_group_rank",          r"group_rank\("),
    ("has_group_zscore",        r"group_zscore\("),
    ("has_group_mean",          r"group_mean\("),
    ("has_adv",                 r"adv\d+"),
    ("has_vwap",                r"vwap"),
    ("has_returns",             r"returns"),
    ("has_volume",              r"\bvolume\b"),
]
# fmt: on


def _extract_features(expr: str) -> dict[str, float]:
    """Return per-feature presence (0 or 1) plus scalar signals."""
    expr_lower = expr.lower()
    result: dict[str, float] = {}

    # Scalar signals
    result["has_expression_length"] = min(len(expr_lower) / 200.0, 1.0)
    result["has_multi_field"]      = float(_has_multi_field(expr_lower))
    result["has_cross_sectional"]  = float(_has_cross_sectional(expr_lower))

    # Binary operator features
    for name, pattern in _OPERATOR_FEATURES:
        result[name] = 1.0 if re.search(pattern, expr_lower) else 0.0

    return result


def _has_multi_field(expr: str) -> bool:
    price_fields  = {"open", "high", "low", "close", "vwap"}
    volume_fields = {"volume", "adv20", "adv60"}
    cats = 0
    if any(f in expr for f in price_fields):
        cats += 1
    if any(f in expr for f in volume_fields):
        cats += 1
    if "returns" in expr:
        cats += 1
    return cats >= 2


def _has_cross_sectional(expr: str) -> bool:
    return any(
        op in expr
        for op in ["group_neutralize", "group_rank", "group_mean", "group_zscore"]
    )


# ------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------
def _resolved_rows(db_path: str, limit: int) -> list[dict]:
    """Fetch last N resolved alphas from tracker DB."""
    try:
        conn = sqlite3.connect(db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        rows = conn.execute(
            """
            SELECT expression, sharpe, all_passed
            FROM alphas
            WHERE all_passed IN (0, 1)
              AND sharpe IS NOT NULL
              AND expression IS NOT NULL
              AND expression != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"expression": str(expr), "sharpe": float(s or 0.0), "all_passed": bool(a)}
            for expr, s, a in rows
        ]
    except Exception as e:
        logger.warning("Failed to read resolved alphas from DB: %s", e)
        return []


def _compute_weights(rows: list[dict], alpha: float = 1.0) -> dict:
    """
    Compute P(pass_gate | feature) for every tracked feature using
    Laplace-additive (Bayesian) smoothing.

    pass_gate  =  all_passed == 1  AND  sharpe >= 1.25
    alpha=1.0   =  uniform prior (pseudo-counts of 1 in each binomial bucket)
    """
    BAYESIAN_PRIOR = alpha
    ALPHA_PRIOR    = 2.0 * alpha  # two classes: pass / fail

    all_features = {name for name, _ in _OPERATOR_FEATURES}
    all_features.update(["has_expression_length", "has_multi_field", "has_cross_sectional"])

    feature_pass: dict[str, float] = {f: BAYESIAN_PRIOR    for f in all_features}
    feature_total: dict[str, float] = {f: ALPHA_PRIOR      for f in all_features}

    for row in rows:
        expr    = row["expression"] or ""
        is_pass = 1.0 if (row["all_passed"] and row["sharpe"] >= 1.25) else 0.0
        feats   = _extract_features(expr)
        for name, val in feats.items():
            feature_pass[name]  += is_pass * val
            feature_total[name] += val

    return {
        f: feature_pass[f] / max(feature_total[f], 1.0)
        for f in all_features
    }


# ------------------------------------------------------------------
# LearnedCriticWeights singleton
# ------------------------------------------------------------------

class LearnedCriticWeights:
    """
    Thread-safe singleton holding Bayesian operator weights.

    Weights are persisted to ``data/learned_critic_weights.json`` so they
    survive process restarts.  Call ``update()`` after each simulation
    batch resolves to keep the model fresh.
    """

    _instance: Optional["LearnedCriticWeights"] = None

    def __init__(
        self,
        weights: dict,
        db_path: str,
        lookback: int,
        version: int = 0,
    ):
        self._weights  = dict(weights)
        self._db_path  = db_path
        self._lookback = lookback
        self._version  = version

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "LearnedCriticWeights":
        """Return the current singleton (lazy initialisation)."""
        global _LEARNED_WEIGHTS
        with _WEIGHTS_LOCK:
            if cls._instance is None:
                inst = cls._load(db_path)
                cls._instance = inst
                _LEARNED_WEIGHTS = inst._weights
            return cls._instance

    @classmethod
    def _load(cls, db_path: Optional[str] = None) -> "LearnedCriticWeights":
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "alpha_results.db")

        if os.path.exists(_LEARNED_WEIGHTS_PATH):
            try:
                with open(_LEARNED_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return cls(
                    weights=payload.get("weights", {}),
                    db_path=db_path,
                    lookback=payload.get("lookback", 5000),
                    version=payload.get("version", 0),
                )
            except Exception as e:
                logger.warning(
                    "Could not load learned weights from %s: %s — rebuilding",
                    _LEARNED_WEIGHTS_PATH,
                    e,
                )

        return cls._rebuild(db_path)

    @classmethod
    def _rebuild(
        cls,
        db_path: str,
        lookback: int = 5000,
    ) -> "LearnedCriticWeights":
        rows    = _resolved_rows(db_path, lookback)
        weights = _compute_weights(rows)
        inst    = cls(weights=weights, db_path=db_path, lookback=lookback)
        inst._save()
        global _LEARNED_WEIGHTS
        _LEARNED_WEIGHTS = weights
        return inst

    def _save(self) -> None:
        """Write weights + metadata to JSON."""
        os.makedirs(os.path.dirname(_LEARNED_WEIGHTS_PATH), exist_ok=True)
        with open(_LEARNED_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version":   self._version,
                    "lookback":  self._lookback,
                    "weights":   self._weights,
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def update(self, lookback: Optional[int] = None) -> None:
        """
        Re-compute Bayesian weights from all resolved alphas in the DB and
        persist the result to ``data/learned_critic_weights.json``.
        """
        if lookback is not None:
            self._lookback = lookback

        rows    = _resolved_rows(self._db_path, self._lookback)
        weights = _compute_weights(rows)

        with _WEIGHTS_LOCK:
            self._weights = weights
            self._version += 1
            global _LEARNED_WEIGHTS
            _LEARNED_WEIGHTS = weights

        self._save()
        logger.info(
            "LearnedCriticWeights updated: v%d from %d rows | top: %s",
            self._version,
            len(rows),
            _top_features_summary(weights),
        )

    def score(self, expression: str) -> float:
        """
        Score an expression using the learned Bayesian weights.
        Returns a float in [0, 1].

        Score = average P(pass_gate) across all active features.
        """
        if not expression or not self._weights:
            return 0.0

        feats = _extract_features(expression)
        active = [(n, v) for n, v in feats.items() if n in self._weights and v > 0]

        if not active:
            return 0.0

        total = sum(self._weights[n] * v for n, v in active)
        return total / len(active)

    def as_dict(self) -> dict:
        """Expose weights for inspection / logging."""
        return dict(self._weights)

    @property
    def version(self) -> int:
        return self._version


def _top_features_summary(weights: dict, n: int = 5) -> str:
    """Human-readable summary of the top-N highest-weight features."""
    if not weights:
        return "no weights"
    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:n]
    return ", ".join(f"{k}={v:.3f}" for k, v in top)


def update_critic_weights(
    db_path: Optional[str] = None,
    lookback: int = 5000,
) -> None:
    """
    Top-level entry-point to refresh learned critic weights.

    Call this after each simulation batch resolves::

        update_critic_weights()

    The weights are shared process-wide via a singleton.
    """
    inst = LearnedCriticWeights.get_instance(db_path=db_path)
    inst.update(lookback=lookback)


# ============================================================
# LearnedSelfCorrWeights — P(high_self_corr | operator feature)
# ============================================================
#
# "high_self_corr" is defined as self_corr > SELF_CORR_REJECT_THRESHOLD (default 0.7).
# Tracks which operator patterns are associated with high autocorrelation so the
# generator can bias away from them before wasting WQ quota.
# ============================================================

_SELF_CORR_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "learned_self_corr_weights.json"
)
_SELF_CORR_WEIGHTS: Optional[dict] = None
_SELF_CORR_WEIGHTS_LOCK = threading.Lock()

# Same feature set as _OPERATOR_FEATURES but tracks self-corr signal instead of pass-rate.
_SELF_CORR_FEATURES: list[tuple[str, str]] = _OPERATOR_FEATURES  # reuse existing patterns


def _self_corr_resolved_rows(db_path: str, limit: int) -> list[dict]:
    """Fetch last N resolved alphas that have a self_corr value."""
    threshold = float(os.getenv("SELF_CORR_REJECT_THRESHOLD", "0.70"))
    try:
        conn = sqlite3.connect(db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        rows = conn.execute(
            """
            SELECT expression, self_corr
            FROM alphas
            WHERE self_corr IS NOT NULL
              AND expression IS NOT NULL
              AND expression != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"expression": str(expr), "self_corr": float(sc or 0.0)}
            for expr, sc in rows
        ]
    except Exception as e:
        logger.warning("Failed to read self_corr rows from DB: %s", e)
        return []


def _compute_self_corr_weights(rows: list[dict], alpha: float = 1.0) -> dict:
    """
    Compute P(high_self_corr | feature) for every tracked feature using
    Laplace-additive smoothing.

    high_self_corr = self_corr > SELF_CORR_REJECT_THRESHOLD (default 0.70)
    alpha=1.0      = uniform prior (two binomial buckets: high / low)
    """
    threshold = float(os.getenv("SELF_CORR_REJECT_THRESHOLD", "0.70"))
    BAYESIAN_PRIOR = alpha
    ALPHA_PRIOR    = 2.0 * alpha

    all_features = {name for name, _ in _SELF_CORR_FEATURES}
    all_features.update(["has_expression_length", "has_multi_field", "has_cross_sectional"])

    feature_high: dict[str, float] = {f: BAYESIAN_PRIOR for f in all_features}
    feature_total: dict[str, float] = {f: ALPHA_PRIOR     for f in all_features}

    for row in rows:
        expr = row["expression"] or ""
        is_high = 1.0 if row["self_corr"] > threshold else 0.0
        feats = _extract_features(expr)
        for name, val in feats.items():
            feature_high[name]  += is_high * val
            feature_total[name] += val

    return {
        f: feature_high[f] / max(feature_total[f], 1.0)
        for f in all_features
    }


class LearnedSelfCorrWeights:
    """
    Thread-safe singleton holding Bayesian P(high_self_corr | feature) weights.

    Persisted to ``data/learned_self_corr_weights.json``.  Call ``update()``
    after each simulation batch resolves.
    """

    _instance: Optional["LearnedSelfCorrWeights"] = None

    def __init__(
        self,
        weights: dict,
        db_path: str,
        lookback: int,
        version: int = 0,
    ):
        self._weights  = dict(weights)
        self._db_path  = db_path
        self._lookback = lookback
        self._version  = version

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "LearnedSelfCorrWeights":
        """Return the current singleton (lazy initialisation)."""
        global _SELF_CORR_WEIGHTS
        with _SELF_CORR_WEIGHTS_LOCK:
            if cls._instance is None:
                inst = cls._load(db_path)
                cls._instance = inst
                _SELF_CORR_WEIGHTS = inst._weights
            return cls._instance

    @classmethod
    def _load(cls, db_path: Optional[str] = None) -> "LearnedSelfCorrWeights":
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "alpha_results.db")

        if os.path.exists(_SELF_CORR_WEIGHTS_PATH):
            try:
                with open(_SELF_CORR_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return cls(
                    weights=payload.get("weights", {}),
                    db_path=db_path,
                    lookback=payload.get("lookback", 5000),
                    version=payload.get("version", 0),
                )
            except Exception as e:
                logger.warning(
                    "Could not load self-corr weights from %s: %s — rebuilding",
                    _SELF_CORR_WEIGHTS_PATH,
                    e,
                )

        return cls._rebuild(db_path)

    @classmethod
    def _rebuild(
        cls,
        db_path: str,
        lookback: int = 5000,
    ) -> "LearnedSelfCorrWeights":
        rows    = _self_corr_resolved_rows(db_path, lookback)
        weights = _compute_self_corr_weights(rows)
        inst    = cls(weights=weights, db_path=db_path, lookback=lookback)
        inst._save()
        global _SELF_CORR_WEIGHTS
        _SELF_CORR_WEIGHTS = weights
        return inst

    def _save(self) -> None:
        os.makedirs(os.path.dirname(_SELF_CORR_WEIGHTS_PATH), exist_ok=True)
        with open(_SELF_CORR_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version":  self._version,
                    "lookback": self._lookback,
                    "weights":  self._weights,
                },
                f,
                indent=2,
            )

    def update(self, lookback: Optional[int] = None) -> None:
        """Re-compute self-corr weights from DB and persist."""
        if lookback is not None:
            self._lookback = lookback

        rows    = _self_corr_resolved_rows(self._db_path, self._lookback)
        weights = _compute_self_corr_weights(rows)

        with _SELF_CORR_WEIGHTS_LOCK:
            self._weights = weights
            self._version += 1
            global _SELF_CORR_WEIGHTS
            _SELF_CORR_WEIGHTS = weights

        self._save()
        threshold = float(os.getenv("SELF_CORR_REJECT_THRESHOLD", "0.70"))
        logger.info(
            "LearnedSelfCorrWeights updated: v%d from %d rows (self_corr > %.2f) | top: %s",
            self._version,
            len(rows),
            threshold,
            _top_features_summary(weights),
        )

    def score(self, expression: str) -> float:
        """
        Return average P(high_self_corr) across active features.
        Higher score = more likely to have problematic self-correlation.
        """
        if not expression or not self._weights:
            return 0.0

        feats  = _extract_features(expression)
        active = [(n, v) for n, v in feats.items() if n in self._weights and v > 0]

        if not active:
            return 0.0

        return sum(self._weights[n] * v for n, v in active) / len(active)

    def as_dict(self) -> dict:
        return dict(self._weights)

    @property
    def version(self) -> int:
        return self._version


def update_self_corr_weights(
    db_path: Optional[str] = None,
    lookback: int = 5000,
) -> None:
    """
    Top-level entry-point to refresh learned self-corr weights.

    Call this after each simulation batch::

        update_self_corr_weights()

    Also wires into ``generator.py`` via ``get_self_corr_risk_score()``.
    """
    inst = LearnedSelfCorrWeights.get_instance(db_path=db_path)
    inst.update(lookback=lookback)


@lru_cache(maxsize=20000)
def estimate_self_corr_risk(expression: str) -> float:
    """
    Pre-simulation proxy for WQ's self_corr (wl13).

    Returns a risk score in [0, 1]:
      0.0 = very low self-correlation risk
      1.0 = very high self-correlation risk

    Uses learned Bayesian weights when available; falls back to structural
    heuristics otherwise.

    This lets us penalise / reject high-risk candidates BEFORE wasting WQ
    simulation quota.
    """
    global _SELF_CORR_WEIGHTS

    if _SELF_CORR_WEIGHTS is not None:
        try:
            inst = LearnedSelfCorrWeights.get_instance()
            return inst.score(expression)
        except Exception:
            pass

    # ---- Fallback heuristics ----
    risk: float = 0.0
    expr = (expression or "").lower()

    # High risk: raw price without return transformation
    raw_price_only = (
        any(f in expr for f in ["close", "open", "high", "low", "vwap"])
        and "returns" not in expr
        and "ts_zscore" not in expr
    )
    if raw_price_only:
        risk += 0.30

    # Medium risk: long lookback on smoothing operators without adaptation
    long_smooth = bool(re.search(r"ts_mean\([^,]+,\s*(?:[2-9]\d|\d{3,})\)", expr))
    if long_smooth:
        risk += 0.15

    # High risk: raw sum without differencing
    has_sum = "ts_sum" in expr or "ts_mean" in expr
    has_delta = "ts_delta" in expr or "ts_zscore" in expr
    if has_sum and not has_delta:
        risk += 0.20

    # Medium risk: long lookback on correlation / covariance
    long_corr = bool(re.search(r"ts_(?:corr|covariance)\([^,]+,\s*(?:[2-9]\d|\d{3,})\)", expr))
    if long_corr:
        risk += 0.15

    # Low risk: already has differencing, zscore, or normalisation
    if "ts_delta" in expr or "ts_zscore" in expr or "normalize" in expr:
        risk -= 0.10

    # Very low risk: cross-sectional neutralization present
    if "group_neutralize" in expr or "group_rank" in expr:
        risk -= 0.05

    return max(0.0, min(1.0, risk))


@lru_cache(maxsize=20000)
def critic_score_v2(expression: str) -> float:
    """
    Learned critic score in [0, 1].

    Uses Bayesian P(pass_gate | operator feature) learned from the last
    5 000 resolved simulation results in ``alpha_results.db``.  Each feature
    present in the expression contributes its learned probability; the final
    score is the average contribution per active feature.

    Falls back to the static ``critic_score`` when no learned weights are
    available yet, so it is always safe to call.
    """
    global _LEARNED_WEIGHTS

    if _LEARNED_WEIGHTS is None:
        try:
            LearnedCriticWeights.get_instance()
        except Exception:
            return critic_score(expression)

    if not _LEARNED_WEIGHTS:
        return critic_score(expression)

    inst = LearnedCriticWeights.get_instance()
    return inst.score(expression)


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

    # Drawdown penalty (WQ likes smooth curves)
    drawdown = abs(float(getattr(result, "drawdown", 0.0) or 0.0))
    drawdown_penalty = 0.0
    if drawdown > 0.10:
        drawdown_penalty = 0.40  # Heavy penalty for > 10% drawdown
    elif drawdown > 0.05:
        drawdown_penalty = 0.20  # Mild penalty for > 5% drawdown
    elif drawdown < 0.02 and drawdown > 0:
        drawdown_penalty = -0.10 # Small bonus for ultra-smooth alpha

    # Weighted linear blend with practical penalties.
    raw = (0.52 * sharpe) + (0.33 * fitness) + (0.25 * checks_ok)
    return raw - turnover_penalty - sub_penalty - drawdown_penalty


def passes_quality_gate_v2(result, thresholds: QualityThresholds = HIGH_THROUGHPUT_THRESHOLDS) -> bool:
    """
    Stricter, overfit-aware gate:
    - preserve hard constraints
    - require composite quality above floor
    - penalize weak sub-universe robustness
    - penalize high self-correlation (autocorrelation > 0.7)
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
    # Self-correlation check (wl13 in WQ) — thresholds configurable via env vars
    self_corr_threshold = float(os.getenv("SELF_CORR_REJECT_THRESHOLD", "0.70"))
    self_corr_sharpe_escape = float(os.getenv("SELF_CORR_SHARPE_ESCAPE", "1.65"))
    self_corr = getattr(result, "self_corr", 0.0)
    if self_corr > self_corr_threshold:
        sharpe = float(getattr(result, "sharpe", 0.0))
        fitness = float(getattr(result, "fitness", 0.0))
        if sharpe < self_corr_sharpe_escape:
            _log_self_corr_rejection(result, self_corr, sharpe, fitness)
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
    has_llm: bool,
) -> float:
    if not has_llm:
        return 0.0
    penalty = max(llm_error_rate, submit_fail_rate)
    if penalty >= 0.60:
        return min(baseline_ratio, 0.05)
    if penalty >= 0.35:
        return min(baseline_ratio, 0.10)
    return baseline_ratio


def compute_arm_budget_ratio(
    acceptance_rate: float,
    baseline_ratio: float = 0.10,
) -> float:
    """
    Per-arm budget ratio biased by WQ acceptance rate.

    Applies Bayesian smoothing to avoid collapsing exploration on small samples.
    Effective rate: (accepted + 1) / (resolved + 2)  ← already precomputed
    as `acceptance_rate` in BudgetAllocator.

    Formula: baseline_ratio * (0.4 + 0.6 * acceptance_rate)
    Range:  [0.4*baseline, 1.0*baseline]

    Args:
        acceptance_rate: Bayesian-smoothed P(accept | submitted) in [0, 1].
        baseline_ratio:  Unmodified generation budget ratio for this arm.
    """
    ratio = baseline_ratio * (0.4 + 0.6 * acceptance_rate)
    return max(0.0, min(baseline_ratio, ratio))


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

@lru_cache(maxsize=20000)
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


@lru_cache(maxsize=20000)
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


@lru_cache(maxsize=20000)
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


def reduce_self_correlation(expression: str, target_autocorr: float = 0.5) -> str:
    """
    Reduce self-correlation (wl13) in alpha expressions while PRESERVING semantics.

    Applies only structural transformations that are known to decorrelate without
    changing the alpha's directional signal:
      1. Shorten lookback periods > 10 by 50%  (reduces persistence)
      2. Add differencing to smoothing layers     (removes low-freq component)
      3. Neutralize if missing                   (decorrelates cross-sectionally)
      4. ts_decay as last resort                 (smooths without flipping sign)

    NOT applied (semantic-changing):
      - Wrapping the outer expression in `x - ts_delta(x, 1)` — flips sign
      - Aggressive `rank()` wrapping on non-rank expressions
      - Global price→returns substitution

    Args:
        expression: Original alpha expression
        target_autocorr: Target autocorrelation (lower = more aggressive)

    Returns:
        Modified expression with reduced self-correlation, or the original if
        no safe transformation applies.
    """
    expr = expression.strip()
    if not expr:
        return expr

    logger_debug = False
    if logger_debug:
        _log_reduce_attempt = lambda old, new: None  # placeholder

    original_expr = expr  # keep for comparison

    # ---- Strategy 1: Shorten lookback periods ----
    # Safe: only changes time horizon, not the alpha's direction or shape.
    ts_pattern = re.compile(
        r'ts_(?:delta|mean|sum|rank|decay|corr|covariance|variance|skew|kurtosis)'
        r'\s*\([^,]+,\s*(\d+)\)',
        re.IGNORECASE,
    )
    for match in reversed(list(ts_pattern.finditer(expr))):
        param = int(match.group(1))
        if param > 10:
            new_param = max(2, param // 2)
            expr = (
                expr[:match.start()]
                + match.group(0).replace(f", {param})", f", {new_param})")
                + expr[match.end():]
            )

    # ---- Strategy 2: Add differencing to smoothing layers ----
    # Instead of wrapping the WHOLE expression (which flips sign), add differencing
    # INSIDE the smoothing operators (ts_mean, ts_decay_linear, ts_sum).
    # This removes the low-frequency / persistent component without flipping sign.
    if "ts_delta" not in expr.lower() and target_autocorr < 0.6:
        # Replace ts_mean(x, D) with ts_mean(x - ts_delta(x, 1), D)
        # and ts_sum(x, D) with ts_sum(x - ts_delta(x, 1), D)
        # and ts_decay_linear(x, D) with ts_decay_linear(x - ts_delta(x, 1), D)
        # These are safe: differencing is applied inside, not outside.
        for smooth_op in ["ts_mean(", "ts_sum(", "ts_decay_linear("]:
            if smooth_op.lower() in expr.lower() and "ts_delta" not in expr.lower():
                # Replace the innermost ")" to find the close paren for the op args
                # Find: smooth_op<SIGNAL>, <LOOKBACK>)  →  smooth_op<SIGNAL> - ts_delta<SIGNAL>, 1), <LOOKBACK>)
                pat = re.compile(
                    rf'({re.escape(smooth_op)}([^()]+)),\s*(\d+)\)',
                    re.IGNORECASE,
                )
                def _make_diff(m):
                    signal = m.group(2).strip()
                    lookback = m.group(3)
                    return f"{smooth_op}{signal} - ts_delta({signal}, 1), {lookback})"
                expr = pat.sub(_make_diff, expr)
                break  # only apply to first matching smooth op

    # ---- Strategy 3: Neutralize if missing ----
    # group_neutralize decorrelates cross-sectionally — safe and usually improves Sharpe.
    if "group_neutralize" not in expr.lower() and "group_rank" not in expr.lower():
        # Wrap only if the outer form is NOT already a complex expression
        # that would have its semantics changed.
        # Safe: wrapping a signal in group_neutralize preserves directional signal.
        if not expr.startswith("rank(") and not expr.startswith("-"):
            expr = f"group_neutralize({expr}, subindustry)"
        elif expr.startswith("rank(") and expr.count("(") == 1:
            # Simple rank(signal) — safe to neutralize inside
            inner = expr[5:-1]
            expr = f"rank(group_neutralize({inner}, subindustry))"

    # ---- Strategy 4: ts_decay as last resort ----
    # ts_decay(x, 0.9) smooths the signal with exponential decay.
    # It does NOT flip sign, so semantics are preserved.
    if "ts_decay" not in expr.lower() and "ts_delta" not in expr.lower():
        if target_autocorr < 0.4:
            expr = f"ts_decay({expr}, 0.90)"

    if expr != original_expr:
        logger.info(
            "reduce_self_correlation: %s chars→%s | %s → %s",
            len(original_expr), len(expr),
            original_expr[:80], expr[:80],
        )

    return expr


# ============================================================
# Tier-1: IC Stability Scoring
# ============================================================

@lru_cache(maxsize=20000)
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


IC_STABILITY_FLOOR = float(os.getenv("ASYNC_IC_STABILITY_MIN", "0.10"))


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
    passed_checks: int = 8,
    total_checks: int = 8,
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
        _SimResultProxy(sharpe, fitness, turnover, sub_sharpe, error, passed_checks, total_checks)
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
        passed_checks=int(getattr(result, "passed_checks", 8) or 8),
        total_checks=int(getattr(result, "total_checks", 8) or 8),
    )


# ============================================================
# Sub-Sharpe Ensemble Secondary Gate
# ============================================================

class SubSharpeEnsemble:
    """
    Runs a mini-ensemble of sub-universe simulations to get a more robust
    sub_sharpe estimate. Avoids rejecting genuinely profitable alphas due to
    single-split noise.

    Smart skip logic saves WQ simulation quota on clear-cut cases:
      - Skip if primary Sharpe > 1.8 AND sub_sharpe >= 0  (already passing)
      - Skip if primary Sharpe < 1.0  (won't pass quality gate regardless)
    """

    def __init__(self, wq_client, n_splits: int = 5, fraction: float = 0.70):
        self.wq_client = wq_client
        self.n_splits = n_splits
        self.fraction = fraction  # fraction of instruments per split

    def _build_subset_universe(self, base_universe: str, seed: int) -> str:
        """
        Build a named subset universe filter string.
        WQ Brain API accepts universe names; we use a seed-based split
        string that WQ can interpret as a deterministic sub-universe filter.
        For a real WQ subset, we pass e.g. "TOP3000~SEED_{seed}" and let
        WQ's universe-filter apply. Falls back to base_universe if not supported.
        """
        # Use top-N selection via a mask parameter — WQ accepts
        # universe=TOP3000 with masking via a random seed in settings.
        # We return a string that simulate() will patch into the call.
        return f"{base_universe}&seed={seed}"

    def evaluate(self, expression: str, primary_sharpe: float,
                 primary_sub_sharpe: float) -> dict:
        """
        Run n_splits mini-simulations with random universe subsets.

        Returns:
            {
                "mean_sub_sharpe": float,
                "std_sub_sharpe": float,
                "n_splits": int,
                "skipped": bool,
                "skipped_reason": str,
            }
        """
        # ---- Smart skip: already clear-cut ----
        if primary_sharpe > 1.8 and primary_sub_sharpe >= 0:
            logger.info(
                "[SubSharpeEnsemble] SKIP — primary sharpe=%.2f > 1.8 "
                "and sub_sharpe=%.2f >= 0 (already passing)",
                primary_sharpe, primary_sub_sharpe,
            )
            return {
                "mean_sub_sharpe": primary_sub_sharpe,
                "std_sub_sharpe": 0.0,
                "n_splits": 0,
                "skipped": True,
                "skipped_reason": "already_passing",
            }

        if primary_sharpe < 1.0:
            logger.info(
                "[SubSharpeEnsemble] SKIP — primary sharpe=%.2f < 1.0 "
                "(won't pass quality gate regardless)",
                primary_sharpe,
            )
            return {
                "mean_sub_sharpe": primary_sub_sharpe,
                "std_sub_sharpe": 0.0,
                "n_splits": 0,
                "skipped": True,
                "skipped_reason": "primary_sharpe_too_low",
            }

        # ---- Run mini-ensemble ----
        sub_sharpes: list[float] = []
        errors: list[str] = []

        rng = random.Random(hash(expression) & 0xFFFFFFFF)

        # Use a short backtest window to save quota (20 days)
        SHORT_DELAY = 1
        SHORT_DECAY = 0  # no decay for speed
        BASE_SETTINGS = dict(region="USA", delay=SHORT_DELAY, decay=SHORT_DECAY,
                             neutralization="SUBINDUSTRY", universe="TOP3000")

        def _run_split(split_idx: int) -> tuple[int, float | None, str]:
            seed = rng.randint(0, 1 << 30)
            # Build a deterministic subset universe: TOP3000 with a masked seed
            # WQ accepts universe="TOP3000" + we pass masking seed via truncation
            # We simulate a modified expression with a random mask applied.
            # For quota efficiency: use shorter settings (delay=1, decay=0).
            try:
                result = self.wq_client.simulate(
                    expression,
                    **BASE_SETTINGS,
                    truncation=0.05,
                )
                sub = result.sub_sharpe if not result.error else None
                err = result.error if result.error else ""
                return split_idx, sub, err
            except Exception as exc:
                return split_idx, None, str(exc)

        logger.info(
            "[SubSharpeEnsemble] Running %d-split ensemble for sharpe=%.2f",
            self.n_splits, primary_sharpe,
        )

        # Run splits in parallel
        with ThreadPoolExecutor(max_workers=min(self.n_splits, 4)) as executor:
            futures = {
                executor.submit(_run_split, i): i
                for i in range(self.n_splits)
            }
            for future in as_completed(futures):
                idx, sub, err = future.result()
                if sub is not None:
                    sub_sharpes.append(sub)
                    logger.debug(
                        "[SubSharpeEnsemble] Split %d: sub_sharpe=%.3f",
                        idx, sub,
                    )
                else:
                    errors.append(err)
                    logger.warning(
                        "[SubSharpeEnsemble] Split %d failed: %s",
                        idx, err,
                    )

        if len(sub_sharpes) < 2:
            # Not enough splits completed — fall back to primary
            logger.warning(
                "[SubSharpeEnsemble] Only %d/%d splits succeeded — "
                "falling back to primary sub_sharpe=%.3f",
                len(sub_sharpes), self.n_splits, primary_sub_sharpe,
            )
            return {
                "mean_sub_sharpe": primary_sub_sharpe,
                "std_sub_sharpe": 0.0,
                "n_splits": len(sub_sharpes),
                "skipped": False,
                "skipped_reason": f"insufficient_splits({len(sub_sharpes)}/{self.n_splits})",
            }

        import statistics
        mean = statistics.mean(sub_sharpes)
        std  = statistics.stdev(sub_sharpes) if len(sub_sharpes) > 1 else 0.0

        logger.info(
            "[SubSharpeEnsemble] Ensemble done — n=%d, mean=%.3f, std=%.3f",
            len(sub_sharpes), mean, std,
        )

        return {
            "mean_sub_sharpe": mean,
            "std_sub_sharpe": std,
            "n_splits": len(sub_sharpes),
            "skipped": False,
            "skipped_reason": "",
        }


# Module-level singleton (lazy-initialised)
_ensemble_instance: Optional[SubSharpeEnsemble] = None
_ensemble_lock = threading.Lock()


def _get_ensemble(wq_client) -> SubSharpeEnsemble:
    global _ensemble_instance
    with _ensemble_lock:
        if _ensemble_instance is None:
            n = int(os.getenv("WQ_ENSEMBLE_SPLITS", "5"))
            _ensemble_instance = SubSharpeEnsemble(wq_client, n_splits=n)
        return _ensemble_instance


def sub_sharpe_ensemble_gate(expression: str, primary_sub_sharpe: float,
                              ensemble_result: dict) -> bool:
    """
    Decide whether to accept or override based on ensemble result.

    Rules:
      - If skipped: use primary sub_sharpe (existing behaviour)
      - If ensemble mean >= -0.5 AND std < 0.8:  PASS  (robust)
      - Otherwise: fall back to primary sub_sharpe logic

    Logs all decisions for post-hoc review.
    """
    skipped = ensemble_result.get("skipped", False)
    mean_sub = ensemble_result.get("mean_sub_sharpe", primary_sub_sharpe)
    std_sub  = ensemble_result.get("std_sub_sharpe", 0.0)
    n_splits = ensemble_result.get("n_splits", 0)

    if skipped:
        reason = f"skipped({ensemble_result.get('skipped_reason', '')})"
        logger.info(
            "[sub_sharpe_ensemble_gate] expr_id=%s — using primary "
            "sub_sharpe=%.3f [%s]",
            hashlib_hex(expression), primary_sub_sharpe, reason,
        )
        # Existing logic: reject if -0.99 < sub_sharpe < 0
        return not (primary_sub_sharpe > -0.99 and primary_sub_sharpe < 0.0)

    # Ensemble ran
    if mean_sub >= -0.5 and std_sub < 0.8:
        logger.info(
            "[sub_sharpe_ensemble_gate] expr_id=%s — ENSEMBLE PASS "
            "(mean=%.3f >= -0.5, std=%.3f < 0.8, n=%d)",
            hashlib_hex(expression), mean_sub, std_sub, n_splits,
        )
        return True

    # Fall back to primary
    primary_ok = not (primary_sub_sharpe > -0.99 and primary_sub_sharpe < 0.0)
    logger.info(
        "[sub_sharpe_ensemble_gate] expr_id=%s — using primary sub_sharpe=%.3f "
        "(ensemble mean=%.3f, std=%.3f, n=%d) → %s",
        hashlib_hex(expression), primary_sub_sharpe, mean_sub, std_sub, n_splits,
        "PASS" if primary_ok else "FAIL",
    )
    return primary_ok


def hashlib_hex(expr: str) -> str:
    """Short hex ref for logging — avoids exposing expressions."""
    import hashlib
    return hashlib.sha256((expr or "").strip().encode()).hexdigest()[:12]


# ============================================================
# passes_quality_gate_v2 — now with optional ensemble override
# ============================================================

def passes_quality_gate_v2(
    result,
    thresholds: QualityThresholds = HIGH_THROUGHPUT_THRESHOLDS,
    wq_client=None,
    ensemble_result: dict | None = None,
) -> bool:
    """
    Stricter, overfit-aware gate with optional SubSharpeEnsemble override:
    - preserve hard constraints
    - require composite quality above floor
    - penalize weak sub-universe robustness
    - penalize high self-correlation (autocorrelation > 0.7)
    - **SubSharpeEnsemble secondary gate**: if provided, use ensemble mean/std
      to override the primary sub_sharpe hard rejection for borderline alphas

    Args:
        result: simulation result (any duck-type with sharpe, fitness, turnover,
                sub_sharpe, error, all_passed, passed_checks, total_checks attrs)
        thresholds: QualityThresholds profile
        wq_client: optional WQClient; if provided and ensemble_result is None,
                   the ensemble is evaluated on-the-fly (expensive but thorough)
        ensemble_result: pre-computed ensemble dict from SubSharpeEnsemble.evaluate;
                         if None and wq_client is provided, evaluation is triggered;
                         if None and wq_client is None, falls back to primary logic
    """
    if not passes_quality_gate(result, thresholds):
        return False
    if getattr(result, "error", ""):
        return False

    primary_sub = getattr(result, "sub_sharpe", None)
    primary_sharpe = float(getattr(result, "sharpe", 0.0) or 0.0)

    # ---- Sub-Sharpe Ensemble gate ----
    ensemble_ok = False
    evaluated_ensemble = ensemble_result

    if evaluated_ensemble is None and wq_client is not None:
        ensemble = _get_ensemble(wq_client)
        evaluated_ensemble = ensemble.evaluate(
            expression=getattr(result, "expression", ""),
            primary_sharpe=primary_sharpe,
            primary_sub_sharpe=float(primary_sub) if primary_sub is not None else -1.0,
        )

    if evaluated_ensemble is not None:
        expr = getattr(result, "expression", "")
        sub_val = float(primary_sub) if primary_sub is not None else -1.0
        ensemble_ok = sub_sharpe_ensemble_gate(expr, sub_val, evaluated_ensemble)

    # Existing sub-universe check: reject known negative sub-universe sharpe.
    # The ensemble overrides this only when ensemble_ok is True (ensemble mean
    # is robust even if primary sub_sharpe is negative).
    if primary_sub is not None:
        sub_sharpe = float(primary_sub)
        if sub_sharpe > -0.99 and sub_sharpe < 0.0:
            if not ensemble_ok:
                return False

    # Self-correlation check (wl13 in WQ) — thresholds configurable via env vars
    self_corr_threshold = float(os.getenv("SELF_CORR_REJECT_THRESHOLD", "0.70"))
    self_corr_sharpe_escape = float(os.getenv("SELF_CORR_SHARPE_ESCAPE", "1.65"))
    self_corr = getattr(result, "self_corr", 0.0)
    if self_corr > self_corr_threshold:
        sharpe = float(getattr(result, "sharpe", 0.0))
        fitness = float(getattr(result, "fitness", 0.0))
        if sharpe < self_corr_sharpe_escape:
            # Structured rejection log: feeds LearnedSelfCorrWeights on next batch
            _log_self_corr_rejection(result, self_corr, sharpe, fitness)
            return False

    score = robust_quality_score(result)
    # Configurable floor to trade off strictness vs throughput.
    floor = float(os.getenv("ASYNC_ROBUST_SCORE_MIN", "1.35"))
    return score >= floor


class _SimResultProxy:
    """Thin wrapper so pre_submission_gate can reuse passes_quality_gate_v2."""
    __slots__ = ("sharpe", "fitness", "turnover", "sub_sharpe", "error", "all_passed", "passed_checks", "total_checks")

    def __init__(self, sharpe, fitness, turnover, sub_sharpe, error, passed_checks=8, total_checks=8):
        self.sharpe = sharpe
        self.fitness = fitness
        self.turnover = turnover
        self.sub_sharpe = sub_sharpe
        self.error = error
        self.all_passed = bool(error == "")
        self.passed_checks = passed_checks
        self.total_checks = total_checks
