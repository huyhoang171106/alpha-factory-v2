"""
local_backtest.py — Local Pre-Simulation Filter
Pre-screens alpha expressions using real market data (yfinance)
BEFORE consuming WQ Brain quota.

Goal: eliminate ~60% of bad candidates locally → higher WQ Brain pass rate.

Key design:
- Download S&P 500 OHLCV data (500 stocks, 3 years)
- Implement core FASTEXPR operators in pandas
- Compute proxy Sharpe, Turnover, Drawdown
- Score each expression 0-100 for pre-filtering

Usage:
    from local_backtest import LocalBacktester
    bt = LocalBacktester()
    results = bt.backtest_batch(expressions, top_n=15)
"""

import os
import re
import time
import hashlib
import logging
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".bt_cache")
CACHE_TTL_DAYS = 7  # refresh market data every 7 days
BT_LOOKBACK_YEARS = 2  # 2 years of backtest
N_STOCKS = 100  # top 100 S&P500 by market cap (fast + representative)
MIN_SHARPE_LOCAL = 0.4  # local Sharpe pre-filter threshold (lenient than WQ)
MAX_TURNOVER_LOCAL = 0.85  # turnover limit (WQ penalizes >80%)
MIN_FITNESS_LOCAL = 0.8  # fitness proxy threshold
MAX_EVAL_TIMEOUT = 5.0  # seconds per expression evaluation


@dataclass
class LocalBTResult:
    expression: str
    sharpe: float = 0.0
    turnover: float = 0.0
    drawdown: float = 0.0
    fitness: float = 0.0
    score: float = 0.0  # composite 0-100
    passed: bool = False
    error: str = ""

    @property
    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} | Sharpe={self.sharpe:.2f} Turnover={self.turnover:.1%} "
            f"Drawdown={self.drawdown:.1%} Score={self.score:.0f}"
        )


# ============================================================
# Market Data Manager
# ============================================================
class MarketData:
    """Download and cache S&P 500 OHLCV data via yfinance"""

    def __init__(self, cache_dir: str = DATA_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._data: Optional[Dict[str, pd.DataFrame]] = None

    def _get_sp500_tickers(self) -> List[str]:
        """Get top N S&P 500 tickers (hardcoded liquid set for speed)"""
        # Top 100 most liquid S&P 500 stocks (stable universe)
        return [
            "AAPL",
            "MSFT",
            "AMZN",
            "NVDA",
            "GOOGL",
            "META",
            "TSLA",
            "BRK-B",
            "UNH",
            "LLY",
            "JPM",
            "V",
            "XOM",
            "MA",
            "JNJ",
            "PG",
            "HD",
            "AVGO",
            "CVX",
            "MRK",
            "ABBV",
            "COST",
            "PEP",
            "BAC",
            "KO",
            "ADBE",
            "CRM",
            "MCD",
            "TMO",
            "WMT",
            "CSCO",
            "ABT",
            "NFLX",
            "ORCL",
            "ACN",
            "DIS",
            "TXN",
            "VZ",
            "INTC",
            "AMD",
            "PM",
            "AMGN",
            "QCOM",
            "HON",
            "IBM",
            "CAT",
            "GS",
            "RTX",
            "UPS",
            "AXP",
            "SPGI",
            "BKNG",
            "LOW",
            "AMAT",
            "MDT",
            "SBUX",
            "NEE",
            "MS",
            "DE",
            "GILD",
            "LMT",
            "CI",
            "C",
            "BLK",
            "NOW",
            "ISRG",
            "SYK",
            "MDLZ",
            "MMC",
            "ZTS",
            "CVS",
            "MO",
            "ADP",
            "TJX",
            "GE",
            "PLD",
            "EOG",
            "SO",
            "CME",
            "D",
            "HCA",
            "NOC",
            "ITW",
            "BDX",
            "EW",
            "AON",
            "REGN",
            "PGR",
            "MCO",
            "WM",
            "APD",
            "MMM",
            "PSA",
            "CCI",
            "KLAC",
            "LRCX",
            "NSC",
            "AEP",
            "TEL",
            "EL",
            "CTSH",
            "PANW",
            "FIS",
            "DXCM",
        ]

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, "market_data.parquet")

    def _is_cache_valid(self) -> bool:
        # Check for at least one of the field parquet files
        sample_path = os.path.join(self.cache_dir, "close.parquet")
        if not os.path.exists(sample_path):
            return False
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(sample_path))
        return age.days < CACHE_TTL_DAYS

    def load(self) -> Dict[str, pd.DataFrame]:
        """Load market data (from cache or download)"""
        if self._data is not None:
            return self._data

        if self._is_cache_valid():
            logger.info("📦 Loading market data from cache...")
            return self._load_from_cache()

        logger.info("📡 Downloading market data (first time, ~30s)...")
        return self._download_and_cache()

    def _download_and_cache(self) -> Dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        tickers = self._get_sp500_tickers()
        end = datetime.now()
        start = end - timedelta(days=365 * BT_LOOKBACK_YEARS + 60)

        logger.info(f"  Downloading {len(tickers)} stocks...")
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        # Build per-field DataFrames (stocks as columns)
        data = {}
        for field_name in ["Open", "High", "Low", "Close", "Volume"]:
            if field_name in raw.columns.get_level_values(0):
                df = raw[field_name].dropna(how="all", axis=1)
                data[field_name.lower()] = df
            else:
                # yfinance sometimes returns flat columns for single ticker
                if field_name.lower() in raw.columns:
                    data[field_name.lower()] = raw[[field_name.lower()]]

        # Derived fields
        if "close" in data and "volume" in data:
            # VWAP approximation (daily OHLCV trick)
            hl2 = (data.get("high", data["close"]) + data.get("low", data["close"])) / 2
            data["vwap"] = hl2
            data["returns"] = data["close"].pct_change()
            data["adv20"] = data["volume"].rolling(20).mean()
            data["adv60"] = data["volume"].rolling(60).mean()
            data["adv120"] = data["volume"].rolling(120).mean()

        # Save cache
        # Store as separate parquets per field
        for fname, df in data.items():
            path = os.path.join(self.cache_dir, f"{fname}.parquet")
            df.to_parquet(path)

        logger.info(
            f"  ✅ Downloaded {data['close'].shape[1]} stocks, {data['close'].shape[0]} days"
        )
        self._data = data
        return data

    def _load_from_cache(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for fname in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "returns",
            "adv20",
            "adv60",
            "adv120",
        ]:
            path = os.path.join(self.cache_dir, f"{fname}.parquet")
            if os.path.exists(path):
                data[fname] = pd.read_parquet(path)
        self._data = data
        return data


# ============================================================
# FASTEXPR Operator Implementations (Pandas)
# ============================================================
class FastExprEval:
    """
    Evaluate simplified FASTEXPR expressions using pandas DataFrames.
    Supports the most common WQ Brain operators.
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        # Pre-compute cross-sectional rank helper
        self._rank_cache: Dict[str, pd.DataFrame] = {}

    def _cs_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank (0-1) across stocks each day"""
        return df.rank(axis=1, pct=True)

    def evaluate(self, expr: str) -> Optional[pd.DataFrame]:
        """
        Evaluate a FASTEXPR expression → returns daily position DataFrame.
        Returns None if evaluation fails or times out.
        """
        try:
            return self._eval(expr)
        except Exception as e:
            logger.debug(f"Eval error for {expr[:40]}: {e}")
            return None

    def _eval(self, expr: str) -> pd.DataFrame:
        """Recursive expression evaluator"""
        expr = expr.strip()

        # ── Primitives ────────────────────────────────────────────
        primitives = {
            "close": self.data.get("close"),
            "open": self.data.get("open"),
            "high": self.data.get("high"),
            "low": self.data.get("low"),
            "volume": self.data.get("volume"),
            "vwap": self.data.get("vwap"),
            "returns": self.data.get("returns"),
            "adv20": self.data.get("adv20"),
            "adv60": (
                self.data.get("adv60")
                if self.data.get("adv60") is not None
                else self.data.get("adv20")
            ),
            "adv120": (
                self.data.get("adv120")
                if self.data.get("adv120") is not None
                else (
                    self.data.get("adv60")
                    if self.data.get("adv60") is not None
                    else self.data.get("adv20")
                )
            ),
        }
        if expr in primitives and primitives[expr] is not None:
            return primitives[expr].copy()

        # ── Numeric literal ────────────────────────────────────────
        try:
            val = float(expr)
            ref = self.data["close"]
            return pd.DataFrame(val, index=ref.index, columns=ref.columns)
        except ValueError:
            pass

        # ── Parse function call ────────────────────────────────────
        m = re.match(r"^(-?)(\w+)\((.+)\)$", expr, re.DOTALL)
        if not m:
            # Binary operation: try to split on top-level * / + -
            return self._eval_binary(expr)

        sign, func, args_str = m.group(1), m.group(2), m.group(3)
        args = self._split_args(args_str)

        result = self._apply_func(func, args)

        if sign == "-" and result is not None:
            result = -result
        return result

    def _apply_func(self, func: str, args: List[str]) -> Optional[pd.DataFrame]:
        """Apply a FASTEXPR function"""
        func = func.lower()

        def arg(i):
            return self._eval(args[i].strip())

        def n(i):
            return int(float(args[i].strip()))

        try:
            # ── Cross-sectional ───
            if func in ("rank",):
                return self._cs_rank(arg(0))
            if func == "zscore":
                a = arg(0)
                return (a - a.mean(axis=1).values.reshape(-1, 1)) / (
                    a.std(axis=1).values.reshape(-1, 1) + 1e-8
                )
            if func == "sigmoid":
                a = arg(0)
                return 1 / (1 + np.exp(-a.clip(-20, 20)))
            if func == "sign":
                return np.sign(arg(0))
            if func == "abs":
                return arg(0).abs()
            if func == "log":
                return np.log(arg(0).clip(lower=1e-8))
            if func == "scale":
                a = arg(0)
                return a.div(a.abs().sum(axis=1), axis=0)
            if func in ("power", "pow"):
                base = arg(0)
                exp = float(args[1].strip())
                return base.clip(lower=0) ** exp

            # ── Time-series ───
            if func in ("ts_delta", "ts_diff"):
                return arg(0).diff(n(1))
            if func in ("ts_delay", "ts_lag"):
                return arg(0).shift(n(1))
            if func == "ts_mean":
                return arg(0).rolling(n(1), min_periods=max(1, n(1) // 2)).mean()
            if func == "ts_sum":
                return arg(0).rolling(n(1), min_periods=1).sum()
            if func in ("ts_std_dev", "ts_std"):
                return arg(0).rolling(n(1), min_periods=max(2, n(1) // 2)).std()
            if func == "ts_min":
                return arg(0).rolling(n(1), min_periods=1).min()
            if func == "ts_max":
                return arg(0).rolling(n(1), min_periods=1).max()
            if func in ("ts_rank", "ts_r"):
                d = n(1)
                return (
                    arg(0)
                    .rolling(d, min_periods=max(1, d // 2))
                    .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
                )
            if func == "ts_zscore":
                a = arg(0)
                d = n(1)
                mu = a.rolling(d, min_periods=max(1, d // 2)).mean()
                sigma = a.rolling(d, min_periods=max(2, d // 2)).std().clip(lower=1e-8)
                return (a - mu) / sigma
            if func in ("ts_corr", "ts_correlation"):
                a, b, d = arg(0), arg(1), n(2)
                return a.rolling(d, min_periods=max(3, d // 2)).corr(b)
            if func in ("ts_covariance", "ts_cov"):
                a, b, d = arg(0), arg(1), n(2)
                return a.rolling(d, min_periods=max(3, d // 2)).cov(b)
            if func == "ts_decay_linear":
                a, d = arg(0), n(1)
                weights = np.arange(1, d + 1, dtype=float)
                weights /= weights.sum()
                return a.rolling(d, min_periods=max(1, d // 2)).apply(
                    lambda x: (
                        np.dot(x[-len(weights) :], weights[-len(x) :])
                        / weights[-len(x) :].sum()
                    ),
                    raw=True,
                )
            if func == "ts_skewness":
                return arg(0).rolling(n(1), min_periods=max(3, n(1) // 2)).skew()
            if func in ("ts_arg_max",):
                d = n(1)
                return arg(0).rolling(d).apply(np.argmax, raw=True)
            if func in ("ts_arg_min",):
                d = n(1)
                return arg(0).rolling(d).apply(np.argmin, raw=True)
            if func == "ts_product":
                return arg(0).rolling(n(1), min_periods=1).apply(np.prod, raw=True)

            # ── Group ops (approximated — all stocks as one group) ───
            if func in ("group_neutralize",):
                a = arg(0)
                # Approximate: subtract cross-sectional mean each day
                daily_mean = a.mean(axis=1)
                return a.sub(daily_mean, axis=0)
            if func in ("group_rank",):
                return self._cs_rank(arg(0))
            if func in ("group_mean",):
                a = arg(0)
                daily_mean = a.mean(axis=1)
                return pd.DataFrame(
                    daily_mean.values.reshape(-1, 1) * np.ones((1, a.shape[1])),
                    index=a.index,
                    columns=a.columns,
                )
            if func in ("group_zscore",):
                a = arg(0)
                mu = a.mean(axis=1)
                sigma = a.std(axis=1).clip(lower=1e-8)
                return a.sub(mu, axis=0).div(sigma, axis=0)

            # ── Math ops ───
            if func == "min":
                return arg(0).combine(arg(1), np.minimum)
            if func == "max":
                return arg(0).combine(arg(1), np.maximum)
            if func == "floor":
                return arg(0).apply(np.floor)
            if func == "pasteurize":
                return arg(0)  # no-op in bt context
            if func in ("signed_power",):
                a, e = arg(0), float(args[1].strip())
                return np.sign(a) * (a.abs() ** e)

            logger.debug(f"Unknown func: {func}")
            return None

        except Exception as e:
            logger.debug(f"func={func} error: {e}")
            return None

    def _eval_binary(self, expr: str) -> Optional[pd.DataFrame]:
        """Evaluate binary operations: A op B"""
        # Find top-level binary operators
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ")":
                depth += 1
            elif expr[i] == "(":
                depth -= 1
            elif depth == 0 and expr[i] in ("+", "-", "*", "/"):
                left = expr[:i].strip()
                right = expr[i + 1 :].strip()
                op = expr[i]
                if left and right:
                    try:
                        L = self._eval(left)
                        R = self._eval(right)
                        if L is None or R is None:
                            return None
                        if op == "+":
                            return L + R
                        if op == "-":
                            return L - R
                        if op == "*":
                            return L * R
                        if op == "/":
                            return L / (R.replace(0, np.nan).clip(lower=1e-8))
                    except Exception:
                        return None
        return None

    def _split_args(self, args_str: str) -> List[str]:
        """Split function args at top-level commas"""
        args = []
        depth = 0
        current = ""
        for ch in args_str:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                args.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current)
        return args


# ============================================================
# Backtester
# ============================================================
class LocalBacktester:
    """
    Pre-simulate alpha expressions locally before using WQ Brain quota.

    Strategy:
        1. Load market data (from cache or download)
        2. Evaluate each expression to get daily position signals
        3. Simulate portfolio PnL from positions
        4. Compute Sharpe, Turnover, Drawdown
        5. Score each expression 0-100
    """

    def __init__(self):
        self.market = MarketData()
        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._evaluator: Optional[FastExprEval] = None

    def _init(self):
        if self._data is None:
            self._data = self.market.load()
            self._evaluator = FastExprEval(self._data)

    def _simulate_positions(self, positions: pd.DataFrame) -> Dict[str, float]:
        """
        Simulate portfolio returns from daily positions (cross-sectional).
        Positions = daily signal (ranked or raw), normalized to sum=1 long/short.
        """
        try:
            returns = self._data.get("returns")
            if returns is None or positions is None:
                return {}

            # Align
            common_idx = positions.index.intersection(returns.index)
            common_cols = positions.columns.intersection(returns.columns)
            if len(common_idx) < 60 or len(common_cols) < 10:
                return {}

            pos = positions.loc[common_idx, common_cols].copy()
            ret = returns.loc[common_idx, common_cols].copy()

            # Normalize positions to be dollar-neutral (long-short)
            pos = pos.sub(pos.mean(axis=1), axis=0)
            pos_abs = pos.abs().sum(axis=1)
            pos = pos.div(pos_abs.clip(lower=1e-6), axis=0)

            # Daily PnL = sum(position_i * return_i)
            pnl = (pos.shift(1) * ret).sum(axis=1)  # shift 1 for lag-1 execution
            pnl = pnl.dropna()

            if len(pnl) < 20:
                return {}

            # Sharpe (annualized, 252 trading days)
            sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)

            # Turnover (daily average % of portfolio traded)
            pos_diff = pos.diff().abs().sum(axis=1)
            turnover = pos_diff.mean()

            # Max Drawdown
            cumret = (1 + pnl).cumprod()
            rolling_max = cumret.cummax()
            drawdown = ((cumret - rolling_max) / rolling_max).min()

            # Fitness proxy (Calmar ratio)
            annual_ret = pnl.mean() * 252
            fitness = annual_ret / (abs(drawdown) + 1e-4)

            return {
                "sharpe": float(sharpe),
                "turnover": float(turnover),
                "drawdown": float(drawdown),
                "fitness": float(fitness),
                "n_days": len(pnl),
            }
        except Exception as e:
            logger.debug(f"Simulation error: {e}")
            return {}

    def _score(
        self, sharpe: float, turnover: float, drawdown: float, fitness: float
    ) -> float:
        """Composite score 0-100 for pre-filtering"""
        score = 50.0

        # Sharpe contribution (most important)
        if sharpe > 1.5:
            score += 30
        elif sharpe > 1.0:
            score += 20
        elif sharpe > 0.6:
            score += 10
        elif sharpe > 0.3:
            score += 0
        else:
            score -= 20

        # Turnover (WQ Brain penalizes very high turnover)
        if turnover < 0.10:
            score += 10
        elif turnover < 0.30:
            score += 5
        elif turnover < 0.60:
            score += 0
        elif turnover < 0.85:
            score -= 10
        else:
            score -= 25

        # Drawdown
        if drawdown > -0.10:
            score += 5
        elif drawdown > -0.20:
            score += 0
        else:
            score -= 10

        # Fitness (risk-adjusted return)
        if fitness > 2.0:
            score += 10
        elif fitness > 1.0:
            score += 5

        return max(0.0, min(100.0, score))

    def backtest_single(self, expr: str) -> LocalBTResult:
        """Backtest a single expression"""
        self._init()
        result = LocalBTResult(expression=expr)

        try:
            positions = self._evaluator.evaluate(expr)
            if positions is None or positions.isnull().all().all():
                result.error = "Eval returned None or all-NaN"
                return result

            metrics = self._simulate_positions(positions)
            if not metrics:
                result.error = "Simulation failed"
                return result

            result.sharpe = metrics.get("sharpe", 0)
            result.turnover = metrics.get("turnover", 0)
            result.drawdown = metrics.get("drawdown", 0)
            result.fitness = metrics.get("fitness", 0)
            result.score = self._score(
                result.sharpe, result.turnover, result.drawdown, result.fitness
            )
            result.passed = (
                result.sharpe >= MIN_SHARPE_LOCAL
                and result.turnover <= MAX_TURNOVER_LOCAL
            )
        except Exception as e:
            result.error = str(e)

        return result

    def backtest_batch(
        self,
        expressions: List[str],
        top_n: int = 15,
        min_score: float = 45.0,
    ) -> List[str]:
        """
        Backtest a batch of expressions and return top candidates.

        Args:
            expressions: list of FASTEXPR strings
            top_n: max to return
            min_score: minimum local score to pass

        Returns:
            Filtered list of expressions sorted by local score (desc)
        """
        self._init()
        logger.info(f"🔬 Local backtest: {len(expressions)} expressions...")

        results = []
        passed = 0
        failed = 0
        errors = 0

        for i, expr in enumerate(expressions, 1):
            res = self.backtest_single(expr)
            results.append(res)
            if res.error:
                errors += 1
            elif res.passed:
                passed += 1
            else:
                failed += 1

            if i % 10 == 0:
                logger.info(
                    f"  Progress: {i}/{len(expressions)} "
                    f"| pass={passed} fail={failed} err={errors}"
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        top_results = [r for r in results if r.score >= min_score][:top_n]

        logger.info(
            f"  ✅ Local BT done: {passed}/{len(expressions)} pass "
            f"| Top {len(top_results)} selected (min_score={min_score})"
        )

        if results:
            top = results[0]
            logger.info(
                f"  🏆 Best local: Sharpe={top.sharpe:.2f} "
                f"Turnover={top.turnover:.1%} Score={top.score:.0f}"
            )

        return [r.expression for r in top_results]

    def show_stats(self, expressions: List[str], limit: int = 10):
        """Show detailed backtest stats for a list of expressions"""
        self._init()
        print(f"\n{'=' * 70}")
        print(f"{'#':3} {'Score':6} {'Sharpe':7} {'Turn':8} {'Draw':8}  Expression")
        print(f"{'=' * 70}")

        results = []
        for expr in expressions:
            res = self.backtest_single(expr)
            results.append(res)

        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results[:limit], 1):
            status = "✅" if r.passed else "❌"
            print(
                f"{i:3}. {status} [{r.score:5.1f}] "
                f"S={r.sharpe:+.2f} T={r.turnover:.0%} D={r.drawdown:.0%} "
                f" {r.expression[:55]}"
            )
        print(f"{'=' * 70}\n")


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )

    bt = LocalBacktester()

    test_exprs = [
        # Expected: decent Sharpe
        "-rank(ts_delta(close, 5))",
        "rank(volume / adv20) * rank(returns)",
        "group_neutralize(rank(ts_mean(returns, 20)), sector)",  # approx
        "rank(ts_corr(returns, ts_delta(volume, 1), 10)) * rank(ts_mean(returns, 20))",
        "-rank(ts_mean(abs(returns) / (volume + 1), 20))",
        "rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
        "rank(ts_std_dev(returns, 5)) * (-rank(ts_delta(close, 5)))",
        # Expected: low quality
        "rank(close)",
        "rank(volume)",
        "-rank(ts_mean(returns, 3))",
    ]

    print("🔬 Local Backtest Test")
    bt.show_stats(test_exprs)

    top = bt.backtest_batch(test_exprs, top_n=5, min_score=45.0)
    print(f"\n✅ Top {len(top)} candidates after local filter:")
    for e in top:
        print(f"  {e}")
