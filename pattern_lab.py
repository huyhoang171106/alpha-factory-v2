"""
pattern_lab.py - Deterministic pattern memory for continuous alpha evolution.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sqlite3
from collections import Counter
from typing import Optional

from alpha_ast import (
    canonicalize_expression,
    operator_set as ast_operator_set,
    parameter_agnostic_signature,
)

logger = logging.getLogger(__name__)


PATTERN_LAB_PATH = os.path.join(os.path.dirname(__file__), "pattern_lab.json")
SELF_CODE_PROPOSALS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "self_code_proposals.md"
)
FRAGMENTS_PATH = os.path.join(os.path.dirname(__file__), "data", "alpha_fragments.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "alpha_results.db")


# ------------------------------------------------------------------
# Fragment storage helpers
# ------------------------------------------------------------------

def _ensure_data_dir() -> None:
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)


def _load_fragments() -> list[dict]:
    """Load fragments from JSON store, creating an empty list if absent."""
    _ensure_data_dir()
    if os.path.exists(FRAGMENTS_PATH):
        try:
            with open(FRAGMENTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_fragments(fragments: list[dict]) -> None:
    """Persist fragment list to JSON store."""
    _ensure_data_dir()
    with open(FRAGMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(fragments, f, indent=2)


# ------------------------------------------------------------------
# Fragment extraction helpers
# ------------------------------------------------------------------

def _extract_sub_expressions(expr: str, min_depth: int = 2, max_depth: int = 4) -> list[str]:
    """
    Pull out bracketed sub-expressions of requested nesting depth.

    Depth is counted as the number of unmatched '(' on the left of any token.
    Operators inside a fragment are also recorded so ablation tagging works.
    """
    fragments: list[str] = []
    if not expr:
        return fragments

    # Track depth at each position by scanning left-to-right
    depth_at: list[int] = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth_at.append(depth)
            depth += 1
        elif ch == ")":
            depth -= 1
            depth_at.append(depth)
        else:
            depth_at.append(depth)

    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            start = i
            # Walk to the matching close paren
            depth_counter = 0
            end = start
            for j in range(start, len(expr)):
                if expr[j] == "(":
                    depth_counter += 1
                elif expr[j] == ")":
                    depth_counter -= 1
                    if depth_counter == 0:
                        end = j + 1
                        break
            sub = expr[start:end]
            frag_depth = sub.count("(")

            if min_depth <= frag_depth <= max_depth and len(sub) > 4:
                # Skip very noisy leaf fragments
                leaf_pattern = re.compile(r"^\([a-zA-Z_][a-zA-Z0-9_]*\)$")
                if not leaf_pattern.match(sub):
                    fragments.append(sub)

            i = end
        else:
            i += 1

    return fragments


def _extract_window_params(expr: str) -> tuple[int, int]:
    """
    Return (min_window, max_window) from ALL ts_* calls in the expression.

    Uses depth-aware scanning so commas inside nested function calls
    (e.g. ts_corr(close, volume, 20)) are not confused with the lookback
    argument of the outer ts_* call.

    For ``ts_mean(ts_corr(close,volume,20),10)`` this returns (10, 20) —
    the lookback of the inner ts_corr AND the lookback of the outer ts_mean.
    """
    windows: list[int] = []
    i = 0
    while i < len(expr):
        # Look for ts_<name>(
        m = re.match(r"(ts_[a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expr[i:])
        if not m:
            i += 1
            continue

        # m.end() is relative to expr[i:]; m.end() == length of "ts_mean("
        # func_arg_start points to the first argument char (skip the opening '(')
        func_arg_start = i + m.end()
        # Walk to the matching close paren of THIS ts_* call only
        depth = 1
        j = func_arg_start
        while j < len(expr) and depth > 0:
            if expr[j] == "(":
                depth += 1
            elif expr[j] == ")":
                depth -= 1
            j += 1
        inner = expr[func_arg_start : j - 1]

        # Scan top-level comma-separated segments to find the last integer,
        # which is the lookback argument of this ts_* call.
        last_num: str = ""
        segment_start = 0
        depth = 0
        for k, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                token = inner[segment_start:k].strip()
                if re.fullmatch(r"\d+", token):
                    last_num = token
                segment_start = k + 1
        token = inner[segment_start:].strip()
        if re.fullmatch(r"\d+", token):
            last_num = token

        if last_num:
            windows.append(int(last_num))

        # Recurse into the inner string to also capture nested ts_* calls
        # (e.g. ts_corr inside ts_mean).
        nested_min, nested_max = _extract_window_params(inner)
        if nested_min != 1 or nested_max != 1:
            windows.append(nested_min)
            windows.append(nested_max)

        i = j

    return (min(windows) if windows else 1, max(windows) if windows else 1)


def _regime_from_theme_or_family(theme: str, family: str) -> Optional[str]:
    """Infer a coarse regime tag from available metadata."""
    combined = f"{theme or ''} {family or ''}".lower()
    if "bull" in combined or "up" in combined:
        return "bull_trending"
    if "bear" in combined or "down" in combined:
        return "bear_trending"
    if "high_vol" in combined or "volatile" in combined:
        return "high_vol"
    if "low_vol" in combined or "stable" in combined:
        return "low_vol"
    if "sideways" in combined or "range" in combined:
        return "sideways"
    return None


# ------------------------------------------------------------------
# Public fragment API
# ------------------------------------------------------------------

def extract_fragments(
    db_path: Optional[str] = None,
    lookback: int = 1000,
    min_sharpe: float = 1.5,
) -> list[dict]:
    """
    Extract sub-expression fragments from passing high-Sharpe alphas.

    For each qualifying alpha in ``alpha_results.db``:
      - Extract sub-expressions of depth 2–4 from the AST.
      - Tag each fragment with operators, window range, regime, and a
        ``sharpe_contribution`` equal to ``alpha_sharpe / num_fragments``.
      - Compute ``complexity_ratio = len(fragment) / len(full_alpha)``.
      - Append to ``data/alpha_fragments.json`` (deduplicated by canonical form).

    Args:
        db_path:      Path to ``alpha_results.db``.  Defaults to the repo default.
        lookback:     How many most-recent alphas to scan.
        min_sharpe:   Minimum Sharpe required to qualify (default 1.5).

    Returns:
        The full fragment list written to the JSON store.
    """
    if db_path is None:
        db_path = DB_PATH

    if not os.path.exists(db_path):
        logger.warning("DB not found at %s — nothing to extract", db_path)
        return _load_fragments()

    fragments_raw: list[dict] = []
    seen_sigs: set[str] = set()

    try:
        conn = sqlite3.connect(db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        rows = conn.execute(
            """
            SELECT expression, sharpe, all_passed, theme, family
            FROM alphas
            WHERE all_passed = 1
              AND sharpe >= ?
              AND expression IS NOT NULL
              AND expression != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (min_sharpe, lookback),
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.warning("Failed to query DB for fragment extraction: %s", e)
        return _load_fragments()

    for (expression, sharpe, all_passed, theme, family) in rows:
        expr = str(expression or "")
        alpha_sharpe = float(sharpe or 0.0)
        regime_tag = _regime_from_theme_or_family(
            str(theme or ""), str(family or "")
        )
        sub_exprs = _extract_sub_expressions(expr, min_depth=2, max_depth=4)
        if not sub_exprs:
            continue

        # Each fragment gets an equal share of the alpha's Sharpe
        sharpe_per_frag = alpha_sharpe / len(sub_exprs)
        win_min, win_max = _extract_window_params(expr)
        frag_len = len(expr)

        for sub in sub_exprs:
            sig = canonicalize_expression(sub)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            operators = frozenset(ast_operator_set(sub))
            complexity_ratio = len(sub) / max(frag_len, 1)

            fragments_raw.append(
                {
                    "expression": sub,
                    "alpha_sharpe": alpha_sharpe,
                    "sharpe_contribution": sharpe_per_frag,
                    "operators": sorted(operators),
                    "lookback_range": [win_min, win_max],
                    "regime_tag": regime_tag,
                    "complexity_ratio": round(complexity_ratio, 4),
                    "source_alpha_len": frag_len,
                }
            )

    # Merge with existing store (deduplicate by canonical signature)
    existing = _load_fragments()
    existing_sigs = {canonicalize_expression(f["expression"]) for f in existing}
    merged = list(existing)
    for frag in fragments_raw:
        sig = canonicalize_expression(frag["expression"])
        if sig not in existing_sigs:
            merged.append(frag)

    _save_fragments(merged)
    logger.info(
        "Fragment extraction done: %d new / %d total",
        len(fragments_raw),
        len(merged),
    )
    return merged


def score_fragment(fragment: dict, regime: str, sector: str) -> float:
    """
    Score a single fragment for a given market context.

    Base score = ``sharpe_contribution`` from past performance.
    If the fragment's ``regime_tag`` matches the current regime → 1.3× boost.

    Args:
        fragment:   A fragment record loaded from ``alpha_fragments.json``.
        regime:     Current market regime name (e.g. ``"bull_trending"``).
        sector:     Sector hint (unused in v1 but reserved for future use).

    Returns:
        Weighted score (higher is better).
    """
    base = float(fragment.get("sharpe_contribution") or 0.0)
    frag_regime = fragment.get("regime_tag")
    if frag_regime is not None and regime and frag_regime.lower() == regime.lower():
        base *= 1.3
    return base


def inject_fragments(n_fragments: int, regime: str, sector: str) -> list[str]:
    """
    Return the top-N fragment expressions ranked by contextual score.

    Loads all fragments from ``data/alpha_fragments.json``, scores each via
    ``score_fragment()`` using the supplied regime/sector context, then returns
    the N highest-scoring expressions.

    Args:
        n_fragments:  How many fragments to return.
        regime:       Current market regime.
        sector:      Sector context (unused v1, reserved).

    Returns:
        List of fragment expression strings, ordered best-first.
    """
    fragments = _load_fragments()
    if not fragments:
        return []

    scored = [(score_fragment(f, regime, sector), f) for f in fragments]
    scored.sort(key=lambda x: x[0], reverse=True)

    return [f["expression"] for _, f in scored[:n_fragments]]


def get_fragment_proposals(regime: str, sector: str, n: int = 10) -> list[str]:
    """
    Return N formatted proposal strings for the safe-mode text interface.

    Each entry is a one-line description suitable for human review.

    Args:
        regime:  Current market regime.
        sector:  Sector context.
        n:       Number of proposals (default 10).

    Returns:
        List of formatted string proposals.
    """
    fragments = _load_fragments()
    if not fragments:
        return [f"- (no fragments yet — run `extract_fragments()` first)"]

    scored = [(score_fragment(f, regime, sector), f) for f in fragments]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n]

    proposals: list[str] = []
    for rank, (score, frag) in enumerate(top, start=1):
        ops = frag.get("operators", [])
        ops_str = ", ".join(ops) if ops else "(none)"
        regime_tag = frag.get("regime_tag") or "(unknown)"
        complexity = frag.get("complexity_ratio", 0.0)
        proposals.append(
            f"{rank:2d}. [{score:+.3f}] reg={regime_tag:<18s} "
            f"ops=[{ops_str}] complexity={complexity:.2f}  "
            f"expr={frag['expression'][:120]}"
        )
    return proposals


# ------------------------------------------------------------------
# PatternLab — original self-code engine (backward-compatible)
# ------------------------------------------------------------------

class PatternLab:
    def __init__(self, path: str = PATTERN_LAB_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"operator_counts": {}, "winning_fragments": [], "updates": 0}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def learn_from_results(self, sim_results):
        winners = [
            r
            for r in sim_results
            if getattr(r, "all_passed", False) and getattr(r, "error", "") == ""
        ]
        if not winners:
            return
        ops = Counter(self._extract_ops(r.expression) for r in winners)
        flat_ops = Counter()
        for op_seq, count in ops.items():
            for op in op_seq:
                flat_ops[op] += count
        for op, count in flat_ops.items():
            self.data["operator_counts"][op] = (
                self.data["operator_counts"].get(op, 0) + count
            )
        for r in sorted(
            winners,
            key=lambda x: float(getattr(x, "sharpe", 0.0) or 0.0),
            reverse=True,
        )[:12]:
            frag = r.expression[:220]
            if frag not in self.data["winning_fragments"]:
                self.data["winning_fragments"].append(frag)
        self.data["winning_fragments"] = self.data["winning_fragments"][-120:]
        self.data["updates"] = int(self.data.get("updates", 0)) + 1
        self.save()

    @staticmethod
    def _extract_ops(expression: str) -> tuple[str, ...]:
        expr = expression or ""
        operators = []
        for token in (
            "group_neutralize(",
            "ts_decay_linear(",
            "ts_zscore(",
            "ts_corr(",
            "trade_when(",
            "hump(",
            "rank(",
        ):
            if token in expr:
                operators.append(token[:-1])
        return tuple(operators)

    def top_operator_bias(self, top_k: int = 4) -> list[str]:
        items = sorted(
            self.data["operator_counts"].items(), key=lambda x: x[1], reverse=True
        )
        return [k for k, _ in items[:top_k]]

    def propose_expressions(self, n: int = 6) -> list[str]:
        """
        Deterministic pattern proposal engine (self-code safe mode).
        """
        base = list(self.data.get("winning_fragments", []))[-40:]
        if not base:
            return []
        ops = self.top_operator_bias(top_k=3)
        proposals: list[str] = []
        tries = 0
        while len(proposals) < n and tries < n * 8:
            tries += 1
            expr = random.choice(base)
            if "ts_decay_linear(" not in expr:
                expr = f"ts_decay_linear({expr}, 6)"
            if "group_neutralize(" not in expr:
                expr = f"group_neutralize({expr}, industry)"
            if ops and random.random() < 0.5:
                op = random.choice(ops)
                if op == "ts_zscore" and "ts_zscore(" not in expr:
                    expr = f"ts_zscore({expr}, 20)"
                if op == "hump" and "hump(" not in expr:
                    expr = f"hump({expr}, hump=0.01)"
            if expr not in proposals:
                proposals.append(expr)
        return proposals

    def emit_self_code_proposal(self):
        """
        Safe mode: output proposal text only, never auto-edit source files.
        """
        top_ops = self.top_operator_bias(top_k=5)
        if not top_ops:
            return
        os.makedirs(os.path.dirname(SELF_CODE_PROPOSALS_PATH), exist_ok=True)
        lines = [
            "## Pattern Lab Self-Code Proposal",
            "",
            f"- Updates: {self.data.get('updates', 0)}",
            f"- Top operators: {', '.join(top_ops)}",
            "- Suggested action: increase template coverage around these operators in generator/evolver.",
            "",
        ]
        with open(SELF_CODE_PROPOSALS_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
