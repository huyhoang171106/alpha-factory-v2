"""
pattern_lab.py - Deterministic pattern memory for continuous alpha evolution.
"""

from __future__ import annotations

import json
import os
from collections import Counter
import random


PATTERN_LAB_PATH = os.path.join(os.path.dirname(__file__), "pattern_lab.json")
SELF_CODE_PROPOSALS_PATH = os.path.join(os.path.dirname(__file__), "results", "self_code_proposals.md")


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
        winners = [r for r in sim_results if getattr(r, "all_passed", False) and getattr(r, "error", "") == ""]
        if not winners:
            return
        ops = Counter(self._extract_ops(r.expression) for r in winners)
        flat_ops = Counter()
        for op_seq, count in ops.items():
            for op in op_seq:
                flat_ops[op] += count
        for op, count in flat_ops.items():
            self.data["operator_counts"][op] = self.data["operator_counts"].get(op, 0) + count
        for r in sorted(winners, key=lambda x: float(getattr(x, "sharpe", 0.0) or 0.0), reverse=True)[:12]:
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
        items = sorted(self.data["operator_counts"].items(), key=lambda x: x[1], reverse=True)
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
