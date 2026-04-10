"""
quality_diversity.py

Quality-diversity search helpers:
- behavior descriptors
- novelty scoring (continuous)
- lightweight MAP-Elites style archive
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

from alpha_ast import parameter_agnostic_signature


def _lookback_bucket(expression: str) -> str:
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", expression or "")]
    if not nums:
        return "none"
    avg = sum(nums) / len(nums)
    if avg <= 10:
        return "short"
    if avg <= 40:
        return "mid"
    return "long"


def behavior_descriptor(expression: str) -> str:
    expr = (expression or "").lower()
    neutralization = "none"
    if "subindustry" in expr:
        neutralization = "subindustry"
    elif "industry" in expr:
        neutralization = "industry"
    elif "sector" in expr:
        neutralization = "sector"
    elif "market" in expr:
        neutralization = "market"

    flags = [
        "g" if "group_" in expr else "ng",
        "c" if "ts_corr(" in expr or "ts_covariance(" in expr else "nc",
        "d" if "ts_decay_linear(" in expr else "nd",
        "e" if "trade_when(" in expr or "hump(" in expr else "ne",
        _lookback_bucket(expr),
        neutralization,
    ]
    return "|".join(flags)


def token_set(expression: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", (expression or "").lower()))


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 1.0
    sim = len(a & b) / union
    return 1.0 - sim


@dataclass
class QDEntry:
    expression: str
    quality: float
    novelty: float


class QualityDiversityArchive:
    def __init__(self, max_recent: int = 5000):
        self.elites: Dict[str, QDEntry] = {}
        self.descriptor_counts: Dict[str, int] = {}
        self.signature_seen: set[str] = set()
        self.recent_tokens: deque[set[str]] = deque(maxlen=max_recent)

    def restore_elite(self, descriptor: str, expression: str, quality: float, novelty: float = 0.0):
        self.elites[descriptor] = QDEntry(expression=expression, quality=float(quality), novelty=float(novelty))
        self.descriptor_counts[descriptor] = self.descriptor_counts.get(descriptor, 0) + 1
        self.signature_seen.add(parameter_agnostic_signature(expression))
        self.recent_tokens.append(token_set(expression))

    def novelty_score(self, expression: str) -> Tuple[float, str]:
        desc = behavior_descriptor(expression)
        sig = parameter_agnostic_signature(expression)
        sig_component = 1.0 if sig not in self.signature_seen else 0.2

        tok = token_set(expression)
        if not self.recent_tokens:
            distance_component = 1.0
        else:
            sample = list(self.recent_tokens)[-80:]
            distances = [jaccard_distance(tok, prev) for prev in sample]
            distance_component = sum(distances) / len(distances)

        total = sum(self.descriptor_counts.values())
        rarity = 1.0 if total == 0 else 1.0 - (self.descriptor_counts.get(desc, 0) / max(1, total))

        novelty = 0.45 * sig_component + 0.35 * distance_component + 0.20 * rarity
        novelty = max(0.0, min(1.0, novelty))
        return novelty, desc

    def maybe_update_archive(self, expression: str, quality: float, novelty: float, descriptor: str) -> bool:
        existing = self.elites.get(descriptor)
        if existing is None or quality > existing.quality:
            self.elites[descriptor] = QDEntry(expression=expression, quality=float(quality), novelty=float(novelty))
            self.descriptor_counts[descriptor] = self.descriptor_counts.get(descriptor, 0) + 1
            self.signature_seen.add(parameter_agnostic_signature(expression))
            self.recent_tokens.append(token_set(expression))
            return True

        # keep novelty memory even when not elite
        self.signature_seen.add(parameter_agnostic_signature(expression))
        self.recent_tokens.append(token_set(expression))
        self.descriptor_counts[descriptor] = self.descriptor_counts.get(descriptor, 0) + 1
        return False

    def stats(self) -> dict:
        return {
            "elite_cells": len(self.elites),
            "seen_signatures": len(self.signature_seen),
            "descriptor_total": sum(self.descriptor_counts.values()),
        }
