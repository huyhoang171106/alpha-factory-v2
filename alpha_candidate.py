"""
alpha_candidate.py — AlphaCandidate Lineage Data Model
=======================================================
Replaces raw `str` passing throughout the pipeline.
Every alpha expression now carries metadata about its origin,
enabling lineage tracking, family crowding control, and theme analytics.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlphaCandidate:
    """
    An alpha expression with full lineage metadata.

    Fields:
        expression:    The alpha formula string.
        theme:         Factor theme (e.g. "microstructure", "quality").
        family:        Family ID grouping related alphas (shared seed).
        seed_id:       ID of the parent alpha that spawned this one.
        mutation_type: How this candidate was created.
    """
    expression: str
    theme: str = "unknown"
    family: str = ""
    seed_id: str = ""
    mutation_type: str = "seed"
    hypothesis: str = ""
    delay: int = 1

    def __post_init__(self):
        if not self.family:
            self.family = self._make_family_id()

    def _make_family_id(self) -> str:
        """Derive a stable family ID from the expression's core structure."""
        core = self.expression.strip().lower().replace(" ", "")
        # Hash first 60 chars to group structurally similar expressions
        prefix = core[:60]
        return hashlib.md5(prefix.encode()).hexdigest()[:12]

    @staticmethod
    def make_id() -> str:
        return uuid.uuid4().hex[:12]

    def derive(
        self,
        new_expression: str,
        mutation_type: str,
        theme: Optional[str] = None,
    ) -> "AlphaCandidate":
        """Create a child candidate inheriting this one's lineage."""
        return AlphaCandidate(
            expression=new_expression,
            theme=theme or self.theme,
            family=self.family,
            seed_id=self.family,
            mutation_type=mutation_type,
        )
