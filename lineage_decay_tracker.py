"""
lineage_decay_tracker.py — Sharpe-trend lineage kill switch.

Tracks Sharpe history per family_id and kills lineages that are consistently
decaying or below the p_accept threshold. This prevents wasted quota on
families that are no longer productive.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LineageStats:
    """Mutable stats snapshot for one lineage (family_id)."""
    family_id: str
    sharpe_history: List[float] = field(default_factory=list)
    p_accept: float = 0.5
    sim_count: int = 0
    is_alive: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class LineageDecayTracker:
    """
    Track Sharpe trend per family_id.
    Kill lineages that are decaying (negative Sharpe slope) and have
    low historical acceptance probability.
    """

    def __init__(
        self,
        decay_slope_threshold: float = -0.1,
        p_accept_threshold: float = 0.3,
        data_dir: str | Path | None = None,
    ):
        self.decay_slope_threshold = decay_slope_threshold
        self.p_accept_threshold = p_accept_threshold
        self.lineages: Dict[str, LineageStats] = {}
        self._data_dir = Path(data_dir) if data_dir else Path("data")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._persist_path = self._data_dir / "lineage_decay_tracker.json"
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self.lineages = {
                fid: LineageStats(
                    family_id=fid,
                    sharpe_history=list(ls.get("sharpe_history", [])),
                    p_accept=float(ls.get("p_accept", 0.5)),
                    sim_count=int(ls.get("sim_count", 0)),
                    is_alive=bool(ls.get("is_alive", True)),
                )
                for fid, ls in raw.get("lineages", {}).items()
            }
            logger.info("Loaded %d lineages from %s", len(self.lineages), self._persist_path)
        except Exception as exc:
            logger.warning("Failed to load lineage tracker state: %s", exc)

    def _save(self) -> None:
        try:
            data = {
                "lineages": {fid: ls.to_dict() for fid, ls in self.lineages.items()}
            }
            self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist lineage tracker state: %s", exc)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_simulation(self, family_id: str, sharpe: float, p_accept: float | None = None) -> None:
        """
        Record a new simulation result for a lineage.

        Args:
            family_id: Unique lineage identifier.
            sharpe: Sharpe ratio from the simulation result.
            p_accept: Optional acceptance probability from submit governor.
        """
        if family_id not in self.lineages:
            self.lineages[family_id] = LineageStats(family_id=family_id)
        ls = self.lineages[family_id]
        ls.sharpe_history.append(sharpe)
        ls.sim_count += 1
        if p_accept is not None:
            ls.p_accept = p_accept
        # Auto-kill if already dead
        ls.is_alive = self.should_kill_lineage(family_id) is False  # noqa: E501
        self._save()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_decay_slope(self, family_id: str, window: int = 10) -> float:
        """
        Compute linear regression slope of Sharpe history (last N sims).
        Negative slope = decaying lineage.

        Uses OLS on (x, y) = (index, sharpe) over the rolling window.
        """
        ls = self.lineages.get(family_id)
        if not ls or len(ls.sharpe_history) < 3:
            return 0.0

        history = ls.sharpe_history[-window:]
        n = len(history)
        if n < 3:
            return 0.0

        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(history) / n

        num = sum((x_vals[i] - x_mean) * (history[i] - y_mean) for i in range(n))
        den = sum((x_vals[i] - x_mean) ** 2 for i in range(n))

        if den == 0:
            return 0.0
        return num / den

    def should_kill_lineage(self, family_id: str) -> bool:
        """
        Kill if decay_slope < decay_slope_threshold AND p_accept < p_accept_threshold.
        A lineage is also killed if it has too few simulations and its
        average Sharpe is deeply negative.
        """
        ls = self.lineages.get(family_id)
        if not ls:
            return False  # unknown lineage — let it run

        if not ls.is_alive:
            return True

        slope = self.compute_decay_slope(family_id)
        avg_sharpe = (sum(ls.sharpe_history) / len(ls.sharpe_history)) if ls.sharpe_history else 0.0

        # Kill conditions
        slope_kill = slope < self.decay_slope_threshold
        p_accept_kill = ls.p_accept < self.p_accept_threshold

        # Immediate kill: very few sims, deeply negative Sharpe
        early_kill = (
            len(ls.sharpe_history) >= 5
            and avg_sharpe < -1.0
            and ls.p_accept < self.p_accept_threshold
        )

        if slope_kill and p_accept_kill:
            logger.info(
                "Killing lineage %s: slope=%.4f (< %.2f) p_accept=%.2f (< %.2f)",
                family_id, slope, self.decay_slope_threshold,
                ls.p_accept, self.p_accept_threshold,
            )
            ls.is_alive = False
            return True

        if early_kill:
            logger.info(
                "Early killing lineage %s: avg_sharpe=%.2f p_accept=%.2f",
                family_id, avg_sharpe, ls.p_accept,
            )
            ls.is_alive = False
            return True

        return False

    def get_alive_lineages(self) -> List[str]:
        """Return family_ids that should still generate candidates."""
        return [fid for fid, ls in self.lineages.items() if ls.is_alive]

    def get_kill_report(self) -> Dict:
        """
        Return summary of killed lineages and reasons.
        """
        killed = []
        for fid, ls in self.lineages.items():
            if not ls.is_alive:
                slope = self.compute_decay_slope(fid)
                avg = sum(ls.sharpe_history) / len(ls.sharpe_history) if ls.sharpe_history else 0.0
                killed.append({
                    "family_id": fid,
                    "sim_count": ls.sim_count,
                    "avg_sharpe": round(avg, 4),
                    "decay_slope": round(slope, 4),
                    "p_accept": round(ls.p_accept, 4),
                })
        return {
            "killed_count": len(killed),
            "alive_count": len(self.get_alive_lineages()),
            "total_count": len(self.lineages),
            "killed_lineages": killed,
        }

    def revive_lineage(self, family_id: str) -> bool:
        """Manually revive a killed lineage (e.g., after a strategy refresh)."""
        ls = self.lineages.get(family_id)
        if ls:
            ls.is_alive = True
            self._save()
            return True
        return False

    def prune_old_history(self, max_history: int = 50) -> None:
        """Trim sharpe_history to the last max_history entries per lineage."""
        for ls in self.lineages.values():
            if len(ls.sharpe_history) > max_history:
                ls.sharpe_history = ls.sharpe_history[-max_history:]
        self._save()


# ----------------------------------------------------------------------
# Convenience standalone functions for pipeline integration
# ----------------------------------------------------------------------

_default_tracker: Optional[LineageDecayTracker] = None


def get_tracker(
    decay_slope_threshold: float = -0.1,
    p_accept_threshold: float = 0.3,
) -> LineageDecayTracker:
    """Singleton tracker for the process lifetime."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = LineageDecayTracker(
            decay_slope_threshold=decay_slope_threshold,
            p_accept_threshold=p_accept_threshold,
        )
    return _default_tracker


def record_lineage_sim(family_id: str, sharpe: float, p_accept: float | None = None) -> None:
    """Quick helper — delegates to the singleton tracker."""
    get_tracker().record_simulation(family_id, sharpe, p_accept)


def is_lineage_alive(family_id: str) -> bool:
    """Check if a lineage is still alive (has not been killed)."""
    tracker = get_tracker()
    ls = tracker.lineages.get(family_id)
    if not ls:
        return True  # unknown lineage is assumed alive
    return ls.is_alive
