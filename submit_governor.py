"""
submit_governor.py - Queue and dispatch submit jobs to WQ.

================================================================================
INTEGRATION: Portfolio Constructor Ensemble Stacking
================================================================================
To enable ensemble submission, import PreSubmissionGate from portfolio_constructor:

    from portfolio_constructor import PreSubmissionGate, build_ensemble_from_candidates

Integration points in SubmitGovernor.enqueue():

    # 1. After quality gate — collect candidates as (expression, sharpe) tuples
    candidates = [
        (getattr(r, "expression", ""), float(getattr(r, "sharpe", 0) or 0))
        for r in sim_results
        if passes_quality_gate_v2(r)
    ]

    # 2. Ask PreSubmissionGate whether an ensemble beats the best individual
    gate = PreSubmissionGate(improvement_threshold=1.1)
    use_ensemble, ensemble = gate.should_submit_ensemble(candidates)

    if use_ensemble and ensemble:
        # Route ensemble to submit queue via tracker
        for alpha_expr, weight in zip(ensemble.sub_alphas, ensemble.weights):
            tracker.transition_submit_state(alpha_id, "gated")
            tracker.mark_queued(alpha_id)
        logger.info("Ensemble queued: %d components, est Sharpe=%.3f",
                    ensemble.num_components, ensemble.ensemble_sharpe)
    else:
        # Fall back to best individual alpha (existing code path)
        ...

State machine (unchanged):
    new → gated → queued → submitted → accepted|rejected
    Failure path: queued|failed → dead_lettered → replay → queued

Ensemble-specific states are tracked per sub-alpha component in the tracker DB.
================================================================================
"""

from __future__ import annotations

import logging
import os
import random
import hashlib
from difflib import SequenceMatcher
from typing import Iterable, Tuple

from alpha_policy import classify_quality_tier, passes_quality_gate_v2

logger = logging.getLogger(__name__)


def _safe_expr_ref(expression: str) -> str:
    return hashlib.sha256((expression or "").strip().encode("utf-8")).hexdigest()[:12]


class SubmitGovernor:
    def __init__(self, tracker, client, max_submits_per_minute: int = 4):
        self.tracker = tracker
        self.client = client
        self.max_submits_per_minute = max(1, int(max_submits_per_minute))

    def enqueue(self, sim_results: Iterable, candidates_map: dict | None = None) -> int:
        queued = 0
        seen_families: set[str] = set()
        selected_exprs: list[str] = []

        # Select high-throughput queue entries with basic diversity.
        for result in sorted(
            sim_results,
            key=lambda r: float(getattr(r, "competition_priority", getattr(r, "sharpe", 0.0)) or 0.0),
            reverse=True,
        ):
            if not passes_quality_gate_v2(result):
                continue
            alpha_id = getattr(result, "alpha_id", "")
            if not alpha_id:
                continue
            candidate = (candidates_map or {}).get(result.expression)
            family = getattr(candidate, "family", "") if candidate else ""
            if family and family in seen_families:
                continue
            if self._too_similar(result.expression, selected_exprs):
                continue
            # Transition record into governed state before queuing submit jobs.
            try:
                self.tracker.transition_submit_state(alpha_id, "gated")
            except Exception:
                pass
            self.tracker.mark_queued(alpha_id)
            queued += 1
            selected_exprs.append(result.expression)
            if family:
                seen_families.add(family)
        return queued

    @staticmethod
    def _too_similar(expression: str, selected: list[str], threshold: float = 0.82) -> bool:
        for expr in selected:
            if SequenceMatcher(None, expression, expr).ratio() >= threshold:
                return True
        return False

    @staticmethod
    def _retry_policy(error_class: str, attempts: int) -> Tuple[bool, int, bool]:
        """
        Returns (retryable, delay_seconds, dead_letter_now).
        """
        c = (error_class or "").lower()
        n = max(0, int(attempts))
        if "semantic_4xx" in c or "payload_4xx" in c:
            return False, 0, True
        if "auth" in c:
            # allow a couple retries in case transient token/session issue
            if n >= 2:
                return False, 0, True
            return True, 120, False
        if "rate" in c or "429" in c:
            delay = min(600, 30 * (2 ** n))
            return True, delay, False
        if "server_5xx" in c or "network" in c or "exception" in c or "no_response" in c:
            if n >= 5:
                return False, 0, True
            delay = min(900, 45 * (2 ** n))
            return True, delay, False
        # unknown errors: limited retries
        if n >= 4:
            return False, 0, True
        return True, 120, False

    def flush_once(self, limit: int | None = None) -> dict:
        quota = limit or self.max_submits_per_minute
        rows = self.tracker.get_submit_queue(limit=max(1, quota * 6))
        if not rows:
            return {"selected": 0, "submitted": 0, "failed": 0}

        selected = []
        seen_themes: set[str] = set()
        seen_families: set[str] = set()
        selected_exprs: list[str] = []
        for row in rows:
            db_id, expr, alpha_id, sharpe, family, theme, submit_attempts, err_class, job_attempts, next_retry_at = row
            if theme and theme in seen_themes:
                continue
            if family and family in seen_families:
                continue
            if self._too_similar(expr, selected_exprs):
                continue
            selected.append(row)
            selected_exprs.append(expr)
            if theme:
                seen_themes.add(theme)
            if family:
                seen_families.add(family)
            if len(selected) >= quota:
                break

        success = 0
        failed = 0
        dead_lettered = 0
        for _, expr, alpha_id, sharpe, _, _, submit_attempts, _, job_attempts, _ in selected:
            logger.info(
                "Submit queued alpha S=%.3f tier=%s expr_id=%s",
                sharpe,
                classify_quality_tier(sharpe, 9.9),
                _safe_expr_ref(expr),
            )
            if hasattr(self.client, "submit_alpha_detailed"):
                ok, error_class = self.client.submit_alpha_detailed(alpha_id)
            else:
                ok = self.client.submit_alpha(alpha_id)
                error_class = "" if ok else "submit_unknown"

            if ok:
                self.tracker.mark_submitted(alpha_id)
                # HTTP 2xx only confirms submit accepted by endpoint, not final review acceptance.
                success += 1
            else:
                attempts = max(int(submit_attempts or 0), int(job_attempts or 0))
                retryable, delay_s, to_dlq = self._retry_policy(error_class, attempts)
                if to_dlq:
                    self.tracker.mark_dead_lettered(alpha_id, reason=error_class or "submit_api_failed", error_class=error_class)
                    dead_lettered += 1
                elif retryable:
                    self.tracker.mark_submit_failed(
                        alpha_id,
                        reason=error_class or "submit_api_failed",
                        error_class=error_class,
                        next_retry_seconds=delay_s,
                    )
                else:
                    self.tracker.mark_submit_failed(
                        alpha_id,
                        reason=error_class or "submit_api_failed",
                        error_class=error_class,
                        next_retry_seconds=300,
                    )
                failed += 1

        return {"selected": len(selected), "submitted": success, "failed": failed, "dead_lettered": dead_lettered}

    @staticmethod
    def _apply_jitter(base_seconds: int, jitter_ratio: float = 0.20) -> int:
        """
        Add symmetric jitter to avoid synchronized polling spikes.
        """
        env_ratio = os.getenv("ASYNC_REVIEW_JITTER_RATIO")
        if env_ratio not in (None, ""):
            try:
                jitter_ratio = float(env_ratio)
            except ValueError:
                pass
        base = max(5, int(base_seconds))
        ratio = max(0.0, min(float(jitter_ratio), 0.45))
        delta = (random.random() * 2.0 - 1.0) * ratio
        jittered = int(round(base * (1.0 + delta)))
        return max(5, jittered)

    @staticmethod
    def _review_backoff_seconds(attempts: int, error_class: str) -> int:
        n = max(0, int(attempts))
        c = (error_class or "").lower()
        if "rate" in c or "429" in c:
            return SubmitGovernor._apply_jitter(min(1800, 60 * (2 ** n)))
        if "auth" in c:
            return SubmitGovernor._apply_jitter(min(1200, 180 * (2 ** min(n, 3))))
        if "server" in c or "timeout" in c or "connection" in c or "exception" in c:
            return SubmitGovernor._apply_jitter(min(1800, 90 * (2 ** n)))
        # Pending-review polling default.
        return SubmitGovernor._apply_jitter(min(1200, 45 * (2 ** n)))

    def reconcile_submitted(self, limit: int = 20) -> dict:
        """
        Poll WQ review states for already-submitted alphas.
        """
        ids = self.tracker.get_submitted_pending_review(limit=max(1, int(limit)))
        if not ids:
            return {"checked": 0, "accepted": 0, "rejected": 0, "pending": 0, "errors": 0}

        accepted = 0
        rejected = 0
        pending = 0
        errors = 0
        for row in ids:
            if isinstance(row, (tuple, list)):
                alpha_id = row[0]
                attempts = int(row[1] or 0) if len(row) > 1 else 0
            else:
                alpha_id = row
                attempts = 0
            if not hasattr(self.client, "get_submission_decision"):
                pending += 1
                continue
            decision, error_class, detail, self_corr = self.client.get_submission_decision(alpha_id)
            if decision in {"accepted", "rejected"}:
                self.tracker.finalize_submit_review(alpha_id, decision=decision, reason=detail or decision, self_corr=self_corr)
                if decision == "accepted":
                    accepted += 1
                else:
                    rejected += 1
            elif error_class:
                errors += 1
                if hasattr(self.tracker, "mark_review_pending"):
                    delay = self._review_backoff_seconds(attempts, error_class)
                    self.tracker.mark_review_pending(alpha_id, reason=error_class or detail, backoff_seconds=delay)
            else:
                pending += 1
                if hasattr(self.tracker, "mark_review_pending"):
                    delay = self._review_backoff_seconds(attempts, "pending")
                    self.tracker.mark_review_pending(alpha_id, reason=detail or "pending_review", backoff_seconds=delay)

        return {
            "checked": len(ids),
            "accepted": accepted,
            "rejected": rejected,
            "pending": pending,
            "errors": errors,
        }
