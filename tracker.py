"""
tracker.py — Results Tracking (SQLite + CSV)
Logs all simulation results for analysis and evolution.
"""

import os
import csv
import sqlite3
import time
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional
from alpha_ast import (
    canonicalize_expression,
    parameter_agnostic_signature,
    token_set,
    operator_set,
)
from alpha_policy import classify_quality_tier, infer_strategy_cluster, build_risk_flags


DB_PATH = os.path.join(os.path.dirname(__file__), "alpha_results.db")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ALLOWED_SUBMIT_TRANSITIONS = {
    "new": {"gated", "rejected", "queued"},
    "gated": {"queued", "rejected"},
    "queued": {"submitted", "failed", "dead_lettered"},
    "failed": {"queued", "dead_lettered", "rejected"},
    "submitted": {"accepted", "rejected"},
    "accepted": set(),
    "rejected": set(),
    "dead_lettered": {"queued"},
}

FINAL_SUBMIT_STATES = {"accepted", "rejected", "dead_lettered"}


class AlphaTracker:
    """Track simulation results in SQLite + daily CSV logs"""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.db_path = db_path
        self._conn_lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        self._recent_structure_cache: list[tuple[set[str], set[str]]] = []

        # Performance PRAGMAs: Prevent lock crashes in concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alphas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expression TEXT NOT NULL,
                sharpe REAL DEFAULT 0,
                fitness REAL DEFAULT 0,
                turnover REAL DEFAULT 0,
                returns REAL DEFAULT 0,
                drawdown REAL DEFAULT 0,
                passed_checks INTEGER DEFAULT 0,
                total_checks INTEGER DEFAULT 0,
                all_passed BOOLEAN DEFAULT 0,
                alpha_id TEXT DEFAULT '',
                alpha_url TEXT DEFAULT '',
                error TEXT DEFAULT '',
                sub_sharpe REAL DEFAULT -1,
                region TEXT DEFAULT 'USA',
                universe TEXT DEFAULT 'TOP3000',
                delay INTEGER DEFAULT 1,
                decay INTEGER DEFAULT 6,
                neutralization TEXT DEFAULT 'SUBINDUSTRY',
                submitted BOOLEAN DEFAULT 0,
                theme TEXT DEFAULT 'unknown',
                family TEXT DEFAULT '',
                mutation_type TEXT DEFAULT 'seed',
                hypothesis TEXT DEFAULT '',
                fail_reason TEXT DEFAULT '',
                nearest_sibling TEXT DEFAULT '',
                canonical_expr TEXT DEFAULT '',
                self_corr REAL DEFAULT NULL,
                quality_tier TEXT DEFAULT 'reject',
                strategy_cluster TEXT DEFAULT 'deterministic',
                risk_flags TEXT DEFAULT '',
                submit_state TEXT DEFAULT 'new',
                submit_attempts INTEGER DEFAULT 0,
                last_submit_error TEXT DEFAULT '',
                run_id TEXT DEFAULT '',
                batch_id TEXT DEFAULT '',
                queued_at TEXT DEFAULT '',
                submitted_at TEXT DEFAULT '',
                accepted_at TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sharpe ON alphas(sharpe DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created ON alphas(created_at)
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alpha_signatures (
                canonical_expr TEXT PRIMARY KEY,
                param_signature TEXT DEFAULT '',
                source_expr TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS submit_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alpha_id TEXT UNIQUE,
                expression TEXT DEFAULT '',
                state TEXT DEFAULT 'queued',
                error_class TEXT DEFAULT '',
                last_error TEXT DEFAULT '',
                attempt_count INTEGER DEFAULT 0,
                next_retry_at TEXT DEFAULT CURRENT_TIMESTAMP,
                review_poll_attempts INTEGER DEFAULT 0,
                next_review_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_review_error TEXT DEFAULT '',
                last_reviewed_at TEXT DEFAULT '',
                queued_at TEXT DEFAULT CURRENT_TIMESTAMP,
                submitted_at TEXT DEFAULT '',
                dead_lettered_at TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS submit_dlq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alpha_id TEXT,
                expression TEXT DEFAULT '',
                error_class TEXT DEFAULT '',
                last_error TEXT DEFAULT '',
                attempt_count INTEGER DEFAULT 0,
                dead_lettered_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS qd_archive (
                descriptor TEXT PRIMARY KEY,
                expression TEXT DEFAULT '',
                quality_score REAL DEFAULT 0,
                novelty_score REAL DEFAULT 0,
                updates INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_param_signature ON alpha_signatures(param_signature)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_submit_jobs_state_retry ON submit_jobs(state, next_retry_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_submit_dlq_alpha_id ON submit_dlq(alpha_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_qd_archive_quality ON qd_archive(quality_score DESC)
        """)
        self.conn.commit()
        # Migrate first (adds columns if missing), then create indexes on new columns
        self._migrate_lineage_columns()
        self._migrate_submit_jobs_columns()
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_theme ON alphas(theme)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_family ON alphas(family)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_canonical_expr ON alphas(canonical_expr)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_submit_state ON alphas(submit_state, submitted, created_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alpha_id ON alphas(alpha_id)
        """)
        self.conn.commit()
        self._backfill_signatures(limit=12000)
        self._rebuild_structure_cache(limit=1500)

    def _rebuild_structure_cache(self, limit: int = 1500):
        cursor = self.conn.execute(
            "SELECT expression FROM alphas ORDER BY id DESC LIMIT ?",
            (max(200, int(limit)),),
        )
        rows = cursor.fetchall()
        cache: list[tuple[set[str], set[str]]] = []
        for (expr,) in rows:
            expression = expr or ""
            cache.append(
                (
                    token_set(expression, strip_numbers=True),
                    operator_set(expression),
                )
            )
        self._recent_structure_cache = cache

    def _backfill_signatures(self, limit: int = 10000):
        """
        Build persistent de-dup memory across restarts.
        Runs quickly with bounded rows and INSERT OR IGNORE semantics.
        """
        cursor = self.conn.execute(
            "SELECT canonical_expr, expression FROM alphas ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        if not rows:
            return
        to_insert = []
        for canonical, expr in rows:
            expression = expr or ""
            canonical_expr = canonical or canonicalize_expression(expression)
            param_sig = parameter_agnostic_signature(expression)
            if canonical_expr:
                to_insert.append((canonical_expr, param_sig, expression[:300]))
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO alpha_signatures(canonical_expr, param_signature, source_expr)
            VALUES (?, ?, ?)
            """,
            to_insert,
        )
        self.conn.commit()

    def _migrate_lineage_columns(self):
        """Add lineage columns to existing DBs that lack them."""
        cursor = self.conn.execute("PRAGMA table_info(alphas)")
        existing = {row[1] for row in cursor.fetchall()}
        migrations = [
            ("theme", "TEXT DEFAULT 'unknown'"),
            ("family", "TEXT DEFAULT ''"),
            ("mutation_type", "TEXT DEFAULT 'seed'"),
            ("hypothesis", "TEXT DEFAULT ''"),
            ("fail_reason", "TEXT DEFAULT ''"),
            ("nearest_sibling", "TEXT DEFAULT ''"),
            ("canonical_expr", "TEXT DEFAULT ''"),
            ("quality_tier", "TEXT DEFAULT 'reject'"),
            ("strategy_cluster", "TEXT DEFAULT 'deterministic'"),
            ("risk_flags", "TEXT DEFAULT ''"),
            ("submit_state", "TEXT DEFAULT 'new'"),
            ("submit_attempts", "INTEGER DEFAULT 0"),
            ("last_submit_error", "TEXT DEFAULT ''"),
            ("run_id", "TEXT DEFAULT ''"),
            ("batch_id", "TEXT DEFAULT ''"),
            ("queued_at", "TEXT DEFAULT ''"),
            ("submitted_at", "TEXT DEFAULT ''"),
            ("accepted_at", "TEXT DEFAULT ''"),
            ("self_corr", "REAL DEFAULT NULL"),
        ]
        for col, typedef in migrations:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE alphas ADD COLUMN {col} {typedef}")
        self.conn.commit()

    def _migrate_submit_jobs_columns(self):
        cursor = self.conn.execute("PRAGMA table_info(submit_jobs)")
        existing = {row[1] for row in cursor.fetchall()}
        migrations = [
            ("review_poll_attempts", "INTEGER DEFAULT 0"),
            ("next_review_at", "TEXT DEFAULT CURRENT_TIMESTAMP"),
            ("last_review_error", "TEXT DEFAULT ''"),
            ("last_reviewed_at", "TEXT DEFAULT ''"),
        ]
        for col, typedef in migrations:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE submit_jobs ADD COLUMN {col} {typedef}")
        self.conn.commit()

    def save_result(
        self, result, candidate=None, run_id: str = "", batch_id: str = ""
    ) -> int:
        """Save a SimResult to DB with optional lineage from AlphaCandidate."""
        with self._conn_lock:
            theme = getattr(candidate, "theme", "unknown") if candidate else "unknown"
            family = getattr(candidate, "family", "") if candidate else ""
            mutation_type = (
                getattr(candidate, "mutation_type", "seed") if candidate else "seed"
            )
            hypothesis = getattr(candidate, "hypothesis", "") if candidate else ""
            nearest_sibling = (
                getattr(candidate, "nearest_sibling", "") if candidate else ""
            )
            canonical_expr = canonicalize_expression(result.expression)
            quality_tier = classify_quality_tier(
                float(result.sharpe or 0), float(result.fitness or 0)
            )
            strategy_cluster = infer_strategy_cluster(theme, mutation_type)
            risk_flags = build_risk_flags(
                result.expression, float(result.turnover or 0), result.error
            )

            # Deduce fail reason natively from Result (or candidate's metadata)
            fail_reason = ""
            if result.error:
                fail_reason = result.error
            elif not result.all_passed:
                fail_reason = f"Checks Failed: Sharpe={result.sharpe:.2f}, Fitness={result.fitness:.2f}"

            initial_state = "new"
            if result.error:
                initial_state = "rejected"

            cursor = self.conn.execute(
                """
                INSERT INTO alphas (
                    expression, sharpe, fitness, turnover, returns, drawdown,
                    passed_checks, total_checks, all_passed,
                    alpha_id, alpha_url, error, sub_sharpe,
                    region, universe, delay, decay, neutralization,
                    theme, family, mutation_type, hypothesis, fail_reason, nearest_sibling, canonical_expr,
                    quality_tier, strategy_cluster, risk_flags, submit_state, run_id, batch_id, self_corr,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.expression,
                    result.sharpe,
                    result.fitness,
                    result.turnover,
                    result.returns,
                    result.drawdown,
                    result.passed_checks,
                    result.total_checks,
                    result.all_passed,
                    result.alpha_id,
                    result.alpha_url,
                    result.error,
                    result.sub_sharpe,
                    result.region,
                    result.universe,
                    result.delay,
                    result.decay,
                    result.neutralization,
                    theme,
                    family,
                    mutation_type,
                    hypothesis,
                    fail_reason,
                    nearest_sibling,
                    canonical_expr,
                    quality_tier,
                    strategy_cluster,
                    risk_flags,
                    initial_state,
                    run_id,
                    batch_id,
                    getattr(result, "self_corr", None),
                    None,  # created_at defaults to CURRENT_TIMESTAMP
                ),
            )
            self.conn.commit()
            param_sig = parameter_agnostic_signature(result.expression)
            self.conn.execute(
                """
                INSERT OR IGNORE INTO alpha_signatures(canonical_expr, param_signature, source_expr)
                VALUES (?, ?, ?)
                """,
                (canonical_expr, param_sig, (result.expression or "")[:300]),
            )
            self.conn.commit()
            self._recent_structure_cache.append(
                (
                    token_set(result.expression, strip_numbers=True),
                    operator_set(result.expression),
                )
            )
            if len(self._recent_structure_cache) > 2000:
                self._recent_structure_cache = self._recent_structure_cache[-1500:]
            return cursor.lastrowid

    def save_batch(
        self,
        results: list,
        candidates_map: dict = None,
        run_id: str = "",
        batch_id: str = "",
    ) -> int:
        """Save multiple results. candidates_map: {expression: AlphaCandidate}."""
        count = 0
        for r in results:
            candidate = (candidates_map or {}).get(r.expression)
            self.save_result(r, candidate=candidate, run_id=run_id, batch_id=batch_id)
            count += 1
        return count

    def mark_submitted(self, alpha_id: str):
        """Mark an alpha as submitted"""
        self.transition_submit_state(alpha_id, "submitted", reason="")
        self.conn.execute(
            """
            UPDATE submit_jobs
            SET state = 'submitted',
                submitted_at = CURRENT_TIMESTAMP,
                review_poll_attempts = 0,
                next_review_at = CURRENT_TIMESTAMP,
                last_review_error = '',
                last_reviewed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE alpha_id = ?
            """,
            (alpha_id,),
        )
        self.conn.commit()

    def _get_submit_state(self, alpha_id: str) -> str:
        cursor = self.conn.execute(
            "SELECT submit_state FROM alphas WHERE alpha_id = ? ORDER BY id DESC LIMIT 1",
            (alpha_id,),
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] else "new"

    def transition_submit_state(
        self, alpha_id: str, next_state: str, reason: str = ""
    ) -> bool:
        current = self._get_submit_state(alpha_id)
        if current == next_state:
            return True
        allowed = ALLOWED_SUBMIT_TRANSITIONS.get(current, set())
        if next_state not in allowed:
            return False

        submitted_flag = 1 if next_state in {"submitted", "accepted"} else 0
        submitted_at = (
            "CURRENT_TIMESTAMP"
            if next_state in {"submitted", "accepted"}
            else "submitted_at"
        )
        accepted_at = "CURRENT_TIMESTAMP" if next_state == "accepted" else "accepted_at"
        self.conn.execute(
            f"""
            UPDATE alphas
            SET submit_state = ?,
                submitted = ?,
                last_submit_error = ?,
                submitted_at = {submitted_at},
                accepted_at = {accepted_at}
            WHERE alpha_id = ?
            """,
            (next_state, submitted_flag, (reason or "")[:500], alpha_id),
        )
        self.conn.commit()
        return True

    def mark_queued(self, alpha_id: str):
        self.transition_submit_state(alpha_id, "queued")
        self.conn.execute(
            """
            INSERT INTO submit_jobs(alpha_id, expression, state, next_retry_at, queued_at, updated_at)
            VALUES (
                ?,
                COALESCE((SELECT expression FROM alphas WHERE alpha_id = ? ORDER BY id DESC LIMIT 1), ''),
                'queued',
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT(alpha_id) DO UPDATE SET
                state = 'queued',
                next_retry_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            """,
            (alpha_id, alpha_id),
        )
        self.conn.execute(
            """
            UPDATE alphas
            SET queued_at = CASE WHEN queued_at = '' THEN CURRENT_TIMESTAMP ELSE queued_at END
            WHERE alpha_id = ? AND alpha_id != ''
            """,
            (alpha_id,),
        )
        self.conn.commit()

    def mark_submit_failed(
        self,
        alpha_id: str,
        reason: str,
        error_class: str = "",
        next_retry_seconds: int = 60,
    ):
        self.transition_submit_state(alpha_id, "failed", reason=reason)
        self.conn.execute(
            """
            UPDATE alphas
            SET submit_attempts = submit_attempts + 1,
                last_submit_error = ?
            WHERE alpha_id = ?
            """,
            ((reason or "")[:500], alpha_id),
        )
        self.conn.execute(
            """
            INSERT INTO submit_jobs(alpha_id, expression, state, error_class, last_error, attempt_count, next_retry_at, updated_at)
            VALUES (
                ?,
                COALESCE((SELECT expression FROM alphas WHERE alpha_id = ? ORDER BY id DESC LIMIT 1), ''),
                'failed',
                ?,
                ?,
                1,
                datetime('now', ?),
                CURRENT_TIMESTAMP
            )
            ON CONFLICT(alpha_id) DO UPDATE SET
                state = 'failed',
                error_class = excluded.error_class,
                last_error = excluded.last_error,
                attempt_count = submit_jobs.attempt_count + 1,
                next_retry_at = datetime('now', ?),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                alpha_id,
                alpha_id,
                (error_class or "")[:100],
                (reason or "")[:500],
                f"+{max(0, int(next_retry_seconds))} seconds",
                f"+{max(0, int(next_retry_seconds))} seconds",
            ),
        )
        self.conn.commit()

    def mark_dead_lettered(self, alpha_id: str, reason: str, error_class: str = ""):
        self.transition_submit_state(alpha_id, "dead_lettered", reason=reason)
        self.conn.execute(
            """
            UPDATE submit_jobs
            SET state = 'dead_lettered',
                error_class = ?,
                last_error = ?,
                dead_lettered_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE alpha_id = ?
            """,
            ((error_class or "")[:100], (reason or "")[:500], alpha_id),
        )
        self.conn.execute(
            """
            INSERT INTO submit_dlq(alpha_id, expression, error_class, last_error, attempt_count)
            VALUES (
                ?,
                COALESCE((SELECT expression FROM alphas WHERE alpha_id = ? ORDER BY id DESC LIMIT 1), ''),
                ?,
                ?,
                COALESCE((SELECT submit_attempts FROM alphas WHERE alpha_id = ? ORDER BY id DESC LIMIT 1), 0)
            )
            """,
            (
                alpha_id,
                alpha_id,
                (error_class or "")[:100],
                (reason or "")[:500],
                alpha_id,
            ),
        )
        self.conn.commit()

    def get_submitted_pending_review(self, limit: int = 50):
        cursor = self.conn.execute(
            """
            SELECT a.alpha_id,
                   COALESCE(j.review_poll_attempts, 0) as review_poll_attempts,
                   COALESCE(j.next_review_at, '') as next_review_at,
                   COALESCE(j.submitted_at, a.submitted_at, a.created_at) as submitted_at
            FROM alphas a
            LEFT JOIN submit_jobs j ON a.alpha_id = j.alpha_id
            WHERE a.alpha_id != ''
              AND a.submit_state = 'submitted'
              AND (
                    j.next_review_at IS NULL
                    OR j.next_review_at = ''
                    OR j.next_review_at <= CURRENT_TIMESTAMP
              )
            ORDER BY COALESCE(j.submitted_at, a.submitted_at, a.created_at) ASC, a.id ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        return cursor.fetchall()

    def mark_review_pending(
        self, alpha_id: str, reason: str = "", backoff_seconds: int = 60
    ):
        delay = max(5, int(backoff_seconds))
        self.conn.execute(
            """
            UPDATE submit_jobs
            SET review_poll_attempts = COALESCE(review_poll_attempts, 0) + 1,
                next_review_at = datetime('now', ?),
                last_review_error = ?,
                last_reviewed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE alpha_id = ?
            """,
            (f"+{delay} seconds", (reason or "")[:500], alpha_id),
        )
        self.conn.commit()

    def finalize_submit_review(
        self,
        alpha_id: str,
        decision: str,
        reason: str = "",
        self_corr: float | None = None,
    ) -> bool:
        """
        Apply post-submit decision from WQ review:
        - accepted
        - rejected
        Also writes self_corr (wl13) back to alphas table for the feedback loop.
        """
        d = (decision or "").strip().lower()
        if d not in {"accepted", "rejected"}:
            return False
        ok = self.transition_submit_state(alpha_id, d, reason=reason or d)
        if not ok:
            return False
        # Backfill self_corr to alphas table (feed for LearnedSelfCorrWeights)
        if self_corr is not None and self_corr > 0:
            self.conn.execute(
                "UPDATE alphas SET self_corr = ?, submitted_at = COALESCE(submitted_at, CURRENT_TIMESTAMP) WHERE alpha_id = ? AND (self_corr IS NULL OR self_corr = 0)",
                (self_corr, alpha_id),
            )
        self.conn.execute(
            """
            UPDATE submit_jobs
            SET state = ?,
                last_error = ?,
                last_review_error = ?,
                last_reviewed_at = CURRENT_TIMESTAMP,
                next_review_at = '',
                updated_at = CURRENT_TIMESTAMP
            WHERE alpha_id = ?
            """,
            (d, (reason or "")[:500], (reason or "")[:500], alpha_id),
        )
        self.conn.commit()
        return True

    def replay_dlq(self, limit: int = 20) -> int:
        cursor = self.conn.execute(
            "SELECT alpha_id FROM submit_dlq ORDER BY id ASC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        for (alpha_id,) in rows:
            self.conn.execute("DELETE FROM submit_dlq WHERE alpha_id = ?", (alpha_id,))
            self.mark_queued(alpha_id)
        self.conn.commit()
        return len(rows)

    def mark_gated_by_id(self, row_id: int):
        self.conn.execute(
            """
            UPDATE alphas
            SET submit_state = CASE
                WHEN submit_state IN ('new', 'failed') THEN 'gated'
                ELSE submit_state
            END
            WHERE id = ?
            """,
            (row_id,),
        )
        self.conn.commit()

    def mark_rejected_by_id(self, row_id: int, reason: str = ""):
        self.conn.execute(
            """
            UPDATE alphas
            SET submit_state = CASE
                WHEN submit_state IN ('new', 'gated', 'failed') THEN 'rejected'
                ELSE submit_state
            END,
                last_submit_error = CASE WHEN ? != '' THEN ? ELSE last_submit_error END
            WHERE id = ?
            """,
            ((reason or "")[:500], (reason or "")[:500], row_id),
        )
        self.conn.commit()

    def get_submit_queue(self, limit: int = 30):
        cursor = self.conn.execute(
            """
            SELECT a.id, a.expression, a.alpha_id, a.sharpe, a.family, a.theme, a.submit_attempts,
                   COALESCE(j.error_class, ''), COALESCE(j.attempt_count, 0), COALESCE(j.next_retry_at, '')
            FROM alphas a
            LEFT JOIN submit_jobs j ON a.alpha_id = j.alpha_id
            WHERE a.alpha_id != ''
              AND a.submitted = 0
              AND a.all_passed = 1
              AND a.error = ''
              AND a.submit_state IN ('queued', 'failed', 'new', 'gated')
              AND (
                    j.next_retry_at IS NULL
                    OR j.next_retry_at = ''
                    OR j.next_retry_at <= CURRENT_TIMESTAMP
                  )
              AND (
                    j.state IS NULL
                    OR j.state IN ('queued', 'failed')
                  )
            ORDER BY a.sharpe DESC, a.created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()

    def submit_funnel_metrics(self, lookback_hours: int = 24) -> dict:
        cursor = self.conn.execute(
            """
            SELECT
                COUNT(*) as generated,
                SUM(CASE WHEN error = '' THEN 1 ELSE 0 END) as simulated,
                SUM(CASE WHEN submit_state = 'queued' THEN 1 ELSE 0 END) as queued,
                SUM(CASE WHEN submit_state = 'submitted' OR submitted = 1 THEN 1 ELSE 0 END) as submitted,
                SUM(CASE WHEN accepted_at != '' THEN 1 ELSE 0 END) as accepted
            FROM alphas
            WHERE created_at >= datetime('now', ?)
            """,
            (f"-{lookback_hours} hour",),
        )
        row = cursor.fetchone()
        return {
            "generated": row[0] or 0,
            "simulated": row[1] or 0,
            "queued": row[2] or 0,
            "submitted": row[3] or 0,
            "accepted": row[4] or 0,
        }

    def minute_kpis(self, lookback_minutes: int = 60) -> dict:
        cursor = self.conn.execute(
            """
            SELECT
                COUNT(*) as generated,
                SUM(CASE WHEN error = '' THEN 1 ELSE 0 END) as simulated,
                SUM(CASE WHEN submit_state = 'gated' THEN 1 ELSE 0 END) as gated,
                SUM(CASE WHEN submit_state = 'queued' THEN 1 ELSE 0 END) as queued,
                SUM(CASE WHEN submit_state = 'submitted' THEN 1 ELSE 0 END) as submitted,
                SUM(CASE WHEN submit_state = 'dead_lettered' THEN 1 ELSE 0 END) as dead_lettered,
                SUM(CASE WHEN submit_state = 'accepted' THEN 1 ELSE 0 END) as accepted,
                SUM(CASE WHEN submit_state = 'rejected' AND submitted_at != '' THEN 1 ELSE 0 END) as rejected_after_submit
            FROM alphas
            WHERE COALESCE(created_at, CURRENT_TIMESTAMP) >= datetime('now', ?)
            """,
            (f"-{max(1, int(lookback_minutes))} minutes",),
        )
        row = cursor.fetchone()
        generated = row[0] or 0
        simulated = row[1] or 0
        gated = row[2] or 0
        queued = row[3] or 0
        submitted_pending = row[4] or 0
        dead_lettered = row[5] or 0
        accepted = row[6] or 0
        rejected_after_submit = row[7] or 0
        submit_ok_total = accepted + rejected_after_submit
        total_submitted_lifecycle = submitted_pending + submit_ok_total
        gate_pass_rate = (gated / generated) if generated else 0.0
        submit_success_rate = (submitted_pending / queued) if queued else 0.0
        submit_ok_rate = (
            (submit_ok_total / total_submitted_lifecycle)
            if total_submitted_lifecycle
            else 0.0
        )
        dlq_rate = (dead_lettered / queued) if queued else 0.0
        true_accept_rate = (accepted / submit_ok_total) if submit_ok_total else 0.0
        true_reject_rate = (
            (rejected_after_submit / submit_ok_total) if submit_ok_total else 0.0
        )
        return {
            "generated": generated,
            "simulated": simulated,
            "gated": gated,
            "queued": queued,
            "submitted": submitted_pending,
            "dead_lettered": dead_lettered,
            "accepted": accepted,
            "rejected_after_submit": rejected_after_submit,
            "gate_pass_rate": gate_pass_rate,
            "submit_success_rate": submit_success_rate,
            "submit_ok_rate": submit_ok_rate,
            "dlq_rate": dlq_rate,
            "true_accept_rate": true_accept_rate,
            "true_reject_rate": true_reject_rate,
        }

    def acceptance_rate_by_arm(
        self, min_submitted: int = 5, lookback_hours: int = 168
    ) -> dict:
        """
        Returns P(true_accept | submitted) per strategy_cluster arm.

        Only counts alphas with a *resolved* WQ review decision
        (submit_state = 'accepted' OR submit_state = 'rejected' with a
        submitted_at timestamp, meaning they went through the full WQ pipeline).
        Arms with fewer than `min_submitted` resolved outcomes get a
        uniform prior of 0.5 so we don't prematurely collapse exploration.

        Returns:
            dict[arm_name] -> {"accepted": int, "resolved": int, "p_accept": float}
        """
        with self._conn_lock:
            cursor = self.conn.execute(
                """
                SELECT
                    strategy_cluster,
                    SUM(CASE WHEN submit_state = 'accepted' THEN 1 ELSE 0 END) AS accepted,
                    SUM(CASE
                            WHEN submit_state IN ('accepted', 'rejected')
                             AND submitted_at != ''
                            THEN 1 ELSE 0
                        END) AS resolved
                FROM alphas
                WHERE submitted_at != ''
                  AND COALESCE(created_at, CURRENT_TIMESTAMP) >= datetime('now', ?)
                GROUP BY strategy_cluster
                """,
                (f"-{max(1, int(lookback_hours))} hours",),
            )
            rows = cursor.fetchall()

        result: dict = {}
        for cluster, accepted, resolved in rows:
            arm = cluster or "deterministic"
            p = 0.5  # Uniform prior for cold-start arms
            if resolved and resolved >= min_submitted:
                p = float(accepted or 0) / float(resolved)
            result[arm] = {
                "accepted": int(accepted or 0),
                "resolved": int(resolved or 0),
                "p_accept": p,
            }
        return result

    def acceptance_rate_dict(
        self,
        min_submitted: int = 5,
        lookback_hours: int = 168,
    ) -> Dict[str, float]:
        """
        Clean dict interface: arm_name → Bayesian-smoothed P(accept | submitted).

        Wraps acceptance_rate_by_arm() so callers get a simple dict without
        having to unpack the inner {"accepted", "resolved", "p_accept"} structure.

        Bayesian smoothing is already applied inside acceptance_rate_by_arm()
        (resolved < min_submitted → uniform 0.5 prior), so no additional
        pseudo-counting is needed here.

        Args:
            min_submitted: minimum resolved count before p_accept is trusted.
            lookback_hours: hours of history to query.

        Returns:
            dict[arm_name] -> float  (Bayesian-smoothed acceptance rate in [0,1])
        """
        raw = self.acceptance_rate_by_arm(
            min_submitted=min_submitted,
            lookback_hours=lookback_hours,
        )
        return {arm: info["p_accept"] for arm, info in raw.items()}

    def load_qd_archive(self, limit: int = 2000) -> list[tuple[str, str, float, float]]:
        cursor = self.conn.execute(
            """
            SELECT descriptor, expression, quality_score, novelty_score
            FROM qd_archive
            ORDER BY quality_score DESC, updated_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        return cursor.fetchall()

    def upsert_qd_archive(
        self,
        descriptor: str,
        expression: str,
        quality_score: float,
        novelty_score: float,
    ):
        self.conn.execute(
            """
            INSERT INTO qd_archive(descriptor, expression, quality_score, novelty_score, updates, updated_at)
            VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(descriptor) DO UPDATE SET
                expression = CASE
                    WHEN excluded.quality_score >= qd_archive.quality_score
                    THEN excluded.expression
                    ELSE qd_archive.expression
                END,
                quality_score = MAX(qd_archive.quality_score, excluded.quality_score),
                novelty_score = excluded.novelty_score,
                updates = qd_archive.updates + 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                (descriptor or "")[:120],
                (expression or "")[:1200],
                float(quality_score or 0.0),
                float(novelty_score or 0.0),
            ),
        )
        self.conn.commit()

    def qd_archive_stats(self) -> dict:
        cursor = self.conn.execute(
            """
            SELECT COUNT(*), AVG(quality_score), MAX(quality_score), SUM(updates)
            FROM qd_archive
            """
        )
        row = cursor.fetchone() or (0, 0.0, 0.0, 0)
        return {
            "cells": int(row[0] or 0),
            "avg_quality": float(row[1] or 0.0),
            "max_quality": float(row[2] or 0.0),
            "updates": int(row[3] or 0),
        }

    def get_top_alphas(self, n: int = 10) -> list:
        """Get top alphas by Sharpe ratio"""
        cursor = self.conn.execute(
            """
            SELECT expression, sharpe, fitness, turnover, alpha_url, all_passed
            FROM alphas
            WHERE error = '' AND sharpe > 0
            ORDER BY sharpe DESC LIMIT ?
        """,
            (n,),
        )
        return cursor.fetchall()

    def get_submittable(self) -> list:
        """Get all alphas that passed but haven't been submitted"""
        cursor = self.conn.execute("""
            SELECT expression, sharpe, fitness, turnover, alpha_id, alpha_url
            FROM alphas
            WHERE all_passed = 1 AND submitted = 0 AND alpha_id != ''
            ORDER BY sharpe DESC
        """)
        return cursor.fetchall()

    def get_daily_stats(self, date: str = None) -> dict:
        """Get statistics for a specific date"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cursor = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN error = '' THEN 1 ELSE 0 END) as simulated,
                SUM(CASE WHEN all_passed = 1 THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN submitted = 1 THEN 1 ELSE 0 END) as submitted,
                AVG(CASE WHEN sharpe > 0 THEN sharpe END) as avg_sharpe,
                MAX(sharpe) as max_sharpe
            FROM alphas
            WHERE created_at LIKE ?
        """,
            (f"{date}%",),
        )

        row = cursor.fetchone()
        return {
            "date": date,
            "total": row[0] or 0,
            "simulated": row[1] or 0,
            "passed": row[2] or 0,
            "submitted": row[3] or 0,
            "avg_sharpe": round(row[4] or 0, 3),
            "max_sharpe": round(row[5] or 0, 3),
        }

    def export_csv(self, date: str = None) -> str:
        """Export daily results to CSV"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        csv_path = os.path.join(RESULTS_DIR, f"alphas_{date}.csv")
        cursor = self.conn.execute(
            """
            SELECT expression, sharpe, fitness, turnover, all_passed,
                   alpha_url, error, region, quality_tier, strategy_cluster,
                   risk_flags, submit_state, submit_attempts, run_id, batch_id, created_at
            FROM alphas
            WHERE created_at LIKE ?
            ORDER BY sharpe DESC
        """,
            (f"{date}%",),
        )

        rows = cursor.fetchall()
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "expression",
                        "sharpe",
                        "fitness",
                        "turnover",
                        "all_passed",
                        "alpha_url",
                        "error",
                        "region",
                        "quality_tier",
                        "strategy_cluster",
                        "risk_flags",
                        "submit_state",
                        "submit_attempts",
                        "run_id",
                        "batch_id",
                        "created_at",
                    ]
                )
                writer.writerows(rows)

        return csv_path

    def is_duplicate(self, expression: str) -> bool:
        """Check exact duplicate via canonicalized expression."""
        with self._conn_lock:
            canonical = canonicalize_expression(expression)
            cursor = self.conn.execute(
                "SELECT 1 FROM alpha_signatures WHERE canonical_expr = ? LIMIT 1",
                (canonical,),
            )
            if cursor.fetchone():
                return True

            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM alphas WHERE expression = ? OR canonical_expr = ?",
                (expression.strip(), canonical),
            )
            if cursor.fetchone()[0] > 0:
                return True

            # Backward compatibility for old rows that predate canonical_expr column usage.
            cursor = self.conn.execute(
                "SELECT expression FROM alphas WHERE canonical_expr = '' ORDER BY id DESC LIMIT 3000"
            )
            for (hist_expr,) in cursor.fetchall():
                if canonicalize_expression(hist_expr) == canonical:
                    return True
            return False

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        union = len(a | b)
        if union == 0:
            return 0.0
        return len(a & b) / union

    def is_collinear(self, expression: str, threshold: float = 0.85) -> bool:
        """
        Check structural collinearity against recent history using optimized SQL.
        """
        with self._conn_lock:
            new_canonical = canonicalize_expression(expression)
            new_sig = parameter_agnostic_signature(expression)

            # 1. Quick check for exact canonical or structural signature match
            cursor = self.conn.execute(
                "SELECT 1 FROM alpha_signatures WHERE canonical_expr = ? OR param_signature = ? LIMIT 1",
                (new_canonical, new_sig),
            )
            if cursor.fetchone():
                return True

            # 2. In-memory structure cache for faster rank-time checks.
            new_tokens = token_set(expression, strip_numbers=True)
            new_ops = operator_set(expression)
            if not new_tokens:
                return False

            for h_tokens, h_ops in reversed(self._recent_structure_cache[-1200:]):
                token_sim = self._jaccard(new_tokens, h_tokens)
                if token_sim >= threshold:
                    return True
                if token_sim >= 0.60:
                    op_sim = self._jaccard(new_ops, h_ops)
                    if op_sim >= 0.85:
                        return True
            return False

    def get_theme_stats(self) -> list:
        """Get pass rate and avg Sharpe grouped by theme."""
        cursor = self.conn.execute("""
            SELECT theme,
                   COUNT(*) as total,
                   SUM(CASE WHEN all_passed = 1 THEN 1 ELSE 0 END) as passed,
                   AVG(CASE WHEN sharpe > 0 THEN sharpe END) as avg_sharpe,
                   MAX(sharpe) as max_sharpe
            FROM alphas
            WHERE theme != 'unknown' AND error = ''
            GROUP BY theme
            ORDER BY passed DESC
        """)
        return cursor.fetchall()

    def get_mutation_stats(self) -> list:
        """Get effectiveness of each mutation type."""
        cursor = self.conn.execute("""
            SELECT mutation_type,
                   COUNT(*) as total,
                   SUM(CASE WHEN all_passed = 1 THEN 1 ELSE 0 END) as passed,
                   AVG(CASE WHEN sharpe > 0 THEN sharpe END) as avg_sharpe
            FROM alphas
            WHERE mutation_type != '' AND error = ''
            GROUP BY mutation_type
            ORDER BY passed DESC
        """)
        return cursor.fetchall()

    def get_family_stats(self) -> list:
        """Get how many passes each family has (for crowding detection)."""
        cursor = self.conn.execute("""
            SELECT family,
                   COUNT(*) as total,
                   SUM(CASE WHEN all_passed = 1 THEN 1 ELSE 0 END) as passed,
                   MAX(sharpe) as best_sharpe
            FROM alphas
            WHERE family != '' AND error = ''
            GROUP BY family
            HAVING passed > 0
            ORDER BY passed DESC
            LIMIT 20
        """)
        return cursor.fetchall()

    def get_family_pass_count(self, family: str) -> int:
        """Return number of passed alphas in a family."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM alphas WHERE family = ? AND all_passed = 1", (family,)
        )
        return cursor.fetchone()[0]

    def get_family_pass_counts(self) -> dict:
        """Return pass counts for all families (for rank-time crowding penalties)."""
        cursor = self.conn.execute("""
            SELECT family, COUNT(*) as passed
            FROM alphas
            WHERE family != '' AND all_passed = 1 AND error = ''
            GROUP BY family
        """)
        return {family: passed for family, passed in cursor.fetchall()}

    def get_passing_expressions(self) -> list[str]:
        """Get all expressions that passed (for evolution)"""
        cursor = self.conn.execute("""
            SELECT expression FROM alphas
            WHERE all_passed = 1 AND error = ''
            ORDER BY sharpe DESC
        """)
        return [row[0] for row in cursor.fetchall()]

    def get_region_performance(self, lookback_days: int = 14) -> list[tuple]:
        """
        Return recent region performance:
        (region, total, passed, pass_rate, avg_sharpe)
        """
        cursor = self.conn.execute(
            """
            SELECT region,
                   COUNT(*) as total,
                   SUM(CASE WHEN all_passed = 1 THEN 1 ELSE 0 END) as passed,
                   AVG(CASE WHEN sharpe > 0 THEN sharpe END) as avg_sharpe
            FROM alphas
            WHERE error = ''
              AND created_at >= datetime('now', ?)
            GROUP BY region
        """,
            (f"-{lookback_days} day",),
        )
        rows = []
        for region, total, passed, avg_sharpe in cursor.fetchall():
            total = total or 0
            passed = passed or 0
            pass_rate = (passed / total) if total else 0.0
            rows.append((region, total, passed, pass_rate, avg_sharpe or 0.0))
        rows.sort(key=lambda x: (x[3], x[4]), reverse=True)
        return rows

    def close(self):
        self.conn.close()


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    tracker = AlphaTracker(":memory:")
    stats = tracker.get_daily_stats()
    print(f"Daily stats: {stats}")
    print(f"Top alphas: {tracker.get_top_alphas()}")
    tracker.close()
