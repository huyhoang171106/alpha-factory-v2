"""wqf/db.py — SQLite persistence for candidates and results."""
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "alpha_finder.db")


class AlphaDB:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expression TEXT UNIQUE NOT NULL,
                score REAL,
                hypothesis TEXT,
                reasons TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expression TEXT NOT NULL,
                sharpe REAL DEFAULT 0,
                fitness REAL DEFAULT 0,
                turnover REAL DEFAULT 0,
                sub_sharpe REAL DEFAULT -1,
                all_passed INTEGER DEFAULT 0,
                error TEXT DEFAULT '',
                simulated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_candidate(self, expr: str, score: float, hypo: str, reasons: str):
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO candidates
                   (expression, score, hypothesis, reasons)
                   VALUES (?, ?, ?, ?)""",
                (expr, score, hypo, reasons),
            )
            self.conn.commit()

    def save_result(self, expr: str, result: dict):
        if not result:
            return
        with self._lock:
            self.conn.execute(
                """INSERT INTO results
                   (expression, sharpe, fitness, turnover, sub_sharpe, all_passed, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    expr,
                    result.get("sharpe", 0),
                    result.get("fitness", 0),
                    result.get("turnover", 0),
                    result.get("sub_sharpe", -1),
                    1 if result.get("all_passed") else 0,
                    result.get("error", ""),
                ),
            )
            self.conn.commit()

    def get_stats(self) -> dict:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM results")
        total = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM results WHERE sharpe >= 1.25 AND error = ''")
        wins = cur.fetchone()[0] or 0
        cur.execute("SELECT AVG(sharpe) FROM results WHERE sharpe != 0 AND error = ''")
        avg = cur.fetchone()[0] or 0.0
        cur.execute("SELECT COUNT(*) FROM results WHERE sharpe >= 2.5 AND error = ''")
        elite = cur.fetchone()[0] or 0
        return {
            "total": total,
            "win_rate": wins / max(total, 1),
            "avg_sharpe": round(avg, 3),
            "elite": elite,
        }

    def get_top(self, n: int = 10) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT r.expression, r.sharpe, r.fitness, r.turnover, c.hypothesis
               FROM results r
               JOIN candidates c ON c.expression = r.expression
               WHERE r.sharpe > 0 AND r.error = ''
               ORDER BY r.sharpe DESC LIMIT ?""",
            (n,),
        )
        cols = ["expression", "sharpe", "fitness", "turnover", "hypothesis"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_hypothesis_breakdown(self) -> dict:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT c.hypothesis,
                   COUNT(*) as total,
                   AVG(CASE WHEN r.sharpe >= 1.25 THEN 1.0 ELSE 0.0 END) as win_rate,
                   AVG(r.sharpe) as avg_sharpe
            FROM candidates c
            LEFT JOIN results r ON r.expression = c.expression
            GROUP BY c.hypothesis
            ORDER BY win_rate DESC
        """)
        return {row[0]: {"total": row[1], "win_rate": row[2], "avg_sharpe": row[3]}
                for row in cur.fetchall()}

    def close(self):
        self.conn.close()
