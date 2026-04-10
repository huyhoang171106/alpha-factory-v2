"""
dashboard.py — Rich Terminal Dashboard for Alpha Factory
Shows real-time alpha formulas, results, progress bars, and ETA.
Uses only stdlib (no rich/curses dependency).
"""

import os
import sys
import time
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────
# ANSI helpers
# ──────────────────────────────────────────────────────
_COLORS = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "red":     "\033[91m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "blue":    "\033[94m",
    "magenta": "\033[95m",
    "cyan":    "\033[96m",
    "white":   "\033[97m",
    "bg_green":  "\033[42m",
    "bg_red":    "\033[41m",
    "bg_blue":   "\033[44m",
    "bg_yellow": "\033[43m",
}

def _c(text, color):
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"

def _bar(current, total, width=30, fill_char="█", empty_char="░"):
    if total == 0:
        return empty_char * width
    ratio = min(current / total, 1.0)
    filled = int(width * ratio)
    return fill_char * filled + empty_char * (width - filled)

def _truncate(s, maxlen=60):
    return s[:maxlen-1] + "…" if len(s) > maxlen else s

def _term_width():
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80


# ──────────────────────────────────────────────────────
# Live Dashboard State
# ──────────────────────────────────────────────────────
@dataclass
class AlphaRow:
    """One row in the live results table."""
    index: int
    expression: str
    theme: str = ""
    status: str = "⏳"         # ⏳ pending, 🔄 running, ✅ pass, ❌ fail, ⚠️ partial
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    checks: str = ""           # e.g. "9/9"
    elapsed: float = 0.0
    error: str = ""


@dataclass
class DashboardState:
    """Full dashboard state for one cycle."""
    cycle: int = 0
    session_start: float = 0.0
    cycle_start: float = 0.0

    # Pipeline stages
    phase: str = "INIT"        # GENERATE, VALIDATE, DEDUP, RANK, SIMULATE, EVOLVE, SUBMIT
    total_seeds: int = 0
    generated: int = 0
    validated: int = 0
    unique: int = 0
    ranked: int = 0
    simulated: int = 0
    passed: int = 0
    submitted: int = 0

    # Current simulation
    sim_total: int = 0
    sim_done: int = 0
    sim_current_expr: str = ""
    sim_current_theme: str = ""

    # Results table
    rows: List[AlphaRow] = field(default_factory=list)

    # Session totals
    session_simulated: int = 0
    session_passed: int = 0
    session_submitted: int = 0
    session_best_sharpe: float = 0.0

    # Timing
    avg_sim_time: float = 30.0  # rolling average per alpha


def clear_screen():
    """Clear terminal (cross-platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')


def render_header(state: DashboardState):
    """Render the top header bar."""
    w = _term_width()
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = time.time() - state.session_start
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    print(_c("═" * w, "cyan"))
    title = f"  🏭 ALPHA FACTORY — Cycle {state.cycle}  |  {now}  |  Session: {elapsed_str}"
    print(_c(title, "bold"))
    print(_c("═" * w, "cyan"))


def render_pipeline(state: DashboardState):
    """Render the pipeline funnel stages."""
    stages = [
        ("GENERATE",  state.generated,  "🌱", "blue"),
        ("VALIDATE",  state.validated,   "✓",  "cyan"),
        ("DEDUP",     state.unique,      "🔍", "magenta"),
        ("RANK",      state.ranked,      "📊", "yellow"),
        ("SIMULATE",  state.simulated,   "⚡", "green"),
        ("PASSED",    state.passed,      "✅", "green"),
        ("SUBMIT",    state.submitted,   "🚀", "bold"),
    ]

    # Current phase indicator
    phase_order = ["GENERATE", "VALIDATE", "DEDUP", "RANK", "SIMULATE", "EVOLVE", "SUBMIT"]
    current_idx = phase_order.index(state.phase) if state.phase in phase_order else -1

    line_parts = []
    for i, (name, count, icon, color) in enumerate(stages):
        s_name = name[:4]
        if name == state.phase:
            part = _c(f" {icon}{count:>3} ", "bold")
        elif i < current_idx:
            part = _c(f" {icon}{count:>3} ", "dim")
        else:
            part = f" {icon}{count:>3} "
        line_parts.append(part)

    arrow = _c(" → ", "dim")
    print("  " + arrow.join(line_parts))
    print()


def render_simulation_progress(state: DashboardState):
    """Render the simulation progress bar with current formula."""
    if state.sim_total == 0:
        return

    pct = (state.sim_done / state.sim_total * 100) if state.sim_total > 0 else 0
    bar = _bar(state.sim_done, state.sim_total, width=35)

    # ETA calculation
    remaining = state.sim_total - state.sim_done
    eta_secs = remaining * state.avg_sim_time
    eta_str = str(timedelta(seconds=int(eta_secs))) if eta_secs < 36000 else "∞"

    print(f"  {_c('SIMULATING', 'yellow')}  [{_c(bar, 'green')}]  "
          f"{_c(f'{state.sim_done}', 'bold')}/{state.sim_total}  "
          f"({pct:.0f}%)  "
          f"ETA: {_c(eta_str, 'cyan')}")

    # Current formula being tested
    if state.sim_current_expr:
        theme_tag = f"[{state.sim_current_theme}]" if state.sim_current_theme else ""
        expr_display = _truncate(state.sim_current_expr, 65)
        print(f"  {_c('NOW:', 'dim')} {_c(theme_tag, 'magenta')} {expr_display}")
    print()


def render_results_table(state: DashboardState, max_rows: int = 15):
    """Render the live results table."""
    if not state.rows:
        return

    w = _term_width()
    # Header
    header = f"  {'#':>3}  {'Status':6}  {'Sharpe':>7}  {'Fit':>5}  {'Turn%':>6}  {'Chk':>5}  {'Time':>5}  Expression"
    print(_c(header, "dim"))
    print(_c("  " + "─" * min(w - 4, 100), "dim"))

    # Show latest rows (most recent first)
    display_rows = state.rows[-max_rows:]
    for row in display_rows:
        # Color by status
        if row.status == "✅":
            color = "green"
        elif row.status == "❌":
            color = "red"
        elif row.status == "⚠️":
            color = "yellow"
        elif row.status == "🔄":
            color = "cyan"
        else:
            color = "dim"

        sharpe_str = f"{row.sharpe:.3f}" if row.sharpe is not None else "  —  "
        fit_str    = f"{row.fitness:.2f}" if row.fitness is not None else " — "
        turn_str   = f"{row.turnover:.1f}" if row.turnover is not None else "  — "
        checks_str = row.checks if row.checks else " — "
        time_str   = f"{row.elapsed:.0f}s" if row.elapsed > 0 else " — "
        expr_max   = min(45, w - 55)
        expr_str   = _truncate(row.expression, expr_max)

        # Highlight high Sharpe
        if row.sharpe and row.sharpe >= 1.5:
            sharpe_str = _c(sharpe_str, "green")
        elif row.sharpe and row.sharpe >= 1.25:
            sharpe_str = _c(sharpe_str, "yellow")

        line = f"  {row.index:>3}  {row.status:6}  {sharpe_str:>7}  {fit_str:>5}  {turn_str:>6}  {checks_str:>5}  {time_str:>5}  {_c(expr_str, color)}"
        print(line)

    if len(state.rows) > max_rows:
        hidden = len(state.rows) - max_rows
        print(_c(f"  ... +{hidden} more rows (showing latest {max_rows})", "dim"))
    print()


def render_session_stats(state: DashboardState):
    """Render session-level cumulative stats."""
    w = _term_width()
    elapsed = time.time() - state.session_start
    rate = state.session_simulated / (elapsed / 3600) if elapsed > 60 else 0

    print(_c("  ─── Session Totals ───", "dim"))
    stats_line = (
        f"  Simulated: {_c(str(state.session_simulated), 'bold')}  │  "
        f"Passed: {_c(str(state.session_passed), 'green')}  │  "
        f"Submitted: {_c(str(state.session_submitted), 'cyan')}  │  "
        f"Best Sharpe: {_c(f'{state.session_best_sharpe:.3f}', 'yellow')}  │  "
        f"Rate: {_c(f'{rate:.0f}/hr', 'magenta')}"
    )
    print(stats_line)
    print()


def render_full(state: DashboardState):
    """Render the complete dashboard (non-destructive, append-mode)."""
    render_header(state)
    render_pipeline(state)
    render_simulation_progress(state)
    render_results_table(state)
    render_session_stats(state)


def render_cycle_header(state: DashboardState):
    """Render a compact cycle start header."""
    w = _term_width()
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = time.time() - state.session_start
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    print()
    print(_c("═" * w, "cyan"))
    print(_c(f"  🔄 Cycle {state.cycle}  │  {now}  │  Session: {elapsed_str}", "bold"))
    print(_c("═" * w, "cyan"))


def render_phase_start(state: DashboardState, phase: str, detail: str = ""):
    """Render a phase transition line."""
    icons = {
        "GENERATE": "🌱", "VALIDATE": "✓ ", "DEDUP": "🔍",
        "RANK": "📊", "SIMULATE": "⚡", "EVOLVE": "🧬",
        "SUBMIT": "🚀", "REPLICATE": "🌍",
    }
    icon = icons.get(phase, "▪")
    state.phase = phase
    msg = f"  {icon} {phase}"
    if detail:
        msg += f" — {detail}"
    print(_c(msg, "cyan"))


def render_sim_row_live(row: AlphaRow):
    """Print a single simulation result row as it arrives."""
    if row.status == "✅":
        color = "green"
    elif row.status == "❌":
        color = "red"
    elif row.status == "⚠️":
        color = "yellow"
    else:
        color = "dim"

    sharpe_str = f"{row.sharpe:.3f}" if row.sharpe is not None else "  —  "
    fit_str    = f"{row.fitness:.2f}" if row.fitness is not None else " — "
    turn_str   = f"{row.turnover:.1f}%" if row.turnover is not None else "  — "
    checks_str = row.checks if row.checks else " — "
    time_str   = f"{row.elapsed:.0f}s" if row.elapsed > 0 else " — "
    expr_str   = _truncate(row.expression, 50)

    # Highlight values
    if row.sharpe and row.sharpe >= 1.5:
        sharpe_str = _c(sharpe_str, "green")
    elif row.sharpe and row.sharpe >= 1.25:
        sharpe_str = _c(sharpe_str, "yellow")

    line = (f"  {row.status}  "
            f"S={sharpe_str}  F={fit_str}  T={turn_str}  "
            f"Chk={checks_str}  "
            f"{time_str}  "
            f"{_c(expr_str, color)}")
    print(line)


def render_sim_progress_inline(done: int, total: int, avg_time: float):
    """Print an inline progress update line (overwrite-friendly)."""
    pct = (done / total * 100) if total > 0 else 0
    remaining = total - done
    eta_secs = remaining * avg_time
    eta_str = str(timedelta(seconds=int(eta_secs))) if eta_secs < 36000 else "∞"
    bar = _bar(done, total, width=25)
    line = f"\r  ⚡ [{bar}] {done}/{total} ({pct:.0f}%)  ETA: {eta_str}  "
    sys.stdout.write(line)
    sys.stdout.flush()


def render_cycle_summary(state: DashboardState):
    """Render end-of-cycle summary."""
    cycle_time = time.time() - state.cycle_start
    cycle_str = str(timedelta(seconds=int(cycle_time)))

    print()
    print(_c("  ┌─ Cycle Summary ─────────────────────────────────┐", "dim"))
    print(f"  │ Generated: {state.generated:>4}  → Validated: {state.validated:>4}  "
          f"→ Unique: {state.unique:>4}   │")
    print(f"  │ Ranked:    {state.ranked:>4}  → Simulated: {state.simulated:>4}  "
          f"→ Passed: {state.passed:>4}   │")
    print(f"  │ Submitted: {state.submitted:>4}  │  Cycle time: {cycle_str:>12}        │")
    print(_c("  └──────────────────────────────────────────────────┘", "dim"))

    # Session cumulative
    render_session_stats(state)


def render_cooldown(seconds: int, shutdown_flag_fn):
    """Render a live cooldown timer."""
    for i in range(seconds, 0, -1):
        if shutdown_flag_fn():
            break
        bar = _bar(seconds - i, seconds, width=20, fill_char="▓", empty_char="░")
        sys.stdout.write(f"\r  😴 Cooldown [{bar}] {i}s remaining  ")
        sys.stdout.flush()
        time.sleep(1)
    print()  # newline after countdown
