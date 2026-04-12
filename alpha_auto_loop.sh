#!/bin/bash
# alpha_auto_loop.sh - Auto-run Alpha Factory with 60-minute cycles and self-improvement
# Usage: ./alpha_auto_loop.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CYCLE_MINUTES=60
MAX_CYCLES=0  # 0 = infinite
LOG_FILE="logs/auto_loop_$(date +%Y%m%d).log"
IMPROVE_SCRIPT="$SCRIPT_DIR/auto_improve.py"

mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Detect venv Python - check if .venv exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    if [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
        PYTHON="$SCRIPT_DIR/.venv/Scripts/python.exe"
    elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
        PYTHON="$SCRIPT_DIR/.venv/bin/python"
    fi
fi

# Ensure venv exists and is set up if not found
if [ -z "$PYTHON" ] || [ ! -f "$PYTHON" ]; then
    log "Creating venv..."
    python -m venv "$SCRIPT_DIR/.venv"
    # Re-detect after creation
    if [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
        PYTHON="$SCRIPT_DIR/.venv/Scripts/python.exe"
    elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
        PYTHON="$SCRIPT_DIR/.venv/bin/python"
    fi
    log "Installing dependencies..."
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

log "Using Python: $PYTHON"

# Check for auto_improve.py or create default improvement logic
if [ ! -f "$IMPROVE_SCRIPT" ]; then
    log "Creating default auto_improve.py..."
    cat > "$IMPROVE_SCRIPT" << 'IMPROVE_EOF'
#!/usr/bin/env python3
"""
Auto-improvement module - analyzes metrics and suggests/adjusts parameters
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
TRACKER_DB = ROOT / "tracker.db"

def get_recent_kpis(minutes=60):
    """Fetch recent KPIs from tracker"""
    try:
        sys.path.insert(0, str(ROOT))
        from tracker import AlphaTracker
        tracker = AlphaTracker()
        kpis = tracker.minute_kpis(minutes)
        qd = tracker.qd_archive_stats()
        tracker.close()
        return {"kpis": kpis, "qd": qd}
    except Exception as e:
        return {"error": str(e)}

def analyze_and_suggest():
    """Main improvement logic"""
    data = get_recent_kpis(60)

    suggestions = []

    if "error" not in data:
        kpis = data.get("kpis", {})
        qd = data.get("qd", {})

        # Analyze submission rate
        submitted = kpis.get("submitted", 0)
        accepted = kpis.get("accepted", 0)
        rejected = kpis.get("rejected", 0)

        if submitted > 0:
            accept_rate = accepted / submitted
            print(f"Acceptance rate: {accept_rate:.1%}")

            if accept_rate < 0.3:
                suggestions.append("LOW_ACCEPT_RATE: Consider increasing ASYNC_MIN_SHARPE threshold")
            elif accept_rate > 0.7:
                suggestions.append("HIGH_ACCEPT_RATE: Can lower ASYNC_MIN_SHARPE for more candidates")

        # Check QD archive health
        archive_size = qd.get("total_alphas", 0)
        print(f"QD Archive size: {archive_size}")

        if archive_size > 0:
            avg_fitness = qd.get("avg_fitness", 0)
            if avg_fitness < 0.7:
                suggestions.append("LOW_AVG_FITNESS: Consider adjusting ranker thresholds")

    # Write suggestions for next run
    if suggestions:
        with open(ROOT / ".auto_improve_suggestions.json", "w") as f:
            json.dump({"suggestions": suggestions, "timestamp": datetime.now().isoformat()}, f, indent=2)
        print("Improvement suggestions:", suggestions)
    else:
        print("No improvements needed - system is healthy")

    return suggestions

if __name__ == "__main__":
    analyze_and_suggest()
IMPROVE_EOF
    log "Created auto_improve.py"
fi

# Main loop
cycle=0
while true; do
    cycle=$((cycle + 1))
    log "=== Starting cycle $cycle ==="

    # Run the auto command
    log "Running: $PYTHON alpha_factory_cli.py auto --profile local"
    "$PYTHON" alpha_factory_cli.py auto --profile local

    exit_code=$?
    log "Auto command exited with code: $exit_code"

    # Run auto-improvement analysis
    log "Running auto-improvement analysis..."
    "$PYTHON" "$IMPROVE_SCRIPT" >> "$LOG_FILE" 2>&1

    # Check for suggestions and apply them
    if [ -f ".auto_improve_suggestions.json" ]; then
        log "Found improvement suggestions"
        # Could parse and apply suggestions here
        # For now, just log them
        cat ".auto_improve_suggestions.json" >> "$LOG_FILE"
    fi

    # Check if we should stop
    if [ $MAX_CYCLES -gt 0 ] && [ $cycle -ge $MAX_CYCLES ]; then
        log "Reached max cycles ($MAX_CYCLES) - stopping"
        break
    fi

    log "Waiting $CYCLE_MINUTES minutes before next cycle..."
    sleep $((CYCLE_MINUTES * 60))
done

log "Auto loop finished after $cycle cycles"
