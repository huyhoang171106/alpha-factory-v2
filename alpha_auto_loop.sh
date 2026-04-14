#!/bin/bash
# alpha_auto_loop.sh - HARD RESET loop (NO SLEEP) with auto-improvement
# Usage: ./alpha_auto_loop.sh
#
# Environment variables:
#   ASYNC_USE_RAG=1          Enable RAG generation (default 0)
#   ALPHA_LOOP_LIMIT=N       Limit number of alphas per cycle (optional)
#   ALPHA_LOOP_SCORE=N       Minimum score threshold (optional)
#   ALPHA_LOOP_NO_HYBRID=1   Disable hybrid supervisor mode (optional)
#   MAX_CYCLES=N             Stop after N cycles (0 = infinite, default 0)
#
# The script runs alpha_factory_cli.py auto in a loop with hard cleanup between cycles.
# It kills related processes, removes lock files, and runs auto-improvement analysis.
# No sleep between cycles (except 1s race condition guard).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CYCLE_MINUTES=60  # kept for compatibility but not used in no-sleep mode
MAX_CYCLES=0  # 0 = infinite
LOG_FILE="logs/auto_loop_$(date +%Y%m%d).log"
IMPROVE_SCRIPT="$SCRIPT_DIR/auto_improve.py"
LOCK_FILE="alpha_factory.lock"

# RAG/LLM settings - set to 1 to enable RAG generation
ASYNC_USE_RAG="${ASYNC_USE_RAG:-0}"

mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

########################################
# 💀 HARD CLEAN
########################################
hard_cleanup() {
    log "💀 Killing ALL related processes..."

    # Kill processes by name patterns
    pkill -9 -f alpha_factory 2>/dev/null || true
    pkill -9 -f run_async_pipeline 2>/dev/null || true
    pkill -9 -f auto_improve 2>/dev/null || true
    pkill -9 -f tracker 2>/dev/null || true

    # Kill any remaining python processes spawned by this script (optional)
    # CURRENT_PID=$$
    # ps -eo pid,cmd | grep python | grep -v grep | while read pid cmd; do
    #     if [ "$pid" != "$CURRENT_PID" ]; then
    #         kill -9 "$pid" 2>/dev/null || true
    #     fi
    # done

    # Remove lock files
    rm -f "$LOCK_FILE" 2>/dev/null || true
    rm -f *.lock 2>/dev/null || true

    log "✅ Cleanup done"
}

########################################
# 📊 MEMORY MONITORING
########################################
log_memory() {
    if command -v free >/dev/null 2>&1; then
        free -m | awk '
            /^Mem:/{printf "Memory: %sMB total, %sMB used, %sMB free\n", $2, $3, $4}
            /^Swap:/{printf "Swap: %sMB total, %sMB used, %sMB free\n", $2, $3, $4}'
    else
        log "free command not available, cannot log memory"
    fi
}

########################################
# 🔒 LOCK FILE
########################################
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        log "⚠️ Lock file exists, another instance may be running. Removing stale lock."
        rm -f "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
    log "🔒 Lock acquired (PID $$)"
}

release_lock() {
    rm -f "$LOCK_FILE" 2>/dev/null || true
    log "🔓 Lock released"
}

cleanup() {
    hard_cleanup
    release_lock
}

trap cleanup EXIT INT TERM

########################################
# 🐍 PYTHON VENV DETECTION
########################################
detect_python() {
    if [ -f ".venv/Scripts/python.exe" ]; then
        echo ".venv/Scripts/python.exe"
    elif [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
    else
        log "No venv found, creating..."
        python -m venv .venv
        PYTHON=".venv/bin/python"
        "$PYTHON" -m pip install -r requirements.txt
        echo "$PYTHON"
    fi
}

PYTHON=$(detect_python)
log "Using Python: $PYTHON"
log "RAG Enabled: $ASYNC_USE_RAG"

########################################
# 🖥️ VPS OPTIMIZATION
########################################
optimize_for_vps() {
    local total_mem_mb
    if command -v free >/dev/null 2>&1; then
        total_mem_mb=$(free -m | awk '/^Mem:/{print $2}')
    else
        total_mem_mb=1024  # assume 1GB if free not available
    fi
    log "Total memory: ${total_mem_mb}MB"

    # Default to vps profile unless overridden
    export ALPHA_PROFILE="${ALPHA_PROFILE:-vps}"

    # Conservative settings for low memory (<=2GB)
    if [ "$total_mem_mb" -le 2048 ]; then
        log "Low memory detected, applying conservative settings"
        # Reduce workers
        export ASYNC_RANKER_WORKERS="${ASYNC_RANKER_WORKERS:-1}"
        export ASYNC_SIMULATOR_WORKERS="${ASYNC_SIMULATOR_WORKERS:-1}"
        # Reduce batch sizes and queue sizes
        export ASYNC_GEN_BATCH_SIZE="${ASYNC_GEN_BATCH_SIZE:-4}"
        export ASYNC_GEN_QUEUE_SIZE="${ASYNC_GEN_QUEUE_SIZE:-60}"
        export ASYNC_SIM_QUEUE_SIZE="${ASYNC_SIM_QUEUE_SIZE:-30}"
        export ASYNC_BATCH_SIZE="${ASYNC_BATCH_SIZE:-1}"
        # Limit concurrent WQ API calls
        export WQ_MAX_CONCURRENT="${WQ_MAX_CONCURRENT:-1}"
        # Reduce QD archive memory pressure
        export ASYNC_NOVELTY_MIN="${ASYNC_NOVELTY_MIN:-0.1}"
        # Disable RAG by default (memory intensive)
        export ASYNC_USE_RAG="${ASYNC_USE_RAG:-0}"
        # Limit CPU threads to reduce memory usage
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    fi

    # If memory is extremely low (<=1GB), further reduce
    if [ "$total_mem_mb" -le 1024 ]; then
        log "Extremely low memory (${total_mem_mb}MB), applying ultra-conservative settings"
        export ASYNC_GEN_BATCH_SIZE="${ASYNC_GEN_BATCH_SIZE:-2}"
        export ASYNC_GEN_QUEUE_SIZE="${ASYNC_GEN_QUEUE_SIZE:-30}"
        export ASYNC_SIM_QUEUE_SIZE="${ASYNC_SIM_QUEUE_SIZE:-15}"
        export ASYNC_RANKER_WORKERS="${ASYNC_RANKER_WORKERS:-1}"
        export ASYNC_SIMULATOR_WORKERS="${ASYNC_SIMULATOR_WORKERS:-1}"
        # Consider limiting the number of alphas in QD archive
        export ASYNC_QD_MAX_ALPHAS="${ASYNC_QD_MAX_ALPHAS:-500}"
        # Ensure thread limits are set
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    fi

    log "VPS optimization applied: profile=$ALPHA_PROFILE, RAG=$ASYNC_USE_RAG, ranker_workers=$ASYNC_RANKER_WORKERS, sim_workers=$ASYNC_SIMULATOR_WORKERS"
}

optimize_for_vps

########################################
# 📝 AUTO-IMPROVE SCRIPT
########################################
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

########################################
# 🔁 LOOP KHÔNG NGỦ
########################################
acquire_lock
cycle=0
while true; do
    cycle=$((cycle + 1))
    log "=============================="
    log "🚀 Cycle $cycle START"

    # Clean trước
    hard_cleanup
    log_memory
    sleep 1   # ⚠️ giữ 1s để tránh race condition

    log "Running Alpha Factory (profile=$ALPHA_PROFILE)..."
    # Build optional flags from environment variables
    RAG_FLAG=""
    if [ "$ASYNC_USE_RAG" = "1" ]; then
        RAG_FLAG="--use-rag"
    fi
    LIMIT_FLAG=""
    if [ -n "$ALPHA_LOOP_LIMIT" ]; then
        LIMIT_FLAG="--limit $ALPHA_LOOP_LIMIT"
    fi
    SCORE_FLAG=""
    if [ -n "$ALPHA_LOOP_SCORE" ]; then
        SCORE_FLAG="--score $ALPHA_LOOP_SCORE"
    fi
    NO_HYBRID_FLAG=""
    if [ "$ALPHA_LOOP_NO_HYBRID" = "1" ]; then
        NO_HYBRID_FLAG="--no-hybrid"
    fi
    log "Flags: RAG=$RAG_FLAG LIMIT=$LIMIT_FLAG SCORE=$SCORE_FLAG NO_HYBRID=$NO_HYBRID_FLAG"
    ASYNC_USE_RAG="$ASYNC_USE_RAG" "$PYTHON" alpha_factory_cli.py auto --profile ${ALPHA_PROFILE} $RAG_FLAG $LIMIT_FLAG $SCORE_FLAG $NO_HYBRID_FLAG

    exit_code=$?
    log "Process exited with code: $exit_code"
    log_memory

    # Run auto-improvement analysis
    log "Running auto-improvement analysis..."
    "$PYTHON" "$IMPROVE_SCRIPT" >> "$LOG_FILE" 2>&1

    # Check for suggestions and apply them
    if [ -f ".auto_improve_suggestions.json" ]; then
        log "Found improvement suggestions"
        cat ".auto_improve_suggestions.json" >> "$LOG_FILE"
    fi

    # Check if we should stop
    if [ $MAX_CYCLES -gt 0 ] && [ $cycle -ge $MAX_CYCLES ]; then
        log "Reached max cycles ($MAX_CYCLES) - stopping"
        break
    fi

    # ❌ KHÔNG SLEEP — chạy lại ngay
done

log "Auto loop finished after $cycle cycles"
release_lock