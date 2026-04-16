#!/usr/bin/env bash
# =============================================================================
# Alpha Factory VPS Runner
# - Auto-restart on crash with exponential backoff
# - Singleton lock to prevent duplicates
# - Health monitoring & logging
# - Graceful shutdown handling
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- Load environment variables ---
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
    echo "Loaded .env file"
else
    echo "WARNING: .env file not found at $ROOT_DIR/.env"
fi

# --- Configuration ---
LOCK_FILE="$ROOT_DIR/results/vps_runner.lock"
LOG_FILE="$ROOT_DIR/results/vps_runner.log"
PID_FILE="$ROOT_DIR/results/vps_runner.pid"
MAX_RESTART_INTERVAL=300       # 5 minutes max backoff
INITIAL_BACKOFF=10             # 10 seconds initial backoff
HEALTH_CHECK_INTERVAL=60       # Check every minute
STALE_LOCK_SECONDS=3600       # 1 hour stale lock threshold

# --- Logging ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# --- Ensure directories ---
mkdir -p "$ROOT_DIR/results"

# --- Singleton lock ---
acquire_lock() {
    mkdir -p "$(dirname "$LOCK_FILE")"
    
    if [ -f "$LOCK_FILE" ]; then
        # Check if existing process is alive
        OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null | head -1)
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            log_error "Another instance is running (PID: $OLD_PID)"
            exit 1
        fi
        # Stale lock - remove it
        log "Removing stale lock from PID $OLD_PID"
        rm -f "$LOCK_FILE"
    fi
    
    # Write our PID
    echo "$$" > "$LOCK_FILE"
    log "Acquired lock: $$"
}

release_lock() {
    rm -f "$LOCK_FILE"
    log "Released lock"
}

# --- Cleanup handlers ---
cleanup() {
    log "Caught signal, shutting down gracefully..."
    RELEASE_LOCK=1
    if [ -f "$PID_FILE" ]; then
        ASYNC_PID=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$ASYNC_PID" ] && kill -0 "$ASYNC_PID" 2>/dev/null; then
            log "Sending SIGINT to async engine (PID: $ASYNC_PID)"
            kill -SIGINT "$ASYNC_PID" 2>/dev/null || true
            # Wait for graceful shutdown
            for i in {1..30}; do
                if ! kill -0 "$ASYNC_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$ASYNC_PID" 2>/dev/null; then
                log "Force killing async engine"
                kill -SIGKILL "$ASYNC_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
    release_lock
    log "Shutdown complete"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# --- Setup virtual environment ---
setup_venv() {
    if [ ! -d "$ROOT_DIR/.venv" ]; then
        log "Creating virtual environment..."
        python3 -m venv "$ROOT_DIR/.venv"
    fi
    
    source "$ROOT_DIR/.venv/bin/activate"
    
    # Upgrade pip quietly
    python -m pip install --upgrade pip setuptools wheel -q 2>/dev/null || true
    
    # Install requirements if needed
    if [ -f "$ROOT_DIR/requirements.txt" ]; then
        pip install -r "$ROOT_DIR/requirements.txt" -q 2>/dev/null || true
    fi
}

# --- Run async pipeline with auto-restart ---
run_with_recovery() {
    source "$ROOT_DIR/.venv/bin/activate"
    
    BACKOFF=$INITIAL_BACKOFF
    RESTART_COUNT=0
    
    while true; do
        log "Starting async pipeline (attempt $((++RESTART_COUNT)))"
        
        # Run the async pipeline
        # --limit 0 = infinite
        # --profile vps uses optimized settings
        python alpha_factory_cli.py auto --profile vps --skip-install &
        ASYNC_PID=$!
        echo "$ASYNC_PID" > "$PID_FILE"
        
        log "Async engine started with PID: $ASYNC_PID"
        
        # Wait for process and capture exit code
        if wait "$ASYNC_PID"; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
        fi
        
        rm -f "$PID_FILE"
        
        # Check if we were terminated intentionally
        if [ "$EXIT_CODE" -eq 130 ] || [ "$EXIT_CODE" -eq 143 ]; then
            log "Intentional shutdown requested"
            return 0
        fi
        
        # Exponential backoff for rapid crashes
        if [ $EXIT_CODE -ne 0 ]; then
            log_error "Async engine exited with code $EXIT_CODE"
            log "Waiting ${BACKOFF}s before restart..."
            sleep "$BACKOFF"
            
            # Exponential increase, capped
            BACKOFF=$((BACKOFF * 2))
            if [ $BACKOFF -gt $MAX_RESTART_INTERVAL ]; then
                BACKOFF=$MAX_RESTART_INTERVAL
            fi
        else
            # Clean exit - reset backoff
            BACKOFF=$INITIAL_BACKOFF
        fi
    done
}

# --- Health check (optional background monitor) ---
health_monitor() {
    while true; do
        sleep $HEALTH_CHECK_INTERVAL
        
        # Check if async process is running
        if [ -f "$PID_FILE" ]; then
            ASYNC_PID=$(cat "$PID_FILE" 2>/dev/null)
            if [ -n "$ASYNC_PID" ] && ! kill -0 "$ASYNC_PID" 2>/dev/null; then
                log_error "Health check: async process dead but no exit detected!"
            fi
        fi
        
        # Log recent KPI if available
        if [ -f "$ROOT_DIR/results/kpi_latest.json" ]; then
            AGE=$(($(date +%s) - $(stat -c %Y "$ROOT_DIR/results/kpi_latest.json" 2>/dev/null || echo "0")))
            if [ $AGE -lt 600 ]; then  # Less than 10 min old
                log "KPI file age: ${AGE}s (healthy)"
            fi
        fi
    done
}

# =============================================================================
# MAIN
# =============================================================================

log "=========================================="
log "Alpha Factory VPS Runner Starting"
log "Root: $ROOT_DIR"
log "=========================================="

# Acquire singleton lock
acquire_lock

# Setup virtual environment
setup_venv

# Start health monitor in background (optional)
health_monitor &
HEALTH_MON_PID=$!

# Run with auto-recovery
run_with_recovery

# Cleanup
kill $HEALTH_MON_PID 2>/dev/null || true
release_lock

log "VPS Runner stopped"

