#!/usr/bin/env bash
# Run all mixed-effect TS experiments overnight. Saves plots to plots/ and logs to run_all_overnight.log.
# Usage: ./run_all_overnight.sh   (run from mixed_effect_ts/ or repo root)

set -e

# Go to package root (directory containing experiments/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p plots

LOG="run_all_overnight.log"
echo "=== Started at $(date) ===" | tee "$LOG"

run_one() {
    local name="$1"
    shift
    echo "" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] Running: $name" | tee -a "$LOG"
    if python "$@" 2>&1 | tee -a "$LOG"; then
        echo "[$(date '+%H:%M:%S')] OK: $name" | tee -a "$LOG"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $name" | tee -a "$LOG"
        return 1
    fi
}

# Synthetic experiments (default params from README)
run_one "run_lin.py" experiments/run_lin.py --save plots/ || true
run_one "run_log.py" experiments/run_log.py --save plots/ || true

# MovieLens experiments (require data/ratings.dat)
run_one "run_lin_movielens.py" experiments/run_lin_movielens.py --save plots/ || true
run_one "run_log_movielens.py" experiments/run_log_movielens.py --save plots/ || true

echo "" | tee -a "$LOG"
echo "=== Finished at $(date) ===" | tee -a "$LOG"
echo "Plots in: $SCRIPT_DIR/plots/"
echo "Full log: $SCRIPT_DIR/$LOG"
