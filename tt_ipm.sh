#!/bin/bash

set -u

usage() {
    echo "Usage: bash tt_ipm.sh <problem> <start_dim> <end_dim> <rank> [--track_mem] [--terminal_only]"
}

if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
    usage
    exit 1
fi

PROBLEM=$1
START_DIM=$2
END_DIM=$3
RANK=$4
TRACK_MEM=""
TERMINAL_ONLY=""

for arg in "${@:5}"; do
    case "$arg" in
        --track_mem)
            TRACK_MEM="--track_mem"
            ;;
        --terminal_only|--terminal-only)
            TERMINAL_ONLY="--terminal_only"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$START_DIM" =~ ^[0-9]+$ && "$END_DIM" =~ ^[0-9]+$ && "$RANK" =~ ^[0-9]+$ ]]; then
    echo "❌ start_dim, end_dim, and rank must be positive integers."
    exit 1
fi
if [ "$RANK" -lt 1 ] || [ "$START_DIM" -lt 1 ] || [ "$END_DIM" -lt "$START_DIM" ]; then
    echo "❌ Invalid range/rank: require start_dim >= 1, end_dim >= start_dim, rank >= 1."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running: ${PROBLEM} (rank=${RANK})"
BASE_TIMEOUT=40000 # 12h changed for graphm rank >2
if [ -z "$TERMINAL_ONLY" ]; then
    mkdir -p results
    LOGFILE="results/tt_ipm_${PROBLEM}_${START_DIM}_${END_DIM}_${RANK}.txt"
    rm -f "$LOGFILE"
fi

# ---------------------------
# Activate conda environment if not already activated
# ---------------------------

# Try to find conda.sh and source it (adjust paths if needed)
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "❌ Conda not found. Please install Miniforge, Miniconda, or Anaconda."
    exit 1
fi

# Now activate environment
ENV_NAME="${TTIPM_CONDA_ENV:-ttipm}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    conda activate "$ENV_NAME"
fi

# ---------------------------
# Fix threads
# ---------------------------
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TMPDIR="${TTIPM_TMPDIR:-/tmp}"

# Cleanup on exit or interrupt
RUN_PID=""
TIMER_PID=""

cleanup() {
    local signal_name="${1:-interrupt}"
    local exit_code="${2:-130}"

    trap - SIGINT SIGTERM

    if [ -n "${TIMER_PID:-}" ]; then
        kill "$TIMER_PID" 2>/dev/null || true
        wait "$TIMER_PID" 2>/dev/null || true
    fi

    if [ -n "${RUN_PID:-}" ]; then
        kill -TERM -- "-$RUN_PID" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-$RUN_PID" 2>/dev/null || true
        wait "$RUN_PID" 2>/dev/null || true
    fi

    echo -e "\n⚠️ Caught ${signal_name}. Cleaned up at $(date). Exiting." >&2
    exit "$exit_code"
}
trap 'cleanup SIGINT 130' SIGINT
trap 'cleanup SIGTERM 143' SIGTERM
trap 'echo -e "\n⚠️ Script resumed (was suspended). Memory may not have been cleaned up."' SIGCONT

run_with_timeout() {
    local timeout_seconds=$1
    shift

    setsid "$@" &
    RUN_PID=$!
    local timeout_marker="${TMPDIR:-/tmp}/tt_ipm_timeout_${$}_${RUN_PID}"
    rm -f "$timeout_marker"

    (
        sleep "$timeout_seconds"
        if kill -0 "$RUN_PID" 2>/dev/null; then
            : > "$timeout_marker"
            kill -TERM -- "-$RUN_PID" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-$RUN_PID" 2>/dev/null || true
        fi
    ) &
    TIMER_PID=$!

    wait "$RUN_PID"
    local status=$?

    if [ -f "$timeout_marker" ]; then
        status=124
    fi

    if kill -0 "$TIMER_PID" 2>/dev/null; then
        kill "$TIMER_PID" 2>/dev/null || true
    fi
    wait "$TIMER_PID" 2>/dev/null || true
    rm -f "$timeout_marker"

    RUN_PID=""
    TIMER_PID=""
    return "$status"
}

# ---------------------------
# Logging setup
# ---------------------------
if [ -z "$TERMINAL_ONLY" ]; then
    exec > >(tee -a "$LOGFILE") 2>&1
fi

echo "==== ${PROBLEM} TT-IPM Batch Run Started at $(date) ===="

cd "psd_system/${PROBLEM}"

# ---------------------------
# Loop through configs
# ---------------------------

for dim in $(seq $START_DIM $END_DIM); do
    CONFIG="configs/${PROBLEM}_${dim}.yaml"
    echo -e "\n▶ Running dim=$dim with config=$CONFIG at $(date)"
    CURRENT_TIMEOUT=$((BASE_TIMEOUT * dim))
    cmd=(python "${PROBLEM}.py" --config "$CONFIG" --rank "$RANK")
    if [ -n "$TRACK_MEM" ]; then
        cmd+=("$TRACK_MEM")
    fi
    if [ -n "$TERMINAL_ONLY" ]; then
        cmd+=("$TERMINAL_ONLY")
    fi
    run_with_timeout "$CURRENT_TIMEOUT" "${cmd[@]}"
    status=$?

    if [ $status -eq 124 ]; then
        echo "⏱ Timeout reached for dim=$dim"
    elif [ $status -ne 0 ]; then
        echo "❌ Error occurred for dim=$dim (exit code $status)"
    else
        echo "✅ Completed dim=$dim successfully"
    fi
done

echo -e "\n==== All runs completed at $(date) ===="
