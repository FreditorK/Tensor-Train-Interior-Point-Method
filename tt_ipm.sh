#!/bin/bash

echo "Running: $1"
# ---------------------------
# Parameters
# ---------------------------
BASE_TIMEOUT=40000 # 12h changed for graphm  rank >2

START_DIM=$2
END_DIM=$3
TXT_IDENTIFIER=$4

# ---------------------------
# Delete log file if it exists
# ---------------------------
LOGFILE="results/tt_ipm_$1_${START_DIM}_${END_DIM}_${TXT_IDENTIFIER}_c1.txt"
if [ -f "$LOGFILE" ]; then
    rm "$LOGFILE"
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
if [ "$CONDA_DEFAULT_ENV" != "ttipm" ]; then
    conda activate ttipm
fi

# ---------------------------
# Fix threads
# ---------------------------
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Cleanup on exit or interrupt
cleanup() {
    echo -e "\n⚠️ Caught interrupt. Cleaning up at $(date)..." >&2
    pkill -P $$ 2>/dev/null
    echo "🧹 Cleaned up. Exiting." >&2
    exit 1
}
trap cleanup SIGINT SIGTERM
trap 'echo -e "\n⚠️ Script resumed (was suspended). Memory may not have been cleaned up."' SIGCONT

# ---------------------------
# Logging setup
# ---------------------------
exec > >(tee -a "$LOGFILE") 2>&1

echo "==== $1 TT-IPM Batch Run Started at $(date) ===="

cd psd_system/$1

# ---------------------------
# Loop through configs
# ---------------------------

for arg in "$@"; do
    if [ "$arg" == "--track_mem" ]; then
        TRACK_MEM="--track_mem"
    fi
done

for dim in $(seq $START_DIM $END_DIM); do
    CONFIG="configs/$1_${dim}c1.yaml"
    echo -e "\n▶ Running dim=$dim with config=$CONFIG at $(date)"
    CURRENT_TIMEOUT=$((BASE_TIMEOUT * dim))

    timeout "$CURRENT_TIMEOUT" python $1.py --config "$CONFIG" --rank $TXT_IDENTIFIER $TRACK_MEM
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

