#!/bin/bash

# Activate conda if not activated

#Fix threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

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
# Parameters
# ---------------------------
BASE_TIMEOUT=1800  # 30 minutes
START_DIM=2
END_DIM=2

# ---------------------------
# Logging setup
# ---------------------------
LOGFILE="tt_ipm_graphm_($START_DIM)_($END_DIM).txt"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==== Graph Matching TT-IPM Batch Run Started at $(date) ===="

cd psd_system/graphm

# ---------------------------
# Loop through configs
# ---------------------------
for dim in $(seq $START_DIM $END_DIM); do
    CONFIG="configs/graphm_${dim}.yaml"
    echo -e "\n▶ Running dim=$dim with config=$CONFIG at $(date)"
    CURRENT_TIMEOUT=$((2 * BASE_TIMEOUT * dim))

    timeout "$CURRENT_TIMEOUT" python graphm.py --config "$CONFIG" --track_mem
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
