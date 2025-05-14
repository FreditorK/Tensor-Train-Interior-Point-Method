#!/bin/bash

# Activate conda if not activated

#Fix threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# ---------------------------
# Parameters
# ---------------------------
BASE_TIMEOUT=1800  # 30 minutes
START_DIM=8
END_DIM=8

# ---------------------------
# Logging setup
# ---------------------------
LOGFILE="tt_ipm_maxcut_($START_DIM)_($END_DIM).txt"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==== Maxcut TT-IPM Batch Run Started at $(date) ===="

cd psd_system/maxcut

# ---------------------------
# Loop through configs
# ---------------------------
for dim in $(seq $START_DIM $END_DIM); do
    CONFIG="configs/maxcut_${dim}.yaml"
    echo -e "\n▶ Running dim=$dim with config=$CONFIG at $(date)"
    CURRENT_TIMEOUT=$((BASE_TIMEOUT * dim))

    timeout "$CURRENT_TIMEOUT" python maxcut.py --config "$CONFIG" --track_mem
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
