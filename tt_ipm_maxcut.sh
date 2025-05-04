#!/bin/bash
#PBS -l select=1:ncpus=22:mem=32gb
#PBS -l walltime=10:00:00
#PBS -N tt_maxcut
#PBS -o tt_ipm_maxcut.out
#PBS -e tt_ipm_maxcut.err

cd "$PBS_O_WORKDIR"

# Activate conda
source .conda/bin/activate
conda activate ttipm

cd TT-IPM/psd_system/maxcut

# ---------------------------
# Logging setup
# ---------------------------
LOGFILE="maxcut_batch_log_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==== Maxcut TT-IPM Batch Run Started at $(date) ===="

# ---------------------------
# Parameters
# ---------------------------
BASE_TIMEOUT=1800  # 30 minutes
START_DIM=6
END_DIM=10

# ---------------------------
# Loop through configs
# ---------------------------
for dim in $(seq $START_DIM $END_DIM); do
    CONFIG="configs/maxcut_${dim}.yaml"
    echo -e "\n▶ Running dim=$dim with config=$CONFIG at $(date)"
    CURRENT_TIMEOUT=$((BASE_TIMEOUT * dim))

    if [ -f "$CONFIG" ]; then
        timeout "$CURRENT_TIMEOUT" python maxcut.py --config "$CONFIG" --track_mem
        status=$?

        if [ $status -eq 124 ]; then
            echo "⏱ Timeout reached for dim=$dim"
        elif [ $status -ne 0 ]; then
            echo "❌ Error occurred for dim=$dim (exit code $status)"
        else
            echo "✅ Completed dim=$dim successfully"
        fi
    else
        echo "⚠️ Config file not found: $CONFIG"
    fi
done

echo -e "\n==== All runs completed at $(date) ===="
