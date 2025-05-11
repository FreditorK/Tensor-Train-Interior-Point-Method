#!/bin/bash
#PBS -l select=1:ncpus=22:mem=32gb
#PBS -l walltime=10:00:00
#PBS -N sdpa_graphm
#PBS -o sdpa_graphm.out
#PBS -e sdpa_graphm.err

cd "$PBS_O_WORKDIR"

# Activate conda
source .conda/bin/activate
conda activate ttipm

cd TT-IPM/psd_system/graphm

# ---------------------------
# Logging setup
# ---------------------------
LOGFILE="graphm_batch_log_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==== Graph Matching SDPA Batch Run Started at $(date) ===="

# ---------------------------
# Parameters
# ---------------------------
BASE_TIMEOUT=1800  # 30 minutes
START_DIM=3
END_DIM=4

# ---------------------------
# Loop through configs
# ---------------------------
for dim in $(seq $START_DIM $END_DIM); do
    CONFIG="configs/graphm_${dim}.yaml"
    echo -e "\n▶ Running dim=$dim with config=$CONFIG at $(date)"
    CURRENT_TIMEOUT=$((2 * BASE_TIMEOUT * dim))

    timeout "$CURRENT_TIMEOUT" python graphm_sdpa.py --config "$CONFIG" --track_mem
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
