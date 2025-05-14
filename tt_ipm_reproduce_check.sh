#!/bin/bash
#PBS -l select=1:ncpus=22:mem=32gb
#PBS -l walltime=10:00:00
#PBS -N tt_reproduce_check
#PBS -o tt_ipm_reproduce_check.out
#PBS -e tt_ipm_reproduce_check.err

cd "$PBS_O_WORKDIR"

# Activate conda
source .conda/bin/activate
conda activate ttipm

export OMP_NUM_THREADS=22
export MKL_NUM_THREADS=22
export OPENBLAS_NUM_THREADS=22
export NUMEXPR_NUM_THREADS=22


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
START_DIM=7
END_DIM=7

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
