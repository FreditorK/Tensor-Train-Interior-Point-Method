#!/usr/bin/env bash
set -euo pipefail

cd /vol/bitbucket/fjk20/projects/Tensor-Train-Interior-Point-Method

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh"
    exit 1
fi

conda activate ttipm

echo "Building on $(hostname)"
lscpu | grep -E 'Model name|Flags' || true

rm -rf build
find . -name "*.so" -delete

python setup.py build_ext --inplace
