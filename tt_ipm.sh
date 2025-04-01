#!/bin/bash
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -l walltime=01:00:00
#PBS -N tt_ipm_010425

cd $PBS_O_WORKDIR

source .conda/bin/activate
conda activate ttipm

cd TT-IPM/psd_system/maxcut

echo "Maxcut (TT-IPM) dim=6:"
python maxcut.py --config configs/maxcut_6.yaml --track_mem

echo "Maxcut (Sketch-CGAL) dim=6:"
timeout 30m python maxcut_sketch_cgal.py --config configs/maxcut_6.yaml --track_mem

echo "Maxcut (SDPA) dim=6:"
python maxcut_sdpa.py --config configs/maxcut_6.yaml --track_mem

cd ../stable_set

echo "MaxStableSet (TT-IPM) dim=6:"
python max_stable_set.py --config configs/max_stable_set_6.yaml --track_mem

echo "MaxStableSet (Sketch-CGAL) dim=6:"
timeout 30m python max_stable_set_sketch_cgal.py --config configs/max_stable_set_6.yaml --track_mem

echo "MaxStableSet (SDPA) dim=6:"
python max_stable_set_sdpa.py --config configs/max_stable_set_6.yaml --track_mem