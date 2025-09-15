![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
## TT-IPM Codebase

This repository contains the reference implementation of the TT-IPM method for solving SDP problems in Tensor-Train (TT) format, along with baselines and reproducibility tooling.

### What’s here
- **Core implementation**: `src/tt_ipm.py`, `src/tt_ops.py`, `src/tt_als.py`
- **Problem generators**: `psd_system/{maxcut,corr_clust,graphm,max_stable_set}`
- **Experiment runner**: `src/utils.py::run_experiment`
- **Configs**: ready-to-run YAMLs in `configs/`
- **Figures/tables**: scripts in repo root to turn results into LaTeX/plots

### Quickstart
1) Create and activate the conda environment
```bash
conda env create -f env.yaml
conda activate ttipm
```

2) Build Cython extensions
```bash
python setup.py build_ext --inplace
```

3) Preferred: run batch experiments over multiple dimensions via shell scripts
- TT-IPM (our method):
```bash
# Usage: ./tt_ipm.sh <problem> <start_dim> <end_dim> <rank> [--track_mem]
# Examples:
bash tt_ipm.sh maxcut 5 7 1 --track_mem
bash tt_ipm.sh corr_clust 6 9 1
bash tt_ipm.sh graphm 2 3 2
bash tt_ipm.sh max_stable_set 6 8 1
```
The script auto-activates the environment, fixes thread counts, iterates `configs/<problem>_<dim>.yaml`, and logs to `results/tt_ipm_<problem>_<start>_<end>_<rank>.txt`. Add `--track_mem` to measure peak memory (slightly slower). Note: setting `verbose: true` in configs can also slow runs due to increased logging.

- Baselines (SCS/SDPA/SC-GAL):
```bash
# Usage: ./scs.sh <problem> <start_dim> <end_dim> [--track_mem]
bash scs.sh maxcut 6 9 --track_mem

# Usage: ./sdpa.sh <problem> <start_dim> <end_dim> [--track_mem]
bash sdpa.sh corr_clust 6 9

# Usage: ./scgal.sh <problem> <start_dim> <end_dim> [--track_mem]
bash scgal.sh graphm 2 4
```

4) Results are written as JSON into `results/` automatically after the run. See “Results and plotting” below.

### Running experiments (details)
Each problem family exposes a `create_problem(dim, rank)` function and a small `__main__` entry that forwards to `run_experiment`. You can also run any single experiment with:
- **MaxCut**:
```bash
python psd_system/maxcut/maxcut.py --config configs/maxcut_10.yaml --rank 1 --track_mem
```
- **Correlation Clustering**:
```bash
python psd_system/corr_clust/corr_clust.py --config configs/corr_clust_6.yaml --rank 1
```
- **Graph Matching**:
```bash
python psd_system/graphm/graphm.py --config configs/graphm_3.yaml --rank 2
```
- **Maximum Stable Set**:
```bash
python psd_system/max_stable_set/max_stable_set.py --config configs/max_stable_set_7.yaml --rank 1
```

The shell wrappers `tt_ipm.sh`, `scs.sh`, `sdpa.sh`, and `scgal.sh` are preferred for batch sweeps across dimensions.

### Configuration files (YAML)
The configs in `configs/` define problem size and solver hyperparameters. Common fields:
- **seeds**: list of integer seeds to generate problem instances
- **dim**: problem dimension in TT (number of TT cores)
- **max_iter, warm_up, max_refinement**: maximum number of iterations, number of warm-up iterations with XZ direction, maximum number refinement iterations
- **gap_tol, op_tol, abs_tol**: relative stopping criteria, delta threshold, absolute error tolerance
- **mals_restarts**: number of restarted AMEn attempts in solving KKT system
- **epsilonDash, epsilonDashineq**: regularization terms

Defaults and tips:
- Configs are provided with a single active seed; additional seeds are included but commented out.
- `verbose: true` is enabled by default to aid inspection; disable for fastest runs.

Example: `configs/maxcut_10.yaml`
```yaml
dim: 10
seeds: [0, 1, 2]
max_iter: 100
warm_up: 3
max_refinement: 5
gap_tol: 1e-4
op_tol: 1e-5
abs_tol: 8e-4
mals_restarts: 3
epsilonDash: 1
epsilonDashineq: 1
```

### Results and plotting
After each run, a JSON summary is saved to `results/` with metrics aggregated over seeds:
- Runtimes, iterations
- Feasibility and dual feasibility errors
- Duality gap (complementary slackness)
- Peak memory (if `--track_mem` used)
- Per-iterate TT ranks for X, Y, Z (and T when in use)

Post-processing scripts:
- `produce_table.py`: converts results JSON to LaTeX rows
- `produce_heatmap.py`: generates heatmap-friendly LaTeX snippets
- `produce_scatter.py`: produces scatter plot LaTeX snippets

Typical usage:
```bash
python produce_table.py
python produce_heatmap.py
python produce_scatter.py
```
Each script prints the LaTeX code and saves intermediary files if needed.

### Reproducibility checklist
- Environment is captured in `env.yaml`. Use the exact versions for artifact reproduction.
- Randomness is controlled via the `seeds` field. All seeds from a config are run and aggregated.
- Threading is stabilized in the shell scripts by setting `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`.
- Cython extensions are compiled. Rebuild when changing Python or NumPy.
- Verbose logging (`verbose: true` in configs) can slow runs and produce large logs; disable unless diagnosing.

### Repository structure
```
TT-IPM/
  configs/                # YAML experiment configs
  psd_system/             # Problem families and baselines
    maxcut/               #   MaxCut
    corr_clust/           #   Correlation Clustering
    graphm/               #   Graph Matching
    max_stable_set/       #   Maximum Stable Set
  src/                    # Core TT-IPM and TT algebra
  cy_src/                 # Optional Cython accelerators
  results/                # Output JSON and logs (created on demand)
  *.sh                    # Batch runners for TT-IPM and baselines
  produce_*.py            # Table and figure generators
```

### Troubleshooting
- If you see import errors, ensure you are running from the repo root with the conda env activated.
- PETSc or sparse CHOLMOD bindings may require system packages (e.g., SuiteSparse). Consult your distro instructions.
- For long jobs, prefer the shell wrappers so logging and cleanup are handled automatically.


## License & Dependencies

This project is released under the [MIT License](LICENSE).

The code depends on several third-party libraries distributed under their own licenses, including:

- [NumPy](https://numpy.org/) (BSD)  
- [SciPy](https://scipy.org/) (BSD)  
- [CVXPY](https://www.cvxpy.org/) (Apache 2.0)  
- [PETSc](https://petsc.org/release/) and [petsc4py](https://petsc4py.readthedocs.io/en/stable/) (LGPL)  
- [SCS](https://github.com/cvxgrp/scs) (MIT)  
- [SDPA-Python](https://github.com/sdpa-python/sdpa-python) (MIT)  

Use of these libraries is governed by their respective licenses. The MIT License applies only to the original code in this repository, not to its dependencies.


### Citation
If you use this code in academic work, please cite the corresponding paper (add BibTeX available shortly).

### Contact
For questions or issues, please open a GitHub issue or contact the authors.