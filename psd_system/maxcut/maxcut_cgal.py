# Import packages.
import sys
import os
import argparse
import tracemalloc
sys.path.append(os.getcwd() + '/../../')
import time
from src.tt_ops import *
from maxcut import Config
from src.baselines import cgal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    args = parser.parse_args()

    np.random.seed(Config.seed)
    t0 = time.time()
    G = tt_rank_reduce(tt_random_graph(Config.dim, Config.max_rank))
    print(np.round(tt_matrix_to_matrix(G), decimals=2))
    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    C = np.round(tt_matrix_to_matrix(G))
    t2 = time.time()
    constraint_matrices = [np.outer(column, column) for column in np.eye(C.shape[0])]
    bias = np.ones((C.shape[0], 1))
    trace_param = np.sum(bias)
    if args.track_mem:
        print("Memory tracking started...")
        tracemalloc.start()  # Start memory tracking
    X, duality_gaps = cgal(-C, constraint_matrices, bias, (trace_param, trace_param), num_iter=100)
    print(np.round(X, decimals=2))
    t3 = time.time()
    if args.track_mem:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10 ** 6:.2f} MB")
        print(f"Peak memory usage: {peak / 10 ** 6:.2f} MB")
        tracemalloc.stop()  # Stop tracking after measuring
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {np.trace(C.T @ X)}")
    chol = robust_cholesky(X, epsilon=1e-3)
    nodes_in_cut = [i for i, v in enumerate(chol.T @ np.random.randn(chol.shape[0], 1)) if v > 0]