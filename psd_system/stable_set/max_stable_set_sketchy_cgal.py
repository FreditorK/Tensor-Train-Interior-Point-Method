import sys
import os
import time
import yaml
import argparse
import tracemalloc

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.baselines import sketchy_cgal


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    print("Creating Problem...")

    np.random.seed(config["seed"])
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    t0 = time.time()
    G = tt_rank_reduce(tt_random_graph(config["dim"], config["max_rank"]))
    adj_matrix = np.round(tt_matrix_to_matrix(G), decimals=1)

    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    constraint_matrices = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            A = np.zeros_like(adj_matrix)
            A[i, j] = adj_matrix[i, j]
            constraint_matrices.append(A)
    bias = np.zeros((len(adj_matrix)**2, 1))
    J = np.ones_like(adj_matrix)
    if args.track_mem:
        print("Memory tracking started...")
        tracemalloc.start()  # Start memory tracking
    t2 = time.time()
    X, duality_gaps = sketchy_cgal(-J, constraint_matrices, bias, (1, 1), feasability_tol=1e-5, duality_tol=0.71, num_iter=1000*2**config["dim"], R=2, verbose=True)
    t3 = time.time()
    if args.track_mem:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10 ** 6:.2f} MB")
        print(f"Peak memory usage: {peak / 10 ** 6:.2f} MB")
        tracemalloc.stop()  # Stop tracking after measuring
    #print(np.round(X, decimals=4))
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {np.trace(J.T @ X)}")
    print(f"Total feasibility error: {np.linalg.norm([np.trace(c.T @ X) - b for c, b in zip(constraint_matrices, bias)])**2}")