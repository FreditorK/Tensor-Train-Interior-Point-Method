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

    problem_creation_times = []
    runtimes = []
    memory = []
    aux_duality_gap = []
    feasibility_errors = []
    num_iters = []
    for seed in config["seeds"]:
        np.random.seed(seed)
        t0 = time.time()
        G = tt_rank_reduce(tt_random_graph(config["dim"], config["max_rank"]))
        adj_matrix = np.round(tt_matrix_to_matrix(G), decimals=1)
        t1 = time.time()
        constraint_matrices = []
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                A = np.zeros_like(adj_matrix)
                A[i, j] = adj_matrix[i, j]
                constraint_matrices.append(A)
        bias = np.zeros((len(adj_matrix) ** 2, 1))
        J = np.ones_like(adj_matrix)
        if args.track_mem:
            tracemalloc.start()  # Start memory tracking
        t2 = time.time()
        X, duality_gaps, info = sketchy_cgal(-J, constraint_matrices, bias, (1, 1), feasability_tol=config["feasibility_tol"], duality_tol=config["duality_tol"],
                                       num_iter=1000 * 2 ** config["dim"], R=config["sketch_cgal_rank"], verbose=config["verbose"])
        t3 = time.time()
        if args.track_mem:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()  # Stop tracking after measuring
            memory.append(peak / 10 ** 6)
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        aux_duality_gap.append(duality_gaps[-1])
        feasibility_errors.append(
            np.linalg.norm([np.trace(c.T @ X) - b for c, b in zip(constraint_matrices, bias.flatten())]) ** 2)
        num_iters.append(info["num_iters"])
    print(f"Converged after avg {np.mean(num_iters):.1f} iterations")
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s")
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s")
    print(f"Peak memory avg {np.mean(memory):.3f} MB")
    print(f"Aux duality gap avg: {np.mean(aux_duality_gap)}")
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}")