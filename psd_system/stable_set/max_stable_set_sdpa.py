import sys
import os
import time

import yaml
import argparse
import tracemalloc
import sdpap

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
import cvxpy as cp


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
    complementary_slackness = []
    feasibility_errors = []
    for seed in config["seeds"]:
        np.random.seed(seed)
        t0 = time.time()
        G = tt_rank_reduce(tt_random_graph(config["dim"], config["max_rank"]))
        adj_matrix = np.round(tt_matrix_to_matrix(G), decimals=1)
        t1 = time.time()
        if args.track_mem:
            tracemalloc.start()
        constraint_matrices = []
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                A = np.zeros_like(adj_matrix)
                A[i, j] = adj_matrix[i, j]
                constraint_matrices.append(A)
        bias = np.zeros((len(adj_matrix)**2, 1))
        J = np.ones_like(adj_matrix)
        X = cp.Variable(J.shape, PSD=True)
        constraints = [X >> 0]
        constraints += [cp.trace(A.T @ X) == b for A, b in zip(constraint_matrices, bias.flatten())]
        constraints += [cp.trace(X) == 1]
        t2 = time.time()
        prob = cp.Problem(cp.Maximize(cp.trace(J.T @ X)), constraints)
        _ = prob.solve(solver=cp.SDPA, epsilonStar=0.1*config["centrality_tol"], epsilonDash=config["feasibility_tol"], verbose=True)
        X = X.value
        for m in prob.solution.dual_vars.values():
            if type(m) == np.ndarray:
                Z = m
        t3 = time.time()
        if args.track_mem:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()  # Stop tracking after measuring
            memory.append(peak/ 10 ** 6)
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        complementary_slackness.append(np.trace(X @ Z))
        feasibility_errors.append(np.linalg.norm([np.trace(c.T @ X) - b for c, b in zip(constraint_matrices, bias.flatten())])**2)
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s")
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s")
    print(f"Peak memory avg {np.mean(memory):.3f} MB")
    print(f"Complementary Slackness avg: {np.mean(complementary_slackness)}")
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}")
    print(X)