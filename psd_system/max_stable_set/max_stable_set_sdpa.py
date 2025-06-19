import sys
import os
import time

import numpy as np
import yaml
import argparse
import tracemalloc
import sdpap

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from memory_profiler import memory_usage
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
        adj_matrix = tt_matrix_to_matrix(G)
        t1 = time.time()
        J = np.ones_like(adj_matrix)
        if args.track_mem:
            start_mem = memory_usage(max_usage=True, include_children=True)
        X = cp.Variable(J.shape, symmetric=True)
        constraints = [X >> 0]
        constraints += [cp.diag(cp.vec(adj_matrix, order="F").flatten()) @ cp.vec(X, order="F") == 0] # for A, b in zip(constraint_matrices, bias.flatten())]
        constraints += [cp.trace(X) == 1]
        t2 = time.time()
        if args.track_mem:
            def wrapper():
                prob = cp.Problem(cp.Maximize(cp.trace(J.T @ X)), constraints)
                _ = prob.solve(solver=cp.SDPA, epsilonDash=1e-6 / 2**config["dim"], epsilonStar=1e-5 / 2**config["dim"], verbose=True, numThreads=1, omegaStar=100, betaStar=0.5, gammaStar=0.9)
                return prob

            res, prob = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
            X = X.value
            for m in prob.solution.dual_vars.values():
                    if type(m) == np.ndarray:
                        if m.shape == (2 ** config["dim"], 2 ** config["dim"]):
                            Z = m
            memory.append(res - start_mem)
        else:
            prob = cp.Problem(cp.Maximize(cp.trace(J.T @ X)), constraints)
            _ = prob.solve(solver=cp.SDPA, epsilonDash=1e-6 / 2**config["dim"], epsilonStar=1e-5 / 2**config["dim"], verbose=True, numThreads=1, omegaStar=100, betaStar=0.5, gammaStar=0.9)
            for m in prob.solution.dual_vars.values():
                if type(m) == np.ndarray:
                    if m.shape == (2 ** config["dim"], 2 ** config["dim"]):
                        Z = m

        t3 = time.time()
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        complementary_slackness.append(np.trace(X @ Z))
        feasibility_errors.append((np.trace(X)-1)**2 + np.linalg.norm(np.diag(adj_matrix.reshape(-1, 1, order="F").flatten()) @ X.reshape(-1, 1, order="F"))**2)
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s")
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s")
    print(f"Peak memory avg {np.mean(memory):.3f} MB")
    print(f"Complementary Slackness avg: {np.mean(complementary_slackness)}")
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}")