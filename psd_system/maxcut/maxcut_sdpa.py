# Import packages.
import sys
import os
import argparse
import tracemalloc
import yaml
import sdpap


sys.path.append(os.getcwd() + '/../../')
import time
from maxcut import *
from src.baselines import *
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
        C = tt_matrix_to_matrix(tt_obj_matrix(config["max_rank"], config["dim"]))
        t1 = time.time()
        if args.track_mem:
            tracemalloc.start()  # Start memory tracking
        t2 = time.time()
        X = cp.Variable(C.shape, PSD=True)
        constraints = [X >> 0]
        constraints += [cp.diag(X) == 1]
        t2 = time.time()
        prob = cp.Problem(cp.Maximize(cp.trace(C.T @ X)), constraints)
        _ = prob.solve(solver=cp.SDPA, epsilonStar=0.1*config["centrality_tol"], epsilonDash=config["feasibility_tol"], verbose=True, numThreads=1)
        X = X.value
        for m in prob.solution.dual_vars.values():
            if type(m) == np.ndarray:
                if m.shape == (2**config["dim"], 2**config["dim"]):
                    Z = m
        t3 = time.time()
        if args.track_mem:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()  # Stop tracking after measuring
            memory.append(peak / 10 ** 6)
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        complementary_slackness.append(np.trace(X @ Z))
        feasibility_errors.append(np.linalg.norm(np.diag(X) - 1 ) ** 2)
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s")
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s")
    if args.track_mem:
        print(f"Peak memory avg {np.mean(memory):.3f} MB")
    print(f"Complementary Slackness avg: {np.mean(complementary_slackness)}")
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}")
    print(X)
    print(np.trace(C.T @ X))
    print(X.shape)
