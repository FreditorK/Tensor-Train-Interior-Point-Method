# Import packages.
import numpy as np
import argparse
import os
import sys
import os
import time
import yaml
from memory_profiler import memory_usage
import cvxpy as cp

sys.path.append(os.getcwd() + '/../../')
from maxcut import *
from src.baselines import *
from src.utils import print_results_summary 
import cvxpy as cp

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    num_seeds = len(config["seeds"])
    problem_creation_times = np.zeros(num_seeds)
    runtimes = np.zeros(num_seeds)
    memory = np.zeros(num_seeds)
    complementary_slackness = np.zeros(num_seeds)
    feasibility_errors = np.zeros(num_seeds)
    dual_feasibility_errors = np.zeros(num_seeds)
    num_failed_seeds = 0
    num_iters = np.zeros(num_seeds)

    for s_i, seed in enumerate(config["seeds"]):
        tried_new_seed = False
        for attempt in range(3):  # At most two tries: original and one new random seed
            if attempt == 0:
                current_seed = seed
            else:
                current_seed = np.random.randint(0, 10000)
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)
            t1 = time.time()
            C = tt_matrix_to_matrix(tt_obj_matrix(1, config["dim"]))
            t2 = time.time()
            if args.track_mem:
                start_mem = memory_usage(max_usage=True, include_children=True)
            X = cp.Variable(C.shape, symmetric=True)
            constraints = [X >> 0, cp.diag(X) == 1]
            try:
                if args.track_mem:
                    def wrapper():
                        prob = cp.Problem(cp.Maximize(cp.trace(C.T @ X)), constraints)
                        _ = prob.solve(solver=cp.SCS, eps=1e-5 / config["dim"], verbose=True)
                        return prob
                    res, prob = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    memory[s_i] = res - start_mem
                else:
                    prob = cp.Problem(cp.Maximize(cp.trace(C.T @ X)), constraints)
                    _ = prob.solve(solver=cp.SCS, eps=1e-5 / config["dim"], verbose=True)
                # If we get here, break out of the attempt loop (success)
                break
            except Exception as e:
                print(e)
                if attempt == 0:
                    print(f"Failed to solve problem with config seed {seed}, trying a new random seed...")
                else:
                    print(f"Failed to solve problem with new random seed {current_seed}")
                    num_failed_seeds += 1
        else:
            # Only runs if both attempts failed
            continue
        X_val = X.value
        Z = constraints[0].dual_value
        y = constraints[1].dual_value
        t3 = time.time()
        problem_creation_times[s_i] = t2 - t1
        runtimes[s_i] = t3 - t2
        complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))
        feasibility_errors[s_i] = np.linalg.norm(np.diag(X_val) - 1) ** 2
        dual_feas = Z + C - np.diag(y)
        dual_feasibility_errors[s_i] = np.sum(dual_feas**2)
        num_iters[s_i] = prob.solver_stats.extra_stats["info"]["iter"]

    # Prepare dummy arrays for missing metrics to match the signature
    ranksX = np.zeros((1, num_seeds, 1))
    ranksY = np.zeros((1, num_seeds, 1))
    ranksZ = np.zeros((1, num_seeds, 1))

    print(f"Number of failed seeds: {num_failed_seeds}")
    # Print summary (adapt as needed)
    print_results_summary(
        config, args,
        runtimes.reshape(1, -1), problem_creation_times.reshape(1, -1), num_iters.reshape(1, -1),
        feasibility_errors.reshape(1, -1), dual_feasibility_errors.reshape(1, -1), complementary_slackness.reshape(1, -1),
        ranksX, ranksY, ranksZ,
        ranksT=None,
        memory=memory.reshape(1, -1) if args.track_mem else None
    )