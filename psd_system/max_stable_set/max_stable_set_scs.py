import sys
import os
import time

import numpy as np
import yaml
import argparse

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from memory_profiler import memory_usage
import cvxpy as cp
from src.utils import print_results_summary
import warnings
warnings.filterwarnings("ignore", message=".*Python recalculation of primal and/or dual feasibility error failed.*")


def safe_dual_term(constr, var):
    dual_val = constr.dual_value
    if dual_val is None:
        return 0
    try:
        grad_dict = constr.args[0].gradient
        if var in grad_dict:
            return grad_dict[var].T @ dual_val
    except:
        pass
    return 0  # fallback if gradient missing or fails

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
        for attempt in range(1):  # At most two tries: original and one new random seed
            if attempt == 0:
                current_seed = seed
            else:
                current_seed = np.random.randint(0, 10000)
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)
            t1 = time.time()
            G = tt_rank_reduce(tt_random_graph(config["dim"], 1))
            adj_matrix = np.round(tt_matrix_to_matrix(G), decimals=1)
            t2 = time.time()
            J = np.ones_like(adj_matrix)
            if args.track_mem:
                start_mem = memory_usage(max_usage=True, include_children=True)
            X = cp.Variable(J.shape, symmetric=True)
            constraints = [X >> 0]
            constraints += [cp.diag(cp.vec(adj_matrix, order="F").flatten(order="F")) @ cp.vec(X, order="F") == 0]
            constraints += [cp.trace(X) == 1]
            try:
                if args.track_mem:
                    def wrapper():
                        try:
                            prob = cp.Problem(cp.Maximize(cp.trace(J.T @ X)), constraints)
                            _ = prob.solve(solver=cp.SCS, eps=1e-5 / config["dim"]) # needed to reduce for stability
                        except:
                            pass
                        return prob
                    res, prob = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    memory[s_i] = res - start_mem
                    X_val = X.value
                else:
                    try:
                        prob = cp.Problem(cp.Maximize(cp.trace(J.T @ X)), constraints)
                        _ = prob.solve(solver=cp.SCS, eps=1e-5 / config["dim"]) # needed to reduce for stability
                    except:
                        pass
                    X_val = X.value
                Z = constraints[0].dual_value
                y_1 = constraints[1].dual_value
                y_2 = constraints[2].dual_value
                t3 = time.time()
                problem_creation_times[s_i] = t2 - t1
                runtimes[s_i] = t3 - t2
                complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))
                # Feasibility error: (trace(X)-1)^2 + norm(diag(adj_matrix) @ X)^2
                feasibility_errors[s_i] = (np.trace(X_val)-1)**2 + np.linalg.norm(np.diag(adj_matrix.reshape(-1, 1, order="F").flatten()) @ X_val.reshape(-1, 1, order="F"))**2
                dual_feas = Z + J - y_2*np.eye(len(Z)) - (np.diag(adj_matrix.flatten()) @ y_1).reshape(*Z.shape, order="F")
                dual_feasibility_errors[s_i] = np.sum(dual_feas**2)
                num_iters[s_i] = prob.solver_stats.extra_stats["info"]["iter"]
                break  # Success, break out of attempt loop
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

    # Prepare dummy arrays for missing metrics to match the signature
    ranksX = np.zeros((1, num_seeds, 1))
    ranksY = np.zeros((1, num_seeds, 1))
    ranksZ = np.zeros((1, num_seeds, 1))

    # Print summary (adapt as needed)
    print(f"Number of failed seeds: {num_failed_seeds}")
    print_results_summary(
        config, args,
        runtimes.reshape(1, -1), problem_creation_times.reshape(1, -1), num_iters.reshape(1, -1),
        feasibility_errors.reshape(1, -1), dual_feasibility_errors.reshape(1, -1), complementary_slackness.reshape(1, -1),
        ranksX, ranksY, ranksZ,
        ranksT=None,
        memory=memory.reshape(1, -1) if args.track_mem else None
    )