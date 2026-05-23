import argparse
import os
import sys
import time

import numpy as np
import yaml
from memory_profiler import memory_usage

sys.path.append(os.getcwd() + '/../../')
from maxcut import *
from psd_system.direct_conic import sdpa_row_from_entries, solve_sdpa_psd_max
from src.utils import print_results_summary


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--rank", type=int, default=1, help="TT-rank used for graph generation")
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

    for s_i, seed in enumerate(config["seeds"]):
        for attempt in range(3):
            current_seed = seed if attempt == 0 else np.random.randint(0, 10000)
            if attempt > 0:
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)

            t1 = time.time()
            C = tt_matrix_to_matrix(tt_obj_matrix(args.rank, config["dim"]))
            t2 = time.time()
            if args.track_mem:
                start_mem = memory_usage(max_usage=True, include_children=True)

            n = C.shape[0]
            eq_rows = [sdpa_row_from_entries(n, [(i, i, 1.0)]) for i in range(n)]
            eq_rhs = np.ones(n)

            try:
                option = {
                    "epsilonDash": 1e-6 / (2 ** config["dim"]),
                    "epsilonStar": 1e-5 / (2 ** config["dim"]),
                    "print": "display",
                }
                if args.track_mem:
                    def wrapper():
                        return solve_sdpa_psd_max(C, eq_rows, eq_rhs, option=option)

                    res_mem, sol = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    memory[s_i] = res_mem - start_mem
                else:
                    sol = solve_sdpa_psd_max(C, eq_rows, eq_rhs, option=option)
                break
            except Exception as e:
                print(e)
                if attempt == 0:
                    print(f"Failed to solve problem with config seed {seed}, trying a new random seed...")
                else:
                    print(f"Failed to solve problem with new random seed {current_seed}")
                    num_failed_seeds += 1
        else:
            continue

        X_val = sol["x_matrix"]
        Z = sol["z_matrix"]
        y = sol["y_eq"]

        t3 = time.time()
        problem_creation_times[s_i] = t2 - t1
        runtimes[s_i] = t3 - t2
        complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))
        feasibility_errors[s_i] = np.linalg.norm(np.diag(X_val) - 1) ** 2
        dual_feas = Z + C + np.diag(y)
        dual_feasibility_errors[s_i] = np.sum(dual_feas ** 2)

    num_iters = np.zeros(num_seeds)
    ranksX = np.zeros((1, num_seeds, 1))
    ranksY = np.zeros((1, num_seeds, 1))
    ranksZ = np.zeros((1, num_seeds, 1))

    print(f"Number of failed seeds: {num_failed_seeds}")
    print_results_summary(
        config,
        args,
        runtimes.reshape(1, -1),
        problem_creation_times.reshape(1, -1),
        num_iters.reshape(1, -1),
        feasibility_errors.reshape(1, -1),
        dual_feasibility_errors.reshape(1, -1),
        complementary_slackness.reshape(1, -1),
        ranksX,
        ranksY,
        ranksZ,
        ranksT=None,
        memory=memory.reshape(1, -1) if args.track_mem else None,
    )
