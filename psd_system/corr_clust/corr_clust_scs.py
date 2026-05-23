import argparse
import os
import sys
import time

import numpy as np
import yaml
from memory_profiler import memory_usage

sys.path.append(os.getcwd() + '/../../')
from corr_clust import *
from src.utils import print_results_summary
from psd_system.direct_conic import scs_row_from_entries, solve_scs_psd_max


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
    num_iters = np.zeros(num_seeds)

    for s_i, seed in enumerate(config["seeds"]):
        for attempt in range(1):
            current_seed = seed if attempt == 0 else np.random.randint(0, 10000)
            if attempt > 0:
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)

            t1 = time.time()
            C, ineq_A = tt_obj_matrix_and_ineq_mask(args.rank, config["dim"])
            C = tt_matrix_to_matrix(C)
            ineq_A = np.round(tt_matrix_to_matrix(ineq_A), decimals=1)
            t2 = time.time()
            if args.track_mem:
                start_mem = memory_usage(max_usage=True, include_children=True)

            n = C.shape[0]
            eq_rows = [scs_row_from_entries(n, [(i, i, 1.0)]) for i in range(n)]
            eq_rhs = np.ones(n)

            ineq_entries = [
                (i, j, ineq_A[i, j])
                for i in range(n)
                for j in range(n)
                if abs(ineq_A[i, j]) > 1e-12
            ]
            ineq_rows = [scs_row_from_entries(n, [entry]) for entry in ineq_entries]
            ineq_rhs = np.zeros(len(ineq_rows))

            try:
                if args.track_mem:
                    def wrapper():
                        return solve_scs_psd_max(
                            C,
                            eq_rows,
                            eq_rhs,
                            ineq_rows=ineq_rows,
                            ineq_rhs=ineq_rhs,
                            eps=1e-5 / config["dim"],
                            verbose=True,
                        )

                    res_mem, sol = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    memory[s_i] = res_mem - start_mem
                else:
                    sol = solve_scs_psd_max(
                        C,
                        eq_rows,
                        eq_rhs,
                        ineq_rows=ineq_rows,
                        ineq_rhs=ineq_rhs,
                        eps=1e-5 / config["dim"],
                        verbose=True,
                    )
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

        t3 = time.time()
        problem_creation_times[s_i] = t2 - t1
        runtimes[s_i] = t3 - t2
        complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))

        diag_err = np.linalg.norm(np.diag(X_val) - 1.0) ** 2
        ineq_vals = ineq_A * X_val
        ineq_violation = np.minimum(ineq_vals, 0.0)
        feasibility_errors[s_i] = diag_err + np.sum(ineq_violation ** 2)

        y_full = np.asarray(sol["sol"]["y"]).reshape(-1)
        dual_residual = sol["A"].T @ y_full + sol["c"]
        dual_feasibility_errors[s_i] = np.sum(dual_residual ** 2)
        num_iters[s_i] = sol["sol"]["info"].get("iter", 0)

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
