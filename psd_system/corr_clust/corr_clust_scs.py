import argparse
import os
import sys
import time

import numpy as np
import yaml
from memory_profiler import memory_usage


sys.path.append(os.getcwd() + '/../../')
from corr_clust import *
from psd_system.direct_conic import scs_row_from_entries, solve_scs_psd_max
from src.utils import print_results_summary


def _build_constraints_scs(ineq_A):
    n = ineq_A.shape[0]
    eq_rows = [scs_row_from_entries(n, [(i, i, 1.0)]) for i in range(n)]
    eq_rhs = np.ones(n)

    ineq_rows = []
    ineq_rhs = []
    for i in range(n):
        for j in range(i, n):
            coef = float(ineq_A[i, j])
            if coef != 0.0:
                # SCS uses Ax + s = b, s >= 0 => Ax <= b. Encode coef*X_ij >= 0 as -coef*X_ij <= 0.
                ineq_rows.append(scs_row_from_entries(n, [(i, j, -coef)]))
                ineq_rhs.append(0.0)
    return eq_rows, np.asarray(eq_rhs), ineq_rows, np.asarray(ineq_rhs)


def _feasibility_error(X, ineq_A):
    diag_err = np.linalg.norm(np.diag(X) - 1.0) ** 2
    ineq_vals = ineq_A * X
    ineq_violation = np.minimum(ineq_vals, 0.0)
    return diag_err + np.sum(ineq_violation ** 2)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--rank", type=int, default=1, help="Graph/objective TT rank")
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
            if attempt == 0:
                current_seed = seed
            else:
                current_seed = np.random.randint(0, 10000)
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)

            if args.track_mem:
                # Baseline before setup so peak delta includes objective/constraint build and solve.
                start_mem = memory_usage(max_usage=True, include_children=True)

            try:
                def build_and_solve():
                    t1 = time.time()
                    C, ineq_A = tt_obj_matrix_and_ineq_mask(args.rank, config["dim"])
                    C = tt_matrix_to_matrix(C)
                    ineq_A = np.round(tt_matrix_to_matrix(ineq_A), decimals=1)
                    eq_rows, eq_rhs, ineq_rows, ineq_rhs = _build_constraints_scs(ineq_A)
                    t2 = time.time()
                    result = solve_scs_psd_max(
                        C,
                        eq_rows,
                        eq_rhs,
                        ineq_rows=ineq_rows,
                        ineq_rhs=ineq_rhs,
                        eps=1e-5 / config["dim"],
                        verbose=True,
                    )
                    t3 = time.time()
                    return C, ineq_A, result, t2 - t1, t3 - t2

                if args.track_mem:
                    peak_mem, payload = memory_usage(
                        proc=build_and_solve,
                        max_usage=True,
                        retval=True,
                        include_children=True,
                    )
                    memory[s_i] = peak_mem - start_mem
                else:
                    payload = build_and_solve()

                C, ineq_A, result, problem_creation_time, runtime = payload
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

        X_val = result["x_matrix"]
        Z = result["z_matrix"]

        problem_creation_times[s_i] = problem_creation_time
        runtimes[s_i] = runtime
        complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))
        feasibility_errors[s_i] = _feasibility_error(X_val, ineq_A)

        info = result.get("sol", {}).get("info", {})
        dual_feasibility_errors[s_i] = float(info.get("res_dual", np.nan)) ** 2
        num_iters[s_i] = float(info.get("iter", 0))

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
