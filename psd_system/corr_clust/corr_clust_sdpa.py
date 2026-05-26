import argparse
import os
import sys
import time

import numpy as np
import yaml
from memory_profiler import memory_usage


sys.path.append(os.getcwd() + '/../../')
from corr_clust import *
from psd_system.direct_conic import sdpa_row_from_entries, solve_sdpa_psd_max
from src.utils import print_results_summary

import warnings

warnings.filterwarnings("ignore", message=".*Python recalculation of primal and/or dual feasibility error failed.*")


def _build_constraints_sdpa(ineq_A):
    n = ineq_A.shape[0]
    eq_rows = [sdpa_row_from_entries(n, [(i, i, 1.0)]) for i in range(n)]
    eq_rhs = np.ones(n)

    ineq_rows = []
    ineq_rhs = []
    for i in range(n):
        for j in range(i, n):
            coef = float(ineq_A[i, j])
            if coef != 0.0:
                # SDPA direct form uses Ax - b in l-cone => Ax >= b.
                ineq_rows.append(sdpa_row_from_entries(n, [(i, j, coef)]))
                ineq_rhs.append(0.0)
    return eq_rows, np.asarray(eq_rhs), ineq_rows, np.asarray(ineq_rhs)


def _feasibility_error(X, ineq_A):
    diag_err = np.linalg.norm(np.diag(X) - 1.0) ** 2
    ineq_vals = ineq_A * X
    ineq_violation = np.minimum(ineq_vals, 0.0)
    return diag_err + np.sum(ineq_violation ** 2)


def _sdpa_dual_error(result):
    info = result.get("sdpapinfo", {})
    if isinstance(info, dict):
        for key in ("dualError", "dual_error", "err_d"):
            if key in info:
                try:
                    return float(info[key]) ** 2
                except Exception:
                    pass
    y_full = np.concatenate([result["y_eq"], result["y_ineq"]])
    dual_res = result["c"] - result["A"].T @ y_full - result["z_matrix"].reshape(-1, order="F")
    return float(np.sum(np.asarray(dual_res).reshape(-1) ** 2))


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

    for s_i, seed in enumerate(config["seeds"]):
        for attempt in range(3):
            if attempt == 0:
                current_seed = seed
            else:
                current_seed = np.random.randint(0, 10000)
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)

            if args.track_mem:
                # Baseline before building data so tracked memory includes matrices/constraints too.
                start_mem = memory_usage(max_usage=True, include_children=True)

            t1 = time.time()
            C, ineq_A = tt_obj_matrix_and_ineq_mask(args.rank, config["dim"])
            C = tt_matrix_to_matrix(C)
            ineq_A = np.round(tt_matrix_to_matrix(ineq_A), decimals=1)
            eq_rows, eq_rhs, ineq_rows, ineq_rhs = _build_constraints_sdpa(ineq_A)
            option = {
                "print": "display",
                "epsilonDash": 1e-6 / (2 ** config["dim"]),
                "epsilonStar": 1e-5 / (2 ** config["dim"]),
                "gammaStar": 0.75,
                "domainMethod": "basis",
            }
            t2 = time.time()


            try:
                if args.track_mem:
                    def wrapper():
                        return solve_sdpa_psd_max(
                            C,
                            eq_rows,
                            eq_rhs,
                            ineq_rows=ineq_rows,
                            ineq_rhs=ineq_rhs,
                            option=option,
                        )

                    res, result = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    memory[s_i] = res - start_mem
                else:
                    result = solve_sdpa_psd_max(
                        C,
                        eq_rows,
                        eq_rhs,
                        ineq_rows=ineq_rows,
                        ineq_rhs=ineq_rhs,
                        option=option,
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

        X_val = result["x_matrix"]
        Z = result["z_matrix"]
        t3 = time.time()

        problem_creation_times[s_i] = t2 - t1
        runtimes[s_i] = t3 - t2
        complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))
        feasibility_errors[s_i] = _feasibility_error(X_val, ineq_A)
        dual_feasibility_errors[s_i] = _sdpa_dual_error(result)

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
