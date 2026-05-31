# Import packages.
import argparse
import os
import sys
import time

import numpy as np
import yaml


sys.path.append(os.getcwd() + '/../../')
from maxcut import *
from psd_system.direct_conic import require_sdpa_optimal, sdpa_dual_error, sdpa_duality_gap, sdpa_num_iters, sdpa_row_from_entries, sdpa_solver_options, solve_sdpa_psd_max
from src.utils import print_results_summary, run_isolated

import warnings

warnings.filterwarnings("ignore", message=".*Python recalculation of primal and/or dual feasibility error failed.*")


def _diag_eq_rows_sdpa(size):
    eq_rows = [sdpa_row_from_entries(size, [(i, i, 1.0)]) for i in range(size)]
    eq_rhs = np.ones(size)
    return eq_rows, eq_rhs


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
    problem_creation_times = np.full(num_seeds, np.nan)
    runtimes = np.full(num_seeds, np.nan)
    memory = np.full(num_seeds, np.nan)
    complementary_slackness = np.full(num_seeds, np.nan)
    feasibility_errors = np.full(num_seeds, np.nan)
    dual_feasibility_errors = np.full(num_seeds, np.nan)
    num_iters = np.full(num_seeds, np.nan)
    num_failed_seeds = 0

    for s_i, seed in enumerate(config["seeds"]):
        for attempt in range(3):
            if attempt == 0:
                current_seed = seed
            else:
                current_seed = np.random.randint(0, 10000)
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)

            try:
                def build_and_solve():
                    t1 = time.time()
                    C = tt_matrix_to_matrix(tt_obj_matrix(args.rank, config["dim"]))
                    n = C.shape[0]
                    eq_rows, eq_rhs = _diag_eq_rows_sdpa(n)
                    option = sdpa_solver_options(config, gamma_star=0.9, domain_method="basis")
                    t2 = time.time()
                    result = solve_sdpa_psd_max(C, eq_rows, eq_rhs, option=option)
                    require_sdpa_optimal(result)
                    t3 = time.time()
                    X_val = result["x_matrix"]
                    return {
                        "problem_creation_time": t2 - t1,
                        "runtime": t3 - t2,
                        "complementary_slackness": sdpa_duality_gap(result),
                        "feasibility_error": np.linalg.norm(np.diag(X_val) - 1) ** 2,
                        "dual_feasibility_error": sdpa_dual_error(result),
                        "num_iters": sdpa_num_iters(result),
                    }

                mem_delta, metrics = run_isolated(build_and_solve, track_mem=args.track_mem)
                if args.track_mem:
                    memory[s_i] = mem_delta
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

        problem_creation_times[s_i] = metrics["problem_creation_time"]
        runtimes[s_i] = metrics["runtime"]
        complementary_slackness[s_i] = metrics["complementary_slackness"]
        feasibility_errors[s_i] = metrics["feasibility_error"]
        dual_feasibility_errors[s_i] = metrics["dual_feasibility_error"]
        num_iters[s_i] = metrics["num_iters"]

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
