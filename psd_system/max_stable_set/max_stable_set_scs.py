import argparse
import os
import sys
import time

import numpy as np
import yaml
from memory_profiler import memory_usage


sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from psd_system.direct_conic import scs_row_from_entries, solve_scs_psd_max
from src.utils import print_results_summary

import warnings

warnings.filterwarnings("ignore", message=".*Python recalculation of primal and/or dual feasibility error failed.*")


def _build_constraints_scs(adj_matrix):
    n = adj_matrix.shape[0]

    eq_rows = []
    eq_rhs = []
    for i in range(n):
        for j in range(i, n):
            coef = float(adj_matrix[i, j])
            if coef != 0.0:
                eq_rows.append(scs_row_from_entries(n, [(i, j, coef)]))
                eq_rhs.append(0.0)

    trace_entries = [(k, k, 1.0) for k in range(n)]
    eq_rows.append(scs_row_from_entries(n, trace_entries))
    eq_rhs.append(1.0)

    return eq_rows, np.asarray(eq_rhs), len(eq_rows) - 1


def _feasibility_error(X, adj_matrix):
    trace_err = (np.trace(X) - 1.0) ** 2
    adj_masked = adj_matrix.reshape(-1, order="F") * X.reshape(-1, order="F")
    return trace_err + np.linalg.norm(adj_masked) ** 2


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
                # Baseline before building data so tracked memory includes matrices/constraints too.
                start_mem = memory_usage(max_usage=True, include_children=True)

            t1 = time.time()
            G = tt_rank_reduce(tt_random_graph(config["dim"], args.rank))
            adj_matrix = np.round(tt_matrix_to_matrix(G), decimals=1)
            J = np.ones_like(adj_matrix)
            eq_rows, eq_rhs, adj_constraint_count = _build_constraints_scs(adj_matrix)
            t2 = time.time()


            try:
                if args.track_mem:
                    def wrapper():
                        return solve_scs_psd_max(
                            J,
                            eq_rows,
                            eq_rhs,
                            eps=1e-5 / config["dim"],
                            verbose=True,
                        )

                    res, result = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    memory[s_i] = res - start_mem
                else:
                    result = solve_scs_psd_max(
                        J,
                        eq_rows,
                        eq_rhs,
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

        X_val = result["x_matrix"]
        Z = result["z_matrix"]
        y_eq = result["y_eq"]
        y_adj = y_eq[:adj_constraint_count]
        y_trace = y_eq[adj_constraint_count] if y_eq.size > adj_constraint_count else 0.0
        t3 = time.time()

        problem_creation_times[s_i] = t2 - t1
        runtimes[s_i] = t3 - t2
        complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))
        feasibility_errors[s_i] = _feasibility_error(X_val, adj_matrix)

        # Aggregate adjacency duals into a matrix for a matrix-space residual.
        adj_dual_mat = np.zeros_like(adj_matrix, dtype=float)
        idx = 0
        for i in range(adj_matrix.shape[0]):
            for j in range(i, adj_matrix.shape[1]):
                coef = float(adj_matrix[i, j])
                if coef != 0.0:
                    val = coef * y_adj[idx]
                    adj_dual_mat[i, j] += val
                    if i != j:
                        adj_dual_mat[j, i] += val
                    idx += 1

        dual_feas = Z + J - y_trace * np.eye(adj_matrix.shape[0]) - adj_dual_mat
        dual_feasibility_errors[s_i] = np.sum(dual_feas ** 2)

        info = result.get("sol", {}).get("info", {})
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
