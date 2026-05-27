# Import packages.
import argparse
import os
import sys
import time

import numpy as np
import yaml
from memory_profiler import memory_usage


sys.path.append(os.getcwd() + '/../../')
from src.tt_ops import *
from psd_system.direct_conic import require_sdpa_optimal, sdpa_dual_error, sdpa_duality_gap, sdpa_row_from_entries, sdpa_solver_options, solve_sdpa_psd_max
from src.utils import print_results_summary

import warnings

warnings.filterwarnings("ignore", message=".*Python recalculation of primal and/or dual feasibility error failed.*")


def _build_constraints_graphm_sdpa(n):
    q_size = n * n
    size = q_size + 1
    last = q_size

    eq_rows = []
    eq_rhs = []
    ineq_rows = []
    ineq_rhs = []

    def add_eq(entries, rhs):
        eq_rows.append(sdpa_row_from_entries(size, entries))
        eq_rhs.append(float(rhs))

    def add_ge(entries, rhs=0.0):
        ineq_rows.append(sdpa_row_from_entries(size, entries))
        ineq_rhs.append(float(rhs))

    add_eq([(last, last, 1.0)], 1.0)

    for a in range(n):
        for b in range(n):
            entries = [(i * n + a, i * n + b, 1.0) for i in range(n)]
            add_eq(entries, 1.0 if a == b else 0.0)

    for i in range(n):
        for j in range(n):
            if i != j:
                entries = [(i * n + a, j * n + a, 1.0) for a in range(n)]
                add_eq(entries, 0.0)

    for i in range(n):
        for j in range(n):
            entries = [(i * n + a, j * n + b, 1.0) for a in range(n) for b in range(n)]
            add_eq(entries, 1.0)

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            add_eq([(idx, idx, 1.0), (idx, last, -1.0)], 0.0)

    for i in range(n):
        entries = [(j + i * n, last, 1.0) for j in range(n)]
        add_eq(entries, 1.0)

    for j in range(n):
        entries = [(j + i * n, last, 1.0) for i in range(n)]
        add_eq(entries, 1.0)

    for i in range(n):
        for j in range(n):
            idx = j + i * n
            add_ge([(idx, last, 1.0)], 0.0)

    for r in range(q_size):
        for c in range(r + 1):
            add_ge([(r, c, 1.0)], 0.0)

    return (
        eq_rows,
        np.asarray(eq_rhs),
        ineq_rows,
        np.asarray(ineq_rhs),
    )


def _feasibility_error(X, n):
    q_size = n * n
    last = q_size
    Q = X[:q_size, :q_size]
    p = X[:q_size, last]

    err = 0.0

    err += (X[last, last] - 1.0) ** 2

    for a in range(n):
        for b in range(n):
            val = 0.0
            for i in range(n):
                val += Q[i * n + a, i * n + b]
            target = 1.0 if a == b else 0.0
            err += (val - target) ** 2

    for i in range(n):
        for j in range(n):
            if i != j:
                val = 0.0
                for a in range(n):
                    val += Q[i * n + a, j * n + a]
                err += val ** 2

    for i in range(n):
        for j in range(n):
            val = 0.0
            for a in range(n):
                for b in range(n):
                    val += Q[i * n + a, j * n + b]
            err += (val - 1.0) ** 2

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            err += (Q[idx, idx] - p[idx]) ** 2

    for i in range(n):
        val = 0.0
        for j in range(n):
            val += p[j + i * n]
        err += (val - 1.0) ** 2

    for j in range(n):
        val = 0.0
        for i in range(n):
            val += p[j + i * n]
        err += (val - 1.0) ** 2

    err += np.sum(np.minimum(p, 0.0) ** 2)
    err += np.sum(np.minimum(Q, 0.0) ** 2)

    return float(err)


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
    num_failed_seeds = 0

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
                    n = 2 ** config["dim"]
                    G_A = tt_matrix_to_matrix(tt_random_graph(config["dim"], args.rank))
                    G_B = tt_matrix_to_matrix(tt_random_graph(config["dim"], args.rank))
                    kron_prod = np.kron(G_B, G_A)
                    q_size = n * n
                    c_matrix = np.zeros((q_size + 1, q_size + 1))
                    c_matrix[:q_size, :q_size] = kron_prod

                    eq_rows, eq_rhs, ineq_rows, ineq_rhs = _build_constraints_graphm_sdpa(n)
                    option = sdpa_solver_options(config, gamma_star=0.9, domain_method=None)
                    t2 = time.time()

                    result = solve_sdpa_psd_max(
                        c_matrix,
                        eq_rows,
                        eq_rhs,
                        ineq_rows=ineq_rows,
                        ineq_rhs=ineq_rhs,
                        option=option,
                    )
                    require_sdpa_optimal(result)
                    t3 = time.time()
                    return n, result, t2 - t1, t3 - t2

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

                n, result, problem_creation_time, runtime = payload
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
        complementary_slackness[s_i] = sdpa_duality_gap(result)
        feasibility_errors[s_i] = _feasibility_error(X_val, n)
        dual_feasibility_errors[s_i] = sdpa_dual_error(result)

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
