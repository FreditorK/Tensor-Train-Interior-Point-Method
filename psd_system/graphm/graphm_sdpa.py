# Import packages.
import sys
import os
import time
import numpy as np
import yaml
import argparse
import sdpap
import cvxpy as cp

sys.path.append(os.getcwd() + '/../../')
from src.tt_ops import *
from memory_profiler import memory_usage
from src.utils import print_results_summary

import warnings
warnings.filterwarnings("ignore", message=".*Python recalculation of primal and/or dual feasibility error failed.*")

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
    # num_iters = np.zeros(num_seeds)  # If available

    for s_i, seed in enumerate(config["seeds"]):
        for attempt in range(1):  # At most two tries: original and one new random seed
            if attempt == 0:
                current_seed = seed
            else:
                current_seed = np.random.randint(0, 10000)
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)
            t1 = time.time()
            n = 2**config["dim"]
            G_A = tt_matrix_to_matrix(tt_random_graph(config["dim"], 1))
            G_B = tt_matrix_to_matrix(tt_random_graph(config["dim"], 1))
            t2 = time.time()
            J_n = np.ones((n, n))
            if args.track_mem:
                start_mem = memory_usage(max_usage=True, include_children=True)
            Q = cp.Variable((n ** 2, n ** 2), symmetric=True)
            P = cp.Variable((n, n))
            kron_prod = np.kron(G_B, G_A)
            objective = cp.Maximize(cp.trace(kron_prod @ Q))
            constraints = []
            QP_mat = cp.bmat([[Q, cp.vec(P, order="F").reshape((n**2, 1), order="F")], [cp.vec(P, order="F").T.reshape((1, n**2), order="F"), np.array([[1]])]])
            constraints.append(QP_mat >> 0)
            Q_blocks = [[Q[i * n:(i + 1) * n, j * n:(j + 1) * n] for j in range(n)] for i in range(n)]
            sum_diag = sum(Q_blocks[i][i] for i in range(n))
            constraints.append(sum_diag == np.eye(n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        constraints.append(cp.trace(Q_blocks[i][j]) == 0)
            for i in range(n):
                for j in range(n):
                    constraints.append(cp.trace(Q_blocks[i][j] @ J_n) == 1)
            for i in range(n):
                for j in range(n):
                    constraints.append(Q_blocks[i][i][j, j] == P[j, i])
            constraints += [cp.sum(P, axis=0) == 1, cp.sum(P, axis=1) == 1, P >= 0]
            for i in range(n):
                for j in range(n):
                    constraints.append(Q_blocks[i][j] >= 0)
            try:
                if args.track_mem:
                    def wrapper():
                        try:
                            prob = cp.Problem(objective, constraints)
                            _ = prob.solve(solver=cp.SDPA, epsilonDash=1e-6 / n, epsilonStar=1e-5 / n, verbose=True, gammaStar=0.9)
                        except:
                            pass
                        return prob
                    res, prob = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    memory[s_i] = res - start_mem
                    X_val = QP_mat.value
                else:
                    try:
                        prob = cp.Problem(objective, constraints)
                        _ = prob.solve(solver=cp.SDPA, epsilonDash=1e-6 / n, epsilonStar=1e-5 / n, verbose=True, gammaStar=0.9)
                    except:
                        pass
                    X_val = QP_mat.value
                # Extract duals (if available)
                Z = None
                for m in prob.solution.dual_vars.values():
                    if isinstance(m, np.ndarray):
                        if 1 <= np.prod(m.shape) <= n ** 2 + 1 and len(m.shape) == 1:
                            y = m
                Z = constraints[0].dual_value
                data, chain, inverse_data = prob.get_problem_data(cp.SDPA)
                soln = chain.solve_via_data(prob, data)
                # unpacks the solution returned by SCS into `problem`
                prob.unpack_results(soln, chain, inverse_data)
                t3 = time.time()
                problem_creation_times[s_i] = t2 - t1
                runtimes[s_i] = t3 - t2
                if Z is not None:
                    complementary_slackness[s_i] = np.abs(np.trace(X_val @ Z))
                else:
                    complementary_slackness[s_i] = np.nan
                # Feasibility error: (sum_diag - I_n)^2 + ... (customize as needed)
                feasibility_errors[s_i] = sum([np.sum(c.residual**2) for c in constraints[1:]])  # Placeholder, customize if needed
                dual_feas_sq = ((data["c"].flatten() - (data["A"].T @ np.concatenate([soln["eq_dual"], soln["ineq_dual"]], axis=0)).flatten()))**2
                dual_feas_diag_sq = dual_feas_sq[[sum(range(i)) for i in range(n**2+1)]]
                for idx, i in enumerate([[sum(range(i)) for i in range(n**2+1)]]):
                    dual_feas_sq -= 0.5*dual_feas_diag_sq[idx]
                # SDPA only stores the lower tri bits of symmetric variables, to make it fair we adjust the error
                dual_feasibility_errors[s_i] = np.sum(2*dual_feas_sq)
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

    # Prepare dummy arrays for missing metrics to match the signature
    num_iters = np.zeros(num_seeds)
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
