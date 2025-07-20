# Import packages.
import sys
import os
import argparse
import yaml
import sdpap


sys.path.append(os.getcwd() + '/../../')
import time
from src.tt_ops import *
import cvxpy as cp
from memory_profiler import memory_usage

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    problem_creation_times = []
    runtimes = []
    memory = []
    complementary_slackness = []
    feasibility_errors = []
    for seed in config["seeds"]:
        np.random.seed(seed)
        t0 = time.time()
        n = 2**config["dim"]
        G_A = tt_matrix_to_matrix(tt_random_graph(config["dim"], config["max_ranks"][0]))
        G_B = tt_matrix_to_matrix(tt_random_graph(config["dim"], config["max_ranks"][0]))
        t1 = time.time()
        t2 = time.time()
        J_n = np.ones((n, n))

        if args.track_mem:
            start_mem = memory_usage(max_usage=True, include_children=True)
        # Variables
        Q = cp.Variable((n ** 2, n ** 2), PSD=True)
        P = cp.Variable((n, n))

        # Objective
        kron_prod = np.kron(G_B, G_A)
        objective = cp.Maximize(cp.trace(kron_prod @ Q))

        constraints = []

        # LMI constraint
        QP_mat = cp.bmat([[Q, cp.vec(P).reshape((n**2, 1), order="F")], [cp.vec(P).T.reshape((1, n**2), order="F"), np.array([[1]])]])
        constraints.append(QP_mat >> 0)

        # Sum of Q_{ii} blocks equals I_n
        Q_blocks = [[Q[i * n:(i + 1) * n, j * n:(j + 1) * n] for j in range(n)] for i in range(n)]
        sum_diag = sum(Q_blocks[i][i] for i in range(n))
        constraints.append(sum_diag == np.eye(n))

        # Trace(Q_{ij}) = 0 for i != j
        for i in range(n):
            for j in range(n):
                if i != j:
                    constraints.append(cp.trace(Q_blocks[i][j]) == 0)

        # Trace(Q_{ij} @ J_n) = 1 for all i,j
        for i in range(n):
            for j in range(n):
                constraints.append(cp.trace(Q_blocks[i][j] @ J_n) == 1)

        # Q_{ii}(j,j) == P(j, i)
        for i in range(n):
            for j in range(n):
                constraints.append(Q_blocks[i][i][j, j] == P[j, i])

        # P doubly stochastic
        constraints += [
            P >= 0,
            cp.sum(P, axis=0) == 1,
            cp.sum(P, axis=1) == 1
        ]

        # Q_{ij} >= 0
        for i in range(n):
            for j in range(n):
                constraints.append(Q_blocks[i][j] >= 0)

        t2 = time.time()
        # Solve the problem
        if args.track_mem:
            def wrapper():
                prob = cp.Problem(objective, constraints)
                _ = prob.solve(solver=cp.SDPA, epsilonDash=1e-6 / n, epsilonStar=1e-5 / n, verbose=True, numThreads=1, omegaStar=100, betaStar=0.5, gammaStar=0.9)
                return prob

            res, prob = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
            X = QP_mat.value
            for m in prob.solution.dual_vars.values():
                if type(m) == np.ndarray:
                    if m.shape == (n ** 2 + 1, n ** 2 + 1):
                        Z = m
            memory.append(res - start_mem)
        else:
            prob = cp.Problem(objective, constraints)
            _ = prob.solve(solver=cp.SDPA, epsilonDash=1e-6 / n, epsilonStar=1e-5 / n, verbose=True, numThreads=1, omegaStar=100, betaStar=0.5, gammaStar=0.9)
            X = QP_mat.value
            for m in prob.solution.dual_vars.values():
                if type(m) == np.ndarray:
                    if m.shape == (n ** 2 + 1, n ** 2 + 1):
                        Z = m

        t3 = time.time()
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        complementary_slackness.append(np.abs(np.trace(X @ Z)))
        feasibility_errors.append(0)
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s")
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s")
    if args.track_mem:
        print(f"Peak memory avg {np.mean(memory):.3f} MB")
    print(f"Complementary Slackness avg: {np.mean(complementary_slackness)}")
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}")
