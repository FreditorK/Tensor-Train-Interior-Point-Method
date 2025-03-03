import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from psd_system.stable_set.max_stable_set import Config
from src.baselines import cgal


if __name__ == "__main__":
    np.random.seed(Config.seed)
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    t0 = time.time()
    G = tt_rank_reduce(tt_random_graph(Config.dim, Config.max_rank))
    adj_matrix = np.round(tt_matrix_to_matrix(G), decimals=1)

    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    constraint_matrices = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            A = np.zeros_like(adj_matrix)
            A[i, j] = adj_matrix[i, j]
            constraint_matrices.append(A)
    bias = np.zeros((len(adj_matrix)**2, 1))
    J = np.ones_like(adj_matrix)
    t2 = time.time()
    X, duality_gaps = cgal(-J, constraint_matrices, bias, (1, 1), num_iter=100)
    t3 = time.time()
    print(np.round(X, decimals=4))
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {np.trace(J.T @ X)}")
    print(f"Total feasibility error: {np.sum([np.abs(np.trace(c.T @  X)-b) for c, b in zip(constraint_matrices, bias)])}")