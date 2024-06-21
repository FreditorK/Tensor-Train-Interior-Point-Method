import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from psd_system.stable_set.max_stable_set import Config
from src.baselines import cgal


if __name__ == "__main__":
    np.random.seed(Config.seed)
    t0 = time.time()
    G = tt_random_graph(Config.ranks)
    adj_matrix = np.round(tt_op_to_matrix(G), decimals=1)
    adj_matrix = 0.5*(adj_matrix + 1)

    #print(np.round(adj_matrix_comp, decimals=2))
    #print(np.round(adj_matrix, decimals=2))

    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    constraint_matrices = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            A = np.zeros_like(adj_matrix)
            A[i, j] = adj_matrix[i, j]
            constraint_matrices.append(A)
    print(adj_matrix)
    bias = np.zeros((len(adj_matrix)**2, 1))
    J = np.ones_like(adj_matrix)
    t2 = time.time()
    X, duality_gaps = cgal(-J, constraint_matrices, bias, (1, 1), num_iter=100)
    t3 = time.time()
    print(np.round(X, decimals=2))
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {np.trace(J.T @ X)}")
    nodes_in_cut = [i for i, v in enumerate(X[0]) if v > 0.05]
    plot_duality_gaps(duality_gaps)
