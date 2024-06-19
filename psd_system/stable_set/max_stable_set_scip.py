import sys
import os
import time
import cvxpy as cp

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from psd_system.stable_set.max_stable_set import Config


if __name__ == "__main__":
    np.random.seed(Config.seed)
    t0 = time.time()
    G = tt_random_graph(Config.ranks)
    adj_matrix = tt_op_to_matrix(G)
    adj_matrix_comp = 0.5*(-adj_matrix + 1)
    adj_matrix = 0.5*(adj_matrix + 1)
    adj_matrix[0] = 0
    adj_matrix[:, 0] = 0
    adj_matrix_comp[:, 0] = 1
    adj_matrix_comp[0, :] = 1
    adj_matrix_comp[0, 0] = 0
    B = np.eye(adj_matrix_comp.shape[0])
    B[0, 0] = 0
    B[1:, 0] = 1
    B[0, 1:] = 1

    print(np.round(adj_matrix_comp, decimals=2))
    print(np.round(adj_matrix, decimals=2))

    """
    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    X = cp.Variable(adj_matrix.shape, symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.multiply(adj_matrix_comp, X) == B]
    t2 = time.time()
    prob = cp.Problem(cp.Minimize(X[0, 0]), constraints)
    prob.solve()
    t3 = time.time()
    print(np.round(X.value, decimals=2))
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {prob.value}")
    #nodes_in_cut = [i for i, v in enumerate(X.value[1, 1:]) if v > 0.01]
    #plot_maxcut(adj_matrix[1:, 1:], nodes_in_cut, [])
    """